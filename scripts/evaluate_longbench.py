"""Resumable, paired full LongBench evaluation with official task metrics."""
import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from kv_distill.engine import load_model
from flash_evict.quality_adapter import PairedController, default_methods, group_key, decode_batch
from flash_evict.streaming_adapter import chunk_mlp
from benchmark_policies import processes
from prepare_longbench import LB_ROOT, sha


def official_metrics():
    spec = importlib.util.spec_from_file_location('official_longbench_metrics', LB_ROOT/'metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['metrics'] = module
    spec = importlib.util.spec_from_file_location('official_longbench_eval', LB_ROOT/'eval.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    return evaluation


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def checkpoint(dest, description, push):
    subprocess.run([sys.executable, 'scripts/report_longbench.py', '--output', str(dest)], check=True)
    if push:
        subprocess.run(['git', 'add', '--', str(dest)], check=True)
        if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode:
            subprocess.run(['git', 'commit', '-m', 'flash-evict: '+description], check=True)
            subprocess.run(['git', 'push', 'origin', 'master'], check=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/longbench_full')
    parser.add_argument('--output', default='results/longbench_full')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--limit', type=int, help='Smoke test only; recorded in the protocol')
    parser.add_argument('--mlp-chunk-size', type=int, default=1024)
    parser.add_argument('--decode-batch-size', type=int, default=6)
    parser.add_argument('--require-idle', action='store_true')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    manifest = json.loads((data/'manifest.json').read_text())
    methods = default_methods()
    if args.methods:
        unknown = set(args.methods)-{m['name'] for m in methods}
        if unknown:
            raise ValueError(f'Unknown methods: {unknown}')
        methods = [m for m in methods if m['name'] in args.methods]
    tasks = args.tasks or list(manifest['tasks'])
    evaluation = official_metrics()
    paths = [Path(__file__), Path('flash_evict/quality_adapter.py'), Path('flash_evict/prefill.py'),
             Path('flash_evict/baselines.py'), Path('flash_evict/streaming_adapter.py'),
             Path('/home/qcb/kv-distill/kv_distill/attention.py'),
             Path('/home/qcb/kv-distill/kv_distill/engine.py')]
    protocol = dict(data_manifest_sha256=sha(data/'manifest.json'), model=manifest['model'],
                    model_revision=manifest['model_revision'], methods=methods, tasks=tasks, limit=args.limit,
                    source_sha256={str(p): sha(p) for p in paths}, seed=42,
                    mlp_chunk_size=args.mlp_chunk_size, decode_batch_size=args.decode_batch_size,
                    generation='Greedy, official per-task max_new_tokens and Samsum newline stop; model generation EOS IDs',
                    compression='Compress prefix excluding final prompt token, then forward final prompt token and decode; '
                    'absolute RoPE positions; budget per KV head in every layer; decode cache grows; '
                    'reserved suffix counts inside budget; uniform_recent is a pure suffix without sink tokens',
                    ablation_reference='recency_scores_only is the existing d01_r0_256 method (including value weighting)',
                    timing='Quality runs share prefill among compatible policies and batch decode; '
                    'shared timings and bank-inclusive allocator peaks are not standalone method latency or memory')
    protocol_path = dest/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Protocol or evaluation sources changed; choose a new output directory')
    else:
        write_json(protocol_path, protocol)
        write_json(dest/'data_manifest.json', manifest)
    protocol_hash = sha(protocol_path)
    raw, others = processes()
    if args.require_idle and others:
        raise RuntimeError(f'Other CUDA processes: {others}')
    torch.manual_seed(42)
    model, tokenizer = load_model()
    if args.mlp_chunk_size:
        chunk_mlp(model, args.mlp_chunk_size)
    ctl = PairedController(model)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    runtime = dict(started_unix=time.time(), pid=os.getpid(), torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(), git_revision=subprocess.check_output(
                       ['git', 'rev-parse', 'HEAD'], text=True).strip(), compute_processes_before=raw)
    write_json(dest/f'runtime_{os.getpid()}.json', runtime)
    try:
        for task in tasks:
            task_path = data/f'{task}.jsonl'
            if sha(task_path) != manifest['tasks'][task]['sha256']:
                raise ValueError(f'Data changed: {task}')
            rows = [json.loads(line) for line in task_path.read_text().splitlines()]
            if args.limit:
                rows = rows[:args.limit]
            done = {}
            for method in methods:
                path = dest/'pred'/method['name']/f'{task}.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                saved = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
                expected = {row['id'] for row in rows}
                ids = {r['id'] for r in saved}
                if len(ids) != len(saved) or not ids <= expected or any(r['protocol_sha256'] != protocol_hash for r in saved):
                    raise ValueError(f'Invalid checkpoint: {path}')
                done[method['name']] = ids
            for index, row in enumerate(rows):
                pending = [m for m in methods if row['id'] not in done[m['name']]]
                if not pending:
                    continue
                _, others = processes()
                if args.require_idle and others:
                    raise RuntimeError(f'Other CUDA processes: {others}')
                ids = torch.tensor([row['input_ids']], device='cuda')
                n = ids.shape[1]
                groups = defaultdict(list)
                for method in pending:
                    groups[group_key(method)].append(method)
                shared_prefill = {}
                ctl.bank = {}
                torch.cuda.reset_peak_memory_stats()
                for key, group in groups.items():
                    ctl.prepare(group, n-1)
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    output = model.model(ids[:, :-1], position_ids=torch.arange(n-1, device='cuda')[None],
                                         use_cache=False).last_hidden_state
                    del output
                    torch.cuda.synchronize()
                    duration = time.monotonic()-start
                    for method in group:
                        shared_prefill[method['name']] = dict(seconds=duration, methods=[m['name'] for m in group])
                prefill_peak = torch.cuda.memory_stats()['allocated_bytes.all.peak']
                decode_groups = defaultdict(list)
                for method in pending:
                    decode_groups[min(n-1, method['budget'] or n)].append(method)
                for retained, group in decode_groups.items():
                    for offset in range(0, len(group), args.decode_batch_size):
                        batch = group[offset:offset+args.decode_batch_size]
                        names = [m['name'] for m in batch]
                        ctl.activate(names)
                        initial_bytes = ctl.bytes()['kv_bytes']//len(names)
                        stops = eos.copy()
                        if task == 'samsum':
                            stops.add(tokenizer.encode('\n', add_special_tokens=False)[-1])
                        torch.cuda.synchronize()
                        start = time.monotonic()
                        generated = decode_batch(model, ctl, row['input_ids'][-1], n, row['max_new_tokens'], stops,
                                                 suppress_first_stop=task == 'samsum')
                        torch.cuda.synchronize()
                        duration = time.monotonic()-start
                        for name, tokens in zip(names, generated):
                            pred = tokenizer.decode(tokens, skip_special_tokens=True)
                            score_pred = pred.lstrip('\n').split('\n')[0] if task in {'trec', 'triviaqa', 'samsum', 'lsht'} else pred
                            score = 100*max(evaluation.dataset2metric[task](score_pred, answer, all_classes=row['all_classes'])
                                            for answer in row['answers'])
                            record = {k: v for k, v in row.items() if k != 'input_ids'}
                            record.update(method=name, pred=pred, output_ids=tokens, score=score,
                                          protocol_sha256=protocol_hash, initial_kv_bytes=initial_bytes,
                                          prefill_shared=shared_prefill[name], prefill_bank_peak_bytes=prefill_peak,
                                          decode_batch=dict(methods=names, seconds=duration), completed_unix=time.time())
                            path = dest/'pred'/name/f'{task}.jsonl'
                            with path.open('a') as f:
                                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                                f.flush()
                                os.fsync(f.fileno())
                            done[name].add(row['id'])
                            print(json.dumps(dict(task=task, example=index+1, total=len(rows), method=name,
                                                  score=score, output_tokens=len(tokens), decode_batch_seconds=duration)), flush=True)
                        ctl.states = {}
                del ids
                ctl.bank = {}
                gc.collect()
                torch.cuda.empty_cache()
                if (index+1) % 25 == 0:
                    checkpoint(dest, f'LongBench {task} checkpoint {index+1}/{len(rows)}', False)
            checkpoint(dest, f'complete LongBench {task} across {len(methods)} policies', args.push)
        write_json(dest/'complete.json', dict(completed_unix=time.time(), protocol_sha256=protocol_hash))
        checkpoint(dest, 'complete full LongBench quality sweep and ablations', args.push)
    except BaseException as error:
        write_json(dest/'failure.json', dict(time=time.time(), error=repr(error)))
        raise
    finally:
        ctl.bank = {}
        ctl.states = {}
        ctl.close()


if __name__ == '__main__':
    main()
