"""Paired Qwen quality pilot using frozen kv-distill data and native metrics.

Run baselines from their common standard prefill and fused policies from their
common fused prefill. Actual cache compaction precedes decoding the last prompt
token, matching the existing evaluator. Full-cache fused control checks drift.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import torch
from kv_distill.engine import Engine,load_model
from kv_distill.metrics import score
from kv_distill.common import MODEL,REVISION
from flash_evict.kv_distill_adapter import FusedController


METHODS=['full','h2o_0.05','h2o_0.1','snap_256','snap_1024',
         'fused_full','fused_0.05','fused_0.1','fused_0.2','fused_256','fused_1024']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_rows(rows,full_pilot):
    if full_pilot:return rows
    chosen=[];seen=set()
    tasks={'narrativeqa','qasper','hotpotqa','passage_retrieval_en'}
    for row in rows:
        if row['suite']=='needle':key=('needle',row['target_length'],row['target_depth'])
        elif row['suite']=='longbench' and row['task'] in tasks:key=('longbench',row['task'])
        elif row['suite']=='gsm8k':key=('gsm8k',row['id'])
        elif row['suite']=='mmlu' and len([r for r in chosen if r['suite']=='mmlu'])<4:key=('mmlu',row['id'])
        else:continue
        if key not in seen:chosen.append(row);seen.add(key)
    return chosen


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='/home/qcb/kv-distill/data/pilot.json')
    p.add_argument('--output',default='results/quality_pilot')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--methods',nargs='+',default=METHODS,choices=METHODS)
    p.add_argument('--full-pilot',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    rows=select_rows(json.loads(Path(args.data).read_text()),args.full_pilot)
    config={"model":MODEL,"model_revision":REVISION,"data_sha256":sha(args.data),
            "baseline_protocol_sha256":sha('/home/qcb/kv-distill/PROTOCOL.md'),
            "methods":args.methods,"seeds":args.seeds,"example_ids":[r['id'] for r in rows],
            "generation":"greedy; original max_new_tokens; no sampling",
            "compression":"pure top-k fused scores per KV head; prefill-only; decode cache grows",
            "scope":"exploratory paired pilot; three seeds are repeats of deterministic decoding",
            "git_revision":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            "kv_distill_revision":subprocess.check_output(['git','-C','/home/qcb/kv-distill','rev-parse','HEAD'],text=True).strip()}
    config_path=dest/'protocol.json'
    if config_path.exists():
        prior=json.loads(config_path.read_text())
        for key in ('model_revision','data_sha256','methods','seeds','example_ids','generation','compression'):
            if prior[key]!=config[key]:raise ValueError(f'Protocol mismatch: {key}; use a new output directory')
    else:config_path.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'event':'protocol_frozen','examples':len(rows),'methods':args.methods}),flush=True)
    if args.prepare_only:return
    runtime={
        'torch':torch.__version__,
        'source_sha256':{str(path):sha(path) for path in [
            Path('flash_evict/prefill.py'),Path('flash_evict/kv_distill_adapter.py'),
            Path('/home/qcb/kv-distill/kv_distill/attention.py'),
            Path('/home/qcb/kv-distill/kv_distill/engine.py'),
            Path('/home/qcb/kv-distill/kv_distill/cache.py'),
            Path('/home/qcb/kv-distill/kv_distill/metrics.py')]},
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'nvidia_smi_before':subprocess.check_output(['nvidia-smi'],text=True),
        'started_unix':time.time()}
    runtime_path=dest/'runtime.json'
    if runtime_path.exists():
        previous=json.loads(runtime_path.read_text())
        if previous['source_sha256']!=runtime['source_sha256']:
            raise ValueError('Evaluation code changed; use a fresh output directory')
    else:runtime_path.write_text(json.dumps(runtime,indent=2)+'\n')
    free,total=torch.cuda.mem_get_info()
    if free<17*2**30:
        raise RuntimeError(f'Need at least 17 GiB free for unchanged BF16 Qwen evaluation; have {free/2**30:.2f} GiB. No other process was stopped.')
    model,tokenizer=load_model()
    engine=Engine(model,tokenizer)
    engine.ctl.close();engine.ctl=FusedController(model)
    for seed in args.seeds:
        random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        path=dest/f'seed_{seed}.jsonl'
        done=set()
        if path.exists():done={(r['id'],r['method']) for r in map(json.loads,path.read_text().splitlines())}
        for i,row in enumerate(rows):
            for backend in ('baseline','fused'):
                pending=[m for m in args.methods if m.startswith('fused_')==(backend=='fused') and (row['id'],m) not in done]
                if not pending:continue
                engine.ctl.prefill_backend=backend
                try:
                    engine.prefill(row)
                    for method in pending:
                        result=engine.generate(row,method)
                        record={k:v for k,v in row.items() if k!='input_ids'}
                        record.update(result,method=method,seed=seed,score=score(row,result['prediction']),
                                      prefill_backend=backend,protocol_sha256=sha(config_path),
                                      prefill_timing_scope='shared once per backend; baseline includes diagnostics',
                                      memory_stats=torch.cuda.memory_stats())
                        with path.open('a') as f:
                            f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                        print(json.dumps({'event':'condition','seed':seed,'example':i+1,'total':len(rows),
                                          'id':row['id'],'method':method,'score':record['score'],
                                          'decode_s':record['decode_seconds']}),flush=True)
                except Exception as e:
                    (dest/'failure.json').write_text(json.dumps({'id':row['id'],'backend':backend,'error':repr(e)},indent=2))
                    raise
                finally:engine.clear()
    (dest/'complete.json').write_text(json.dumps({'time':time.time(),'records':len(rows)*len(args.methods)*len(args.seeds)},indent=2))
    engine.ctl.close()


if __name__=='__main__':main()
