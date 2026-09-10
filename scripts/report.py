"""Generate a transparent pilot summary; partial data is always labelled."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--quality',default='results/quality_pilot')
    p.add_argument('--output',default='results/REPORT.md')
    args=p.parse_args()
    lines=['# Flash Evict experimental report','',
           'Exact tiled replay preserves linear memory but recomputes QK. This is not a single-traversal or zero-overhead implementation.',
           'Dense attention still uses quadratic arithmetic. No novelty or quality-superiority claim follows from correctness alone.','',
           '## Correctness','',
           'Materialized FP32 tests cover 512, 1024, and 4096 tokens, 5/10/20% masks, GQA, causal/noncausal masks, noncontiguous inputs, and large logits. Analytic uniform-attention tests cover 8K, 32K, and 128K. A small Qwen2 model verifies prefill, physical compaction, and decoding.','',
           '## Kernel measurements','',
           'CUDA events measure warmed calls. Peak additional allocations come from `torch.cuda.memory_stats()`, with Q/K/V already allocated. Outputs, scores, and selection are included. This is one layer, not whole-model memory. Driver/context and other-process allocations are outside PyTorch allocator peaks; raw reports also record free device memory.','',
           'The first two sweeps encountered external GPU activity (including activity that resumed after a sweep began idle). Treat their timings as diagnostic, not uncontended publication measurements. Later sweeps record per-method process lists.','']
    summaries=[]
    for path in sorted(Path('results').glob('benchmark*.json')):
        report=json.loads(path.read_text());a=report['args']
        lines += [f'### {path.name}', '',f'BF16, B=1, Hq={a["heads"]}, Hkv={a["kv_heads"]}, D={a["dim"]}, retention=10%.', '',
                  '| Tokens | Method | Median ms | Peak additional MiB |','|---:|---|---:|---:|']
        for r in sorted(report['rows'],key=lambda r:(r['n'],r['method'])):
            if 'median_ms' not in r:
                lines.append(f'| {r["n"]} | {r["method"]} | {r.get("error")} | — |');continue
            lines.append(f'| {r["n"]} | {r["method"]} | {r["median_ms"]:.3f} | {r["peak_increment_bytes"]/2**20:.3f} |')
        lines.append('')
    qpath=Path(args.quality)
    rows=[json.loads(line) for path in sorted(qpath.glob('seed_*.jsonl')) for line in path.read_text().splitlines()]
    protocol=json.loads((qpath/'protocol.json').read_text()) if (qpath/'protocol.json').exists() else {}
    expected=len(protocol.get('example_ids',[]))*len(protocol.get('methods',[]))*len(protocol.get('seeds',[]))
    complete=len(rows)==expected and expected>0 and (qpath/'complete.json').exists()
    lines += ['## Qwen-2.5-7B quality','',f'Status: {"complete" if complete else "incomplete"}; {len(rows)} / {expected} planned records.','',
              'The frozen exploratory pilot uses nine needle prompts (three lengths × three depths), four LongBench prompts, two GSM8K prompts, and four MMLU prompts. It is too small to establish a general quality advantage. Greedy decoding is repeated across seeds; seed spread is a reproducibility check, not independent sampling uncertainty.','',
              'Fused retention is pure top-k with no reserved recent tokens. Fused and SnapKV compress only prefill and then allow cache growth. H2O uses its existing half-recent policy and dynamically evicts during decode; equal initial retention is not equal lifetime memory. Fixed budgets 256 and 1024 directly pair fused scoring with SnapKV.','']
    groups=defaultdict(list)
    for row in rows:groups[(row['suite'],row['method'],row['seed'])].append(row)
    aggregate=defaultdict(list)
    for (suite,method,seed),records in groups.items():
        # Macro average tasks inside each suite, matching kv-distill reporting.
        task_scores=defaultdict(list)
        for r in records:task_scores[r['task']].append(r['score'])
        aggregate[(suite,method)].append({'seed':seed,'quality':statistics.mean(statistics.mean(s) for s in task_scores.values()),
                                         'examples':len(records),'decode_seconds':statistics.mean(r['decode_seconds'] for r in records),
                                         'initial_kv_bytes':statistics.mean(r['initial_cache']['kv_bytes'] for r in records)})
    if aggregate:
        lines+=['| Suite | Method | Quality mean ± seed std | Records | Mean decode s | Initial KV MiB |',
                '|---|---|---:|---:|---:|---:|']
        for (suite,method),values in sorted(aggregate.items()):
            quality=[v['quality'] for v in values]
            summary={'suite':suite,'method':method,'quality_mean':statistics.mean(quality),
                     'quality_seed_std':statistics.stdev(quality) if len(quality)>1 else 0.,'per_seed':values}
            summaries.append(summary)
            lines.append(f'| {suite} | {method} | {summary["quality_mean"]:.2f} ± {summary["quality_seed_std"]:.2f} | {sum(v["examples"] for v in values)} | {statistics.mean(v["decode_seconds"] for v in values):.3f} | {statistics.mean(v["initial_kv_bytes"] for v in values)/2**20:.2f} |')
        lines+=['','Prefill wall times in raw quality records include all model layers. Baseline prefill includes the existing evaluator’s scoring diagnostics; they must not be used for an end-to-end speedup claim. Use the kernel benchmarks for latency comparisons.','']
        paired={(r['id'],r['seed'],r['method']):r for r in rows}
        controls=[]
        for (rid,seed,method),r in paired.items():
            if method=='full' and (rid,seed,'fused_full') in paired:
                other=paired[(rid,seed,'fused_full')]
                controls.append({'id':rid,'seed':seed,'output_ids_equal':r['output_ids']==other['output_ids'],
                                 'full_score':r['score'],'fused_full_score':other['score']})
        lines+=[f'Full-cache control: {sum(c["output_ids_equal"] for c in controls)} / {len(controls)} paired generations have identical token IDs. Finite-precision kernel changes can alter greedy output; controls are retained in the summary JSON.','']
    else:controls=[]
    historical=[]
    for path in sorted(Path('/home/qcb/kv-distill/results/original_pilot/pilot').glob('seed_*.jsonl')):
        historical.extend(json.loads(line) for line in path.read_text().splitlines())
    old={(r['id'],r['seed'],r['method']):r for r in historical}
    comparisons=[]
    for r in rows:
        key=(r['id'],r['seed'],r['method'])
        if key in old:
            comparisons.append({'id':r['id'],'seed':r['seed'],'method':r['method'],
                                'score_equal':r['score']==old[key]['score'],
                                'output_ids_equal':r['output_ids']==old[key]['output_ids']})
    if comparisons:
        lines += [f'Historical baseline reproduction: {sum(c["score_equal"] for c in comparisons)} / {len(comparisons)} task scores and {sum(c["output_ids_equal"] for c in comparisons)} / {len(comparisons)} token sequences match the saved original pilot on the same IDs, methods, and seeds.','']
    Path(args.output).write_text('\n'.join(lines)+'\n')
    Path(args.output).with_suffix('.json').write_text(json.dumps({'complete':complete,'records':len(rows),'expected':expected,'quality':summaries,'full_controls':controls,'historical_comparisons':comparisons},indent=2)+'\n')
    print(f'Wrote {args.output}: {len(rows)}/{expected} quality records')


if __name__=='__main__':main()
