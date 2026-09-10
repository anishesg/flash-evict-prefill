"""Audit follow-up runs and export tables, Pareto points, and standalone figures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

SUITES=('longbench','gsm8k','mmlu','needle')


def records(folder):
    return [json.loads(line) for path in sorted(folder.glob('seed_*.jsonl')) for line in path.read_text().splitlines()]


def aggregate(rows,method):
    result={}
    for suite in SUITES:
        group=[r for r in rows if r['method']==method and r['suite']==suite]
        seeds=defaultdict(list)
        for r in group:seeds[r['seed']].append(r['score'])
        means=[statistics.mean(v) for v in seeds.values()]
        result[suite]=dict(mean=statistics.mean(means) if means else None,
                           seed_std=statistics.stdev(means) if len(means)>1 else None,
                           records=len(group),unique_prompts=len({r['id'] for r in group}))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    root=Path('results');initial=root/'recency_d01'
    runs=[('d01_r0',initial)]
    for sweep in ('query_sweep','combined_sweep'):
        manifest=root/sweep/'policies.json'
        if manifest.exists():runs.extend((p['name'],root/sweep/p['name']) for p in json.loads(manifest.read_text()))
    bench_path=root/'policy_benchmark.json'
    benchmark=json.loads(bench_path.read_text()) if bench_path.exists() else {}
    measurements={(r['method'],r['budget'],r['n']):r for r in benchmark.get('rows',[]) if 'error' not in r}
    primary=records(initial)
    history=records(root/'quality_pilot')
    history_index={(r['seed'],r['id'],r['method']):r for r in history}
    baseline=[r for r in primary if not r['method'].startswith('fused_')]
    match=sum(r['output_ids']==history_index[(r['seed'],r['id'],r['method'])]['output_ids'] for r in baseline)
    report=dict(scope='Adaptive exploratory tuning on the same 19 prompts; three greedy repeats are not independent samples.',
                suite_sizes=dict(longbench=4,gsm8k=2,mmlu=4,needle=9),
                historical_baseline_control=dict(records=len(baseline),identical_token_sequences=match),
                policies=[],points=[],frontiers={})
    for name,folder in runs:
        protocol=json.loads((folder/'protocol.json').read_text());rows=records(folder)
        keys={(r['seed'],r['id'],r['method']) for r in rows}
        expected={(s,i,m) for s in protocol['seeds'] for i in protocol['example_ids'] for m in protocol['methods']}
        complete=keys==expected and len(keys)==len(rows) and (folder/'complete.json').exists()
        if not complete and not args.partial:raise RuntimeError(f'Incomplete policy {name}: {len(rows)}/{len(expected)}')
        full={(r['seed'],r['id']):r for r in rows if r['method']=='full'}
        controls=[r for r in rows if r['method']=='fused_full']
        info=dict(name=name,path=str(folder),scoring=protocol['scoring'],selection=protocol['selection'],
                  complete=complete,records=len(rows),expected_records=len(expected),
                  shared_baseline_records=sum('reused_from' in r for r in rows),
                  full_cache_controls=len(controls),identical_full_cache_controls=sum(r['output_ids']==full[r['seed'],r['id']]['output_ids'] for r in controls),
                  quality={m:aggregate(rows,m) for m in protocol['methods']})
        report['policies'].append(info)
        if not complete:continue
        for budget in (256,1024):
            quality={suite:info['quality'][f'fused_{budget}'][suite]['mean'] for suite in SUITES}
            perf=measurements.get((name,budget,131072))
            report['points'].append(dict(name=name,budget=budget,**quality,
                                         memory_mib=perf['peak_increment_bytes']/2**20 if perf else None,
                                         latency_ms=perf['median_ms'] if perf else None))
    for budget in (256,1024):
        quality={suite:value['mean'] for suite,value in aggregate(primary,f'snap_{budget}').items()}
        perf=measurements.get(('snap_w32',budget,131072))
        report['points'].append(dict(name='SnapKV',budget=budget,**quality,
                                     memory_mib=perf['peak_increment_bytes']/2**20 if perf else None,
                                     latency_ms=perf['median_ms'] if perf else None))
    # A point dominates another only if no worse on any of the four suites and
    # uses no more peak allocation, with at least one strict improvement.
    for budget in (256,1024):
        points=[p for p in report['points'] if p['budget']==budget and p['memory_mib'] is not None]
        def dominates(a,b):
            weak=a['memory_mib']<=b['memory_mib'] and all(a[s]>=b[s] for s in SUITES)
            strict=a['memory_mib']<b['memory_mib'] or any(a[s]>b[s] for s in SUITES)
            return weak and strict
        report['frontiers'][budget]=[p for p in points if not any(dominates(other,p) for other in points)]
    report['target_passes']=[p for p in report['points'] if p['budget']==1024 and p['longbench']>=85 and p['needle']>=85
                              and p['memory_mib'] is not None and p['memory_mib']<1000]
    report['benchmark_complete']=benchmark.get('complete',False)
    report['benchmark_contended_rows']=sum(bool(r['other_compute_pids']) for r in benchmark.get('rows',[]))
    model_path=root/'model_benchmark.json'
    model=json.loads(model_path.read_text()) if model_path.exists() else {}
    report['model_benchmark_complete']=model.get('complete',False)
    report['complete']=all(p['complete'] for p in report['policies']) and report['benchmark_complete'] and report['model_benchmark_complete']
    if not args.partial and not report['complete']:raise RuntimeError('Benchmarks incomplete')
    (root/'FOLLOWUP.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Query recency and observation-window follow-up','',f"Status: {'complete' if report['complete'] else 'in progress'}.",'',
           'Quality uses the same 19 prompts (4 LongBench, 2 GSM8K, 4 MMLU, 9 needle) and three greedy repetitions. '
           'Parameters were tuned on these prompts; these results do not establish held-out benchmark quality. '
           'Quality prompts are at most 8192 tokens. The 131K measurements use synthetic inputs and do not measure answer quality at that length.','',
           'The independent decay=0.1 run reruns all 627 conditions. Each axis/pinning/control policy has 627 conditions '
           '(342 new fused conditions and 285 explicitly shared baselines). The 18-point Cartesian refinement evaluates '
           'both fixed budgets on all prompts and seeds: 114 new fused conditions plus the same 285 baselines per point.','',
           f"Historical baseline token sequences reproduced: {match}/{len(baseline)}.",'',
           'Reservation is inside the total budget. All compressed policies apply to every model layer, compact prompt K/V, '
           'preserve absolute RoPE positions, and let the decode cache grow. H2O-0.1 retains its original dynamic eviction behavior. '
           'Prompts shorter than a budget keep their full prompt cache; the GSM8K/MMLU pilot prompts do not undergo eviction at budget 1024.','',
           '## Quality and 131K kernel memory','',
           'Memory is peak additional PyTorch allocation for one BF16 attention layer, B=1, Hq=28, Hkv=4, D=128. '
           'Outputs, scores, selection, and masks are included; input Q/K/V and model weights are excluded.','',
           '| Policy | Budget | LongBench | GSM8K | MMLU | Needle % | Peak MiB | 131K ms |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'pending' if x is None else f'{x:.2f}'
    for point in report['points']:
        lines.append('| '+' | '.join([point['name'],str(point['budget']),*[fmt(point[s]) for s in SUITES],fmt(point['memory_mib']),fmt(point['latency_ms'])])+' |')
    for method in ('full','h2o_0.1','fused_256','fused_1024'):
        source=history if method.startswith('fused_') else primary
        label='original uniform '+method if method.startswith('fused_') else method
        quality=aggregate(source,method)
        lines.append('| '+label+' | — | '+' | '.join(fmt(quality[s]['mean']) for s in SUITES)+' | — | — |')
    lines+=['','## Latency at 8K, 32K, and 131K','',
            'Five warmed CUDA-event samples per measurement; medians below. The process lists before and after each '
            'measurement are in the raw JSON. Zero additional CUDA processes are required for uncontended measurements.','',
            '| Policy | Budget | 8K ms | 32K ms | 131K ms |','|---|---:|---:|---:|---:|']
    for point in report['points']:
        method='snap_w32' if point['name']=='SnapKV' else point['name']
        samples=[measurements.get((method,point['budget'],n),{}).get('median_ms') for n in (8192,32768,131072)]
        lines.append(f"| {point['name']} | {point['budget']} | "+' | '.join(fmt(x) for x in samples)+' |')
    lines+=['','## Pareto frontier','',
            'Dominance considers all four quality scores (higher is better) and 131K additional allocation (lower is better). '
            'Latency is reported separately. No confidence intervals are inferred from repeated greedy seeds.','']
    for budget,frontier in report['frontiers'].items():lines.append(f"- Budget {budget}: "+', '.join(p['name'] for p in frontier))
    lines+=['','## Whole-model prefill','',
            'All 28 Qwen layers, BF16 weights, no scoring diagnostics, and immediate per-layer compaction. '
            'Synthetic token sequences measure shape-dependent cost; timings exclude tokenization, the LM head, and decode. '
            'Absolute allocator peaks include weights, caches, attention outputs, and MLP activations.','',
            '| Tokens | Method | Median seconds | Absolute peak GiB | Incremental GiB |','|---:|---|---:|---:|---:|']
    for row in model.get('rows',[]):
        if 'error' in row:lines.append(f"| {row['n']} | {row['method']} | OOM | — | — |")
        else:lines.append(f"| {row['n']} | {row['method']} | {row['median_seconds']:.3f} | {row['peak_allocated_bytes']/2**30:.3f} | {row['peak_increment_bytes']/2**30:.3f} |")
    lines+=['','## Interpretation','',
            'Recency weights are exp(decay * (q - (N-1))) per token, and windowing masks exactly the final w query positions. '
            'Attention outputs use the original online softmax. Scores use its final normalizers in a tiled replay; '
            'window policies skip replay for earlier query programs. Pooling and value weighting are explicit ablations.','',
            'This is one attention-kernel launch with a second QK traversal for scored queries, followed by separate selection. '
            'It is not a single QK traversal, zero scoring cost, or O(N) dense-attention arithmetic. '
            'Live score storage is O(N); fixed-window SnapKV storage O(Nw) is also linear in N for fixed w. '
            'The measured memory ratio describes additional allocations in this implementation, not total-model memory or a novelty proof.','']
    (root/'FOLLOWUP.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=report['complete'],completed_policies=sum(p['complete'] for p in report['policies']),
                          total_policies=len(report['policies']),target_passes=report['target_passes'])),flush=True)
    if not report['complete']:return
    figures(report,benchmark,model,root)


def figures(report,benchmark,model,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for col,budget in enumerate((256,1024)):
        points=[p for p in report['points'] if p['budget']==budget and p['memory_mib'] is not None]
        for row,suite in enumerate(('longbench','needle')):
            ax=axes[row,col]
            for p in points:
                snap=p['name']=='SnapKV';color='#d97706' if snap else '#2563eb'
                ax.scatter(p['memory_mib'],p[suite],c=color,s=55 if snap else 24,marker='s' if snap else 'o',alpha=.75)
            ax.axhline(85,color='#777777',linestyle='--',linewidth=.8)
            ax.set_title(f'{suite.capitalize()}, budget {budget}')
            ax.set_ylabel('Quality (0–100)');ax.set_xlabel('131K peak additional allocation (MiB)')
            ax.set_ylim(-3,105);ax.grid(alpha=.15)
    fig.suptitle('Query scoring policies (blue) and SnapKV (orange): exploratory pilot')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(root/f'query_pareto.{ext}',dpi=220)
    plt.close(fig)
    names=['d01_r0','w128_r0','d01_r64','w128_r64','SnapKV']
    points={(p['name'],p['budget']):p for p in report['points']}
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,suite in zip(axes,('longbench','needle')):
        vals=[points[name,1024][suite] for name in names]
        bars=ax.bar(range(len(names)),vals,color=['#2563eb','#0891b2','#4f46e5','#059669','#d97706'])
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set_xticks(range(len(names)),['Decay .1','Window 128','Decay .1\n+ recent 64','Window 128\n+ recent 64','SnapKV'])
        ax.axhline(85,color='#777777',linestyle='--',linewidth=.8)
        ax.set_ylim(0,110);ax.set_title(suite.capitalize());ax.set_ylabel('Quality (0–100)')
    fig.suptitle('Qwen-2.5-7B, budget 1024; 4 LongBench and 9 needle prompts')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(root/f'query_quality.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,label in [('snap_w32','Flash + SnapKV'),('d01_r0','Decay .1'),('w128_r64','Window 128 + recent 64')]:
        rows=sorted([r for r in benchmark['rows'] if r['method']==name and r['budget']==1024 and 'error' not in r],key=lambda r:r['n'])
        axes[0].plot([r['n']/1024 for r in rows],[r['median_ms'] for r in rows],marker='o',label=label)
        axes[1].plot([r['n']/1024 for r in rows],[r['peak_increment_bytes']/2**20 for r in rows],marker='o',label=label)
    for ax in axes:ax.set_xlabel('Context (Ki tokens)');ax.set_xscale('log',base=2);ax.set_xticks([8,32,128],['8','32','128']);ax.grid(alpha=.2)
    axes[0].set_ylabel('One-layer latency (ms)');axes[0].set_yscale('log');axes[0].legend()
    axes[1].set_ylabel('Peak additional allocation (MiB)')
    fig.suptitle('A10G, Qwen attention dimensions, budget 1024; model weights excluded')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(root/f'query_performance.{ext}',dpi=220)
    plt.close(fig)


if __name__=='__main__':main()
