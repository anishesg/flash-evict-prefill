"""Audit coverage and summarize official LongBench metrics without hiding gaps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

ZH = {'multifieldqa_zh', 'dureader', 'vcsum', 'lsht', 'passage_retrieval_zh'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/longbench_full')
    args = parser.parse_args()
    root = Path(args.output)
    protocol = json.loads((root/'protocol.json').read_text())
    manifest = json.loads((root/'data_manifest.json').read_text())
    tasks = protocol['tasks']
    expected = {task: min(manifest['tasks'][task]['examples'], protocol['limit'] or 10**9) for task in tasks}
    report = dict(expected_per_method=sum(expected.values()), tasks=tasks, methods={}, comparisons={})
    records = {}
    for method in protocol['methods']:
        name = method['name']
        task_scores = {}
        untuned_scores = {}
        counts = {}
        records[name] = {}
        for task in tasks:
            path = root/'pred'/name/f'{task}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            ids = {r['id'] for r in rows}
            if len(rows) != len(ids):
                raise ValueError(f'Duplicate records: {path}')
            records[name][task] = {r['id']: r for r in rows}
            counts[task] = len(rows)
            if len(rows) == expected[task]:
                task_scores[task] = statistics.mean(r['score'] for r in rows)
                untuned = [r['score'] for r in rows if not r['used_in_tuning']]
                untuned_scores[task] = statistics.mean(untuned) if untuned else None
        complete = counts == expected
        english = [task_scores[t] for t in tasks if t not in ZH and t in task_scores]
        report['methods'][name] = dict(complete=complete, records=sum(counts.values()), counts=counts,
                                       completed_task_scores=task_scores,
                                       macro_all=statistics.mean(task_scores.values()) if complete else None,
                                       macro_english16=statistics.mean(english) if complete and english else None,
                                       macro_excluding_tuned=statistics.mean(v for v in untuned_scores.values() if v is not None)
                                       if complete else None)
    report['complete'] = all(r['complete'] for r in report['methods'].values())
    pairs = []
    for budget in (256, 1024):
        for label in ('d01_r0', 'd0.1_w128_r0', 'd0.1_w128_r64'):
            pairs.append((f'{label}_{budget}', f'snap_{budget}'))
    pairs += [('d01_r0_256', f'{name}_256') for name in ('value_norm_only', 'uniform_recent')]
    for a, b in pairs:
        if a not in report['methods'] or b not in report['methods']:
            continue
        if not (report['methods'][a]['complete'] and report['methods'][b]['complete']):
            continue
        rng = np.random.default_rng(42)
        replicates = np.zeros(2000)
        differences = []
        for task in tasks:
            ra, rb = records[a][task], records[b][task]
            if ra.keys() != rb.keys():
                raise ValueError(f'Unpaired records: {task}')
            delta = np.array([ra[i]['score']-rb[i]['score'] for i in sorted(ra)])
            differences.append(float(delta.mean()))
            replicates += delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)/len(tasks)
        report['comparisons'][a+' minus '+b] = dict(macro_difference=statistics.mean(differences),
                                                  paired_stratified_bootstrap_95ci=np.quantile(replicates, [.025, .975]).tolist())
    (root/'SUMMARY.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Full LongBench evaluation', '',
             f"Status: {'complete' if report['complete'] else 'in progress'}. "
             f"{len(tasks)} tasks, {report['expected_per_method']} examples per method, {len(protocol['methods'])} methods.", '',
             'All original test examples are evaluated once with greedy decoding. Task scores use the pinned '
             'LongBench author metrics and prescribed generation lengths. Aggregate scores are equal-weight means '
             'of task scores. The English/code aggregate covers the 16 tasks excluding the five Chinese tasks. '
             'No full-benchmark score is emitted before every requested example is present.', '',
             f"Context limit: {manifest['context_limit']:,} tokens including the reserved answer budget. "
             'Official task templates use Qwen chat wrapping except the six official no-chat tasks. '
             'Middle truncation is applied at token level after wrapping, as in the earlier pilot. '
             'The pilot used an 8,192-token prompt cap; its scores are not directly comparable.', '',
             'Budgets count retained prompt tokens per KV head in all 28 layers. The final prompt token is '
             'forwarded after prefix compaction; the decode cache then grows. Recency attention includes value '
             'weighting. The value-only control uses L1 value magnitude; the suffix control retains only the '
             'most recent budget tokens and has no attention sinks.', '',
             'Compatible methods share prefill, and compatible caches batch decoding. MLP chunks of '
             f"{protocol['mlp_chunk_size']} tokens apply to every method. Quality-run timings and peaks include "
             'shared work and live policy banks; use the separate whole-model benchmark for performance comparisons.', '',
             '| Method | Records | All-task macro | English/code macro | Macro excluding tuning prompts |',
             '|---|---:|---:|---:|---:|']
    fmt = lambda value: 'pending' if value is None else f'{value:.2f}'
    for name, result in report['methods'].items():
        lines.append(f"| {name} | {result['records']}/{report['expected_per_method']} | " +
                     ' | '.join(fmt(result[key]) for key in ('macro_all', 'macro_english16', 'macro_excluding_tuned'))+' |')
    lines += ['', '## Per-task quality', '', '| Task | '+' | '.join(report['methods'])+' |',
              '|---|'+'---:|'*len(report['methods'])]
    for task in tasks:
        lines.append('| '+task+' | '+' | '.join(fmt(r['completed_task_scores'].get(task))
                                                for r in report['methods'].values())+' |')
    lines += ['', '## Paired comparisons', '',
              'Pointwise intervals (without a multiple-comparison adjustment) use 2,000 paired bootstrap resamples '
              'within each task, with tasks held fixed. '
              'They measure example variability, not variability across new tasks or model training seeds.', '']
    for pair, result in report['comparisons'].items():
        lo, hi = result['paired_stratified_bootstrap_95ci']
        lines.append(f"- {pair}: {result['macro_difference']:+.2f} points, 95% interval [{lo:+.2f}, {hi:+.2f}].")
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=report['complete'], records={m: r['records'] for m, r in report['methods'].items()})), flush=True)


if __name__ == '__main__':
    main()
