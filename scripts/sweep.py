"""Run complete paired pilots for query-scoring policies, sharing baseline records.

Each policy has 627 conditions: 285 freshly verified baseline conditions from
--baseline-run and 342 new fused conditions. Suffix reservations share the same
prefill snapshot. This reuse is explicit in every copied record and manifest.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import torch
from kv_distill.engine import Engine, load_model
from kv_distill.metrics import score
from flash_evict.kv_distill_adapter import FusedController
from evaluate import METHODS, sha, select_rows


def default_policies():
    policies=[]
    # Order tests the requested window alternative immediately after decay .1.
    for name,decay,window in [('w128',0.,128),('d001',.01,0),('d1',1.,0),
                              ('d01',.1,0),('w32',0.,32),('w64',0.,64)]:
        for recent in (0,64):
            if name=='d01' and recent==0:continue  # Already run independently.
            policies.append(dict(name=f'{name}_r{recent}',recency_decay=decay,
                                 observation_window=window,value_weighted=True,
                                 reserved_recent=recent,pool_kernel=1))
    # Prespecified SnapKV-metric controls isolate value weighting and pooling.
    for weighted,pool,recent in [(False,1,64),(True,5,64),(False,5,64),(False,5,32)]:
        policies.append(dict(name=f'w32_v{int(weighted)}_p{pool}_r{recent}',recency_decay=0.,
                             observation_window=32,value_weighted=weighted,
                             reserved_recent=recent,pool_kernel=pool))
    return policies


def write_json(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def append(path,record):
    with path.open('a') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-run',default='results/recency_d01')
    p.add_argument('--output',default='results/query_sweep')
    p.add_argument('--data',default='/home/qcb/kv-distill/data/pilot.json')
    p.add_argument('--policies',help='JSON list; default is the prespecified axis ablations')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--fixed-budgets-only',action='store_true',help='Cartesian refinement: evaluate 256/1024 plus shared baselines')
    args=p.parse_args()
    baseline=Path(args.baseline_run);dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    prior=json.loads((baseline/'protocol.json').read_text())
    policies=json.loads(Path(args.policies).read_text()) if args.policies else default_policies()
    methods=[m for m in METHODS if not m.startswith('fused_') or m in ('fused_256','fused_1024')] if args.fixed_budgets_only else METHODS
    conditions=len(prior['example_ids'])*len(prior['seeds'])*len(methods)
    if len({p['name'] for p in policies})!=len(policies):raise ValueError('Duplicate policy names')
    rows=select_rows(json.loads(Path(args.data).read_text()),False)
    assert sha(args.data)==prior['data_sha256']
    assert [r['id'] for r in rows]==prior['example_ids'] and prior['methods']==METHODS
    manifest=dest/'policies.json'
    if manifest.exists():assert json.loads(manifest.read_text())==policies
    else:write_json(manifest,policies)
    paths=[Path('flash_evict/prefill.py'),Path('flash_evict/kv_distill_adapter.py'),Path(__file__),
           *[Path('/home/qcb/kv-distill/kv_distill')/f for f in ('engine.py','attention.py','cache.py','metrics.py')]]
    hashes={str(f):sha(f) for f in paths}
    configs={};done={};groups=defaultdict(list)
    for policy in policies:
        name=policy['name'];folder=dest/name;folder.mkdir(exist_ok=True)
        config={**prior,'methods':methods,'scoring':{k:policy[k] for k in ('recency_decay','observation_window','value_weighted')},
                'selection':{k:policy[k] for k in ('reserved_recent','pool_kernel')},
                'compression':'top-k with a reserved suffix inside budget; prefill-only; decode cache grows',
                'baseline_source':str(baseline.resolve()),'baseline_protocol_sha256_reuse':sha(baseline/'protocol.json'),
                'source_sha256':hashes,'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
        protocol=folder/'protocol.json'
        if protocol.exists():
            old=json.loads(protocol.read_text())
            for key in ('methods','scoring','selection','source_sha256','baseline_protocol_sha256_reuse'):assert config[key]==old[key],key
        else:write_json(protocol,config)
        configs[name]=config
        groups[json.dumps(config['scoring'],sort_keys=True)].append(policy)
        for seed in prior['seeds']:
            path=folder/f'seed_{seed}.jsonl'
            records=list(map(json.loads,path.read_text().splitlines())) if path.exists() else []
            done[name,seed]={(r['id'],r['method']) for r in records}
    print(json.dumps({'event':'sweep_frozen','policies':len(policies),'conditions_per_policy':conditions}),flush=True)
    if args.prepare_only:return
    assert (baseline/'complete.json').exists(),'Complete the independent baseline run first'
    original_runtime=json.loads((baseline/'runtime.json').read_text())
    for key,value in original_runtime['source_sha256'].items():
        assert sha(key)==value,f'Baseline source changed: {key}'
    for policy in policies:
        name=policy['name'];folder=dest/name
        for seed in prior['seeds']:
            source=baseline/f'seed_{seed}.jsonl'
            baseline_records=[r for r in map(json.loads,source.read_text().splitlines()) if not r['method'].startswith('fused_')]
            assert len(baseline_records)==len(rows)*5
            for record in baseline_records:
                key=record['id'],record['method']
                if key in done[name,seed]:continue
                record={**record,'reused_from':str(source.resolve()),'reused_source_sha256':sha(source),
                        'source_protocol_sha256':record['protocol_sha256'],'protocol_sha256':sha(folder/'protocol.json')}
                append(folder/f'seed_{seed}.jsonl',record);done[name,seed].add(key)
    free,_=torch.cuda.mem_get_info()
    if free<17*2**30:raise RuntimeError('Need 17 GiB free for BF16 Qwen evaluation')
    model,tokenizer=load_model();engine=Engine(model,tokenizer)
    engine.ctl.close();engine.ctl=FusedController(model);engine.ctl.prefill_backend='fused'
    runtime=dict(started_unix=time.time(),source_sha256=hashes,torch=torch.__version__,
                 nvidia_smi_before=subprocess.check_output(['nvidia-smi'],text=True))
    write_json(dest/'runtime.json',runtime)
    # Finish each scoring family across all seeds before moving to the next.
    for scoring,variants in groups.items():
        engine.ctl.scoring=json.loads(scoring)
        for seed in prior['seeds']:
            random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            for i,row in enumerate(rows):
                pending=[(policy,method) for policy in variants for method in methods
                         if method.startswith('fused_') and (row['id'],method) not in done[policy['name'],seed]]
                if not pending:continue
                try:
                    engine.prefill(row)
                    for policy,method in pending:
                        name=policy['name'];folder=dest/name
                        engine.ctl.reserved_recent=policy['reserved_recent'];engine.ctl.pool_kernel=policy['pool_kernel']
                        result=engine.generate(row,method)
                        record={k:v for k,v in row.items() if k!='input_ids'}
                        record.update(result,method=method,seed=seed,score=score(row,result['prediction']),
                                      policy=name,prefill_backend='fused',protocol_sha256=sha(folder/'protocol.json'),
                                      prefill_timing_scope='shared within scoring family; may include first-use JIT',
                                      memory_stats=torch.cuda.memory_stats())
                        append(folder/f'seed_{seed}.jsonl',record);done[name,seed].add((row['id'],method))
                        print(json.dumps(dict(event='condition',policy=name,seed=seed,example=i+1,
                                              method=method,score=record['score'],decode_s=record['decode_seconds'])),flush=True)
                finally:engine.clear()
        for policy in variants:
            name=policy['name']
            assert all(len(done[name,seed])==len(rows)*len(methods) for seed in prior['seeds'])
            write_json(dest/name/'complete.json',dict(time=time.time(),records=conditions,new_fused_records=conditions-285,shared_baseline_records=285))
    engine.ctl.close()
    write_json(dest/'complete.json',dict(time=time.time(),policies=len(policies),conditions=conditions*len(policies)))


if __name__=='__main__':main()
