"""Warmed one-layer policy latency and allocator peaks at fixed cache budgets."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import torch
import triton
from flash_evict import evict, prefill
from flash_evict.baselines import flash_attention, flash_snap
from flash_evict.prefill import _prefill
from benchmark import measure


def processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                 '--format=csv,noheader'],text=True)
    others=[int(line.split(',')[0]) for line in raw.splitlines()
            if line.strip() and int(line.split(',')[0])!=os.getpid()]
    return raw,others


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--policies',default='results/query_sweep/policies.json')
    p.add_argument('--output',default='results/policy_benchmark.json')
    p.add_argument('--lengths',type=int,nargs='+',default=[8192,32768,131072])
    p.add_argument('--budgets',type=int,nargs='+',default=[256,1024])
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--require-idle',action='store_true')
    p.add_argument('--only-policies',nargs='+',help='Measure a subset now; resume the rest with the same output later')
    args=p.parse_args()
    policies=json.loads(Path(args.policies).read_text())
    if not any(p['name']=='d01_r0' for p in policies):
        policies=[dict(name='d01_r0',recency_decay=.1,observation_window=0,
                       value_weighted=True,reserved_recent=0,pool_kernel=1),*policies]
    torch.set_num_threads(8);torch.manual_seed(20260910)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    paths=[Path('flash_evict/prefill.py'),Path('flash_evict/baselines.py'),Path(__file__),Path('scripts/benchmark.py')]
    report=dict(args={k:v for k,v in vars(args).items() if k!='only_policies'},policies=policies,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                triton=triton.__version__,shape=dict(batch=1,query_heads=28,kv_heads=4,dim=128,dtype='bfloat16'),
                source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                scope='one layer; incremental PyTorch allocations, inputs/model weights excluded; output, scores, selection included',
                rows=[])
    if dest.exists():
        old=json.loads(dest.read_text())
        for key in ('args','policies','source_sha256'):assert old[key]==report[key],key
        report=old
    done={(r['n'],r['method'],r['budget']) for r in report['rows']}
    for n in args.lengths:
        q=torch.randn(1,28,n,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,4,n,128,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
        methods=[('torch_flash',None,lambda:flash_attention(q,k,v)),
                 ('triton_attention_only',None,lambda:prefill(q,k,v,score=False))]
        for budget in args.budgets:
            for window in (32,128):
                methods.append((f'snap_w{window}',budget,lambda b=budget,w=window:flash_snap(q,k,v,b,w)))
            for policy in policies:
                kwargs={key:value for key,value in policy.items() if key!='name'}
                methods.append((policy['name'],budget,lambda b=budget,kw=kwargs:evict(q,k,v,b,**kw)))
        random.Random(n).shuffle(methods)
        for name,budget,fn in methods:
            if (n,name,budget) in done:continue
            if args.only_policies and name not in (*args.only_policies,'torch_flash','triton_attention_only','snap_w32','snap_w128'):continue
            before,others=processes()
            if args.require_idle and others:raise RuntimeError(f'Other CUDA processes: {others}')
            try:
                row=dict(n=n,method=name,budget=budget,**measure(fn,args.repeats))
            except torch.OutOfMemoryError as e:
                row=dict(n=n,method=name,budget=budget,error='CUDA OOM',detail=str(e))
                gc.collect();torch.cuda.empty_cache()
            after,others_after=processes()
            row.update(compute_processes_before=before,compute_processes_after=after,
                       other_compute_pids=sorted(set(others+others_after)))
            report['rows'].append(row)
            done.add((n,name,budget))
            report['kernel_resources']=[dict(hash=kernel.hash,registers_per_thread=kernel.n_regs,
                spills=kernel.n_spills,shared_bytes=kernel.metadata.shared,
                global_scratch_bytes=kernel.metadata.global_scratch_size)
                for bundle in _prefill.device_caches.values() for kernel in bundle[0].values()]
            dest.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            if args.require_idle and row['other_compute_pids']:raise RuntimeError('GPU contention during measurement')
        del q,k,v
        gc.collect();torch.cuda.empty_cache()
    expected=len(args.lengths)*(2+len(args.budgets)*(2+len(policies)))
    report['complete']=len(done)==expected
    dest.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
