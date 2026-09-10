"""Whole-model prefill with prompt caches compacted as each layer completes.

No quality metrics or diagnostic scoring run here. Model weights, live caches,
attention output, and MLP activations count toward the absolute allocator peak.
Synthetic tokens exercise the model shape; this is not a generation-quality run.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch
from kv_distill.engine import load_model
from flash_evict.streaming_adapter import StreamingController, chunk_mlp
from benchmark_policies import processes


@torch.inference_mode()
def measure(model,ctl,ids,repeats):
    n=ids.shape[1];pos=torch.arange(n,device='cuda')[None]
    def clear():
        ctl.states={};gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    def call():
        ctl.reset();ctl.position_end=n
        return model.model(ids,position_ids=pos,use_cache=False).last_hidden_state
    clear()
    # JIT and model warmup are excluded from the samples.
    result=call();del result;clear()
    times=[]
    for _ in range(repeats):
        start=time.monotonic();result=call();torch.cuda.synchronize()
        times.append(time.monotonic()-start)
        del result;clear()
    torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_stats()
    result=call();torch.cuda.synchronize();after=torch.cuda.memory_stats()
    cache=ctl.bytes()
    record=dict(median_seconds=statistics.median(times),samples_seconds=times,
                allocated_before_bytes=before['allocated_bytes.all.current'],
                peak_allocated_bytes=after['allocated_bytes.all.peak'],
                peak_increment_bytes=after['allocated_bytes.all.peak']-before['allocated_bytes.all.current'],
                cache=cache,memory_stats=after)
    del result;clear()
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lengths',type=int,nargs='+',default=[8192,32768,131072])
    p.add_argument('--budget',type=int,default=1024)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--policies',default='results/query_sweep/policies.json')
    p.add_argument('--names',nargs='+',default=['d01_r0','w128_r64'])
    p.add_argument('--output',default='results/model_benchmark.json')
    p.add_argument('--require-idle',action='store_true')
    p.add_argument('--mlp-chunk-size',type=int,default=0)
    args=p.parse_args()
    policies=json.loads(Path(args.policies).read_text())
    policies.append(dict(name='d01_r0',recency_decay=.1,observation_window=0,
                         value_weighted=True,reserved_recent=0,pool_kernel=1))
    policy_by_name={p['name']:p for p in policies}
    methods=[dict(name='full',backend='full'),dict(name='snap_1024',backend='snap')]
    for name in args.names:
        policy=policy_by_name[name]
        methods.append(dict(name=name,backend='fused',
                            scoring={k:policy[k] for k in ('recency_decay','observation_window','value_weighted')},
                            reserved_recent=policy['reserved_recent'],pool_kernel=policy['pool_kernel']))
    torch.manual_seed(20260910)
    raw,others=processes()
    if args.require_idle and others:raise RuntimeError(f'Other CUDA processes: {others}')
    model,_=load_model()
    if args.mlp_chunk_size:chunk_mlp(model,args.mlp_chunk_size)
    paths=[Path(__file__),Path('flash_evict/streaming_adapter.py'),Path('flash_evict/prefill.py'),
           Path('flash_evict/kv_distill_adapter.py'),Path('flash_evict/baselines.py')]
    report=dict(args=vars(args),methods=methods,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                scope='28-layer BF16 Qwen-2.5-7B model.model; synthetic tokens; no lm_head or decode; immediate per-layer cache compaction; weights and MLP activations included; optional MLP chunking applies equally to every method',
                rows=[])
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    for n in args.lengths:
        ids=torch.randint(0,model.config.vocab_size,(1,n),device='cuda')
        for method in methods:
            name=method['name'];kwargs={k:v for k,v in method.items() if k!='name'}
            ctl=StreamingController(model,budget=args.budget,**kwargs)
            before,others=processes()
            if args.require_idle and others:raise RuntimeError(f'Other CUDA processes: {others}')
            try:
                row=dict(n=n,method=name,budget=None if name=='full' else args.budget,**measure(model,ctl,ids,args.repeats))
            except torch.OutOfMemoryError as e:
                row=dict(n=n,method=name,budget=args.budget,error='CUDA OOM',detail=str(e),memory_stats=torch.cuda.memory_stats())
            finally:
                ctl.states={};ctl.close();gc.collect();torch.cuda.empty_cache()
            after,others_after=processes()
            row.update(compute_processes_before=before,compute_processes_after=after,
                       other_compute_pids=sorted(set(others+others_after)))
            report['rows'].append(row);dest.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='memory_stats'}),flush=True)
            if args.require_idle and row['other_compute_pids']:raise RuntimeError('Contention observed')
        del ids
    report['complete']=True;dest.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
