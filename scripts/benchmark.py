"""Reproducible latency and actual PyTorch allocator peak measurements."""
import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
import torch
import triton
from flash_evict import prefill,evict
from flash_evict.prefill import _prefill
from flash_evict.baselines import flash_attention,flash_snap


@torch.inference_mode()
def measure(fn,repeats):
    for _ in range(2):
        result=fn()
        del result
    torch.cuda.synchronize()
    times=[]
    for _ in range(repeats):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        result=fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del result
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before=torch.cuda.memory_stats()
    free_before,total=torch.cuda.mem_get_info()
    result=fn()
    torch.cuda.synchronize()
    after=torch.cuda.memory_stats()
    record={"median_ms":statistics.median(times),"samples_ms":times,
            "allocated_before_bytes":before['allocated_bytes.all.current'],
            "peak_allocated_bytes":after['allocated_bytes.all.peak'],
            "peak_increment_bytes":after['allocated_bytes.all.peak']-before['allocated_bytes.all.current'],
            "peak_reserved_bytes":after['reserved_bytes.all.peak'],
            "free_device_bytes_before":free_before,"total_device_bytes":total}
    del result
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lengths',type=int,nargs='+',default=[8192,32768,131072])
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--kv-heads',type=int,default=2)
    p.add_argument('--dim',type=int,default=64)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',default='results/benchmark.json')
    p.add_argument('--require-idle',action='store_true',help='Abort if another CUDA process is observed')
    args=p.parse_args()
    torch.manual_seed(20260910)
    torch.set_num_threads(8)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report={"args":vars(args),"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
            "triton":triton.__version__,"dtype":"bfloat16","batch":1,
            "git_revision":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            "source_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                             [Path('flash_evict/prefill.py'),Path('flash_evict/baselines.py')]},
            "nvidia_smi_before":subprocess.check_output(['nvidia-smi'],text=True),
            "scope":"single attention layer; allocations include output and selection, exclude inputs",
            "rows":[]}
    for n in args.lengths:
        q=torch.randn((1,args.heads,n,args.dim),device='cuda',dtype=torch.bfloat16)
        k=torch.randn((1,args.kv_heads,n,args.dim),device='cuda',dtype=q.dtype)
        v=torch.randn_like(k)
        budget=math.ceil(n*.1)
        methods={
            "torch_flash":lambda:flash_attention(q,k,v),
            "triton_attention_only":lambda:prefill(q,k,v,score=False),
            "fused_replay_scores":lambda:prefill(q,k,v),
            "fused_replay_topk":lambda:evict(q,k,v,budget),
            "flash_snap_w32":lambda:flash_snap(q,k,v,budget,32),
            "flash_snap_w128":lambda:flash_snap(q,k,v,budget,128),
        }
        # Seeded order reduces a systematic warmup/temperature ordering effect.
        import random
        items=list(methods.items());random.Random(n).shuffle(items)
        for name,fn in items:
            other_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
            pids_before=[int(line.split(',')[0]) for line in other_before.splitlines() if line.strip()]
            if args.require_idle and any(pid!=os.getpid() for pid in pids_before):
                raise RuntimeError('Another CUDA process is present; refusing an uncontended timing claim')
            try:
                row={"n":n,"method":name,"budget":budget,**measure(fn,args.repeats)}
            except torch.OutOfMemoryError as e:
                row={"n":n,"method":name,"error":"CUDA OOM","detail":str(e)}
                gc.collect();torch.cuda.empty_cache()
            row['compute_processes_before']=other_before
            row['compute_processes_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
            pids_after=[int(line.split(',')[0]) for line in row['compute_processes_after'].splitlines() if line.strip()]
            row['other_compute_pids']=sorted(set(pid for pid in pids_before+pids_after if pid!=os.getpid()))
            report['rows'].append(row)
            report['kernel_resources']=[
                {'hash':kernel.hash,'registers_per_thread':kernel.n_regs,
                 'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,
                 'global_scratch_bytes':kernel.metadata.global_scratch_size,
                 'constants':{str(key):value for key,value in kernel.src.constants.items()
                              if isinstance(value,(str,int,float,bool))}}
                for bundle in _prefill.device_caches.values()
                for kernel in bundle[0].values()]
            output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            if args.require_idle and row['other_compute_pids']:
                raise RuntimeError('Another CUDA process appeared during measurement; row is marked contended')
        del q,k,v
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
