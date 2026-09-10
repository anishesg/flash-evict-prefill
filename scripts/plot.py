"""Export standalone plots from raw kernel measurements (no smoothing)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/benchmark.json')
    p.add_argument('--output',default='results/kernel_measurements')
    p.add_argument('--timing-note',default='GPU activity resumed')
    args=p.parse_args()
    report=json.loads(Path(args.input).read_text());config=report['args']
    labels={'fused_replay_topk':'Fused replay + top-k',
            'flash_snap_w32':'Flash + SnapKV (w=32)',
            'flash_snap_w128':'Flash + SnapKV (w=128)'}
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for method,label in labels.items():
        rows=sorted([r for r in report['rows'] if r['method']==method and 'median_ms' in r],key=lambda r:r['n'])
        xs=[r['n']/1024 for r in rows]
        axs[0].plot(xs,[r['peak_increment_bytes']/2**20 for r in rows],marker='o',label=label)
        axs[1].plot(xs,[r['median_ms'] for r in rows],marker='o',label=label)
    lengths=sorted(set(r['n']/1024 for r in report['rows']))
    for ax in axs:
        ax.set_xscale('log',base=2);ax.set_yscale('log')
        ax.set_xticks(lengths,[f'{n:g}K' for n in lengths])
        ax.set_xlabel('Context tokens');ax.grid(True,alpha=.25)
    axs[0].set_ylabel('Peak additional allocated memory (MiB)')
    axs[1].set_ylabel(f'Median latency (ms; {args.timing_note})')
    axs[0].legend(fontsize=8)
    fig.suptitle(f'A10G, BF16, B=1, Hq={config["heads"]}, Hkv={config["kv_heads"]}, D={config["dim"]}; single layer, 10% retention',fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output+'.png',dpi=180)
    fig.savefig(args.output+'.pdf')


if __name__=='__main__':main()
