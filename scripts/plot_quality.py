"""Export the completed exploratory quality comparison, with sample sizes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    report=json.loads(Path('results/REPORT.json').read_text())
    if not report['complete']:
        raise RuntimeError('Refusing to plot an incomplete pilot as a final comparison')
    values={(r['suite'],r['method']):r['quality_mean'] for r in report['quality']}
    methods=['full','h2o_0.05','h2o_0.1','snap_256','fused_256','snap_1024','fused_1024']
    labels=['Full','H2O 5%','H2O 10%','Snap 256','Fused 256','Snap 1024','Fused 1024']
    colors=['#555555','#888888','#aaaaaa','#d97706','#2563eb','#d97706','#2563eb']
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for ax,suite,title in zip(axes,['longbench','needle'],['LongBench: 4 prompts / 4 tasks','Needle retrieval: 9 prompts']):
        ys=[values[(suite,m)] for m in methods]
        bars=ax.bar(range(len(methods)),ys,color=colors)
        ax.bar_label(bars,labels=[f'{y:.1f}' for y in ys],padding=3,fontsize=8)
        ax.set_xticks(range(len(methods)),labels,rotation=35,ha='right')
        ax.set_ylim(0,115);ax.set_ylabel('Quality (0–100)');ax.set_title(title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Qwen-2.5-7B exploratory pilot; three greedy repeats, not independent quality samples',fontsize=10)
    fig.tight_layout()
    fig.savefig('results/quality_comparison.png',dpi=180)
    fig.savefig('results/quality_comparison.pdf')


if __name__=='__main__':main()
