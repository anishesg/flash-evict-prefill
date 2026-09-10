"""Export the completed performance comparison without mixing memory scopes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path('results')
    kernel = json.loads((root/'policy_benchmark.json').read_text())
    model = json.loads((root/'model_benchmark_chunked.json').read_text())
    if not model.get('complete'):
        raise ValueError('Whole-model benchmark is incomplete')
    names = ['snap_w32', 'd01_r0', 'd0.1_w128_r0', 'd0.1_w128_r64']
    labels = ['SnapKV', 'Decay 0.1, all queries', 'Decay 0.1, window 128', 'Window 128, recent 64']
    colors = ['#d97706', '#2563eb', '#0891b2', '#7c3aed']
    selected = []
    for name in names:
        row = next(r for r in kernel['rows'] if (r['n'], r['method'], r['budget']) == (131072, name, 1024))
        if row.get('other_compute_pids') or 'error' in row:
            raise ValueError(f'Invalid kernel measurement: {name}')
        selected.append(row)
    model_names = ['full', 'snap_1024', *names[1:]]
    whole = []
    for name in model_names:
        row = next(r for r in model['rows'] if (r['n'], r['method']) == (32768, name))
        if row.get('other_compute_pids') or 'error' in row:
            raise ValueError(f'Invalid model measurement: {name}')
        whole.append(row)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    def panel(ax, values, names, palette, title, suffix, decimals):
        ax.barh(range(len(values)), values, color=palette, height=.62)
        ax.set_yticks(range(len(values)), names)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values)*1.22)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
        ax.set_xlabel(suffix)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        for index, value in enumerate(values):
            ax.text(value+max(values)*.018, index, f'{value:.{decimals}f}', va='center')

    panel(axes[0, 0], [r['median_ms']/1000 for r in selected], labels, colors,
          'One layer · 131,072 tokens', 'Median seconds', 2)
    panel(axes[0, 1], [r['peak_increment_bytes']/2**20 for r in selected], ['']*4, colors,
          'One layer · additional allocation', 'Peak MiB (inputs and weights excluded)', 1)
    panel(axes[1, 0], [r['median_seconds'] for r in whole], ['Full cache', *labels], ['#64748b', *colors],
          'All 28 layers · 32,768 tokens', 'Median seconds', 2)
    panel(axes[1, 1], [r['peak_allocated_bytes']/2**30 for r in whole], ['']*5, ['#64748b', *colors],
          'All 28 layers · absolute allocation', 'Peak GiB (weights and activations included)', 2)
    fig.suptitle('Qwen-2.5-7B shapes · A10G · BF16 · cache budget 1,024', fontsize=13, fontweight='bold', y=.99)
    fig.text(.5, .015, 'Whole model: common MLP chunk size 1,024; default allocator; excludes LM head and decode.\n'
             'All tested whole-model configurations OOM at 131,072 tokens. Quality is evaluated separately.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .96), h_pad=2.5, w_pad=2.0)
    for extension in ('png', 'pdf'):
        fig.savefig(root/f'followup_performance.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    source = dict(kernel=[{k: r[k] for k in ('n', 'method', 'budget', 'median_ms', 'peak_increment_bytes')} for r in selected],
                  whole_model=[{k: r[k] for k in ('n', 'method', 'budget', 'median_seconds', 'peak_allocated_bytes')} for r in whole],
                  whole_model_configuration=model['args'])
    (root/'followup_performance_data.json').write_text(json.dumps(source, indent=2)+'\n')


if __name__ == '__main__':
    main()
