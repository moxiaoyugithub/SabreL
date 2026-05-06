from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

path = 'logs/tensorboard_logs/palmtree/events.out.tfevents.1776325879.p6000.1984868.0'
out_svg = Path('paper_figures/palmtree_loss_curves.svg')
out_pdf = Path('paper_figures/palmtree_loss_curves.pdf')
out_png = Path('paper_figures/palmtree_loss_curves.png')
out_tsv = Path('paper_figures/palmtree_loss_curves.tsv')

ea = event_accumulator.EventAccumulator(path)
ea.Reload()

tags = ['Loss/Value_Loss', 'Loss/Action_Loss', 'Analysis/Entropy']
labels = {
    'Loss/Value_Loss': 'Value Loss',
    'Loss/Action_Loss': 'Action Loss',
    'Analysis/Entropy': 'Entropy',
}
colors = {
    'Loss/Value_Loss': '#1f77b4',
    'Loss/Action_Loss': '#d55e00',
    'Analysis/Entropy': '#009e73',
}
linestyles = {
    'Loss/Value_Loss': '-',
    'Loss/Action_Loss': '--',
    'Analysis/Entropy': '-.',
}

def moving_average(y, window=15):
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    for i in range(len(y)):
        start = max(0, i - window + 1)
        out[i] = y[start:i+1].mean()
    return out

series = {}
for tag in tags:
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=int)
    values = np.array([e.value for e in events], dtype=float)
    series[tag] = {
        'steps': steps,
        'values': values,
        'smooth': moving_average(values, window=15),
    }

with out_tsv.open('w', encoding='utf-8') as f:
    f.write('tag\tstep\tvalue\tsmoothed\n')
    for tag in tags:
        s = series[tag]
        for step, value, smooth in zip(s['steps'], s['values'], s['smooth']):
            f.write(f'{tag}\t{step}\t{value}\t{smooth}\n')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10.5,
})

fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=300)

for tag in tags:
    s = series[tag]
    ax.plot(
        s['steps'],
        s['values'],
        color=colors[tag],
        linestyle=linestyles[tag],
        linewidth=1.0,
        alpha=0.18,
    )
    ax.plot(
        s['steps'],
        s['smooth'],
        color=colors[tag],
        linestyle=linestyles[tag],
        linewidth=2.2,
        label=labels[tag],
    )

ax.set_xlabel('Training Episodes (x16 Parallel Environments)')
ax.set_ylabel('Scalar Value')
ax.set_title('Training Loss Curves', pad=10)
ax.grid(True, which='major', color='#d9d9d9', linewidth=0.8, alpha=0.8)
ax.grid(True, which='minor', color='#eeeeee', linewidth=0.5, alpha=0.8)
ax.minorticks_on()
ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=2.8)

all_steps = np.concatenate([series[t]['steps'] for t in tags])
ax.set_xlim(all_steps.min(), all_steps.max())

fig.tight_layout()
fig.savefig(out_svg, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight')

for tag in tags:
    s = series[tag]
    print(tag, 'points=', len(s['steps']), 'step_range=', (int(s['steps'].min()), int(s['steps'].max())), 'value_range=', (float(s['values'].min()), float(s['values'].max())))
print('Wrote', out_svg)
print('Wrote', out_pdf)
print('Wrote', out_png)
print('Wrote', out_tsv)