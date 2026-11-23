"""统一字体、尺寸、矢量导出及渲染后的质量检查。"""
from runtime import configure
configure()
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from config import ROOT
sys.path.insert(0, str(ROOT/'figure_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10.5, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9.5,
    'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def clean_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out')


def save_figure(fig, out, stem, axes, *, exclude=(), row_groups=None, column_groups=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    qa = out/'qa'
    qa.mkdir(exist_ok=True)
    fig.canvas.draw()
    for k, ax in enumerate(axes):
        if not ax.get_gid():
            ax.set_gid(f'panel{k+1}')
    options = {}
    if row_groups is not None:
        options['row_groups'] = row_groups
    if column_groups is not None:
        options['column_groups'] = column_groups
    require_matplotlib_panel_alignment(fig, axes=axes,
        panel_ids=[a.get_gid() for a in axes], exclude_axes=list(exclude),
        json_out=qa/f'{stem}.alignment.json', overlay_svg=qa/f'{stem}.alignment.svg',
        strict=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5, **options)
    fig.savefig(out/f'{stem}.pdf')
    fig.savefig(out/f'{stem}.svg')
    fig.savefig(out/f'{stem}.png', dpi=600)
    plt.close(fig)
    audit = subprocess.run([sys.executable, str(ROOT/'figure_qa/audit_pdf_text.py'),
        str(out/f'{stem}.pdf'), '--min-pt', '7', '--json'], capture_output=True, text=True)
    (qa/f'{stem}.fonts.json').write_text(audit.stdout, encoding='utf-8')
    collision = subprocess.run([sys.executable, str(ROOT/'figure_qa/audit_figure_collisions.py'),
        str(out/f'{stem}.pdf'), '--json-out', str(qa/f'{stem}.collisions.json')],
        capture_output=True, text=True)
    (qa/f'{stem}.collisions.txt').write_text(collision.stdout+collision.stderr, encoding='utf-8')
    if audit.returncode or collision.returncode:
        raise RuntimeError(f'{stem}: 字体/重叠检查未通过，请查看{qa}，不要把未通过文件当作最终图。')


def panel_grid(n, title, ylabel):
    """每页最多6个面板，以保证字体和每个绘图区尺寸。"""
    if not 1 <= n <= 6:
        raise ValueError('每页应包含1—6个面板。')
    rows = (n+2)//3
    height = 25 + 62*rows
    fig = plt.figure(figsize=(183/25.4, height/25.4))
    axes = []
    for k in range(n):
        r, c = divmod(k, 3)
        ax = fig.add_axes([(20+58*c)/183, (22+62*(rows-r-1))/height,
                           40/183, 40/height])
        ax.set_gid(f'p{k+1}')
        clean_axis(ax)
        ax.annotate(chr(97+k), (0, 1), xycoords='axes fraction', xytext=(0, 6),
                    textcoords='offset points', fontsize=11, fontweight='bold', va='bottom')
        axes.append(ax)
    fig.text(.5, (height-6)/height, title, ha='center', va='top', fontsize=11)
    fig.text(4/183, .5, ylabel, ha='center', va='center', rotation=90,
             rotation_mode='anchor', fontsize=11)
    row_groups = [{'panels': [f'p{k+1}' for k in range(r*3, min(r*3+3, n))]} for r in range(rows)]
    col_groups = [{'panels': [f'p{k+1}' for k in range(c, n, 3)]} for c in range(min(3, n))]
    return fig, axes, row_groups, col_groups


def packed_offsets(values, bins=60):
    """仅调整点的纵向排布；不改变任何SHAP横坐标或删除观测。"""
    values = np.asarray(values)
    if np.ptp(values) == 0:
        labels = np.zeros(len(values), dtype=int)
    else:
        labels = np.minimum(((values-values.min())/np.ptp(values)*bins).astype(int), bins-1)
    offsets = np.zeros(len(values))
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        if len(ids) > 1:
            lanes = np.array([0 if j==0 else ((j+1)//2)*(1 if j%2 else -1) for j in range(len(ids))])
            offsets[ids] = lanes/max(abs(lanes).max(), 1)*.28
    return offsets
