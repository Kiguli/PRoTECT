"""
Render every JSON figure in submit/results/ as a PNG using matplotlib.
The JSON schema is the portal's tikz-conversion format
(see memory/arch_comp_nln_figure_format.md). We render each PlotData
entry: filled polygons for Fill=True, polylines for Fill=False.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def render(json_path, png_path):
    with open(json_path) as f:
        fig = json.load(f)

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # Plot in a sensible Z-order: filled regions first, lines on top.
    for pd in fig.get('PlotData', []):
        name = pd.get('DisplayName', '')
        if not pd.get('Data'):
            continue
        if pd.get('Fill'):
            ec = pd.get('EdgeColor', [0, 0, 0])
            fc = pd.get('FaceColor', [0.7, 0.7, 0.7])
            patches = []
            for seg in pd['Data']:
                xs = seg.get('x', [])
                ys = seg.get('y', [])
                if len(xs) >= 3:
                    patches.append(Polygon(list(zip(xs, ys)), closed=True))
            if patches:
                pc = PatchCollection(patches, edgecolor=ec, facecolor=fc,
                                     linewidths=1.5, alpha=0.45,
                                     label=name)
                ax.add_collection(pc)
                # add a proxy artist for the legend
                ax.plot([], [], 's', color=fc, markeredgecolor=ec,
                        markersize=10, label=name)
        else:
            color = pd.get('Color', pd.get('EdgeColor', [0, 0, 0]))
            for i, seg in enumerate(pd['Data']):
                xs = seg.get('x', [])
                ys = seg.get('y', [])
                if len(xs) < 2:
                    continue
                ax.plot(xs, ys, '-', color=color, linewidth=2,
                        label=name if i == 0 else None)

    xlim = fig.get('XLim') or None
    ylim = fig.get('YLim') or None
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_xlabel('$' + fig.get('XLabel', '') + '$' if fig.get('XLabel') else '')
    ax.set_ylabel('$' + fig.get('YLabel', '') + '$' if fig.get('YLabel') else '')
    title = fig.get('Title', '')
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='best', fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(png_path, dpi=130)
    plt.close()


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, 'results')
    paths = sorted(p for p in glob.glob(os.path.join(results_dir, '*.json'))
                   if not p.endswith('.result.json'))
    if not paths:
        print('no JSON figures found under', results_dir)
        sys.exit(1)
    for p in paths:
        png = p.replace('.json', '.png')
        try:
            render(p, png)
            print(f'  {os.path.basename(png)}')
        except Exception as exc:
            print(f'  FAIL {os.path.basename(p)}: {exc}')
