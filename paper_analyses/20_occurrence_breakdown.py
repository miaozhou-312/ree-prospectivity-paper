"""
Stage 20 -- where the Superior skill comes from, and prediction-area plots.

  A  per-occurrence breakdown: for every Superior positive, its out-of-fold
     percentile among the evaluation background (mean over the six
     repeats) for DevNet, RF, bagging-PU, nnPU (quantile inputs) and, when
     Stage 17 has run, MLP-BCE, under random CV and 100 km blocks; positives
     are grouped into the eastern-Abitibi cluster (the 50 km cluster holding
     Montviel and Lac Shortt) and the dispersed occurrences.
  B  prediction-area (P-A) curves (Yousefi & Carranza 2015) from the same
     out-of-fold ranks: share of held-out occurrences above a score
     percentile against the share of the evaluation background (a random
     sample of the domain, i.e. area) above it; the intersection is the
     usual P-A summary (occurrences captured = 1 - area).

    python -m paper_analyses.20_occurrence_breakdown
"""
import glob
import importlib
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paper_analyses.common import load_pickle, load_region, out_path  # noqa: E402

_g = importlib.import_module('paper_analyses.06b_figures_tables_revised')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, 'paper', 'manuscript', 'figures')
LABEL = {'DevNet': 'DevNet', 'RF': 'RF', 'BaggingPU': 'Bagging-PU', 'nnPU': 'nnPU', 'MLP-BCE': 'MLP-BCE'}
COLORS = {**_g.COLORS, 'MLP-BCE': '#e08a1e'}


def stores():
    rnd, blk = _g.stores_superior()
    ctl = sorted(glob.glob(out_path('controls_*Superior*.pkl')))
    for f in ctl:
        st = load_pickle(f)
        if ('Superior', 'MLP-BCE', 'random') in st:
            rnd['MLP-BCE'], blk['MLP-BCE'] = st[('Superior', 'MLP-BCE', 'random')], st[('Superior', 'MLP-BCE', 100.0)]
    return rnd, blk


def percentile_of_positives(y, oof):
    """Mean over repeats of each positive's percentile among the background."""
    bg = y == 0
    out = []
    for o in oof:
        b = np.sort(o[bg])
        out.append(np.searchsorted(b, o[y == 1], side='left') / len(b))
    return np.mean(out, axis=0)


def pa_curve(y, oof, thresholds):
    pred, area = [], []
    for o in oof:
        b, p = o[y == 0], o[y == 1]
        cut = np.quantile(b, thresholds)
        pred.append([(p >= c).mean() for c in cut])
        area.append([(b >= c).mean() for c in cut])
    return np.mean(pred, 0), np.mean(area, 0)


def main():
    rec = load_region('Superior')
    y = rec['pruned']['y']
    diag = pd.read_csv(out_path('region_diagnostics_positives.csv'))
    diag = diag[diag.region == 'Superior'].reset_index(drop=True)
    cluster_id = diag.loc[diag.name == 'Montviel', 'cluster_50km'].iloc[0]
    group = np.where(diag.cluster_50km == cluster_id, 'Abitibi cluster', 'dispersed')
    rnd, blk = stores()
    rows = pd.DataFrame({'name': diag.name, 'class': diag['class'], 'group': group})
    for proto, st in [('random', rnd), ('100 km', blk)]:
        for m, res in st.items():
            rows[f'{m} | {proto}'] = percentile_of_positives(y, res['oof'])
    rows.to_csv(out_path('occurrence_breakdown_Superior.csv'), index=False)
    summ = []
    for proto in ['random', '100 km']:
        for m in rnd:
            for gname in ['Abitibi cluster', 'dispersed']:
                v = rows.loc[rows.group == gname, f'{m} | {proto}']
                summ.append({'protocol': proto, 'model': m, 'group': gname, 'n': len(v),
                             'median_pct': float(v.median()), 'top1': float((v >= 0.99).mean()),
                             'top5': float((v >= 0.95).mean())})
    summ = pd.DataFrame(summ)
    summ.to_csv(out_path('occurrence_breakdown_summary_Superior.csv'), index=False)
    print(summ.round(3).to_string(index=False))

    th = np.linspace(0, 0.999, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    pa = []
    for ax, (title, st) in zip(axes, [('(a) random stratified CV', rnd), ('(b) 100 km spatial blocks', blk)]):
        for m in ['DevNet', 'RF', 'BaggingPU', 'nnPU', 'MLP-BCE']:
            if m not in st:
                continue
            p, a = pa_curve(y, st[m]['oof'], th)
            k = np.argmin(np.abs(p - (1 - a)))
            pa.append({'protocol': title[4:], 'model': m, 'threshold_pct': th[k], 'captured_at_intersection': p[k],
                       'area_at_intersection': a[k]})
            ax.plot(th * 100, p * 100, color=COLORS[m], lw=1.6, label=f'{LABEL[m]} occurrences')
            ax.plot(th * 100, a * 100, color=COLORS[m], lw=1.0, ls='--')
            ax.plot(th[k] * 100, p[k] * 100, 'o', color=COLORS[m], ms=5)
        ax.set_xlim(80, 100)
        ax.set_xlabel('score percentile threshold (%)')
        ax.set_title(title, loc='left', fontsize=10)
    axes[0].set_ylabel('occurrences above threshold (solid, %)\narea above threshold (dashed, %)')
    axes[1].legend(frameon=False, fontsize=7.5, loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_pa_superior_rev.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    pa = pd.DataFrame(pa)
    pa.to_csv(out_path('pa_intersections_Superior.csv'), index=False)
    print(pa.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
