"""
Section 12b -- Capture-rate curve & table (OOF, leakage-free).

Reports a capture-rate curve rather than a Prediction-Area (P-A) plot
(Yousefi & Carranza, 2015): a P-A plot needs to locate a crossing point and
draw an area-rate line, both prone to degenerate cases (no-crossing,
near-flat curves) with a dataset this small. The curve and table below never
compute an intersection and never draw an area-rate line, avoiding that
failure mode entirely while answering the same practical question -- how
much of the known deposit inventory sits inside the top-ranked share of the
study area.

- `capture_rate_at_area_oof` -- the primary, sample-stable metric. Ranks
  the labeled pool (positive + eval_neg) by pooled-OOF score, takes the
  count-based top area_frac fraction of that pool (eval_neg-dominated, so
  this approximates "top area_frac of the mapped area"), and reports what
  fraction of KNOWN POSITIVES land inside that top slice.
- `capture_curve_oof` -- the same rule swept over a fine grid of x instead
  of four fixed thresholds.
- `plot_capture_curves` -- one panel per region, RF + DevNet as solid
  lines, y = x random-baseline dashed line, marker dots at 1/5/10% read
  from the table so they match it exactly.

Citations: Chung, C.J.F. and Fabbri, A.G. (1999). Probabilistic prediction
models for landslide hazard mapping. Photogrammetric Engineering and Remote
Sensing, 65(12), pp. 1389-1399. Agterberg, F.P. and Bonham-Carter, G.F.
(2005). Measuring the performance of mineral-potential maps. Natural
Resources Research, 14, pp. 1-17.
"""
import numpy as np
import pandas as pd


def capture_rate_at_area_oof(oof_rank, oof_y, area_fracs=(0.01, 0.02, 0.03, 0.04, 0.05, 0.10)):
    """Primary exploration-efficiency metric, OOF-only. Ranks the labeled
    pool (positive + eval_neg) by pooled-OOF score, takes the count-based
    top area_frac fraction of that pool, and reports what fraction of KNOWN
    POSITIVES land inside that top slice. No production model involved --
    can't leak, can't degenerate the way a crossing-point search can under
    a small sample."""
    m = np.isfinite(oof_rank)
    r, y = np.asarray(oof_rank)[m], np.asarray(oof_y)[m].astype(int)
    n_total = len(r)
    n_pos_total = int((y == 1).sum())
    order = np.argsort(-r)   # descending by OOF score
    y_sorted = y[order]

    rows = []
    for a in area_fracs:
        k = max(1, int(round(a * n_total)))
        n_pos_in_top = int((y_sorted[:k] == 1).sum())
        capture = n_pos_in_top / n_pos_total if n_pos_total > 0 else float('nan')
        rows.append({'area_pct': a * 100, 'n_top': k, 'n_pos_captured': n_pos_in_top,
                     'n_pos_total': n_pos_total, 'capture_rate': capture})
    return rows


def capture_curve_oof(oof_rank, oof_y, x_max_pct=20.0, n_points=201):
    """Continuous capture-rate CURVE over x in [0, x_max_pct]%. Same
    OOF-only, leakage-free data source and count-based top-k rule as
    capture_rate_at_area_oof, swept over a fine grid instead of four fixed
    thresholds."""
    m = np.isfinite(oof_rank)
    r, y = np.asarray(oof_rank)[m], np.asarray(oof_y)[m].astype(int)
    n_total = len(r)
    n_pos_total = int((y == 1).sum())
    order = np.argsort(-r)
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted == 1)

    x_vals = np.linspace(0.0, x_max_pct, n_points)
    k_vals = np.clip(np.round(x_vals / 100.0 * n_total).astype(int), 1, n_total)
    if n_pos_total > 0:
        capture = cum_pos[k_vals - 1] / n_pos_total
    else:
        capture = np.full(n_points, np.nan)
    return x_vals, capture


def build_capture_table(results, run_regions, area_fracs=(0.01, 0.02, 0.03, 0.04, 0.05, 0.10)):
    """Builds the capture-rate table from RESULTS[name][model]['pruned']
    (the dict produced by src/evaluation/run_experiment.py), one row per
    (region, model, area_pct)."""
    capture_rows = []
    for name in run_regions:
        for model in results[name].keys():
            res = results[name][model].get('pruned')
            if res is None or 'oof_rank' not in res:
                continue
            for row in capture_rate_at_area_oof(res['oof_rank'], res['oof_y'], area_fracs):
                capture_rows.append({'region': name, 'model': model, **row})
    return pd.DataFrame(capture_rows)


def plot_capture_curves(capture_df, results, run_regions, x_max_pct=10.0):
    """Publication-quality capture-rate curves. One panel per region,
    RF + DevNet as solid lines, y = x random baseline dashed. Marker dots at
    1/5/10% are read from capture_df so they match the table exactly."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {'RF': '#2f6fb0', 'DevNet(deviation)': '#c0392b'}
    mark_pcts = [1, 2, 3, 4, 5, 10]

    plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)

    for ax, name in zip(axes, run_regions):
        for model in results[name].keys():
            res = results[name][model].get('pruned')
            if res is None or 'oof_rank' not in res:
                continue
            c = colors.get(model, '#444444')
            x_vals, capture = capture_curve_oof(res['oof_rank'], res['oof_y'], x_max_pct=x_max_pct)
            ax.plot(x_vals, capture, color=c, lw=1.8, zorder=3)
            sub = capture_df[(capture_df['region'] == name) & (capture_df['model'] == model)]
            for pct in mark_pcts:
                row = sub[sub['area_pct'] == pct]
                if len(row):
                    ax.scatter(row['area_pct'], row['capture_rate'], color=c, s=32,
                               zorder=4, edgecolors='white', linewidths=0.6)
        ax.plot([0, x_max_pct], [0, x_max_pct / 100.0], color='0.55', lw=1.0,
                ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0, x_max_pct); ax.set_ylim(0, 1.0)
        ax.set_xlabel('Top-ranked search area (%)')
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.18, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(length=3)

    axes[0].set_ylabel('Fraction of occurrences\ncaptured (OOF)')
    handles = [Line2D([], [], color=colors['RF'], lw=1.8, label='RF'),
               Line2D([], [], color=colors['DevNet(deviation)'], lw=1.8, label='DevNet'),
               Line2D([], [], color='0.55', lw=1.0, ls=(0, (4, 3)), label='Random')]
    axes[0].legend(handles=handles, fontsize=8.5, loc='lower right',
                   frameon=True, framealpha=0.9, edgecolor='0.8')

    plt.tight_layout()
    plt.show()
