"""
Superseded / dead code, kept for the record rather than deleted.

Section 12b of the original research notebook defines `plot_capture_curves`
TWICE, back-to-back, in the same cell: the version below (verbose technical
title, boxed legend, per-panel legend) is immediately overwritten by a
second, cleaner "publication-quality" definition before the function is
ever called -- so this first version never actually ran. Only the second
definition made it into src/evaluation/capture_rate.py.

Not reviewed or maintained -- kept only so the notebook's history is
traceable. See src/evaluation/capture_rate.py for the version actually used.
"""
import numpy as np


def plot_capture_curves(capture_df, results, run_regions, capture_curve_oof, x_max_pct=10.0):
    """Dead code -- shadowed by the second definition of the same name in
    the original notebook cell before ever being invoked."""
    import matplotlib.pyplot as plt

    colors = {'RF': '#2a78d6', 'DevNet(deviation)': '#d03b3b'}
    mark_pcts = [1, 2, 3, 4, 5, 10]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, name in zip(axes, run_regions):
        for model in results[name].keys():
            res = results[name][model].get('pruned')
            if res is None or 'oof_rank' not in res:
                continue
            c = colors.get(model)
            x_vals, capture = capture_curve_oof(res['oof_rank'], res['oof_y'], x_max_pct=x_max_pct)
            ax.plot(x_vals, capture, color=c, lw=2, label=model)

            sub = capture_df[(capture_df['region'] == name) & (capture_df['model'] == model)]
            for pct in mark_pcts:
                row = sub[sub['area_pct'] == pct]
                if len(row):
                    ax.scatter(row['area_pct'], row['capture_rate'], color=c, s=45,
                               zorder=5, edgecolors='black', linewidths=0.5)

        ax.plot([0, x_max_pct], [0, x_max_pct / 100.0], color='grey', lw=1.3, ls='--',
                label='Random baseline (y = x)')
        ax.set_xlim(0, x_max_pct); ax.set_ylim(0, 1.02)
        ax.set_xlabel('Top-ranked search area (%)')
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7.5, loc='lower right')
        ax.grid(alpha=0.25); ax.set_axisbelow(True)
    axes[0].set_ylabel('Fraction of known occurrences captured (OOF)')

    fig.suptitle('Capture-rate curves (pooled out-of-fold, leakage-free; '
                 'truncated to 0-20% search area)', fontsize=11)
    plt.tight_layout()
    plt.show()
