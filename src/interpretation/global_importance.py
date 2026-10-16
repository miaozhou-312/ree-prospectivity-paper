"""
Sections 13c / 13e -- Global feature importance (RF vs. DevNet) and the
direction-consistency check.

`plot_global_importance_comparison` plots mean(|SHAP value|) per feature,
the standard global-importance summary: large values mean the feature moves
the prediction a lot (in either direction) across the explain set.

`direction_consistency_table` reports, for the features that matter most to
EITHER model, each model's average SHAP sign -- whether a higher value of
that feature pushes the prediction toward "more prospective" or "less
prospective" on average -- and flags whether the two models agree. Caveat:
"average sign" is a simplification -- a feature's effect need not be
monotonic for every point.
"""
import numpy as np
import pandas as pd


def plot_global_importance_comparison(shap_rf, shap_dn, feat_cols, name):
    import matplotlib.pyplot as plt

    imp_rf = np.abs(shap_rf).mean(axis=0)
    imp_dn = np.abs(shap_dn).mean(axis=0)
    order = np.argsort(-(imp_rf + imp_dn))
    feats_sorted = [feat_cols[i] for i in order]

    y = np.arange(len(feats_sorted))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, max(4, len(feats_sorted) * 0.28)))
    ax.barh(y + h / 2, imp_rf[order], height=h, color='#2a78d6', label='RF (TreeSHAP)')
    ax.barh(y - h / 2, imp_dn[order], height=h, color='#d03b3b', label='DevNet (SHAP)')
    ax.set_yticks(y); ax.set_yticklabels(feats_sorted, fontsize=8)
    ax.invert_yaxis()  # highest combined importance at top
    ax.set_xlabel('mean(|SHAP value|)')
    ax.set_title(f'{name}: global feature importance — RF vs DevNet')
    ax.legend(frameon=False)
    ax.grid(axis='x', color='#e1e0d9', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def direction_consistency_table(shap_rf, shap_dn, feat_cols, top_k=10):
    imp_rf = np.abs(shap_rf).mean(axis=0)
    imp_dn = np.abs(shap_dn).mean(axis=0)
    norm_rf = imp_rf / imp_rf.max() if imp_rf.max() > 0 else imp_rf
    norm_dn = imp_dn / imp_dn.max() if imp_dn.max() > 0 else imp_dn
    top_idx = np.argsort(-(norm_rf + norm_dn))[:top_k]

    # KNOWN ISSUE #1: sign(mean SHAP) is not a feature direction (baseline-dependent) -- see KNOWN_ISSUES.md
    mean_rf = shap_rf.mean(axis=0)
    mean_dn = shap_dn.mean(axis=0)

    rows = []
    for i in top_idx:
        dir_rf = 'higher -> more prospective' if mean_rf[i] > 0 else 'higher -> less prospective'
        dir_dn = 'higher -> more prospective' if mean_dn[i] > 0 else 'higher -> less prospective'
        rows.append({
            'feature': feat_cols[i],
            'RF_importance': round(imp_rf[i], 4), 'RF_direction': dir_rf,
            'DevNet_importance': round(imp_dn[i], 4), 'DevNet_direction': dir_dn,
            'direction_agree': (mean_rf[i] > 0) == (mean_dn[i] > 0),
        })
    return pd.DataFrame(rows)
