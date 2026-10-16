"""
Section 5a -- Unsupervised collinearity pruning (per region).

Each region has its own correlation structure among the ~50 candidate
features, so pruning is done independently per region and can keep a
different number/set of features in each. What must stay identical *within*
a region is that RF and DevNet see the same pruned set.

`prune_collinear(pts, feat_cols, thresh)`:

1. Computes a Spearman correlation matrix over the feature columns only --
   the label is never included, so this is purely unsupervised and cannot
   leak CV information.
2. Median-imputes the features only for this correlation calculation (does
   not alter `pts` or anything downstream) so NaNs don't break the
   correlation matrix.
3. Clusters features with hierarchical clustering (average-linkage,
   distance 1 - |corr|, cut at t = 1 - thresh) -- any two features
   correlated above `thresh` (absolute value) end up in the same cluster.
4. Keeps one representative per cluster: lowest missing-value rate first,
   ties broken by highest variance. Never by which one gives a better AUC --
   no supervised signal enters this choice at all.
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer


def prune_collinear(pts, feat_cols, thresh):
    """Unsupervised collinearity pruning. Clusters feat_cols by
    1 - |Spearman correlation| (average-linkage hierarchical clustering, cut
    at 1 - thresh) and keeps one representative per cluster: lowest
    missing-rate first, ties broken by highest variance. Never looks at the
    label column or any model metric -- purely a feature-to-feature
    statistic, so there is no CV leakage and no supervised feature
    selection. Returns the kept feature names, in their original relative
    order."""
    X = pts[feat_cols].astype(float).values
    X_imp = SimpleImputer(strategy='median').fit_transform(X)  # for the correlation calc only

    corr = pd.DataFrame(X_imp, columns=feat_cols).corr(method='spearman').values
    # A zero-variance column (e.g. a one-hot era/lithology dummy that never
    # occurs in this region's candidate pool) makes Spearman correlation
    # undefined (NaN) against every other column. Treat that as "no
    # relationship" (it can't be redundant with anything if it never
    # varies) rather than letting NaN reach linkage(), which rejects any
    # non-finite value outright.
    const_cols = [feat_cols[i] for i in np.where(np.nanvar(X_imp, axis=0) == 0)[0]]
    if const_cols:
        print(f'  prune_collinear: {len(const_cols)} zero-variance column(s) in this region '
              f'(kept as standalone features, treated as uncorrelated with everything): {const_cols}')
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # guard against float asymmetry
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method='average')
    cluster_ids = fcluster(Z, t=1.0 - thresh, criterion='distance')

    missing_rate = np.isnan(X).mean(axis=0)
    variance     = np.nanvar(X, axis=0)

    kept = []
    print(f'  prune_collinear(thresh={thresh}): {len(feat_cols)} features -> '
          f'{len(np.unique(cluster_ids))} groups')
    for cid in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 1:
            kept.append(feat_cols[idx[0]])
            continue
        # objective, fixed rule: lowest missingness wins; ties -> highest variance
        rep = sorted(idx, key=lambda i: (missing_rate[i], -variance[i]))[0]
        kept.append(feat_cols[rep])
        members = [feat_cols[i] for i in idx]
        print(f'    group {members} -> kept {feat_cols[rep]!r} '
              f'(missing={missing_rate[rep]:.2f}, var={variance[rep]:.3g})')
    # preserve the original master-list order rather than cluster-id order
    kept_set = set(kept)
    return [c for c in feat_cols if c in kept_set]


def plot_collinearity_heatmap(pts, feat_cols, kept_cols, name, thresh):
    """Full Spearman correlation heatmap for a region's candidate features,
    with the features prune_collinear kept bold-labeled -- a visual check
    that the unsupervised pruning is behaving sensibly."""
    X = pts[feat_cols].astype(float).values
    X_imp = SimpleImputer(strategy='median').fit_transform(X)
    corr = pd.DataFrame(X_imp, columns=feat_cols).corr(method='spearman').values
    corr = np.nan_to_num(corr, nan=0.0)  # zero-variance columns -> undefined correlation -> 0

    div_cmap = mcolors.LinearSegmentedColormap.from_list(
        'blue_red_div', ['#2a78d6', '#f0efec', '#d03b3b'])

    fig, ax = plt.subplots(figsize=(max(8, len(feat_cols) * 0.22), max(7, len(feat_cols) * 0.22)))
    im = ax.imshow(corr, cmap=div_cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat_cols))); ax.set_xticklabels(feat_cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(feat_cols))); ax.set_yticklabels(feat_cols, fontsize=6)
    kept_set = set(kept_cols)
    for tick, col in zip(ax.get_xticklabels(), feat_cols):
        tick.set_fontweight('bold' if col in kept_set else 'normal')
        tick.set_color('#0b0b0b' if col in kept_set else '#898781')
    for tick, col in zip(ax.get_yticklabels(), feat_cols):
        tick.set_fontweight('bold' if col in kept_set else 'normal')
        tick.set_color('#0b0b0b' if col in kept_set else '#898781')
    fig.colorbar(im, ax=ax, label='Spearman correlation', shrink=0.8)
    ax.set_title(f'{name}: feature correlation — bold = kept after '
                 f'|corr|>{thresh} pruning ({len(kept_cols)}/{len(feat_cols)})',
                 fontsize=10)
    plt.tight_layout()
    plt.show()
