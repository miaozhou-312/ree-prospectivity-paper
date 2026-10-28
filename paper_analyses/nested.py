"""
Nested selection inside an outer training fold.

The scheme of the Stage-10 nested early stopping, generalised to any finite
list of candidates (design choices, hyperparameters): the labelled rows of
the outer training fold are split into 100 km blocks (3 inner folds,
StratifiedGroupKFold), training background falling in an inner validation
block is purged, every candidate is fitted on each inner training part and
scored on the inner validation part, and the candidate with the highest
pooled inner AP is refitted on the whole outer training fold. The outer
test fold is never used. Inner folds without a training positive are
skipped; with fewer than two positive blocks the first candidate is kept.

Scorers built here receive rows whose last two columns are projected x, y
(km), used only for the inner split (as in Stage 10).
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from paper_analyses.common import config
from paper_analyses.protocol import block_ids

INNER_BLOCK_KM, INNER_FOLDS = 100.0, 3


def inner_select(candidates, fit_score, X_tr, y_tr, xy_tr, X_bg, xy_bg, seed_counter):
    """fit_score(candidate, X_pos, X_bg, X_eval, seed) -> scores of X_eval.
    Returns (chosen candidate, {candidate: pooled inner AP})."""
    rng = np.random.default_rng(config.RANDOM_SEED + seed_counter)
    offset = rng.uniform(0, INNER_BLOCK_KM, size=2)
    g_tr, g_bg = block_ids(xy_tr, INNER_BLOCK_KM, offset), block_ids(xy_bg, INNER_BLOCK_KM, offset)
    k = min(INNER_FOLDS, len(np.unique(g_tr[y_tr == 1])))
    if k < 2:
        return candidates[0], {}
    inner = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_SEED + seed_counter)
    ranks = {c: np.full(len(y_tr), np.nan) for c in candidates}
    for i, (itr, iva) in enumerate(inner.split(X_tr, y_tr, groups=g_tr)):
        if not (y_tr[itr] == 1).any():
            continue
        keep = ~np.isin(g_bg, np.unique(g_tr[iva]))
        X_pos = X_tr[itr][y_tr[itr] == 1]
        for c in candidates:
            s = fit_score(c, X_pos, X_bg[keep], X_tr[iva], 7919 * (seed_counter + 1) + 100 * i)
            ranks[c][iva] = rankdata(s) / len(s)
    scored = ~np.isnan(ranks[candidates[0]])
    if not (y_tr[scored] == 1).any():
        return candidates[0], {}
    ap = {c: average_precision_score(y_tr[scored], ranks[c][scored]) for c in candidates}
    return max(candidates, key=lambda c: (ap[c], -candidates.index(c))), ap   # ties -> earlier candidate


def nested_factory(candidates, fit_score, final_fit_score, n_feat, log):
    """Scorer factory for repeated_random_cv / repeated_spatial_cv. `log`
    collects the chosen candidate and inner APs of every outer fold."""
    def make(X_bg_aug):
        X_bg, xy_bg = X_bg_aug[:, :n_feat], X_bg_aug[:, n_feat:]

        def scorer(X_tr_aug, y_tr, X_va_aug, seed_counter=0):
            X_tr, xy_tr = X_tr_aug[:, :n_feat], X_tr_aug[:, n_feat:]
            chosen, ap = inner_select(candidates, fit_score, X_tr, y_tr, xy_tr, X_bg, xy_bg, seed_counter)
            log.append({'chosen': chosen, 'inner_ap': ap})
            return final_fit_score(chosen, X_tr[y_tr == 1], X_bg, X_va_aug[:, :n_feat], seed_counter)
        return scorer
    return make
