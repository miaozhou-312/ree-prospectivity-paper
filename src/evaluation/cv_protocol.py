"""
Section 6 -- Evaluation protocol: fold-safe pooled out-of-fold (OOF)
scoring, repeated cross-validation.

One evaluation engine shared between Random Forest and DevNet so the two
are judged under an identical protocol:

- `pooled_oof_eval` runs one StratifiedKFold, converts each fold's
  validation scores to within-fold percentile ranks (rankdata / n), and
  pools all folds' ranks + labels into a SINGLE AUC/AP -- never an average
  of per-fold AUCs.
- `repeated_pooled_auc` repeats that whole pooled-OOF procedure
  `n_repeats` times (RepeatedStratifiedKFold) and returns the array of
  per-repeat pooled AUCs, from which the median and a [2.5, 97.5]
  percentile interval are reported.
- `repeated_pooled_auc_ap` is the same idea but returns both per-repeat
  pooled AUC and AP (used by the positive-count ablation).
- `repeated_pooled_oof_scores` returns each point's out-of-fold score
  averaged across all repeats, instead of collapsing to a pooled AUC (used
  by the label-confidence stratification analysis, which needs a per-point
  score rather than a single summary statistic).

A `fold_scorer(X_tr_raw, y_tr, X_va_raw, seed_counter)` callable is the only
thing that differs between models -- src/models/random_forest.py and
src/models/devnet.py each provide one.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def pooled_oof_eval(X_raw, y, fold_scorer, n_splits, seed):
    """One stratified CV pass. Returns (auc, ap, oof_rank) where oof_rank is
    the per-fold percentile rank of every sample's out-of-fold score, pooled
    across folds before computing a single AUC/AP."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(y)
    oof_rank = np.full(n, np.nan)
    for k, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
        va_scores = fold_scorer(X_raw[tr_idx], y[tr_idx], X_raw[va_idx], seed_counter=k)
        oof_rank[va_idx] = rankdata(va_scores) / len(va_scores)
    auc = roc_auc_score(y, oof_rank)
    ap  = average_precision_score(y, oof_rank)
    return auc, ap, oof_rank


def repeated_pooled_auc(X_raw, y, fold_scorer, n_splits, n_repeats, seed):
    """Distribution of the pooled-OOF AUC across independent n_splits-fold
    splits. Returns an array of length n_repeats (one pooled AUC per
    repeat) -- report median + [2.5, 97.5] percentile, not a single mean."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    n = len(y)
    repeat_aucs = []
    oof_rank = np.full(n, np.nan)
    counter = 0
    for k, (tr_idx, va_idx) in enumerate(rskf.split(X_raw, y)):
        if k > 0 and k % n_splits == 0:
            repeat_aucs.append(roc_auc_score(y, oof_rank))
            oof_rank = np.full(n, np.nan)
        va_scores = fold_scorer(X_raw[tr_idx], y[tr_idx], X_raw[va_idx], seed_counter=counter)
        oof_rank[va_idx] = rankdata(va_scores) / len(va_scores)
        counter += 1
    repeat_aucs.append(roc_auc_score(y, oof_rank))
    return np.array(repeat_aucs)


def repeated_pooled_auc_ap(X_raw, y, fold_scorer, n_splits, n_repeats, seed):
    """Like repeated_pooled_auc but returns BOTH per-repeat pooled AUC and
    AP. Returns (aucs, aps), each an array of length n_repeats. Used by the
    Cordillera positive-count ablation (src/evaluation/ablation_positive_count.py)."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    n = len(y)
    aucs, aps = [], []
    oof_rank = np.full(n, np.nan)
    counter = 0
    for k, (tr_idx, va_idx) in enumerate(rskf.split(X_raw, y)):
        if k > 0 and k % n_splits == 0:
            aucs.append(roc_auc_score(y, oof_rank))
            aps.append(average_precision_score(y, oof_rank))
            oof_rank = np.full(n, np.nan)
        va_scores = fold_scorer(X_raw[tr_idx], y[tr_idx], X_raw[va_idx], seed_counter=counter)
        oof_rank[va_idx] = rankdata(va_scores) / len(va_scores)
        counter += 1
    aucs.append(roc_auc_score(y, oof_rank))
    aps.append(average_precision_score(y, oof_rank))
    return np.array(aucs), np.array(aps)


def repeated_pooled_oof_scores(X_raw, y, fold_scorer, n_splits, n_repeats, seed):
    """Same RepeatedStratifiedKFold(seed=seed) split sequence as
    repeated_pooled_auc, but returns each point's out-of-fold score
    averaged over all n_repeats repeats (instead of collapsing to a pooled
    AUC). Calling this with the same (X_raw, y, seed) for two different
    fold_scorers guarantees both see identical folds -- required for a fair
    RF-vs-DevNet per-point comparison (see
    src/evaluation/label_confidence_stratification.py). Only ever pass OOF
    scores here, never a production model's training-set scores -- see that
    module's docstring for why."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    n = len(y)
    score_sum = np.zeros(n)
    score_cnt = np.zeros(n)
    for k, (tr_idx, va_idx) in enumerate(rskf.split(X_raw, y)):
        va_scores = fold_scorer(X_raw[tr_idx], y[tr_idx], X_raw[va_idx], seed_counter=k)
        score_sum[va_idx] += rankdata(va_scores) / len(va_scores)
        score_cnt[va_idx] += 1
    return score_sum / score_cnt


def top_decile_flags(oof_scores):
    """True where oof_scores ranks in the top 10% of the WHOLE array passed
    in -- used against a region's full labeled pool (positive + eval_neg) to
    define "hit rate" (src/evaluation/label_confidence_stratification.py)."""
    pct = rankdata(oof_scores) / len(oof_scores)
    return pct >= 0.90
