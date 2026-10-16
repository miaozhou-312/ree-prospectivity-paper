"""
Evaluation protocols for the paper.

`repeated_random_cv` is src.evaluation.cv_protocol.repeated_pooled_auc_ap
with the per-repeat pooled OOF ranks kept as well (same
RepeatedStratifiedKFold split sequence, same seed_counter sequence, same
within-fold rank normalisation), so the AUC values reproduce the thesis
table and every model sees identical folds.

`repeated_spatial_cv` replaces the random split with spatial blocks:
labelled points (positives + eval_neg) are assigned to square blocks of
side `block_km` in EPSG:3978, whole blocks are held out together
(StratifiedGroupKFold), and the shared training background is purged of
every point that falls in a held-out block, so neither held-out positives
nor their spatial neighbourhood are seen in training. The block grid gets a
random offset per repeat so results do not hinge on one grid alignment.

`paired_bootstrap` compares two models' repeat-averaged OOF scores on the
same points (stratified resampling of positives and background).
"""
import numpy as np
import pyproj
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold


def repeated_random_cv(X_lab, y, X_bg, make_scorer, n_splits, n_repeats, seed):
    scorer = make_scorer(X_bg)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = np.full((n_repeats, len(y)), np.nan)
    for k, (tr, va) in enumerate(rskf.split(X_lab, y)):
        s = scorer(X_lab[tr], y[tr], X_lab[va], seed_counter=k)
        oof[k // n_splits, va] = rankdata(s) / len(s)
    return _summarise(y, oof)


def project_xy(lat, lon, crs='EPSG:3978'):
    tr = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    return np.column_stack(tr.transform(np.asarray(lon), np.asarray(lat)))


def block_ids(xy_km, block_km, offset):
    ij = np.floor((xy_km - offset) / block_km).astype(np.int64)
    return ij[:, 0] * 1_000_003 + ij[:, 1]


def repeated_spatial_cv(X_lab, y, xy_lab_km, X_bg, xy_bg_km, make_scorer, block_km,
                        n_splits, n_repeats, seed):
    rng = np.random.default_rng(seed)
    oof = np.full((n_repeats, len(y)), np.nan)
    info = []
    counter = 0
    for r in range(n_repeats):
        offset = rng.uniform(0, block_km, size=2)
        g_lab = block_ids(xy_lab_km, block_km, offset)
        g_bg = block_ids(xy_bg_km, block_km, offset)
        n_pos_groups = len(np.unique(g_lab[y == 1]))
        k = min(n_splits, n_pos_groups)
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        purged = []
        for tr, va in sgkf.split(X_lab, y, groups=g_lab):
            keep_bg = ~np.isin(g_bg, np.unique(g_lab[va]))
            purged.append(int((~keep_bg).sum()))
            scorer = make_scorer(X_bg[keep_bg])
            s = scorer(X_lab[tr], y[tr], X_lab[va], seed_counter=counter)
            oof[r, va] = rankdata(s) / len(s)
            counter += 1
        info.append({'repeat': r, 'n_folds': k, 'n_pos_blocks': n_pos_groups,
                     'mean_bg_purged': float(np.mean(purged))})
    out = _summarise(y, oof)
    out['fold_info'] = info
    return out


def _summarise(y, oof):
    aucs = np.array([roc_auc_score(y, o) for o in oof])
    aps = np.array([average_precision_score(y, o) for o in oof])
    return {'aucs': aucs, 'aps': aps, 'oof': oof, 'oof_mean': oof.mean(axis=0),
            'auc_med': float(np.median(aucs)), 'auc_ci': tuple(np.percentile(aucs, [2.5, 97.5])),
            'ap_med': float(np.median(aps)), 'ap_ci': tuple(np.percentile(aps, [2.5, 97.5]))}


def paired_bootstrap(y, s_a, s_b, n_boot=5000, seed=0):
    """Stratified paired bootstrap of AP and AUC differences (a - b).
    Returns point differences, 95% percentile CIs and one-sided p-values
    P(diff <= 0)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d_ap, d_auc = [], []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        yy = y[idx]
        d_ap.append(average_precision_score(yy, s_a[idx]) - average_precision_score(yy, s_b[idx]))
        d_auc.append(roc_auc_score(yy, s_a[idx]) - roc_auc_score(yy, s_b[idx]))
    d_ap, d_auc = np.array(d_ap), np.array(d_auc)
    return {
        'd_ap': average_precision_score(y, s_a) - average_precision_score(y, s_b),
        'd_ap_ci': tuple(np.percentile(d_ap, [2.5, 97.5])), 'p_ap': float((d_ap <= 0).mean()),
        'd_auc': roc_auc_score(y, s_a) - roc_auc_score(y, s_b),
        'd_auc_ci': tuple(np.percentile(d_auc, [2.5, 97.5])), 'p_auc': float((d_auc <= 0).mean()),
    }
