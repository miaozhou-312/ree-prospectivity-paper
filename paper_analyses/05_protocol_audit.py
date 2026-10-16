"""
Stage 5 -- protocol audit: how much can the validation design of Parsa et
al. (2024) inflate test AUC on these data?

Parsa et al. (2024, Nat. Resour. Res. 33:1941-1965) replaced each S2 level-12
cell containing an occurrence by its four level-13 child cells ("we
quadrupled the number of positive labels"), drew an equal number of
negatives from the 5% of cells least cosine-similar to the positives, and
evaluated with a stratified random 70/30 split of the augmented labels,
repeated 50 times. The paper does not describe grouping child cells by their
parent, so sibling children of one occurrence can sit on both sides of the
split.

This script rebuilds that design on our data with the same S2 library
(s2sphere) and our feature pipeline (features extracted at every child-cell
centre with the src/ functions), then toggles one factor at a time:

  split      child-level random (as described) | grouped by parent cell
             | no augmentation (one label per occurrence)
  negatives  cosine-dissimilar (lowest 5% mean similarity, as described)
             | random from the A∧B background

Model: Random Forest (1000 trees) on min-max scaled features, balanced
labels, 50 stratified 70/30 iterations -- as in Parsa et al. Reported:
mean and 95% interval of test AUC, the share of test positives with a
sibling in training, and how similar siblings are in feature space.

    python -m paper_analyses.05_protocol_audit [--regions Superior ...]
"""
import argparse

import numpy as np
import pandas as pd
import s2sphere as s2
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from paper_analyses.common import (GLCM_RASTERS, REGION_ORDER, config, load_region, out_path,
                                   save_pickle, setup_regions)
from src.features import raster_preprocessing
from src.features.glcm import add_glcm_features
from src.features.proximity import (add_proximity_line_layers, add_proximity_worms, add_rasters,
                                    build_worm_trees)
from src.features.terrane import add_terrane_h3

N_ITER, TEST_FRAC, S2_PARENT_LEVEL, LOW_SIM_FRAC = 50, 0.30, 12, 0.05


def s2_children(lat, lon, level=S2_PARENT_LEVEL):
    cell = s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lon)).parent(level)
    out = []
    for ch in cell.children():
        ll = s2.LatLng.from_point(s2.Cell(ch).get_center())
        out.append((ll.lat().degrees, ll.lng().degrees))
    return out


def child_feature_table(region, cfg, clipped_r, clipped_v, worm_trees, cols):
    rec = load_region(region)
    pos = rec['lab'][rec['lab']['pool'] == 'positive'].reset_index(drop=True)
    rows = []
    for pid, (la, lo) in enumerate(zip(pos['lat'], pos['lon'])):
        for cla, clo in s2_children(la, lo):
            rows.append({'lat': cla, 'lon': clo, 'label': 1, 'parent': pid})
    ch = pd.DataFrame(rows)
    meta = ch[['parent']].copy()
    pts = ch[['lat', 'lon', 'label']].copy()
    pts = add_terrane_h3(pts, clipped_v.get('h3_grid'))
    pts = add_proximity_line_layers(pts, cfg['vectors'], config.PROJECTED_CRS)
    pts = add_proximity_worms(pts, worm_trees, config.PROJECTED_CRS)
    pts = add_rasters(pts, clipped_r)
    pts = add_glcm_features(pts, clipped_r, GLCM_RASTERS)
    for c in cols:                      # a terrane dummy absent at every child
        if c not in pts:
            pts[c] = 0.0
    return pts[cols].astype(float).values, meta['parent'].values, pos


def cosine_dissimilar_negatives(X_pos, X_cand, n, rng):
    imp = SimpleImputer(strategy='median').fit(np.vstack([X_pos, X_cand]))
    X_pos, X_cand = imp.transform(X_pos), imp.transform(X_cand)
    mm = MinMaxScaler().fit(np.vstack([X_pos, X_cand]))
    P, C = mm.transform(X_pos), mm.transform(X_cand)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    mean_sim = (C @ P.T).mean(1)
    low = np.where(mean_sim <= np.quantile(mean_sim, LOW_SIM_FRAC))[0]
    return rng.choice(low, n, replace=len(low) < n), mean_sim


def run_rf(X_tr, y_tr, X_te, seed):
    imp = SimpleImputer(strategy='median').fit(X_tr)
    mm = MinMaxScaler().fit(imp.transform(X_tr))
    rf = RandomForestClassifier(n_estimators=1000, random_state=seed, n_jobs=-1)
    rf.fit(mm.transform(imp.transform(X_tr)), y_tr)
    return rf.predict_proba(mm.transform(imp.transform(X_te)))[:, 1]


def audit_region(region, X_child, parent, X_parent, X_cand):
    rng = np.random.default_rng(config.RANDOM_SEED)
    n_aug, n_par = len(X_child), len(X_parent)
    neg_sets = {}
    idx_cos, _ = cosine_dissimilar_negatives(X_child, X_cand, n_aug, rng)
    neg_sets[('cosine', 'aug')] = X_cand[idx_cos]
    neg_sets[('random', 'aug')] = X_cand[rng.choice(len(X_cand), n_aug, replace=False)]
    idx_cos_p, _ = cosine_dissimilar_negatives(X_parent, X_cand, n_par, rng)
    neg_sets[('cosine', 'none')] = X_cand[idx_cos_p]
    neg_sets[('random', 'none')] = X_cand[rng.choice(len(X_cand), n_par, replace=False)]

    rows = []
    for neg in ['cosine', 'random']:
        for split in ['child-level random', 'grouped by parent', 'no augmentation']:
            aug = 'none' if split == 'no augmentation' else 'aug'
            X_pos = X_parent if aug == 'none' else X_child
            X_neg = neg_sets[(neg, aug)]
            X = np.vstack([X_pos, X_neg])
            y = np.r_[np.ones(len(X_pos)), np.zeros(len(X_neg))]
            aucs, sib = [], []
            for it in range(N_ITER):
                if split == 'grouped by parent':
                    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=it)
                    p_tr, p_te = next(gss.split(X_pos, groups=parent))
                    n_idx = np.random.default_rng(it).permutation(len(X_neg))
                    n_te = n_idx[:int(round(TEST_FRAC * len(X_neg)))]
                    n_tr = n_idx[len(n_te):]
                    tr = np.r_[p_tr, len(X_pos) + n_tr]
                    te = np.r_[p_te, len(X_pos) + n_te]
                else:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=it)
                    tr, te = next(sss.split(X, y))
                if split == 'child-level random':
                    tr_par = set(parent[tr[tr < len(X_pos)]])
                    te_pos = te[te < len(X_pos)]
                    sib.append(np.mean([parent[i] in tr_par for i in te_pos]))
                s = run_rf(X[tr], y[tr], X[te], seed=it)
                aucs.append(roc_auc_score(y[te], s))
            aucs = np.array(aucs)
            rows.append({'region': region, 'negatives': neg, 'split': split,
                         'n_pos_labels': len(X_pos), 'n_neg_labels': len(X_neg),
                         'auc_mean': aucs.mean(), 'auc_lo': np.percentile(aucs, 2.5),
                         'auc_hi': np.percentile(aucs, 97.5),
                         'test_pos_with_sibling_in_train': np.mean(sib) if sib else np.nan})
            print(f"[{region}] neg={neg:7s} split={split:20s} AUC {aucs.mean():.3f} "
                  f"[{np.percentile(aucs, 2.5):.3f},{np.percentile(aucs, 97.5):.3f}]"
                  + (f"  test positives with a sibling in train: {np.mean(sib):.0%}" if sib else ''),
                  flush=True)
    return rows


def sibling_similarity(region, X_child, parent, X_parent, X_cand):
    sc = StandardScaler().fit(np.vstack([X_parent, X_cand]))
    imp = SimpleImputer(strategy='median').fit(np.vstack([X_parent, X_cand]))
    Zc = sc.transform(imp.transform(X_child))
    Zp = sc.transform(imp.transform(X_parent))
    sib = [np.linalg.norm(Zc[i] - Zc[j]) for i in range(len(Zc)) for j in range(i + 1, len(Zc))
           if parent[i] == parent[j]]
    d_pp = NearestNeighbors(n_neighbors=2).fit(Zp).kneighbors(Zp)[0][:, 1]
    rng = np.random.default_rng(0)
    Zb = sc.transform(imp.transform(X_cand[rng.choice(len(X_cand), 500, replace=False)]))
    d_bb = NearestNeighbors(n_neighbors=2).fit(Zb).kneighbors(Zb)[0][:, 1]
    out = {'region': region, 'sibling_dist_med': float(np.median(sib)),
           'nearest_other_occurrence_dist_med': float(np.median(d_pp)),
           'nearest_background_neighbour_dist_med': float(np.median(d_bb))}
    print(f"[{region}] feature-space distance (standardised): siblings {out['sibling_dist_med']:.2f} | "
          f"nearest other occurrence {out['nearest_other_occurrence_dist_med']:.2f} | "
          f"nearest background neighbour {out['nearest_background_neighbour_dist_med']:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    args = ap.parse_args()
    regions = setup_regions()
    sub = {r: regions[r] for r in args.regions}
    clipped_r, clipped_v = raster_preprocessing.preprocess_all(sub)
    worm_trees = build_worm_trees(list(regions.values())[0]['points'], config.PROJECTED_CRS)

    rows, sims, store = [], [], {}
    for r in args.regions:
        rec = load_region(r)
        cols = rec['full']['cols']               # Parsa et al. used their full vector set
        X_child, parent, _ = child_feature_table(r, regions[r], clipped_r[r], clipped_v[r], worm_trees, cols)
        X_parent = rec['full']['X_lab'][rec['full']['y'] == 1]
        X_cand = np.vstack([rec['full']['X_lab'][rec['full']['y'] == 0], rec['full']['X_unl']])
        store[r] = {'X_child': X_child, 'parent': parent}
        sims.append(sibling_similarity(r, X_child, parent, X_parent, X_cand))
        rows += audit_region(r, X_child, parent, X_parent, X_cand)
    pd.DataFrame(rows).to_csv(out_path('protocol_audit.csv'), index=False)
    pd.DataFrame(sims).to_csv(out_path('protocol_audit_sibling_similarity.csv'), index=False)
    save_pickle(store, out_path('protocol_audit_children.pkl'))


if __name__ == '__main__':
    main()
