"""
Stage 43 -- the S2 level-12 experiment of Parsa et al. (2024) rebuilt at the
national scale on feature set 3, then corrected one step at a time.

--step grid: every S2 L12 cell whose centre lies in the national domain of
    Stage 41 (outline, A∧B, on the H3 geology map), found by subdividing a
    level-5 covering of Canada and pruning cells too far from the domain;
    the four L13 children of every cell holding a national occurrence. The
    Stage-41 extractor computes feature set 3 (national columns and H3
    classes) at all cell and child centres in one call.

--step ladder: each rung changes one thing relative to the one before.
    R0  Parsa et al. as described: each occurrence cell replaced by its four
        L13 children; as many negatives, drawn from the 5% of cells least
        cosine-similar to the positives; stratified random 70/30 split of
        the augmented labels; RF, 1000 trees, min-max inputs; 50 splits; AUC.
    R1  negatives drawn at random from cells more than 10 km from every
        occurrence (the regional background rule).
    R2  split grouped by parent cell (siblings never on both sides).
    R3  no augmentation: one label per occurrence cell.
    R4  the R3 models scored against realistic background: test positives
        plus a random 30% of a fixed evaluation sample of background cells
        (prevalence as in the national table); AUC, AP, AP lift, top-10%.
    R5  trained on the positives and a fixed training sample of background
        cells (the shared-background design), class-balanced; tested as R4.
        DevNet, Bagging-PU and the paper's RF are added here.
    R6  as R5 under 100 km spatial blocks (Stage 02 protocol, 5 x 6).
    R7  as R5, leave one province out (Superior, Cordillera, Grenville,
        rest of Canada).
    The training and evaluation background samples have the sizes of the
    national table's pools and are disjoint; R1-R3 negatives come from the
    training sample's side.

--clf CNN runs R0-R4 with the best model of Parsa et al. (2024), their 1D CNN,
    instead of their RF; R5 takes it as the model CNN-Parsa.

    python -m paper_analyses.43_parsa_ladder --step grid
    python -m paper_analyses.43_parsa_ladder --step ladder [--rungs R0 R1 ...] [--clf CNN]
"""
import argparse
import copy
import importlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import s2sphere as s2
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from paper_analyses.common import config, load_pickle, save_pickle, seed_everything
from paper_analyses.national import (H3_IN_KM, LABELS, NAME, STUDY, Extractor, encode_geology, metrics, nat_path,
                                     provinces_of, summarise)
from paper_analyses.protocol import project_xy, run_protocol
from src.features import sampleable_domain

_b = importlib.import_module('paper_analyses.42_national_benchmark')
N_ITER, TEST_FRAC, LOW_SIM_FRAC, START_LEVEL, LEVEL = 50, 0.30, 0.05, 5, 12
MAX_DIAG_RAD = 2.438654594434021          # S2 max cell diagonal at level 0 (quadratic projection), radians
EARTH_KM = 6371.0
GRID_FILE = 's2_l12_features.npz'


# -- step grid ------------------------------------------------------------------
def domain_distance():
    """Lower bound (km) on the distance from any point to the national A∧B
    domain, from a distance transform of the domain mask (EPSG:3978, 2 km)."""
    info = sampleable_domain.SAMPLEABLE[NAME]
    tf, mask = info['ref_tf'], info['sampleable']
    px = abs(tf.a) / 1000.0
    edt = distance_transform_edt(~mask) * px
    H, W = mask.shape

    def lower_bound(lat, lon):
        x, y = (project_xy(lat, lon, config.PROJECTED_CRS) / 1000.0).T
        col = (x * 1000.0 - tf.c) / tf.a
        row = (y * 1000.0 - tf.f) / tf.e
        r, c = np.clip(np.floor(row), 0, H - 1).astype(int), np.clip(np.floor(col), 0, W - 1).astype(int)
        off = np.hypot((row - (r + 0.5)) * px, (col - (c + 0.5)) * px)
        return edt[r, c] - off - px
    return lower_bound


def l12_cells(ex):
    lower_bound = domain_distance()
    rect = s2.LatLngRect.from_point_pair(s2.LatLng.from_degrees(41.0, -142.0), s2.LatLng.from_degrees(84.0, -52.0))
    cov = s2.RegionCoverer()
    cov.min_level = cov.max_level = START_LEVEL
    cov.max_cells = 100000
    cells = cov.get_covering(rect)
    for level in range(START_LEVEL + 1, LEVEL + 1):
        ch = [c for cell in cells for c in cell.children()]
        ll = [c.to_lat_lng() for c in ch]
        lat = np.array([p.lat().degrees for p in ll])
        lon = np.array([p.lng().degrees for p in ll])
        diag_km = 1.1 * MAX_DIAG_RAD * EARTH_KM / 2 ** level
        keep = lower_bound(lat, lon) <= diag_km if level < LEVEL else ex.in_domain(lat, lon)
        cells = [c for c, k in zip(ch, keep) if k]
        print(f'  level {level}: {len(ch):,} children -> {len(cells):,} kept', flush=True)
    ll = [c.to_lat_lng() for c in cells]
    lat = np.array([p.lat().degrees for p in ll])
    lon = np.array([p.lng().degrees for p in ll])
    on_map = ex.h3_distance(lat, lon) <= H3_IN_KM
    print(f'  {int(on_map.sum()):,} L12 cells in the national domain ({int((~on_map).sum())} off the H3 map dropped)')
    ids = np.array([c.id() for c in cells], dtype=np.uint64)
    return ids[on_map], lat[on_map], lon[on_map]


def grid():
    t0 = time.time()
    seed_everything()
    ex = Extractor()
    meta = load_pickle(nat_path('national_table.pkl'))
    nat = meta['pts']
    pos = nat[nat.in_national & (nat.pool == 'positive')].reset_index(drop=True)
    ids, lat, lon = l12_cells(ex)
    pos_cell = np.array([s2.CellId.from_lat_lng(s2.LatLng.from_degrees(a, b)).parent(LEVEL).id()
                         for a, b in zip(pos.lat, pos.lon)], dtype=np.uint64)
    upos = np.unique(pos_cell)
    is_pos = np.isin(ids, upos)
    missing = np.setdiff1d(upos, ids)
    print(f'{len(pos)} national occurrences -> {len(upos)} positive L12 cells '
          f'({len(missing)} with a centre outside the domain, added)', flush=True)
    if len(missing):
        ll = [s2.CellId(int(i)).to_lat_lng() for i in missing]
        ids = np.r_[ids, missing]
        lat = np.r_[lat, [p.lat().degrees for p in ll]]
        lon = np.r_[lon, [p.lng().degrees for p in ll]]
        is_pos = np.r_[is_pos, np.ones(len(missing), bool)]
    ch_lat, ch_lon, ch_parent = [], [], []
    for pid in upos:
        for c in s2.CellId(int(pid)).children():
            p = c.to_lat_lng()
            ch_lat.append(p.lat().degrees)
            ch_lon.append(p.lng().degrees)
            ch_parent.append(pid)
    pts = pd.DataFrame({'lat': np.r_[lat, ch_lat], 'lon': np.r_[lon, ch_lon]})
    print(f'feature extraction: {len(lat):,} cells + {len(ch_lat)} children', flush=True)
    prov_all = provinces_of(pts.lat.values, pts.lon.values)
    raw = ex.raw(pts, prov_all)
    geo = encode_geology(raw, meta['lith_classes'], meta['era_classes'])
    raw = pd.concat([raw, geo], axis=1)
    X = raw[meta['cols3']].values.astype(np.float32)
    X2 = raw[meta['cols2']].values.astype(np.float32)
    n = len(lat)
    occ = pd.read_csv(LABELS)
    d_occ = cKDTree(project_xy(occ.Latitude.values, occ.Longitude.values) / 1000.0).query(
        project_xy(lat, lon) / 1000.0)[0]
    prov = prov_all[:n]
    np.savez_compressed(nat_path(GRID_FILE), ids=ids, lat=lat, lon=lon, is_pos=is_pos, d_occ_km=d_occ,
                        province=prov, X=X[:n], X2=X2[:n], X_child=X[n:], X2_child=X2[n:],
                        child_parent=np.array(ch_parent, dtype=np.uint64), cols=np.array(meta['cols3']),
                        cols2=np.array(meta['cols2']))
    print(f'saved {nat_path(GRID_FILE)} in {(time.time() - t0) / 60:.1f} min', flush=True)


# -- step ladder ----------------------------------------------------------------
def load_grid():
    z = np.load(nat_path(GRID_FILE), allow_pickle=True)    # our own file; 'province' is an object array
    return {k: z[k] for k in z.files}


def run_rf(X_tr, y_tr, X_te, seed, balanced=False):
    imp = SimpleImputer(strategy='median').fit(X_tr)
    mm = MinMaxScaler().fit(imp.transform(X_tr))
    rf = RandomForestClassifier(n_estimators=1000, random_state=seed, n_jobs=-1,
                                class_weight='balanced' if balanced else None)
    rf.fit(mm.transform(imp.transform(X_tr)), y_tr)
    return rf.predict_proba(mm.transform(imp.transform(X_te)))[:, 1]


CNN_POOL, CNN_DROPOUT, CNN_BATCH, CNN_LR, CNN_EPOCHS, CNN_PATIENCE, CNN_VAL = 2, 0.3, 32, 1e-3, 200, 20, 0.2
CNN_EPOCH_HALF = 1024                     # balanced=True: positives and background drawn per epoch, each
CNN_CHUNK = 1024                          # rows per forward pass when scoring


def run_cnn(X_tr, y_tr, X_te, seed, balanced=False):
    """The 1D CNN of Parsa et al. (2024) on min-max inputs: Conv1D (512 filters,
    kernel 5) with batch normalisation, ReLU and dropout, max pooling, flatten,
    dense 256-128-64 (ReLU), one sigmoid output; MSE loss, Adam, early stopping
    on validation loss. Not stated there and chosen here: pool size 2, dropout
    0.3, batch 32, learning rate 1e-3, at most 200 epochs, patience 20, a
    stratified 20% of the training labels for validation, best weights kept.
    With balanced=True every epoch draws 1,024 positives and 1,024 background
    rows with replacement, and the validation loss is the mean of the two
    classes' losses."""
    import torch
    from torch import nn
    imp = SimpleImputer(strategy='median').fit(X_tr)
    mm = MinMaxScaler().fit(imp.transform(X_tr))
    A_tr = torch.from_numpy(mm.transform(imp.transform(X_tr)).astype(np.float32)[:, None, :])
    A_te = torch.from_numpy(mm.transform(imp.transform(X_te)).astype(np.float32)[:, None, :])
    y = torch.from_numpy(y_tr.astype(np.float32))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=CNN_VAL, random_state=seed).split(X_tr, y_tr))
    p = X_tr.shape[1]
    net = nn.Sequential(nn.Conv1d(1, 512, 5), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(CNN_DROPOUT),
                        nn.MaxPool1d(CNN_POOL), nn.Flatten(), nn.Linear(512 * ((p - 4) // CNN_POOL), 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    opt = torch.optim.Adam(net.parameters(), lr=CNN_LR)
    pos_va = y_tr[va] == 1

    def predict(A):                              # in chunks: the conv layer's output is 512 x (p - 4) per row
        net.eval()
        with torch.no_grad():
            return torch.cat([net(A[s:s + CNN_CHUNK]).squeeze(1) for s in range(0, len(A), CNN_CHUNK)])

    def val_loss():
        e = (predict(A_tr[va]) - y[va]) ** 2
        return float((e[pos_va].mean() + e[~pos_va].mean()) / 2) if balanced else float(e.mean())

    pos, neg = tr[y_tr[tr] == 1], tr[y_tr[tr] == 0]
    best, best_state, wait = np.inf, None, 0
    for _ in range(CNN_EPOCHS):
        net.train()
        idx = rng.permutation(np.r_[rng.choice(pos, CNN_EPOCH_HALF), rng.choice(neg, CNN_EPOCH_HALF)]
                              if balanced else tr)
        for s in range(0, len(idx), CNN_BATCH):
            b = idx[s:s + CNN_BATCH]
            if len(b) < 2:                            # batch normalisation needs two rows
                continue
            opt.zero_grad()
            ((net(A_tr[b]).squeeze(1) - y[b]) ** 2).mean().backward()
            opt.step()
        v = val_loss()
        if v < best:
            best, best_state, wait = v, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= CNN_PATIENCE:
                break
    net.load_state_dict(best_state)
    return predict(A_te).numpy()


CLASSIFIERS = {'RF': run_rf, 'CNN': run_cnn}


def parsa_rf_factory(region=None, clf='RF'):
    """Parsa et al.'s RF (1000 trees, min-max inputs), or with clf='CNN' their
    1D CNN, on positives against the shared training background,
    class-balanced."""
    def make(X_bg):
        def scorer(X_tr, y_tr, X_va, seed_counter=0):
            X_pos = X_tr[y_tr == 1]
            return CLASSIFIERS[clf](np.vstack([X_pos, X_bg]), np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg))], X_va,
                                    config.RANDOM_SEED + seed_counter, balanced=True)
        return scorer
    return make


def mean_cosine(X_pos, X, cand, chunk=200000):
    """Mean cosine similarity of every candidate row X[cand] to the positives
    (median imputation and min-max scaling fitted on positives + candidates,
    as Stage 05), in chunks; mean_j(c . p_j) = c . mean_j(p_j)."""
    p = X.shape[1]
    med, lo, hi = np.empty(p), np.empty(p), np.empty(p)
    for j in range(p):
        v = np.r_[X_pos[:, j], X[cand, j]]
        med[j] = np.nanmedian(v)
        v = np.where(np.isnan(v), med[j], v)
        lo[j], hi[j] = v.min(), v.max()
    span = np.where(hi > lo, hi - lo, 1.0)

    def scaled(A):
        A = np.where(np.isnan(A), med, A)
        A = (A - lo) / span
        return A / np.linalg.norm(A, axis=1, keepdims=True)
    pbar = scaled(X_pos).mean(axis=0)
    return np.concatenate([scaled(X[cand[s:s + chunk]]) @ pbar for s in range(0, len(cand), chunk)])


def balanced_runs(X_pos, X_neg, parent, split, models=('RF-Parsa',), X_eval=None, clf='RF'):
    """50 splits of a balanced positive/negative set. Returns per-split AUC on
    the balanced test set, and with X_eval also metrics against realistic
    background (test positives + 30% of X_eval)."""
    X = np.vstack([X_pos, X_neg])
    y = np.r_[np.ones(len(X_pos)), np.zeros(len(X_neg))]
    out = []
    for it in range(N_ITER):
        if split == 'grouped':
            p_tr, p_te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=it)
                              .split(X_pos, groups=parent))
            n_idx = np.random.default_rng(it).permutation(len(X_neg))
            n_te = n_idx[:int(round(TEST_FRAC * len(X_neg)))]
            tr, te = np.r_[p_tr, len(X_pos) + n_idx[len(n_te):]], np.r_[p_te, len(X_pos) + n_te]
        else:
            tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=it).split(X, y))
        rec = {'iter': it}
        if parent is not None and split == 'random':
            tr_par = set(parent[tr[tr < len(X_pos)]])
            rec['sibling_in_train'] = float(np.mean([parent[i] in tr_par for i in te[te < len(X_pos)]]))
        X_te = X[te] if X_eval is None else np.vstack([X[te], X_eval[np.random.default_rng(1000 + it).choice(
            len(X_eval), int(round(TEST_FRAC * len(X_eval))), replace=False)]])
        s = CLASSIFIERS[clf](X[tr], y[tr], X_te, seed=it)
        rec['auc_balanced'] = roc_auc_score(y[te], s[:len(te)])
        rec['ap_balanced'] = average_precision_score(y[te], s[:len(te)])
        if X_eval is not None:
            pos_te = y[te] == 1
            yy = np.r_[np.ones(pos_te.sum()), np.zeros(len(X_te) - len(te))]
            ss = np.r_[s[:len(te)][pos_te], s[len(te):]]
            rec.update(metrics(yy.astype(int), ss))
        out.append(rec)
    return pd.DataFrame(out)


def shared_split(it, Xp, Xb_tr, Xb_ev, fac):
    """One R5 split: a stratified 70/30 split of the positive cells; trained
    on the training positives and the training background sample, tested on
    the test positives and a random 30% of the evaluation sample."""
    rng = np.random.default_rng(it)
    perm = rng.permutation(len(Xp))
    n_te = int(round(TEST_FRAC * len(Xp)))
    te_p, tr_p = perm[:n_te], perm[n_te:]
    ev = np.random.default_rng(1000 + it).choice(len(Xb_ev), int(round(TEST_FRAC * len(Xb_ev))), replace=False)
    X_te = np.vstack([Xp[te_p], Xb_ev[ev]])
    y_te = np.r_[np.ones(len(te_p)), np.zeros(len(ev))].astype(int)
    s = fac(Xb_tr)(Xp[tr_p], np.ones(len(tr_p)), X_te, seed_counter=it)
    return {'iter': it, **metrics(y_te, s)}


def _parsa_split(args):
    it, Xp, Xb_tr, Xb_ev, clf = args
    return shared_split(it, Xp, Xb_tr, Xb_ev, parsa_rf_factory(clf=clf))


def shared_runs(Xp, Xb_tr, Xb_ev, model, workers=1):
    """R5: 50 splits (shared_split). The models of Parsa et al. can run the
    splits in a process pool; each split is seeded by its index, so the
    result does not depend on the number of workers."""
    if workers > 1 and model.endswith('-Parsa'):
        with ProcessPoolExecutor(workers) as ex:
            return pd.DataFrame(list(ex.map(_parsa_split, [(it, Xp, Xb_tr, Xb_ev, model.split('-')[0])
                                                           for it in range(N_ITER)])))
    fac = parsa_rf_factory(clf=model.split('-')[0]) if model.endswith('-Parsa') else _b.FACTORIES[model](NAME)
    return pd.DataFrame([shared_split(it, Xp, Xb_tr, Xb_ev, fac) for it in range(N_ITER)])


def ladder(rungs, models=('RF-Parsa', 'RF', 'DevNet', 'BaggingPU'), clf='RF', workers=1):
    g = load_grid()
    rng = np.random.default_rng(config.RANDOM_SEED)
    X, is_pos = g['X'].astype(float), g['is_pos']
    pos_ids = g['ids'][is_pos]
    X_par = X[is_pos]
    X_ch = g['X_child'].astype(float)
    parent = np.searchsorted(np.sort(pos_ids), g['child_parent'])
    cand = np.where(~is_pos)[0]
    far = np.where(~is_pos & (g['d_occ_km'] > config.BUFFER_KM))[0]
    meta = load_pickle(nat_path('national_table.pkl'))
    nat = meta['pts'][meta['pts'].in_national]
    n_ev, n_tr = int((nat.pool == 'eval_neg').sum()), int((nat.pool == 'train_unlabeled').sum())
    perm = rng.permutation(far)
    ev_idx, tr_idx = perm[:n_ev], perm[n_ev:n_ev + n_tr]
    side = perm[n_ev:]                                  # R1-R3 negatives: the training side of the background
    print(f'{len(g["ids"]):,} L12 cells | {int(is_pos.sum())} positive cells, {len(X_ch)} children | '
          f'{len(far):,} cells > {config.BUFFER_KM:.0f} km from occurrences | evaluation sample {n_ev:,}, '
          f'training sample {n_tr:,}', flush=True)
    rows = []

    def report(rung, label, df, model=f'{clf}-Parsa', **extra):
        r = {'rung': rung, 'design': label, 'model': model, **extra}
        for c in df.columns:
            if c != 'iter':
                r[f'{c}_mean'] = float(df[c].mean())
                r[f'{c}_lo'], r[f'{c}_hi'] = (float(v) for v in np.percentile(df[c], [2.5, 97.5]))
        rows.append(r)
        shown = ', '.join(f"{c} {r[f'{c}_mean']:.3f}" for c in ('auc_balanced', 'ap_balanced', 'auc', 'ap', 'lift', 'cap10',
                                                                 'sibling_in_train') if f'{c}_mean' in r)
        print(f'[{rung} | {model}] {label}: {shown}', flush=True)

    # separate generators so that any subset of rungs draws the same samples
    rng_neg = np.random.default_rng(config.RANDOM_SEED + 1)
    neg_aug = rng_neg.choice(side, len(X_ch), replace=False)
    neg_par = rng_neg.choice(side, len(X_par), replace=False)
    if 'R0' in rungs:
        sim = mean_cosine(X_ch, X, cand)
        low = cand[sim <= np.quantile(sim, LOW_SIM_FRAC)]
        neg = np.random.default_rng(config.RANDOM_SEED + 2).choice(low, len(X_ch), replace=False)
        print(f'  R0 cosine pool: {len(low):,} cells; median distance of chosen negatives to the nearest '
              f'occurrence {np.median(g["d_occ_km"][neg]):.0f} km (random cells: '
              f'{np.median(g["d_occ_km"][far]):.0f} km)', flush=True)
        report('R0', 'Parsa et al.: children, cosine-dissimilar negatives, child-level random split',
               balanced_runs(X_ch, X[neg], parent, 'random', clf=clf), n_pos=len(X_ch), n_neg=len(neg))
    if 'R2c' in rungs or 'R3c' in rungs:          # cosine-dissimilar negatives without the sibling leak
        rng_c = np.random.default_rng(config.RANDOM_SEED + 3)
        sim_ch = mean_cosine(X_ch, X, cand)
        neg_c_aug = rng_c.choice(cand[sim_ch <= np.quantile(sim_ch, LOW_SIM_FRAC)], len(X_ch), replace=False)
        sim_par = mean_cosine(X_par, X, cand)
        neg_c_par = rng_c.choice(cand[sim_par <= np.quantile(sim_par, LOW_SIM_FRAC)], len(X_par), replace=False)
        if 'R2c' in rungs:
            report('R2c', 'R2 with cosine-dissimilar negatives',
                   balanced_runs(X_ch, X[neg_c_aug], parent, 'grouped', clf=clf), n_pos=len(X_ch), n_neg=len(neg_c_aug))
        if 'R3c' in rungs:
            df = balanced_runs(X_par, X[neg_c_par], None, 'random', X_eval=X[ev_idx], clf=clf)
            report('R3c', 'R3 with cosine-dissimilar negatives', df[['iter', 'auc_balanced', 'ap_balanced']],
                   n_pos=len(X_par), n_neg=len(neg_c_par))
            report('R4c', 'R3c models against realistic background', df.drop(columns=['auc_balanced', 'ap_balanced']),
                   n_pos=len(X_par), n_neg=len(neg_c_par))
    if 'R1' in rungs:
        report('R1', 'random negatives (> 10 km)', balanced_runs(X_ch, X[neg_aug], parent, 'random', clf=clf),
               n_pos=len(X_ch), n_neg=len(neg_aug))
    if 'R2' in rungs:
        report('R2', 'split grouped by parent cell', balanced_runs(X_ch, X[neg_aug], parent, 'grouped', clf=clf),
               n_pos=len(X_ch), n_neg=len(neg_aug))
    if 'R3' in rungs or 'R4' in rungs:
        df = balanced_runs(X_par, X[neg_par], None, 'random', X_eval=X[ev_idx], clf=clf)
        report('R3', 'no augmentation (one label per occurrence cell)', df[['iter', 'auc_balanced', 'ap_balanced']],
               n_pos=len(X_par), n_neg=len(neg_par))
        report('R4', 'R3 models against realistic background', df.drop(columns=['auc_balanced', 'ap_balanced']),
               n_pos=len(X_par), n_neg=len(neg_par))
    if 'R5' in rungs:
        for m in models:
            seed_everything()
            report('R5', 'shared training background, realistic test',
                   shared_runs(X_par, X[tr_idx], X[ev_idx], m, workers), model=m, n_pos=len(X_par), n_bg_train=n_tr)
    xy = project_xy(g['lat'], g['lon']) / 1000.0
    lab_idx = np.r_[np.where(is_pos)[0], ev_idx]
    y_lab = np.r_[np.ones(int(is_pos.sum())), np.zeros(n_ev)].astype(int)
    if 'R6' in rungs:
        for m in models:
            fac = parsa_rf_factory() if m == 'RF-Parsa' else _b.FACTORIES[m](NAME)
            seed_everything()
            res = run_protocol(100.0, X[lab_idx], y_lab, xy[lab_idx], X[tr_idx], xy[tr_idx], fac,
                               config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
            s = summarise(y_lab, res['oof'])
            rows.append({'rung': 'R6', 'design': '100 km spatial blocks', 'model': m,
                         **{k.replace('_med', '_mean'): v for k, v in s.items() if k.endswith(('_med', '_lo', '_hi'))}})
            print(f"[R6 | {m}] AUC {s['auc_med']:.3f} AP {s['ap_med']:.3f} lift {s['lift_med']:.0f} "
                  f"top-10% {s['cap10_med']:.2f}", flush=True)
    if 'R7' in rungs:
        prov = g['province']
        grp = np.where(np.isin(prov, STUDY), prov, 'Other')
        for m in models:
            fac = parsa_rf_factory() if m == 'RF-Parsa' else _b.FACTORIES[m](NAME)
            for gname in STUDY + ['Other']:
                te = lab_idx[grp[lab_idx] == gname]
                trp = np.where(is_pos & (grp != gname))[0]
                trb = tr_idx[grp[tr_idx] != gname]
                seed_everything()
                S = _b.fit_reps(fac, X[trp], X[trb], X[te])
                s = summarise(is_pos[te].astype(int), S)
                rows.append({'rung': 'R7', 'design': f'leave one province out: {gname}', 'model': m,
                             **{k.replace('_med', '_mean'): v for k, v in s.items()
                                if k.endswith(('_med', '_lo', '_hi'))}})
                print(f"[R7 | {m} | {gname}] AUC {s['auc_med']:.3f} AP {s['ap_med']:.3f} "
                      f"lift {s['lift_med']:.0f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=['grid', 'ladder'], required=True)
    ap.add_argument('--rungs', nargs='+', default=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'])
    ap.add_argument('--models', nargs='+', default=['RF-Parsa', 'RF', 'DevNet', 'BaggingPU'],
                    help='models for R5-R7 (R0-R4 use the model of Parsa et al. given by --clf)')
    ap.add_argument('--clf', choices=['RF', 'CNN'], default='RF', help='R0-R4: their RF or their 1D CNN')
    ap.add_argument('--workers', type=int, default=1, help='R5: processes for the models of Parsa et al.')
    args = ap.parse_args()
    if args.step == 'grid':
        grid()
    else:
        df = ladder(args.rungs, args.models, args.clf, args.workers)
        default = args.models == ['RF-Parsa', 'RF', 'DevNet', 'BaggingPU']
        models = args.models if args.clf == 'RF' else [f'{args.clf}-Parsa']
        df.to_csv(_b.out(f"parsa_ladder_{'_'.join(args.rungs)}"
                         f"{'' if default and args.clf == 'RF' else '_' + '_'.join(models)}.csv"), index=False)


if __name__ == '__main__':
    main()
