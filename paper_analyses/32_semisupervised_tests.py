"""
Stage 32 -- three tests of the semi-supervised premise: does treating the
background as unlabelled (DevNet, and among the trees bagging-PU) help where
treating it as barren (MLP-BCE, RF) should hurt?

DevNet and MLP-BCE share network, quantile-normal inputs, batches of 16
positives + 240 background rows, 300 steps and the five-restart ensemble;
they differ only in the loss. RF and bagging-PU are the tree-based pair.
nnPU (same network, PU risk) is run alongside. Superior; feature set 3 (main)
and set 2 (no point-level map classes); random 5-fold CV and 100 km blocks;
20 repeats (instead of the 6 of the main results) for power. All models in a
condition use identical folds, and folds are identical across conditions.

Written down before the first run (PREDICTIONS below is also written to
out/semisup_predictions.txt with a timestamp when the stage starts):

  E1  background near occurrences (stress test of mislabelled ground).
      The main design excludes background within 10 km of every occurrence.
      Here n = 0, 20 or 50 extra background points per occurrence are
      drawn uniformly in a 2-10 km annulus around each of the 15 Superior
      occurrences (inside the sampleable domain, > 2 km from every
      occurrence) and added to the training background. In random CV this
      includes ground around the held-out occurrences, as in reality the
      unlabelled pool contains ground around undiscovered deposits; under
      spatial blocks most of it is purged with the held-out blocks.
  E2  size of the unlabelled pool: the training background is a nested
      random subset of 625, 1,250, 2,500 or 5,000 points.
  E3  label scarcity: all but k = 1, 2, 3, 5 or 8 training positives per
      fold are dropped (same withheld positives for every model; Stage 22),
      plus the full label set.

Cordillera and Grenville (--region) repeat the three tests with the same
design; their predictions were added to out/semisup_predictions.txt before
those runs (ADDENDUM below). Outputs for them carry the region in the file
name; Superior keeps the original names.

Outputs: out/semisup_<experiment>_<featureset>[_<region>].csv (one row per condition,
model, protocol and repeat), out/semisup_<experiment>_<featureset>.pkl
(repeat-averaged out-of-fold ranks), out/near_occurrence_points_Superior.pkl.

    python -m paper_analyses.32_semisupervised_tests --experiment prepare
    nice -n 10 python -m paper_analyses.32_semisupervised_tests --experiment E1 --featureset reencoded
"""
import argparse
import datetime
import importlib
import io
import contextlib
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from paper_analyses.common import GLCM_RASTERS, config, load_pickle, out_path, save_pickle, seed_everything, setup_regions
from paper_analyses.models import bagging_pu_factory, rf_factory
from paper_analyses.protocol import project_xy, run_protocol
from src.features import raster_preprocessing, sampleable_domain
from src.features.proximity import build_worm_trees

_fin = importlib.import_module('paper_analyses.11_devnet_final')
_c17 = importlib.import_module('paper_analyses.17_controls')
_s14 = importlib.import_module('paper_analyses.14_final_suite')
_s22 = importlib.import_module('paper_analyses.22_label_contamination')
_s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
_p12 = importlib.import_module('paper_analyses.12_final_production')
_p29 = importlib.import_module('paper_analyses.29_production_fs3')

REGION = 'Superior'
N_REPEATS = 20
PROTOCOLS = ['random', 100.0]
FACTORIES = {'DevNet': _fin.final_factory, 'MLP-BCE': _c17.mlp_bce_factory, 'nnPU': _s14.nnpu_qt_factory,
             'RF': rf_factory, 'BaggingPU': bagging_pu_factory}
NEAR_N = [0, 20, 50]
NEAR_MAX, NEAR_RING_KM = 50, (2.0, 10.0)
POOL_N = [625, 1250, 2500, 5000]
KEEP_K = [1, 2, 3, 5, 8, None]          # None: all training positives
PREDICTIONS = """\
P1 (E1) From n = 0 to n = 50 near-occurrence background points per occurrence, MLP-BCE loses more AP
   than DevNet (difference in differences > 0 in most repeats), under both protocols and on both
   feature sets; likewise RF loses more than bagging-PU.
P2 (E2) The DevNet - MLP-BCE AP gap is larger with 5,000 than with 625 training-background points.
P3 (E3) The DevNet - MLP-BCE AP gap is positive at k <= 3 training positives and shrinks as k grows;
   it is largest on feature set 2 (no point-level map classes).
All results are reported whichever way they come out.
"""
ADDENDUM = """\
ADDENDUM (Cordillera, Grenville): the same three tests, design and predictions P1-P3 are applied unchanged
   to the Cordilleran Orogen (27 occurrences) and the Grenville Province (12). Expected power is low under
   100 km blocks, where every model is close to chance in these provinces, so the random-CV results carry
   most of the information there. Superior results were known when this addendum was written.
All results are reported whichever way they come out.
"""
T = Transformer.from_crs('EPSG:3978', 'EPSG:4326', always_xy=True)


def near_points():
    """Up to NEAR_MAX points per occurrence in the 2-10 km annulus, with set-2 and set-3 features."""
    path = out_path(f'near_occurrence_points_{REGION}.pkl')
    if os.path.exists(path):
        return load_pickle(path)
    rec = load_region_quiet()
    pos = rec['lab'][rec['lab']['pool'] == 'positive'].reset_index(drop=True)
    xy_pos = project_xy(pos.lat, pos.lon, config.PROJECTED_CRS)
    regions = setup_regions()
    cfg = regions[REGION]
    clipped_r, clipped_v = raster_preprocessing.preprocess_all({REGION: cfg})
    sampleable_domain.build_all({REGION: cfg}, clipped_r, config.PROJECTED_CRS, GLCM_RASTERS)
    worm_trees = build_worm_trees(cfg['points'], config.PROJECTED_CRS)
    rng = np.random.default_rng(config.RANDOM_SEED)
    tree = cKDTree(xy_pos)
    rows = []
    r1, r2 = (1000.0 * r for r in NEAR_RING_KM)
    for i, (x0, y0) in enumerate(xy_pos):
        r = np.sqrt(rng.uniform(r1 ** 2, r2 ** 2, 400))
        th = rng.uniform(0, 2 * np.pi, 400)
        xy = np.c_[x0 + r * np.cos(th), y0 + r * np.sin(th)]
        ok = tree.query(xy)[0] > r1
        lon, lat = T.transform(xy[:, 0], xy[:, 1])
        ok &= sampleable_domain.sampleable_lookup(lat, lon, REGION)
        idx = np.where(ok)[0][:NEAR_MAX]
        rows += [{'occ': i, 'rank': j, 'lat': lat[k], 'lon': lon[k], 'x_km': xy[k, 0] / 1000, 'y_km': xy[k, 1] / 1000}
                 for j, k in enumerate(idx)]
    pts = pd.DataFrame(rows)
    pruned = rec['pruned']['cols']
    _, X1 = _p12.features_at(pts[['lat', 'lon']].assign(label=0), cfg, clipped_r[REGION], clipped_v[REGION],
                             worm_trees, pruned)
    out = {'points': pts}
    for fs in ('extended', 'reencoded'):
        with contextlib.redirect_stdout(io.StringIO()):
            _, cols, _, _, _ = _s25.extended_matrices(REGION, fs)
        base = [c for c in cols if c in pruned]
        out[fs] = _p29.set3_rows(REGION, rec, cols, {'p': (pd.DataFrame(X1, columns=pruned)[base],
                                                           pts.lon.values, pts.lat.values)})['p']
        out[f'{fs}_cols'] = cols
    good = np.isfinite(out['extended']).all(1) & np.isfinite(out['reencoded']).all(1)
    out = {'points': pts[good].reset_index(drop=True), 'extended': out['extended'][good],
           'reencoded': out['reencoded'][good], 'extended_cols': out['extended_cols'],
           'reencoded_cols': out['reencoded_cols']}
    print(f'near-occurrence points: {len(out["points"])} kept, per occurrence '
          f'{out["points"].groupby("occ").size().to_dict()}', flush=True)
    save_pickle(out, path)
    return out


def load_region_quiet():
    from paper_analyses.common import load_region
    return load_region(REGION)


def matrices(featureset):
    with contextlib.redirect_stdout(io.StringIO()):
        rec, cols, X_lab, X_unl, _ = _s25.extended_matrices(REGION, featureset)
    xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
    xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
    return rec, cols, X_lab, X_unl, xy_lab, xy_bg


def conditions(experiment, featureset, X_unl, xy_bg):
    """Yield (condition label, X_bg, xy_bg, factory wrapper)."""
    ident = lambda f: f
    if experiment == 'E1':
        near = near_points()
        assert near[f'{featureset}_cols'] == matrices(featureset)[1]
        pts = near['points']
        for n in NEAR_N:
            sel = (pts['rank'] < n).values
            yield (f'near={n}', np.vstack([X_unl, near[featureset][sel]]),
                   np.vstack([xy_bg, pts.loc[sel, ['x_km', 'y_km']].values]), ident)
    elif experiment == 'E2':
        order = np.random.default_rng(config.RANDOM_SEED).permutation(len(X_unl))
        for n in POOL_N:
            idx = np.sort(order[:n])
            yield f'pool={n}', X_unl[idx], xy_bg[idx], ident
    elif experiment == 'E3':
        for k in KEEP_K:
            wrap = ident if k is None else (lambda f, k=k: _s22.withholding_factory(f, k, 'drop'))
            yield f'k={"all" if k is None else k}', X_unl, xy_bg, wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', choices=['prepare', 'E1', 'E2', 'E3'], required=True)
    ap.add_argument('--featureset', choices=['extended', 'reencoded'], default='reencoded')
    ap.add_argument('--region', choices=['Superior', 'Cordillera', 'Grenville'], default='Superior')
    args = ap.parse_args()
    global REGION
    REGION = args.region
    if args.experiment == 'prepare':
        with open(out_path('semisup_predictions.txt'), 'a') as f:
            when = datetime.datetime.now().isoformat(timespec="seconds")
            if REGION == 'Superior':
                f.write(f'# written {when}, before any E1-E3 run\n' + PREDICTIONS + '\n')
            else:
                f.write(f'# written {when}, before any E1-E3 run in {REGION}\n' + ADDENDUM + '\n')
        near_points()
        return
    rec, cols, X_lab, X_unl, xy_lab, xy_bg = matrices(args.featureset)
    y = rec['pruned']['y']
    rows, store = [], {}
    tag = f'{args.experiment}_{args.featureset}' + ('' if REGION == 'Superior' else f'_{REGION}')
    for cond, X_bg, xyb, wrap in conditions(args.experiment, args.featureset, X_unl, xy_bg):
        for proto in PROTOCOLS:
            pname = 'random' if proto == 'random' else f'{proto:.0f} km'
            for model, fac in FACTORIES.items():
                seed_everything()
                res = run_protocol(proto, X_lab, y, xy_lab, X_bg, xyb, wrap(fac(REGION)), config.CV_N_SPLITS,
                                   N_REPEATS, config.RANDOM_SEED)
                store[(cond, pname, model)] = {'oof_mean': res['oof_mean'], 'aps': res['aps'], 'aucs': res['aucs']}
                for r, (a, u) in enumerate(zip(res['aps'], res['aucs'])):
                    rows.append({'region': REGION, 'experiment': args.experiment, 'featureset': args.featureset, 'condition': cond,
                                 'protocol': pname, 'model': model, 'repeat': r, 'ap': a, 'auc': u,
                                 'n_bg': len(X_bg)})
                print(f"[{tag} | {cond} | {pname}] {model}: AP {res['ap_med']:.3f} AUC {res['auc_med']:.3f}",
                      flush=True)
            pd.DataFrame(rows).to_csv(out_path(f'semisup_{tag}.csv'), index=False)
            save_pickle(store, out_path(f'semisup_{tag}.pkl'))


if __name__ == '__main__':
    main()
