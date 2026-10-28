"""
Stage 21 -- how stable are the Superior maps? Jackknife over occurrences.

Each of the 15 Superior positives is left out in turn; the final DevNet
(Stage 11, five restarts), MLP-BCE (Stage 17: same network, inputs and
batches, supervised loss) and RF (fixed seed, as in production) are refitted
on the remaining 14 positives plus the training background and score the
0.1 degree prediction grid of Stage 12. Per grid node we report the share
of the 15 jackknife models that place it in the province's top 5% (class
stability) and the standard deviation of its percentile rank. The grid
features are rebuilt with the Stage-12 code and cached in out/ (git-ignored,
derived from licensed grids).

    nice -n 10 python -m paper_analyses.21_map_uncertainty [--features all|no-terrane]
"""
import argparse
import importlib
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from paper_analyses.common import config, load_region, out_path, seed_everything, setup_regions
from src.evaluation.prediction_map import build_grid
from src.features import raster_preprocessing, sampleable_domain
from src.features.proximity import build_worm_trees
from src.models.random_forest import make_rf_params, make_rf_pipeline

_p12 = importlib.import_module('paper_analyses.12_final_production')
_fin = importlib.import_module('paper_analyses.11_devnet_final')
_tb = importlib.import_module('paper_analyses.10_training_budget')
_c17 = importlib.import_module('paper_analyses.17_controls')
REGION = 'Superior'
GRID_CACHE = out_path('superior_grid_features.npz')


def grid_features():
    if os.path.exists(GRID_CACHE):
        z = np.load(GRID_CACHE)
        return z['lat'], z['lon'], z['X']
    regions = setup_regions()
    cfg = regions[REGION]
    clipped_r, clipped_v = raster_preprocessing.preprocess_all({REGION: cfg})
    sampleable_domain.build_all({REGION: cfg}, clipped_r, config.PROJECTED_CRS, _p12.GLCM_RASTERS)
    worm_trees = build_worm_trees(cfg['points'], config.PROJECTED_CRS)
    cols = load_region(REGION)['pruned']['cols']
    grid, _ = build_grid(cfg, REGION, config.GRID_DEG)
    gpts, Xg = _p12.features_at(pd.DataFrame({'lat': grid['lat'].values, 'lon': grid['lon'].values}),
                                cfg, clipped_r[REGION], clipped_v[REGION], worm_trees, cols)
    np.savez_compressed(GRID_CACHE, lat=gpts['lat'].values, lon=gpts['lon'].values, X=Xg)
    return gpts['lat'].values, gpts['lon'].values, Xg


def pct(s):
    return rankdata(s) / len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', choices=['all', 'no-terrane'], default='all')
    args = ap.parse_args()
    seed_everything()
    lat, lon, Xg = grid_features()
    d = load_region(REGION)['pruned']
    j = [i for i, c in enumerate(d['cols']) if args.features == 'all' or not c.startswith(('geo_era_', 'lith_'))]
    Xg = Xg[:, j]
    X_pos, X_bg = d['X_lab'][d['y'] == 1][:, j], d['X_unl'][:, j]
    cfg = _c17.model_cfg(REGION)
    out = {'lat': lat, 'lon': lon}
    for model in ['devnet', 'mlp', 'rf']:
        runs = []
        for leave in [None] + list(range(len(X_pos))):
            P = X_pos if leave is None else np.delete(X_pos, leave, axis=0)
            if model == 'devnet':
                tf, models = _fin.fit_final(P, X_bg, REGION, config.DN_SEED)
                s = _tb._ensemble(models, tf(Xg))
            elif model == 'mlp':
                X_tr = np.vstack([P, X_bg])
                y_tr = np.r_[np.ones(len(P)), np.zeros(len(X_bg))]
                tf = _fin.QuantileNormal(X_tr)
                models = [_c17.train_bce(tf(X_tr), y_tr, config.DN_SEED + k, cfg) for k in range(config.DN_RESTARTS)]
                s = _tb._ensemble(models, tf(Xg))
            else:
                X_tr = np.vstack([P, X_bg])
                y_tr = np.r_[np.ones(len(P)), np.zeros(len(X_bg))]
                s = make_rf_pipeline(make_rf_params(config.RANDOM_SEED)).fit(X_tr, y_tr).predict_proba(Xg)[:, 1]
            runs.append(pct(s))
        full, jk = runs[0], np.array(runs[1:])
        out[f'{model}_pct'] = full
        out[f'{model}_top5_freq'] = (jk >= 0.95).mean(0)
        out[f'{model}_pct_sd'] = jk.std(0)
        top5 = full >= 0.95
        print(f'[{model}] nodes in full-model top 5%: {top5.sum()} | of these, in the top 5% of >= 80% of '
              f'jackknife models: {(out[f"{model}_top5_freq"][top5] >= 0.8).mean():.2f} | median sd of percentile '
              f'in top 5%: {np.median(out[f"{model}_pct_sd"][top5]):.3f}', flush=True)
    suffix = '' if args.features == 'all' else f'_{args.features}'
    pd.DataFrame(out).to_csv(out_path(f'superior_grid_uncertainty{suffix}.csv'), index=False)


if __name__ == '__main__':
    main()
