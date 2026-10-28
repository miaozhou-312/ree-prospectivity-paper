"""
Stage 29 -- production models on feature set 3 (the main analysis of the
second revision).

  * production DevNet (final configuration), MLP-BCE and RF (fixed seed),
    fitted to all positives + the training background of each region;
  * SHAP: DeepSHAP for DevNet (ensemble mean, quantile-normal inputs) and
    TreeSHAP for RF on an explain set of all positives + evaluation-
    background points (200 in total; 100 training-background points as the
    DeepSHAP reference); directions by the Spearman rule of Stage 3;
  * Superior: 0.1 degree grid scores (maps), the Clay-Howells blind check,
    and leave-one-occurrence-out stability of the top-5% class.

    nice -n 10 python -m paper_analyses.29_production_fs3
"""
import importlib
import os

import numpy as np
import pandas as pd
import shap
from scipy.stats import rankdata

from paper_analyses.common import (GLCM_RASTERS, REGION_ORDER, config, load_pickle, load_region, model_cfg,
                                   out_path, save_pickle, seed_everything, setup_regions)
from paper_analyses.figstyle import pretty
from paper_analyses.models import DEVICE
from src.features import raster_preprocessing, sampleable_domain
from src.features.proximity import build_worm_trees
from src.features.sampling import haversine_km
from src.interpretation.shap_explainers import explain_devnet
from src.models.random_forest import make_rf_params, make_rf_pipeline

_fin = importlib.import_module('paper_analyses.11_devnet_final')
_tb = importlib.import_module('paper_analyses.10_training_budget')
_c17 = importlib.import_module('paper_analyses.17_controls')
_s3 = importlib.import_module('paper_analyses.03_shap_directions')
_p12 = importlib.import_module('paper_analyses.12_final_production')
_s24 = importlib.import_module('paper_analyses.24_extended_features')
_s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
_s26 = importlib.import_module('paper_analyses.26_geology_reencode')
TOP_K = 10


def fit_models(X_pos, X_bg, region):
    X_tr = np.vstack([X_pos, X_bg])
    y_tr = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg))]
    tf, dn = _fin.fit_final(X_pos, X_bg, region, config.DN_SEED)
    mlp = [_c17.train_bce(tf(X_tr), y_tr, config.DN_SEED + k, model_cfg(region)) for k in range(config.DN_RESTARTS)]
    rf = make_rf_pipeline(make_rf_params(config.RANDOM_SEED)).fit(X_tr, y_tr)
    return {'tf': tf, 'devnet': dn, 'mlp': mlp, 'rf': rf}


def score(m, X, which):
    if which == 'rf':
        return m['rf'].predict_proba(X)[:, 1]
    return _tb._ensemble(m[which], m['tf'](X))


def shap_region(region, rec, cols, X_lab, X_unl, m):
    y = rec['pruned']['y']
    rng = np.random.default_rng(config.RANDOM_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = np.r_[pos, rng.choice(neg, 200 - len(pos), replace=False)]
    X_ex = X_lab[idx]
    X_ref = X_unl[rng.choice(len(X_unl), 100, replace=False)]
    sv = [explain_devnet(net, m['tf'](X_ref), m['tf'](X_ex), DEVICE, config.DN_SEED)[0] for net in m['devnet']]
    sh_dn = np.mean(sv, axis=0)
    rfm = m['rf'].named_steps['rf']
    Xs = m['rf'].named_steps['scaler'].transform(m['rf'].named_steps['imputer'].transform(X_ex))
    s = shap.TreeExplainer(rfm).shap_values(Xs)
    sh_rf = s[1] if isinstance(s, list) else (s[..., 1] if s.ndim == 3 else s)
    X_raw = m['rf'].named_steps['imputer'].transform(X_ex)
    imp_dn, imp_rf = np.abs(sh_dn).mean(0), np.abs(sh_rf).mean(0)
    rk = lambda v: pd.Series(v).rank(ascending=False).astype(int).values
    rows = []
    for j, c in enumerate(cols):
        rho_dn, dir_dn = _s3.direction(X_raw[:, j], sh_dn[:, j])
        rho_rf, dir_rf = _s3.direction(X_raw[:, j], sh_rf[:, j])
        rows.append({'region': region, 'feature': c, 'label': pretty(c), 'dn_mean_abs_shap': imp_dn[j],
                     'dn_rank': rk(imp_dn)[j], 'dn_rho': rho_dn, 'dn_direction': dir_dn, 'rf_mean_abs_shap': imp_rf[j],
                     'rf_rank': rk(imp_rf)[j], 'rf_rho': rho_rf, 'rf_direction': dir_rf})
    df = pd.DataFrame(rows)
    df['joint'] = df.rf_mean_abs_shap / df.rf_mean_abs_shap.max() + df.dn_mean_abs_shap / df.dn_mean_abs_shap.max()
    return df, {'shap_dn': sh_dn, 'shap_rf': sh_rf, 'X_raw': X_raw, 'cols': cols}


def set3_rows(region, rec, cols, sets):
    """Set-3 feature matrix for arbitrary points of `region` (base pruned
    columns supplied by the caller as a DataFrame per set)."""
    cells = _s24.h3_cells(region)
    cells26 = _s26.h3_cells(region)
    tree = __import__('scipy.spatial', fromlist=['cKDTree']).cKDTree(
        _s24.xy_km(cells26.LongWGS84.values, cells26.LatWGS84.values))
    unl_att = _s26.attributes(cells26, tree, rec['unl']['lon'].values, rec['unl']['lat'].values)
    lc = unl_att.LithMajor.fillna('Unknown').value_counts()
    ec = unl_att.EraMaxMjr.fillna('Unknown').value_counts()
    lith_classes = sorted(lc[lc >= _s26.MIN_TRAIN_BG].index)
    era_classes = sorted(ec[ec >= _s26.MIN_TRAIN_BG].index)
    out = {}
    for name, (base_df, lon, lat) in sets.items():
        feats, _ = _s24.geology_distances(cells, (lon, lat))
        for k, path in _s24.NEIGHBOURHOOD_RASTERS.items():
            mu, sd = _s24.neighbourhood(_s24.raster_file(path), (lon, lat), _s24.RADIUS_KM)
            feats[f'{k}_mean{_s24.RADIUS_KM:.0f}km'], feats[f'{k}_std{_s24.RADIUS_KM:.0f}km'] = mu, sd
        geo = _s26.encode(_s26.attributes(cells26, tree, lon, lat), lith_classes, era_classes)
        full = pd.concat([base_df.reset_index(drop=True), pd.DataFrame(feats), geo], axis=1)
        out[name] = full.reindex(columns=cols).values.astype(float)
    return out


def superior_grid(rec, cols, region='Superior'):
    """Set-3 inputs at the 0.1 degree prediction-grid nodes (Stage 21) and at
    Clay-Howells."""
    z = np.load(out_path('superior_grid_features.npz'))
    pruned = rec['pruned']['cols']
    base = [c for c in cols if c in pruned]
    grid_base = pd.DataFrame(z['X'], columns=pruned)[base]
    cfg = setup_regions()[region]
    clipped_r, clipped_v = raster_preprocessing.preprocess_all({region: cfg})
    sampleable_domain.build_all({region: cfg}, clipped_r, config.PROJECTED_CRS, GLCM_RASTERS)
    worm_trees = build_worm_trees(cfg['points'], config.PROJECTED_CRS)
    lat_c, lon_c = _p12.CLAY_HOWELLS
    _, Xc = _p12.features_at(pd.DataFrame([{'lat': lat_c, 'lon': lon_c, 'label': 1}]), cfg, clipped_r[region],
                             clipped_v[region], worm_trees, pruned)
    clay_base = pd.DataFrame(Xc, columns=pruned)[base]
    M = set3_rows(region, rec, cols, {'grid': (grid_base, z['lon'], z['lat']),
                                      'clay': (clay_base, np.array([lon_c]), np.array([lat_c]))})
    return z, M['grid'], M['clay']


def main():
    seed_everything()
    shap_rows, shap_store, models = [], {}, {}
    for region in REGION_ORDER:
        rec, cols, X_lab, X_unl, _ = _s25.extended_matrices(region, 'reencoded')
        y = rec['pruned']['y']
        m = fit_models(X_lab[y == 1], X_unl, region)
        models[region] = (rec, cols, X_lab, X_unl, m)
        df, st = shap_region(region, rec, cols, X_lab, X_unl, m)
        shap_rows.append(df)
        shap_store[region] = st
        top = df.sort_values('joint', ascending=False).head(TOP_K)
        agree = sum(a.split('->')[-1].strip() == b.split('->')[-1].strip() and 'clear' not in a
                    for a, b in zip(top.rf_direction, top.dn_direction))
        top.to_csv(out_path(f'shap_fs3_top{TOP_K}_{region}.csv'), index=False)
        print(f'[{region}] SHAP: top-{TOP_K} directions agree {agree}/{TOP_K}', flush=True)
    pd.concat(shap_rows).to_csv(out_path('shap_fs3_directions_all.csv'), index=False)
    save_pickle(shap_store, out_path('shap_fs3_results.pkl'))

    # Superior: grid, Clay-Howells, stability
    region = 'Superior'
    rec, cols, X_lab, X_unl, m = models[region]
    y = rec['pruned']['y']
    z, Xg, Xclay = superior_grid(rec, cols)
    lat_c, lon_c = _p12.CLAY_HOWELLS
    pct = lambda s: rankdata(s) / len(s)
    grid = pd.DataFrame({'lat': z['lat'], 'lon': z['lon']})
    X_bg_all = np.vstack([X_lab[y == 0], X_unl])
    ext = {'target': 'Clay-Howells'}
    pos = rec['lab'][rec['lab']['pool'] == 'positive']
    ext['nearest_training_occurrence_km'] = float(haversine_km(lat_c, lon_c, pos['lat'].values, pos['lon'].values).min())
    for w in ('devnet', 'mlp', 'rf'):
        grid[w] = score(m, Xg, w)
        sb = score(m, np.vstack([X_bg_all, Xclay]), w)
        ext[f'{w}_percentile'] = float(rankdata(sb)[-1] / len(sb))
        ext[f'{w}_grid_rank'] = float((grid[w].values < sb[-1]).mean())
    grid.to_csv(out_path('superior_grid_scores_fs3.csv'), index=False)
    pd.DataFrame([ext]).to_csv(out_path('external_check_fs3.csv'), index=False)
    print(ext, flush=True)

    X_pos = X_lab[y == 1]
    unc = {'lat': z['lat'], 'lon': z['lon']}
    runs = {w: [pct(score(m, Xg, w))] for w in ('devnet', 'mlp', 'rf')}
    for leave in range(len(X_pos)):
        mm = fit_models(np.delete(X_pos, leave, axis=0), X_unl, region)
        for w in runs:
            runs[w].append(pct(score(mm, Xg, w)))
    for w in ('devnet', 'mlp', 'rf'):
        full, jk = runs[w][0], np.array(runs[w][1:])
        unc[f'{w}_pct'], unc[f'{w}_top5_freq'], unc[f'{w}_pct_sd'] = full, (jk >= 0.95).mean(0), jk.std(0)
        top5 = full >= 0.95
        print(f'[stability | {w}] top-5% nodes stable in >=80% of jackknife models: '
              f'{(unc[f"{w}_top5_freq"][top5] >= 0.8).mean():.2f}', flush=True)
    pd.DataFrame(unc).to_csv(out_path('superior_grid_uncertainty_fs3.csv'), index=False)


if __name__ == '__main__':
    main()
