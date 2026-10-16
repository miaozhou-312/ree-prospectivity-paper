"""
Stage 12 -- production models with the final DevNet (Stage 11): SHAP directions (all
regions), the Clay-Howells blind check, and Superior prediction-grid scores
for the maps.

Production models are fit on all positives + the training background of a
region (as in src.evaluation.run_experiment): RF (fixed seed), the
z-scored DevNet ensemble (ablation baseline) and the final DevNet. SHAP
uses src.interpretation.shap_pipeline unchanged; for the final DevNet the
"scaler" passed to it is the fitted quantile transformer. Directions use the Spearman rule
of Stage 3.

The Superior prediction grid is the 0.1 degree, A∧B-restricted grid of
src.evaluation.prediction_map.build_grid, with features extracted by the
same src functions as the training points.

    nice -n 10 python -m paper_analyses.12_final_production
"""
import importlib

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from paper_analyses.common import (GLCM_RASTERS, REGION_ORDER, config, load_region, out_path,  # noqa: E402
                                   save_pickle, seed_everything, setup_regions)
from paper_analyses.figstyle import pretty  # noqa: E402
from paper_analyses.models import DEVICE  # noqa: E402
from src.evaluation.prediction_map import build_grid  # noqa: E402
from src.features import raster_preprocessing, sampleable_domain  # noqa: E402
from src.features.glcm import add_glcm_features  # noqa: E402
from src.features.proximity import (add_proximity_line_layers, add_proximity_worms, add_rasters,  # noqa: E402
                                    build_worm_trees)
from src.features.sampling import haversine_km  # noqa: E402
from src.features.terrane import add_terrane_h3  # noqa: E402
from src.interpretation.shap_explainers import explain_devnet  # noqa: E402
from src.interpretation.shap_pipeline import compute_shap_results  # noqa: E402
from src.models.devnet import ensemble_score  # noqa: E402

_s3 = importlib.import_module('paper_analyses.03_shap_directions')
_fin = importlib.import_module('paper_analyses.11_devnet_final')
_tb = importlib.import_module('paper_analyses.10_training_budget')
CLAY_HOWELLS = (49 + 49 / 60 + 10.61 / 3600, -(82 + 3 / 60 + 10.37 / 3600))
TOP_K = 10


def fit_all(region):
    t = _s3.fit_production(region)                       # RF + z-scored DevNet
    rec = load_region(region)
    d = rec['pruned']
    tf, models = _fin.fit_final(d['X_lab'][d['y'] == 1], d['X_unl'], region, config.DN_SEED)
    t_fin = {**t, 'dn_ensemble': models, 'dn_scaler': tf.qt, 'dn_imp': tf.imp}
    return t, t_fin, tf


def features_at(pts, region_cfg, clipped_r, clipped_v, worm_trees, cols):
    pts = add_terrane_h3(pts, clipped_v.get('h3_grid'))
    pts = add_proximity_line_layers(pts, region_cfg['vectors'], config.PROJECTED_CRS)
    pts = add_proximity_worms(pts, worm_trees, config.PROJECTED_CRS)
    pts = add_rasters(pts, clipped_r)
    pts = add_glcm_features(pts, clipped_r, GLCM_RASTERS)
    for c in cols:
        if c not in pts:
            pts[c] = 0.0
    return pts, pts[cols].astype(float).values


def scores(t, t_fin, tf, X):
    rf = t['rf_model'].predict_proba(t['rf_scaler'].transform(t['rf_imp'].transform(X)))[:, 1]
    dn = ensemble_score(t['dn_ensemble'], t['dn_scaler'].transform(t['dn_imp'].transform(X)), DEVICE)
    fin = _tb._ensemble(t_fin['dn_ensemble'], tf(X))
    return rf, dn, fin


def main():
    seed_everything()
    regions = setup_regions()
    fitted = {r: fit_all(r) for r in REGION_ORDER}

    # ── SHAP for the final DevNet vs RF, all regions────────────────────────────────────
    trained_fin = {r: fitted[r][1] for r in REGION_ORDER}
    shap_res = compute_shap_results(config, trained_fin, REGION_ORDER, 100, 200)
    rows = []
    for r in REGION_ORDER:
        sr = shap_res[r]
        sv = [explain_devnet(m, sr['X_bg_dn'], sr['X_explain_dn'], DEVICE, config.DN_SEED)[0]
              for m in trained_fin[r]['dn_ensemble']]
        sr['shap_dn_ens'] = np.mean(sv, axis=0)
        cols = sr['feat_cols']
        imp_rf, imp_dn = np.abs(sr['shap_rf']).mean(0), np.abs(sr['shap_dn_ens']).mean(0)
        rk = lambda v: pd.Series(v).rank(ascending=False).astype(int).values
        for j, c in enumerate(cols):
            rho_rf, dir_rf = _s3.direction(sr['X_explain_rf'][:, j], sr['shap_rf'][:, j])
            rho_dn, dir_dn = _s3.direction(sr['X_explain_dn'][:, j], sr['shap_dn_ens'][:, j])
            rows.append({'region': r, 'feature': c, 'label': pretty(c),
                         'rf_mean_abs_shap': imp_rf[j], 'rf_rank': rk(imp_rf)[j], 'rf_rho': rho_rf,
                         'rf_direction': dir_rf, 'dn_mean_abs_shap': imp_dn[j], 'dn_rank': rk(imp_dn)[j],
                         'dn_rho': rho_dn, 'dn_direction': dir_dn})
    df = pd.DataFrame(rows)
    df.to_csv(out_path('shap_final_directions_all.csv'), index=False)
    for r in REGION_ORDER:
        sub = df[df.region == r].copy()
        sub['joint'] = sub.rf_mean_abs_shap / sub.rf_mean_abs_shap.max() + sub.dn_mean_abs_shap / sub.dn_mean_abs_shap.max()
        top = sub.sort_values('joint', ascending=False).head(TOP_K)
        agree = sum(a.split('->')[-1].strip() == b.split('->')[-1].strip() and 'clear' not in a
                    for a, b in zip(top.rf_direction, top.dn_direction))
        print(f'[{r}] DevNet vs RF: top {TOP_K} directions agree on {agree}/{TOP_K}; importance rank rho='
              f'{spearmanr(sub.rf_rank, sub.dn_rank)[0]:.2f}', flush=True)
        print(top[['label', 'rf_rank', 'rf_direction', 'dn_rank', 'dn_rho', 'dn_direction']].to_string(index=False))
        top.to_csv(out_path(f'shap_final_top{TOP_K}_{r}.csv'), index=False)
    sr = shap_res['Superior']
    names = [pretty(c) for c in sr['feat_cols']]
    fig = plt.figure(figsize=(14, 6.5))
    for i, (key, X, title) in enumerate([('shap_dn_ens', sr['X_explain_dn'], '(a) DevNet (ensemble-mean SHAP)'),
                                         ('shap_rf', sr['X_explain_rf'], '(b) Random Forest (TreeSHAP)')]):
        ax = fig.add_subplot(1, 2, i + 1)
        plt.sca(ax)
        np.random.seed(config.RANDOM_SEED)  # summary_plot shuffles and jitters dots with np.random
        shap.summary_plot(sr[key], X, feature_names=names, max_display=12, show=False, plot_size=None,
                          color_bar=(i == 1))
        ax.set_title(title, fontsize=12, loc='left')
        ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(out_path('fig_shap_superior_final.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── Superior: prediction grid + Clay-Howells ───────────────────────────
    r = 'Superior'
    cfg = regions[r]
    clipped_r, clipped_v = raster_preprocessing.preprocess_all({r: cfg})
    sampleable_domain.build_all({r: cfg}, clipped_r, config.PROJECTED_CRS, GLCM_RASTERS)
    worm_trees = build_worm_trees(cfg['points'], config.PROJECTED_CRS)
    t, t_fin, tf = fitted[r]
    cols = load_region(r)['pruned']['cols']

    grid, _ = build_grid(cfg, r, config.GRID_DEG)
    gpts, Xg = features_at(pd.DataFrame({'lat': grid['lat'].values, 'lon': grid['lon'].values}),
                           cfg, clipped_r[r], clipped_v[r], worm_trees, cols)
    rf_g, dn_g, fin_g = scores(t, t_fin, tf, Xg)
    pd.DataFrame({'lat': gpts['lat'], 'lon': gpts['lon'], 'rf': rf_g, 'devnet_zscore': dn_g, 'devnet': fin_g}).to_csv(
        out_path('superior_grid_scores.csv'), index=False)
    print(f'[{r}] scored prediction grid: {len(Xg)} nodes', flush=True)

    rec = load_region(r)
    X_bg = np.vstack([rec['pruned']['X_lab'][rec['pruned']['y'] == 0], rec['pruned']['X_unl']])
    _, Xc = features_at(pd.DataFrame([{'lat': CLAY_HOWELLS[0], 'lon': CLAY_HOWELLS[1], 'label': 1}]),
                        cfg, clipped_r[r], clipped_v[r], worm_trees, cols)
    rf_b, dn_b, fin_b = scores(t, t_fin, tf, np.vstack([X_bg, Xc]))
    pct = lambda s: float(rankdata(s)[-1] / len(s))
    pos = rec['lab'][rec['lab']['pool'] == 'positive']
    ext = {'target': 'Clay-Howells', 'nearest_training_occurrence_km':
           float(haversine_km(CLAY_HOWELLS[0], CLAY_HOWELLS[1], pos['lat'].values, pos['lon'].values).min()),
           'rf_percentile': pct(rf_b), 'devnet_zscore_percentile': pct(dn_b), 'devnet_percentile': pct(fin_b),
           'grid_rank_devnet': float((fin_g < fin_b[-1]).mean())}
    print(ext, flush=True)
    pd.DataFrame([ext]).to_csv(out_path('external_check_final.csv'), index=False)
    save_pickle({'shap': shap_res}, out_path('shap_final_results.pkl'))


if __name__ == '__main__':
    main()
