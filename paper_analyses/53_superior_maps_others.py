"""
Stage 53 -- Superior prospectivity maps of the other three models of the
benchmark (XGBoost, bagging-PU, nnPU), made as Stage 29 makes those of
DevNet and RF: each model, with the settings of the benchmark (Stages 25 and
47), is fitted to all Superior positives against the training background and
scores the 0.1 degree prediction grid (inputs from Stage 29's
superior_grid); Clay-Howells is ranked among the province's background
points, as in Stage 29.

Writes out/superior_grid_scores_fs3_others.csv (lat, lon, xgboost,
baggingpu, nnpu), out/external_check_fs3_others.csv (per model: Clay-Howells
percentile, Spearman correlation with DevNet over the grid) and
figures_r7/fig_map_superior_others_r7.png in the style of the DevNet and RF maps (Stage 13),
lettered c-e to follow them in one figure.

`ksz` describes the largest DevNet top-1% patch inside the frame of Fig. 12
(classes as drawn there, on the 1 km raster of Stage 13): its area, centre
and extent, the distance from its centre to the nearest training occurrence,
and the grid percentile of every model at the Kidd Creek mine (48.69 N,
81.37 W) -> out/ksz_top1_patch.csv.

    source ../work_public/env.sh
    (unset REE_FEATURESET; python -m paper_analyses.53_superior_maps_others [ksz|draw])
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from paper_analyses.common import OUT_DIR, out_path, seed_everything
from paper_analyses.models import bagging_pu_factory, xgb_factory

_s14 = importlib.import_module('paper_analyses.14_final_suite')
_s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
_p29 = importlib.import_module('paper_analyses.29_production_fs3')
_m = importlib.import_module('paper_analyses.13_maps')
REGION = 'Superior'
MODELS = [('xgboost', 'XGBoost', xgb_factory), ('baggingpu', 'bagging-PU', bagging_pu_factory),
          ('nnpu', 'nnPU', _s14.nnpu_qt_factory)]
FIG = os.path.join(_m.REPO, 'paper', 'manuscript', 'figures_r7')


def main():
    rec, cols, X_lab, X_unl, _ = _s25.extended_matrices(REGION, 'reencoded')
    y = rec['pruned']['y']
    X_pos = X_lab[y == 1]
    z, Xg, Xclay = _p29.superior_grid(rec, cols)
    X_bg_all = np.vstack([X_lab[y == 0], X_unl])
    dn = pd.read_csv(out_path('superior_grid_scores_fs3.csv'))
    assert np.allclose(dn.lat, z['lat']) and np.allclose(dn.lon, z['lon'])
    grid = pd.DataFrame({'lat': z['lat'], 'lon': z['lon']})
    ext = []
    for col, name, factory in MODELS:
        scorer = factory(REGION)(X_unl)
        seed_everything()
        grid[col] = scorer(X_pos, np.ones(len(X_pos)), Xg, seed_counter=0)
        seed_everything()
        sb = scorer(X_pos, np.ones(len(X_pos)), np.vstack([X_bg_all, Xclay]), seed_counter=0)
        ext.append({'model': name, 'clay_howells_percentile': float(rankdata(sb)[-1] / len(sb)),
                    'spearman_with_devnet': float(spearmanr(grid[col], dn.devnet)[0]),
                    'share_zero': float((grid[col] == 0).mean())})
        print(ext[-1], flush=True)
    grid.to_csv(out_path('superior_grid_scores_fs3_others.csv'), index=False)
    pd.DataFrame(ext).to_csv(out_path('external_check_fs3_others.csv'), index=False)
    draw()


def draw():
    _m.OUT_DIR, _m.FIG = OUT_DIR, FIG
    _m.superior_maps(panels=[(col, f'Superior {name} MPM', letter) for (col, name, _), letter in zip(MODELS, 'cde')],
                     outname='fig_map_superior_others_r7.png', grid_file='superior_grid_scores_fs3_others.csv')


KIDD_CREEK = (48.687, -81.371)                   # Kidd mine, Kidd Township (Mindat; Wikidata 48 41'13"N 81 22'16"W)


def ksz_patch():
    import geopandas as gpd
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from src.features.sampling import haversine_km
    _m.OUT_DIR = OUT_DIR
    X0, X1, Y0, Y1 = _m.ksz_frame()
    g = pd.read_csv(out_path('superior_grid_scores_fs3.csv'))
    o = pd.read_csv(out_path('superior_grid_scores_fs3_others.csv'))
    outline = gpd.read_file(os.path.join(_m.REPO, 'data', 'outlines', 'Superior.geojson')).to_crs(_m.CRS)
    gx, gy, cls, _ = _m.raster(g, 'devnet', outline, res=1000.0)
    top1 = (cls == 3) & (gx >= X0) & (gx <= X1) & (gy >= Y0) & (gy <= Y1)
    lab, n = ndimage.label(top1)
    sizes = ndimage.sum(top1, lab, range(1, n + 1))
    big = lab == int(np.argmax(sizes)) + 1
    lon, lat = _m.TO_G.transform(gx[big], gy[big])
    clon, clat = _m.TO_G.transform(gx[big].mean(), gy[big].mean())
    dep = _m.occurrences(REGION)
    row = {'area_km2': float(big.sum()), 'centre_lat': clat, 'centre_lon': clon, 'lat_min': lat.min(),
           'lat_max': lat.max(), 'lon_min': lon.min(), 'lon_max': lon.max(),
           'nearest_occurrence_km': float(haversine_km(clat, clon, dep.Latitude.values, dep.Longitude.values).min())}
    x, y = _m.TO_P.transform(g.lon.values, g.lat.values)
    d, i = cKDTree(np.c_[x, y]).query(_m.TO_P.transform(KIDD_CREEK[1], KIDD_CREEK[0]))
    row['kidd_creek_node_km'] = d / 1000.0
    for col in ('devnet', 'rf'):
        row[f'kidd_creek_pct_{col}'] = float(rankdata(g[col])[i] / len(g))
    for col, _, _ in MODELS:
        row[f'kidd_creek_pct_{col}'] = float(rankdata(o[col])[i] / len(o))
    pd.DataFrame([row]).to_csv(out_path('ksz_top1_patch.csv'), index=False)
    print(row, flush=True)


if __name__ == '__main__':
    {'ksz': ksz_patch, 'draw': draw}.get(sys.argv[1] if len(sys.argv) > 1 else '', main)()
