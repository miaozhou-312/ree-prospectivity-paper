"""
Stage 26 -- geology re-encoded from the raw H3 attributes (feature set 3).

The thesis encoding grouped the H3 lithologies into four classes (plus era).
Here every raw LithMajor class gets its own one-hot column (no grouping),
together with the raw era (EraMaxMjr) one-hot and the H3 lithological-
contact and age-contact flags (LithCntact, PrdContact). To keep the number
of inputs manageable, a class gets a column only if at least MIN_TRAIN_BG
of the region's 5,000 training-background points fall in it (fixed before
running); rarer classes share an "other" column. Each point takes the
attributes of the H3 cell whose centre is nearest (= the containing cell on
a hexagonal grid).

Outputs out/geology_raw_<region>.pkl with the same row alignment as the
region cache (lab, unl, and for Superior the prediction grid).

    python -m paper_analyses.26_geology_reencode
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.spatial import cKDTree

from paper_analyses.common import REGION_ORDER, load_region, out_path, save_pickle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H3 = os.path.join(os.path.dirname(REPO), 'dataset', 'h3_grid', 'Geology_H3Grid_Canada.shp')
MIN_TRAIN_BG = 10
_s24 = __import__('importlib').import_module('paper_analyses.24_extended_features')


def h3_cells(region):
    out = gpd.read_file(os.path.join(REPO, 'data', 'outlines', f'{region}.geojson')).to_crs('EPSG:4326')
    x0, y0, x1, y1 = out.total_bounds
    return pyogrio.read_dataframe(H3, columns=['LongWGS84', 'LatWGS84', 'LithMajor', 'EraMaxMjr', 'LithCntact',
                                               'PrdContact'], read_geometry=False,
                                  where=f'LatWGS84 > {y0 - 1} AND LatWGS84 < {y1 + 1} AND '
                                        f'LongWGS84 > {x0 - 2} AND LongWGS84 < {x1 + 2}').reset_index(drop=True)


def attributes(cells, tree, lon, lat):
    _, i = tree.query(_s24.xy_km(lon, lat))
    return cells.iloc[i].reset_index(drop=True)


def encode(att, lith_classes, era_classes):
    f = {}
    lith = att.LithMajor.fillna('Unknown')
    for c in lith_classes:
        f[f'lith_{c}'] = (lith == c).astype(float).values
    f['lith_other'] = (~lith.isin(lith_classes)).astype(float).values
    era = att.EraMaxMjr.fillna('Unknown')
    for c in era_classes:
        f[f'era_{c}'] = (era == c).astype(float).values
    f['lith_contact'] = (att.LithCntact == 'Present').astype(float).values
    f['age_contact'] = (att.PrdContact == 'Present').astype(float).values
    return pd.DataFrame(f)


def main():
    for region in REGION_ORDER:
        rec = load_region(region)
        cells = h3_cells(region)
        tree = cKDTree(_s24.xy_km(cells.LongWGS84.values, cells.LatWGS84.values))
        sets = {'lab': (rec['lab']['lon'].values, rec['lab']['lat'].values),
                'unl': (rec['unl']['lon'].values, rec['unl']['lat'].values)}
        grid_file = out_path('superior_grid_features.npz')
        if region == 'Superior' and os.path.exists(grid_file):
            z = np.load(grid_file)
            sets['grid'] = (z['lon'], z['lat'])
        att = {k: attributes(cells, tree, *v) for k, v in sets.items()}
        lc = att['unl'].LithMajor.fillna('Unknown').value_counts()
        ec = att['unl'].EraMaxMjr.fillna('Unknown').value_counts()
        lith_classes = sorted(lc[lc >= MIN_TRAIN_BG].index)
        era_classes = sorted(ec[ec >= MIN_TRAIN_BG].index)
        out = {k: encode(a, lith_classes, era_classes) for k, a in att.items()}
        y = rec['pruned']['y']
        pos = out['lab'][y == 1]
        print(f'[{region}] {len(lith_classes)} lithology + {len(era_classes)} era classes kept '
              f'(of {len(lc)} / {len(ec)}); {out["lab"].shape[1]} columns', flush=True)
        print('   occurrences per column:', {c: int(v) for c, v in pos.sum().items() if v > 0}, flush=True)
        save_pickle(out, out_path(f'geology_raw_{region}.pkl'))


if __name__ == '__main__':
    main()
