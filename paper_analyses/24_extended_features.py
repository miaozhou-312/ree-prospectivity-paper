"""
Stage 24 -- additional evidence layers in the spirit of DEEP-SEAM (Luo et
al., 2026): distance-encoded regional geology and neighbourhood statistics.

1. Regional geology distances (from the H3 geology grid). Point-level map
   classes mark the known complexes themselves (Stage 18/19), so the map is
   first generalised: for every lithology class, connected H3 cells (res. 7,
   ~5 km2) are grouped and units smaller than MIN_UNIT_CELLS (~100 km2) are
   removed, as is the alkalic-intrusive class altogether. Distances (km) are
   then measured to the generalised map's lithological contacts and age
   (era) contacts and to regional greenstone (extrusive), mafic-ultramafic,
   high-grade metamorphic and sedimentary units. Individual intrusions of
   carbonatite-complex size no longer exist on this map.
2. Neighbourhood statistics: mean and standard deviation of the Bouguer
   anomaly, its 40 km high-pass, the DRTP and the TMI within RADIUS_KM of
   each point (native grid and projection).
3. Radiometrics (K, eU, eTh, dose rate; Buckle et al. 2014, 250 m) when the
   grids are present under dataset/rasters/radiometric/: point value and
   neighbourhood mean.

Outputs out/extended_<region>.pkl (columns aligned with the region cache:
labelled rows, unlabelled rows, and for Superior the 0.1 degree prediction
grid). Licensed-grid derivatives stay in out/ (git-ignored).

    python -m paper_analyses.24_extended_features
"""
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from paper_analyses.common import REGION_ORDER, load_region, out_path, save_pickle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(REPO), 'dataset')
H3 = os.path.join(DATA, 'h3_grid', 'Geology_H3Grid_Canada.shp')
MIN_UNIT_CELLS = 20          # ~100 km2 at H3 resolution 7
NEIGHBOUR_KM = 2.6           # adjacent res-7 cells (centre spacing ~2.1 km)
RADIUS_KM = 10.0
TO_3978 = Transformer.from_crs('EPSG:4326', 'EPSG:3978', always_xy=True)
UNITS = {
    'greenstone': lambda l: l == 'Igneous_Extrusive',
    'mafic_ultramafic': lambda l: l in ('Igneous_Intrusive_Mafic', 'Igneous_Intrusive_Ultramafic'),
    'highgrade': lambda l: l.startswith(('Metamorphic_Migmatite', 'Metamorphic_Gneiss', 'Metamorphic_Charnockite',
                                         'Metamorphic_Granulite')),
    'sedimentary': lambda l: l.startswith('Sedimentary_'),
}
NEIGHBOURHOOD_RASTERS = {'BA': 'Canada_BA', 'BA_HP40km': 'Canada_BA_HP40km', 'DRTP': 'DRTP', 'TMI': 'TMI'}
RASTER_ROOT = os.path.abspath(os.environ.get('REE_RASTER_DIR', os.path.join(DATA, 'rasters')))


def raster_file(name):
    """The grid called `name` (.flt or .tif) anywhere under the raster root."""
    hits = [p for ext in ('flt', 'tif') for p in glob.glob(os.path.join(RASTER_ROOT, '**', f'{name}.{ext}'),
                                                           recursive=True)]
    if len(hits) != 1:
        raise FileNotFoundError(f'{name}: {hits or "not found"} under {RASTER_ROOT}')
    return hits[0]


def xy_km(lon, lat):
    x, y = TO_3978.transform(np.asarray(lon), np.asarray(lat))
    return np.c_[x, y] / 1000.0


def h3_cells(region):
    out = gpd.read_file(os.path.join(REPO, 'data', 'outlines', f'{region}.geojson')).to_crs('EPSG:4326')
    x0, y0, x1, y1 = out.total_bounds
    df = pyogrio.read_dataframe(H3, columns=['LongWGS84', 'LatWGS84', 'LithMajor', 'EraMaxMjr'], read_geometry=False,
                                where=f'LatWGS84 > {y0 - 2} AND LatWGS84 < {y1 + 2} AND '
                                      f'LongWGS84 > {x0 - 3} AND LongWGS84 < {x1 + 3}')
    return df.reset_index(drop=True)


def generalise(df):
    """Label of each cell on the generalised map ('' = removed)."""
    xy = xy_km(df.LongWGS84.values, df.LatWGS84.values)
    pairs = cKDTree(xy).query_pairs(NEIGHBOUR_KM, output_type='ndarray')
    lith = df.LithMajor.fillna('').values
    same = lith[pairs[:, 0]] == lith[pairs[:, 1]]
    p = pairs[same]
    n = len(df)
    g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    size = np.bincount(comp)[comp]
    keep = (size >= MIN_UNIT_CELLS) & (lith != '') & (lith != 'Igneous_Intrusive_Alkalic')
    gen = np.where(keep, lith, '')
    era = np.where(keep, df.EraMaxMjr.fillna('').values, '')
    a, b = pairs[:, 0], pairs[:, 1]
    both = (gen[a] != '') & (gen[b] != '')
    lith_contact = np.zeros(n, bool)
    age_contact = np.zeros(n, bool)
    for mask, flag in [(both & (gen[a] != gen[b]), lith_contact), (both & (era[a] != era[b]) & (era[a] != '') &
                                                                  (era[b] != ''), age_contact)]:
        flag[a[mask]] = True
        flag[b[mask]] = True
    return xy, gen, lith_contact, age_contact, keep


def geology_distances(df, pts_lonlat):
    xy, gen, lc, ac, keep = generalise(df)
    q = xy_km(*pts_lonlat)
    feats = {}
    targets = {'dist_regional_contact': lc, 'dist_regional_age_contact': ac}
    targets.update({f'dist_{k}': np.array([f(l) if l else False for l in gen]) for k, f in UNITS.items()})
    for name, m in targets.items():
        feats[name] = cKDTree(xy[m]).query(q)[0] if m.any() else np.full(len(q), np.nan)
    info = {'cells': len(df), 'kept_share': float(keep.mean()), **{k: int(v.sum()) for k, v in targets.items()}}
    return feats, info


def neighbourhood(path, pts_lonlat, radius_km):
    lon, lat = pts_lonlat
    with rasterio.open(path) as r:
        a = r.read(1, masked=True).astype('float64').filled(np.nan)
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        if r.crs.to_epsg() == 3978:
            x, y = TO_3978.transform(lon, lat)
            ry = rx = np.full(len(lon), radius_km * 1000.0 / abs(r.res[1]))
        else:
            x, y = np.asarray(lon), np.asarray(lat)
            ry = np.full(len(lon), radius_km / 111.2 / abs(r.res[1]))
            rx = radius_km / (111.2 * np.cos(np.radians(lat))) / abs(r.res[0])
        rows, cols = rasterio.transform.rowcol(r.transform, x, y)
    rows, cols = np.asarray(rows), np.asarray(cols)
    mean, std = np.full(len(lon), np.nan), np.full(len(lon), np.nan)
    for i in range(len(lon)):
        dy, dx = int(np.ceil(ry[i])), int(np.ceil(rx[i]))
        r0, r1, c0, c1 = max(rows[i] - dy, 0), min(rows[i] + dy + 1, a.shape[0]), max(cols[i] - dx, 0), \
            min(cols[i] + dx + 1, a.shape[1])
        if r0 >= r1 or c0 >= c1:
            continue
        w = a[r0:r1, c0:c1]
        yy, xx = np.mgrid[r0:r1, c0:c1]
        inside = ((yy - rows[i]) / max(ry[i], 1e-9)) ** 2 + ((xx - cols[i]) / max(rx[i], 1e-9)) ** 2 <= 1.0
        v = w[inside]
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            mean[i], std[i] = v.mean(), v.std()
    return mean, std


def point_values(path, pts_lonlat):
    lon, lat = pts_lonlat
    with rasterio.open(path) as r:
        x, y = (TO_3978.transform(lon, lat) if r.crs and r.crs.to_epsg() == 3978 else (lon, lat))
        v = np.array([s[0] for s in r.sample(zip(x, y))], dtype='float64')
        if r.nodata is not None:
            v[v == r.nodata] = np.nan
    return v


def build(region, sets):
    df = h3_cells(region)
    out = {}
    for name, (lon, lat) in sets.items():
        feats, info = geology_distances(df, (lon, lat))
        for k, path in NEIGHBOURHOOD_RASTERS.items():
            m, s = neighbourhood(raster_file(path), (lon, lat), RADIUS_KM)
            feats[f'{k}_mean{RADIUS_KM:.0f}km'], feats[f'{k}_std{RADIUS_KM:.0f}km'] = m, s
        for path in sorted(glob.glob(os.path.join(DATA, 'rasters', 'radiometric', '*.tif')) +
                           glob.glob(os.path.join(DATA, 'rasters', 'radiometric', '*.grd'))):
            k = 'rad_' + os.path.splitext(os.path.basename(path))[0]
            feats[k] = point_values(path, (lon, lat))
            feats[f'{k}_mean{RADIUS_KM:.0f}km'] = neighbourhood(path, (lon, lat), RADIUS_KM)[0]
        out[name] = pd.DataFrame(feats)
        print(f'[{region} | {name}] {len(lon)} points, {out[name].shape[1]} features; generalised map: {info}',
              flush=True)
    return out


def main():
    for region in REGION_ORDER:
        rec = load_region(region)
        sets = {'lab': (rec['lab']['lon'].values, rec['lab']['lat'].values),
                'unl': (rec['unl']['lon'].values, rec['unl']['lat'].values)}
        grid_file = out_path('superior_grid_features.npz')
        if region == 'Superior' and os.path.exists(grid_file):
            z = np.load(grid_file)
            sets['grid'] = (z['lon'], z['lat'])
        save_pickle(build(region, sets), out_path(f'extended_{region}.pkl'))


if __name__ == '__main__':
    main()
