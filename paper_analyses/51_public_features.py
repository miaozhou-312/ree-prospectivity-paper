"""
Stage 51: the national feature table and the S2 grid with the public-data
gravity and magnetic grids of Stage 50 in place of Getech's, and GLCM
textures of all eleven gravity and magnetic grids (REE_GLCM=all).

Run with ../work_public/env.sh. The regional tables are rebuilt from scratch
by Stages 00, 24 and 26 in the new work directory. The national table and
the S2 grid keep their points and every column that does not come from a
gravity or magnetic grid; the raster values of the seven derived grids, the
GLCM textures and the 10 km statistics of BA_HP40km and DRTP are recomputed
the way Stages 41 and 43 compute them (rasters and GLCM per geological
province), and the feature sets are pruned again with Stage 41's rules.

    python -m paper_analyses.51_public_features national   # -> $REE_PAPER_WORK/national/national_table.pkl
    python -m paper_analyses.51_public_features s2         # -> $REE_PAPER_WORK/national/s2_l12_features.npz
"""
import argparse
import importlib
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from paper_analyses.common import GLCM_RASTERS, WORK_DIR, config, load_pickle, save_pickle, setup_regions
from paper_analyses.national import EXT_COLS, TERRANE, encode_geology, nat_path, province_extent
from src.features.collinearity import prune_collinear

ROOT = os.path.dirname(WORK_DIR)
OLD_NAT = os.path.join(ROOT, 'work', 'national')   # the tables built with Getech's grids
DERIVED = ['Canada_BA_HP40km', 'Canada_BA_HP100km', 'Canada_BA_THD_Tilt', 'DRTP', 'DRTP_1VD', 'DRTP_THD',
           'DRTP_THD_Tilt']
NEIGH = {'BA_HP40km': 'Canada_BA_HP40km', 'DRTP': 'DRTP'}   # the 10 km statistics that change
WORKERS = int(os.environ.get('REE_WORKERS', '6'))


def glcm_cols(rasters=None):
    from src.features.glcm import GLCM_PROPS
    return [f'{k}_glcm_{p}' for k in (rasters or GLCM_RASTERS) for p in GLCM_PROPS]


def _province_job(args):
    """Derived-grid values and GLCM textures at points of one province, from
    that province's buffered clips (Stage 41's rasters_by_province and
    glcm_by_province)."""
    name, lat, lon, glcm_rasters = args
    from src.features.glcm import add_glcm_features
    from src.features.proximity import add_rasters
    from src.features.raster_preprocessing import preprocess_rasters
    cfg = dict(next(iter(setup_regions().values())))
    outline, clip = province_extent(name)
    rasters = preprocess_rasters(clip, dict(cfg, outline=outline))
    pts = pd.DataFrame({'lat': lat, 'lon': lon})
    pts = add_rasters(pts, {k: rasters[k] for k in DERIVED})
    pts = add_glcm_features(pts, rasters, glcm_rasters)
    return name, pts.drop(columns=['lat', 'lon'])


def grid_columns(pts, province, glcm_rasters=None, neigh=NEIGH):
    """Columns that come from the gravity and magnetic grids, at `pts`."""
    glcm_rasters = list(glcm_rasters if glcm_rasters is not None else GLCM_RASTERS)
    out = pd.DataFrame(index=pts.index, columns=DERIVED + glcm_cols(glcm_rasters), dtype=float)
    jobs = []
    for name in pd.unique(province):
        m = np.asarray(province == name)
        jobs.append((name, pts.lat.values[m], pts.lon.values[m], glcm_rasters))
    jobs.sort(key=lambda j: -len(j[1]))
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for name, df in ex.map(_province_job, jobs):
            m = np.asarray(province == name)
            out.loc[m, df.columns] = df.values
            print(f'  {name}: {int(m.sum()):,} points', flush=True)
    s24 = importlib.import_module('paper_analyses.24_extended_features')
    for k, stem in neigh.items():
        mu, sd = s24.neighbourhood(s24.raster_file(stem), (pts.lon.values, pts.lat.values), s24.RADIUS_KM)
        out[f'{k}_mean{s24.RADIUS_KM:.0f}km'], out[f'{k}_std{s24.RADIUS_KM:.0f}km'] = mu, sd
    return out


def national():
    t0 = time.time()
    meta = load_pickle(os.path.join(OLD_NAT, 'national_table.pkl'))
    pts = meta['pts'].copy()
    old_glcm = [c for c in meta['set1_cols'] if '_glcm_' in c]
    print(f'{len(pts):,} points; GLCM on {len(GLCM_RASTERS)} grids', flush=True)
    new = grid_columns(pts, pts.province.values)
    pts = pts.drop(columns=old_glcm)
    for c in new.columns:
        pts[c] = new[c].values

    # feature sets exactly as in Stage 41, with the GLCM block replaced
    i = meta['set1_cols'].index(old_glcm[0])
    set1_cols = [c for c in meta['set1_cols'] if c not in old_glcm]
    set1_cols = set1_cols[:i] + glcm_cols() + set1_cols[i:]
    nat = pts[pts.in_national].reset_index(drop=True)
    set1_pruned = prune_collinear(nat, set1_cols, thresh=config.COLLINEARITY_THRESH)
    base = [c for c in set1_pruned if c not in TERRANE]
    geo_cols = list(encode_geology(pts.head(5), meta['lith_classes'], meta['era_classes']).columns)
    cols2 = prune_collinear(nat, base + EXT_COLS, thresh=config.COLLINEARITY_THRESH)
    cols3 = prune_collinear(nat, base + EXT_COLS + geo_cols, thresh=config.COLLINEARITY_THRESH)
    const = [c for c in cols3 if nat[c].nunique() < 2]
    cols2, cols3 = [c for c in cols2 if c not in const], [c for c in cols3 if c not in const]
    print(f'set 1: {len(set1_cols)} -> {len(set1_pruned)}; without map classes: {len(cols2)}; '
          f'feature set 3: {len(cols3)} features')
    print('  set 3:', cols3)
    print('  new in set 3:', sorted(set(cols3) - set(meta['cols3'])))
    print('  gone from set 3:', sorted(set(meta['cols3']) - set(cols3)))
    out = dict(meta, pts=pts, set1_cols=set1_cols, set1_pruned=set1_pruned, cols2=cols2, cols3=cols3)
    save_pickle(out, nat_path('national_table.pkl'))
    print(f'saved {nat_path("national_table.pkl")} in {(time.time() - t0) / 60:.1f} min', flush=True)


def s2():
    """The S2 grid of Stage 43 in the new national feature sets: unchanged
    columns are copied, the grid-derived ones recomputed at the cell centres."""
    import s2sphere as s2l
    t0 = time.time()
    meta = load_pickle(nat_path('national_table.pkl'))
    old = np.load(os.path.join(OLD_NAT, 's2_l12_features.npz'), allow_pickle=True)
    ocols, ocols2 = list(old['cols']), list(old['cols2'])
    s24 = importlib.import_module('paper_analyses.24_extended_features')
    neigh_cols = {f'{k}_{s}{s24.RADIUS_KM:.0f}km': k for k in NEIGH for s in ('mean', 'std')}
    # textures of the unchanged grids (BA, TMI) are copied where the old grid has them
    grid_derived = set(DERIVED) | set(neigh_cols) | {c for c in glcm_cols() if c.split('_glcm_')[0] in DERIVED
                                                     or (c not in ocols and c not in ocols2)}
    need = [c for c in dict.fromkeys(meta['cols3'] + meta['cols2']) if c in grid_derived]
    missing = [c for c in dict.fromkeys(meta['cols3'] + meta['cols2'])
               if c not in grid_derived and c not in ocols and c not in ocols2]
    if missing:
        raise RuntimeError(f'columns not in the old S2 grid and not grid-derived: {missing}')
    ch_lat, ch_lon = [], []
    parents = old['child_parent']
    for pid in parents[::4]:
        for c in s2l.CellId(int(pid)).children():
            p = c.to_lat_lng()
            ch_lat.append(p.lat().degrees)
            ch_lon.append(p.lng().degrees)
    n = len(old['lat'])
    pts = pd.DataFrame({'lat': np.r_[old['lat'], ch_lat], 'lon': np.r_[old['lon'], ch_lon]})
    from paper_analyses.national import provinces_of
    prov = np.r_[old['province'], provinces_of(np.asarray(ch_lat), np.asarray(ch_lon))]
    glcm_need = [k for k in GLCM_RASTERS if any(c.startswith(f'{k}_glcm_') for c in need)]
    neigh_need = {k: v for k, v in NEIGH.items() if any(neigh_cols.get(c) == k for c in need)}
    print(f'{len(pts):,} points; recomputing {need} (GLCM on {glcm_need})', flush=True)
    new = grid_columns(pts, prov, glcm_need, neigh_need)

    def pick(c):  # a column absent from one old matrix is in the other (checked above)
        return ('X', ocols) if c in ocols else ('X2', ocols2)
    X = np.empty((len(pts), len(meta['cols3'])), np.float32)
    X2 = np.empty((len(pts), len(meta['cols2'])), np.float32)
    for M, cols in ((X, meta['cols3']), (X2, meta['cols2'])):
        for k, c in enumerate(cols):
            if c in grid_derived:
                M[:, k] = new[c].values
            else:
                key, names = pick(c)
                j = names.index(c)
                M[:, k] = np.r_[old[key][:, j], old[f'{key}_child'][:, j]]
    np.savez_compressed(nat_path('s2_l12_features.npz'), ids=old['ids'], lat=old['lat'], lon=old['lon'],
                        is_pos=old['is_pos'], d_occ_km=old['d_occ_km'], province=old['province'],
                        X=X[:n], X2=X2[:n], X_child=X[n:], X2_child=X2[n:], child_parent=parents,
                        cols=np.array(meta['cols3']), cols2=np.array(meta['cols2']))
    print(f'saved {nat_path("s2_l12_features.npz")} in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['national', 's2'])
    a = ap.parse_args()
    national() if a.part == 'national' else s2()
