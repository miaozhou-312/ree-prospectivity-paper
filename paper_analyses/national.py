"""
National (Canada) scale, shared by Stages 40-45: paths, and a feature
extractor that computes feature set 3 at arbitrary points with the same
functions and rules as the regional tables.

What is identical to the regional pipeline: the src/ raster sampling, GLCM
texture (buffer-then-clip rasters, 2nd-98th percentile quantisation over
the clipped raster), worm proximity and A∧B sampleable domain, all run on a
'Canada' region whose outline is the national one (Stage 40); the Stage-24
generalised-geology distances and 10 km neighbourhood statistics; the
Stage-26 raw H3 lithology / era / contact encoding.

What differs, and why:
- H3 attributes are looked up at the nearest H3 cell centre (Stage 26 does
  the same) instead of a point-in-polygon join against an H3 layer clipped
  to the outline. On a hexagonal grid the nearest centre is the containing
  cell; clipping the 1.8 GB polygon layer to the whole country is what the
  lookup avoids. A point more than H3_IN_KM from every centre lies outside
  the geology map (the set-1 dummies are then NaN, as the join would give).
- Distances to line layers come from an STR-tree nearest query over the
  layer's parts instead of a distance to their union, point by point; the
  two are the same number (checked in check_lines()) and the tree is fast
  enough for the S2 grid of Stage 43.
- Scale-dependent features are computed province by province, so that a
  point's value does not depend on whether Canada or one province is being
  analysed (Stage 48 showed that computing them over the whole country
  costs Superior's DevNet a third of its AP): rasters are sampled from, and
  GLCM grey levels quantised over, the point's geological province (its
  buffer-then-clip rasters, as regionally; near a province edge a coarse
  pixel whose centre falls outside the 8 km buffer is nodata, as
  regionally), and the generalised geology map is built from the H3 cells
  in that province's bounding box plus the regional margin. For the three
  study provinces these are the regional rasters and boxes, so their points
  take exactly the regional values (checked in Stage 41).
"""
import contextlib
import importlib
import io
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from scipy.spatial import cKDTree

from paper_analyses.common import GLCM_RASTERS, WORK_DIR, config, load_pickle, out_path, setup_regions
from src.features import raster_preprocessing, sampleable_domain
from src.features.glcm import GLCM_PROPS, add_glcm_features
from src.features.proximity import add_proximity_lines, add_proximity_worms, add_rasters, build_worm_trees
from src.features.sampling import normalize_proximity
from src.features.terrane import ERA_DUMMY_GROUPS, LITH_DUMMY_GROUPS, LITH_GROUP_MAP

_s24 = importlib.import_module('paper_analyses.24_extended_features')
_s26 = importlib.import_module('paper_analyses.26_geology_reencode')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAT_DATA = os.path.join(REPO, 'data', 'national')
OUTLINE = os.path.join(NAT_DATA, 'Canada.geojson')
PROVINCES = os.path.join(NAT_DATA, 'provinces.geojson')
LABELS = os.path.join(NAT_DATA, 'Canada_REE_deposits_deduped_2.0km.csv')
NAT_DIR = os.path.join(WORK_DIR, 'national')
NAME = 'Canada'
STUDY = ['Superior', 'Cordillera', 'Grenville']
H3_IN_KM = 1.5          # H3 res-7 circumradius is ~1.2 km
H3_COLS = ['LongWGS84', 'LatWGS84', 'LithMajor', 'EraMaxMjr', 'LithCntact', 'PrdContact']
TERRANE = [f'geo_era_{e}' for e in ERA_DUMMY_GROUPS] + [f'lith_{g}' for g in LITH_DUMMY_GROUPS]


def nat_path(*parts):
    p = os.path.join(NAT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def provinces_of(lat, lon):
    """Province of each point; points on no polygon (edge rounding) take the nearest one."""
    prov = gpd.read_file(PROVINCES).to_crs(config.PROJECTED_CRS)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs='EPSG:4326').to_crs(config.PROJECTED_CRS)
    j = gpd.sjoin(pts, prov, how='left', predicate='within')
    out = j.loc[~j.index.duplicated(), 'province'].reindex(pts.index)
    miss = out.isna().values
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss], prov, how='left')
        out[miss] = near.loc[~near.index.duplicated(), 'province'].values
    return out.values


def province_extent(name):
    """(outline file, clip-cache name) of a province: a study province's
    regional outline and raster cache, else its polygon in provinces.geojson."""
    if name in STUDY:
        return os.path.join(REPO, 'data', 'outlines', f'{name}.geojson'), name
    slug = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    path = nat_path('provinces', f'{slug}.geojson')
    if not os.path.exists(path):
        prov = gpd.read_file(PROVINCES)
        prov[prov.province == name][['geometry']].to_file(path, driver='GeoJSON')
    return path, f'prov_{slug}'


class Extractor:
    def __init__(self):
        regions = setup_regions()
        cfg = dict(next(iter(regions.values())))
        cfg.update(outline=OUTLINE, dep=LABELS)
        self.cfg = cfg
        print('[national] buffer-then-clip rasters to the national outline ...', flush=True)
        self.rasters = raster_preprocessing.preprocess_rasters(NAME, cfg)
        sampleable_domain.build_all({NAME: cfg}, {NAME: self.rasters}, config.PROJECTED_CRS, GLCM_RASTERS)
        self.worms = build_worm_trees(cfg['points'], config.PROJECTED_CRS)
        self.lines = {}
        for k, p in cfg['vectors'].items():
            parts = gpd.read_file(p).to_crs(config.PROJECTED_CRS).geometry.explode(index_parts=False)
            parts = parts[~parts.is_empty].values
            self.lines[k] = (shapely.STRtree(parts), parts)
        print('[national] reading the H3 geology grid (attributes only) ...', flush=True)
        self.cells = pyogrio.read_dataframe(_s26.H3, columns=H3_COLS, read_geometry=False).reset_index(drop=True)
        self.tree = cKDTree(_s24.xy_km(self.cells.LongWGS84.values, self.cells.LatWGS84.values))
        print(f'[national] {len(self.cells):,} H3 cells', flush=True)

    # -- domain -------------------------------------------------------------
    def in_domain(self, lat, lon):
        return sampleable_domain.sampleable_lookup(np.asarray(lat), np.asarray(lon), NAME)

    def h3_distance(self, lat, lon):
        return self.tree.query(_s24.xy_km(lon, lat))[0]

    # -- features -----------------------------------------------------------
    def line_distances(self, lat, lon):
        tr = __import__('pyproj').Transformer.from_crs('EPSG:4326', config.PROJECTED_CRS, always_xy=True)
        x, y = tr.transform(np.asarray(lon), np.asarray(lat))
        pts = shapely.points(x, y)
        out = {}
        for k, (tree, _) in self.lines.items():
            idx, d = tree.query_nearest(pts, return_distance=True, all_matches=False)
            dist = np.full(len(pts), np.nan)
            dist[idx[0]] = d
            out[k] = dist
        return out

    def check_lines(self, lat, lon):
        """Largest difference (m) between the tree distances and the src/
        union distances on the given points."""
        pts = pd.DataFrame({'lat': lat, 'lon': lon})
        ref = pts.copy()
        for k, p in self.cfg['vectors'].items():
            ref = add_proximity_lines(ref, p, k, config.PROJECTED_CRS)
        fast = self.line_distances(lat, lon)
        return {k: float(np.nanmax(np.abs(ref[f'{k}_dist_m'].values - fast[k]))) for k in fast}

    def terrane(self, pts):
        d, i = self.tree.query(_s24.xy_km(pts.lon.values, pts.lat.values))
        att = self.cells.iloc[i].reset_index(drop=True)
        inside = d <= H3_IN_KM
        era = att.EraMaxMjr.where(att.EraMaxMjr.isin(ERA_DUMMY_GROUPS) & inside).values
        for e in ERA_DUMMY_GROUPS:
            pts[f'geo_era_{e}'] = np.where(pd.isna(era), np.nan, (era == e).astype(float))
        grp = att.LithMajor.replace({'NA': None, '': None}).map(LITH_GROUP_MAP).where(inside).values
        for g in LITH_DUMMY_GROUPS:
            pts[f'lith_{g}'] = np.where(pd.isna(grp), np.nan, (grp == g).astype(float))
        return att

    def rasters_by_province(self, pts, province):
        cols = list(self.rasters)
        out = np.full((len(pts), len(cols)), np.nan)
        for name in pd.unique(province):
            m = province == name
            outline, clip = province_extent(name)
            rasters = raster_preprocessing.preprocess_rasters(clip, dict(self.cfg, outline=outline))
            out[m] = add_rasters(pts.loc[m, ['lat', 'lon']].copy(), rasters)[cols].values
        return pd.DataFrame(out, columns=cols)

    def glcm_by_province(self, pts, province):
        cols = [f'{k}_glcm_{p}' for k in GLCM_RASTERS for p in GLCM_PROPS]
        out = np.full((len(pts), len(cols)), np.nan)
        for name in pd.unique(province):
            m = province == name
            outline, clip = province_extent(name)
            cfg = dict(self.cfg, outline=outline, raster={k: self.cfg['raster'][k] for k in GLCM_RASTERS})
            rasters = raster_preprocessing.preprocess_rasters(clip, cfg)
            sub = add_glcm_features(pts.loc[m, ['lat', 'lon']].copy(), rasters, GLCM_RASTERS)
            out[m] = sub[cols].values
            print(f'    GLCM over {name}: {int(m.sum()):,} points', flush=True)
        return pd.DataFrame(out, columns=cols)

    def geology_by_province(self, pts, province):
        out = None
        for name in pd.unique(province):
            m = province == name
            x0, y0, x1, y1 = gpd.read_file(province_extent(name)[0]).to_crs('EPSG:4326').total_bounds
            c = self.cells    # the Stage-24 selection (h3_cells), in file order
            sel = (c.LatWGS84 > y0 - 2) & (c.LatWGS84 < y1 + 2) & (c.LongWGS84 > x0 - 3) & (c.LongWGS84 < x1 + 3)
            df = c.loc[sel, ['LongWGS84', 'LatWGS84', 'LithMajor', 'EraMaxMjr']].reset_index(drop=True)
            feats, _ = _s24.geology_distances(df, (pts.lon.values[m], pts.lat.values[m]))
            if out is None:
                out = pd.DataFrame(np.nan, index=range(len(pts)), columns=list(feats))
            out.loc[m, list(feats)] = pd.DataFrame(feats).values
        return out

    def raw(self, pts, province):
        """Set-1 columns (in the order build_region writes them), the Stage-24
        columns, and the raw H3 attributes for Stage-26 encoding, at points
        lying in the given provinces. Proximity columns are min-max scaled
        over `pts`, as in build_region (an affine map, which no model here is
        sensitive to)."""
        province = np.asarray(province)
        pts = pts[['lat', 'lon']].reset_index(drop=True).copy()
        print(f'  [{len(pts):,} points] H3 terrane ...', flush=True)
        att = self.terrane(pts)
        print('  line proximity ...', flush=True)
        for k, d in self.line_distances(pts.lat.values, pts.lon.values).items():
            pts[f'{k}_dist_m'] = d
            pts[f'{k}_proximity'] = normalize_proximity(d)
        print('  worm proximity, rasters, GLCM ...', flush=True)
        pts = add_proximity_worms(pts, self.worms, config.PROJECTED_CRS)
        pts = pd.concat([pts, self.rasters_by_province(pts, province), self.glcm_by_province(pts, province)], axis=1)
        print('  generalised-geology distances (by province), 10 km neighbourhood statistics ...', flush=True)
        pts = pd.concat([pts, self.geology_by_province(pts, province)], axis=1)
        for k, path in _s24.NEIGHBOURHOOD_RASTERS.items():
            m, s = _s24.neighbourhood(_s24.raster_file(path), (pts.lon.values, pts.lat.values),
                                      _s24.RADIUS_KM)
            pts[f'{k}_mean{_s24.RADIUS_KM:.0f}km'], pts[f'{k}_std{_s24.RADIUS_KM:.0f}km'] = m, s
        att = att[['LithMajor', 'EraMaxMjr', 'LithCntact', 'PrdContact']].add_prefix('h3_')
        return pd.concat([pts, att], axis=1)


def encode_geology(raw, lith_classes, era_classes):
    att = raw[['h3_LithMajor', 'h3_EraMaxMjr', 'h3_LithCntact', 'h3_PrdContact']]
    att.columns = ['LithMajor', 'EraMaxMjr', 'LithCntact', 'PrdContact']
    return _s26.encode(att.reset_index(drop=True), lith_classes, era_classes)


def regional_spec(region):
    """A study province's own feature-set-3 columns and H3 class lists
    (Stages 25-26), for province-level models."""
    s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
    with contextlib.redirect_stdout(io.StringIO()):
        _, cols, _, _, _ = s25.extended_matrices(region, 'reencoded')
    geo = load_pickle(out_path(f'geology_raw_{region}.pkl'))['unl'].columns
    lith = [c[5:] for c in geo if c.startswith('lith_') and c not in ('lith_other', 'lith_contact')]
    era = [c[4:] for c in geo if c.startswith('era_')]
    return cols, lith, era


def province_matrix(df, spec):
    """Rows of the national table in a province's own columns (H3 classes
    re-encoded with that province's class lists)."""
    cols, lith, era = spec
    geo = encode_geology(df, lith, era)
    base = df.reset_index(drop=True).drop(columns=[c for c in df.columns if c in geo.columns])
    return pd.concat([base, geo], axis=1)[cols].values.astype(float)


EXT_COLS = (['dist_regional_contact', 'dist_regional_age_contact'] + [f'dist_{k}' for k in _s24.UNITS]
            + [f'{k}_{s}{_s24.RADIUS_KM:.0f}km' for k in _s24.NEIGHBOURHOOD_RASTERS for s in ('mean', 'std')])


def load_national(featureset='set3', table='national_table.pkl'):
    """The Stage-41 table as matrices. 'nat' are the national rows (the
    national table), 'all' every row (with the full regional pools)."""
    from paper_analyses.common import load_pickle
    from paper_analyses.protocol import project_xy
    d = load_pickle(nat_path(table))
    pts = d['pts']
    cols = d['cols3'] if featureset == 'set3' else d['cols2']
    pts['group'] = np.where(pts.province.isin(STUDY), pts.province, 'Other')
    pts['xy_x'], pts['xy_y'] = (project_xy(pts.lat.values, pts.lon.values, config.PROJECTED_CRS) / 1000.0).T
    nat = pts[pts.in_national]
    lab = nat[nat.pool.isin(['positive', 'eval_neg'])].reset_index(drop=True)
    unl = nat[nat.pool == 'train_unlabeled'].reset_index(drop=True)
    return {'pts': pts, 'cols': cols, 'lab': lab, 'unl': unl, 'y': lab.label.values.astype(int),
            'X_lab': lab[cols].values.astype(float), 'X_unl': unl[cols].values.astype(float),
            'xy_lab': lab[['xy_x', 'xy_y']].values, 'xy_unl': unl[['xy_x', 'xy_y']].values, 'meta': d}


def regional_rows(d, region):
    """A study province's labelled rows (its positives + full regional
    evaluation pool) and training pool, all with national features: the
    regional evaluation and training sets exactly, including the few pool
    points that fail a national filter."""
    pts = d['pts']
    m = pts.province.eq(region)
    lab = pts[m & pts.pool.isin(['positive', 'eval_neg'])].reset_index(drop=True)
    unl = pts[m & (pts.pool == 'train_unlabeled')].reset_index(drop=True)
    return lab, unl


def metrics(y, s, top=0.10):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ap = average_precision_score(y, s)
    thr = np.quantile(s, 1.0 - top)
    return {'auc': roc_auc_score(y, s), 'ap': ap, 'lift': ap / y.mean(), 'cap10': float((s[y == 1] >= thr).mean())}


def summarise(y, oofs, **meta):
    """Median and 95% range over repeats of AUC, AP, AP lift and the share
    of positives in the top 10% of scored points."""
    m = pd.DataFrame([metrics(y, o) for o in np.atleast_2d(oofs)])
    out = dict(meta, n_pos=int(y.sum()), n_bg=int((y == 0).sum()), prevalence=float(y.mean()))
    for c in m:
        out[f'{c}_med'] = float(m[c].median())
        out[f'{c}_lo'], out[f'{c}_hi'] = (float(v) for v in np.percentile(m[c], [2.5, 97.5]))
    return out
