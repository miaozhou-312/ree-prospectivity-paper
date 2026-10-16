"""
Section 3b (line-feature proximity: faults, passive margins), Section 3c
(point-feature proximity: geophysical "worms", via KDTree), and Section 3d
(raster sampling at point locations).
"""
import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from scipy.spatial import cKDTree

from src.features.sampling import normalize_proximity


# ── Section 3b: proximity to line features (faults, passive margins, ...) ──
def add_proximity_lines(pts, geojson_path, col_prefix, projected_crs):
    gdf  = gpd.read_file(geojson_path).to_crs(projected_crs)
    su   = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union
    g    = gpd.GeoDataFrame(pts.copy(),
               geometry=gpd.points_from_xy(pts['lon'], pts['lat']),
               crs='EPSG:4326').to_crs(projected_crs)
    dist = g.geometry.apply(lambda p: p.distance(su)).values.astype(float)
    pts[f'{col_prefix}_dist_m']    = dist
    pts[f'{col_prefix}_proximity'] = normalize_proximity(dist)
    return pts


def add_proximity_line_layers(pts, vector_paths, projected_crs):
    """Apply add_proximity_lines once per discovered line-vector layer (see
    src/features/data_discovery.py) -- the key of each {key: path} entry
    becomes that layer's col_prefix. Replaces the old two hardcoded calls
    (one for 'fault', one for 'margin') with a loop over however many line
    layers a given run's vector folder actually turned up."""
    for key, path in vector_paths.items():
        pts = add_proximity_lines(pts, path, key, projected_crs)
    return pts


# ── Section 3c: proximity to point features (worms) using KDTree ───────────
def build_worm_trees(worm_paths, projected_crs):
    """Load each worm file once, project, and build a cKDTree. Returns dict of trees."""
    trees = {}
    for name, path in worm_paths.items():
        print(f'  loading worms: {name} ...')
        w = gpd.read_file(path).to_crs(projected_crs)
        coords = np.column_stack([w.geometry.x.values, w.geometry.y.values])
        trees[name] = cKDTree(coords)
        print(f'    {len(w):,} points  tree built')
    return trees


def add_proximity_worms(pts, worm_trees, projected_crs):
    g = gpd.GeoDataFrame(pts.copy(),
            geometry=gpd.points_from_xy(pts['lon'], pts['lat']),
            crs='EPSG:4326').to_crs(projected_crs)
    qcoords = np.column_stack([g.geometry.x.values, g.geometry.y.values])
    for name, tree in worm_trees.items():
        dist, _ = tree.query(qcoords, k=1)
        dist = dist.astype(float)
        pts[f'worm_{name}_dist_m']    = dist
        pts[f'worm_{name}_proximity'] = normalize_proximity(dist)
    return pts


# ── Section 3d: raster sampling ─────────────────────────────────────────────
def add_rasters(pts, raster_layers):
    for feat, path in raster_layers.items():
        with rasterio.open(path) as src:
            if src.crs and src.crs.to_epsg() != 4326:
                tr = pyproj.Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
                xs, ys = tr.transform(pts['lon'].values, pts['lat'].values)
            else:
                xs, ys = pts['lon'].values, pts['lat'].values
            vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
        pts[feat] = vals
    return pts
