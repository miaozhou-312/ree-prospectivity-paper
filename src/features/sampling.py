"""
Section 2 (helper functions) + Section 3a (positive/negative sample points)
+ Section 3f (negative-selection strategies).

`make_points` is the core sampling step: positives are REE deposits inside
the study-area outline (deduplicated, then dropped if outside the A∧B
sampleable domain -- see src/features/sampleable_domain.py); negative
candidates are grid points inside the outline that are far enough from any
deposit (BUFFER_KM) and inside the A∧B domain.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import BUFFER_KM, GRID_DEG, PROJECTED_CRS
from src.features.sampleable_domain import sampleable_lookup


# ── Section 2: generic helpers ──────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def normalize_proximity(dist_arr):
    """Return 1 - (d-dmin)/(dmax-dmin); closer = higher value."""
    dmin, dmax = dist_arr.min(), dist_arr.max()
    return 1.0 - (dist_arr - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(dist_arr)


# ── Section 3a: positive / negative sample points ───────────────────────────
def dedup_close_positives(dep_in, radius_km):
    """Collapse REE deposit points within radius_km of each other (chained,
    single-linkage) down to one representative per cluster -- the point
    closest to the cluster centroid. radius_km<=0/None disables this."""
    if not radius_km or radius_km <= 0 or len(dep_in) < 2:
        return dep_in

    proj = dep_in.to_crs(PROJECTED_CRS)
    coords = np.column_stack([proj.geometry.x.values, proj.geometry.y.values])
    cluster_ids = fcluster(linkage(coords, method='single', metric='euclidean'),
                            t=radius_km * 1000.0, criterion='distance')

    keep_idx = []
    for cid in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 1:
            keep_idx.append(idx[0])
            continue
        centroid = coords[idx].mean(axis=0)
        rep = idx[np.argmin(np.linalg.norm(coords[idx] - centroid, axis=1))]
        keep_idx.append(rep)
        dropped = [dep_in.iloc[i].get('Name', i) for i in idx if i != rep]
        print(f'    dedup @ {radius_km}km: kept {dep_in.iloc[rep].get("Name", rep)!r} '
              f'<- dropped {dropped}')

    keep_idx = sorted(keep_idx)
    print(f'  positive dedup @ {radius_km}km: {len(dep_in)} -> {len(keep_idx)} '
          f'(removed {len(dep_in) - len(keep_idx)})')
    return dep_in.iloc[keep_idx].reset_index(drop=True)


def make_points(cfg, rng, name, positive_dedup_radius_km, buffer_km=BUFFER_KM, grid_deg=GRID_DEG):
    """Positive = REE deposits within outline, deduplicated within
    positive_dedup_radius_km (see dedup_close_positives), THEN dropped if
    outside the A∧B sampleable domain -- a positive there has no guarantee
    of valid features, so keeping it would train on a point with
    degenerate/NaN feature values. Dropped positives still count toward the
    buffer_km exclusion zone below (dlat/dlon are captured from the
    undeduped-by-domain dep_in), so nearby candidates are never mislabeled
    as background just because the known deposit itself was excluded as a
    training example.

    Negative candidates = grid points within outline that are > buffer_km
    from any deposit AND inside the region's A∧B sampleable domain (see
    src/features/sampleable_domain.py: build_all() must have populated
    SAMPLEABLE[name] before this is called)."""
    boundary = gpd.read_file(cfg['outline']).to_crs('EPSG:4326')
    study_area = boundary.unary_union

    dep = pd.read_csv(cfg['dep']).dropna(subset=['Latitude', 'Longitude'])
    dep_gdf = gpd.GeoDataFrame(dep,
        geometry=gpd.points_from_xy(dep['Longitude'], dep['Latitude']),
        crs='EPSG:4326')
    dep_in = dep_gdf[dep_gdf.within(study_area)].reset_index(drop=True)
    if len(dep_in) == 0:
        raise ValueError('No REE deposits found within the study area outline.')
    dep_in = dedup_close_positives(dep_in, positive_dedup_radius_km)

    pos = pd.DataFrame({'lat': dep_in.geometry.y, 'lon': dep_in.geometry.x, 'label': 1})
    dlat = dep_in.geometry.y.values
    dlon = dep_in.geometry.x.values

    # ── Positive sampleable check: DROP any positive outside A∧B ───────────
    pos_in_dom = sampleable_lookup(pos['lat'].values, pos['lon'].values, name)
    if not pos_in_dom.all():
        bad_idx = np.where(~pos_in_dom)[0]
        dep_names = dep_in['Name'] if 'Name' in dep_in.columns else pd.Series(['Unknown'] * len(dep_in))
        print(f'  WARNING: dropping {len(bad_idx)} positive(s) that fall OUTSIDE the A∧B sampleable domain:')
        for i in bad_idx:
            print(f"    {dep_names.iloc[i]!r} at (lat={pos['lat'].iloc[i]:.4f}, lon={pos['lon'].iloc[i]:.4f})")
        pos = pos[pos_in_dom].reset_index(drop=True)
    else:
        print(f'  All {len(pos)} positives confirmed inside the A∧B sampleable domain.')

    # Generate regular grid inside boundary
    xmin, ymin, xmax, ymax = boundary.total_bounds
    xs = np.arange(xmin, xmax, grid_deg)
    ys = np.arange(ymin, ymax, grid_deg)
    xx, yy = np.meshgrid(xs, ys)
    grid_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs='EPSG:4326')
    grid_in = grid_gdf[grid_gdf.within(study_area)].copy()
    grid_in['lat'] = grid_in.geometry.y
    grid_in['lon'] = grid_in.geometry.x

    # Filter: must be > buffer_km from any deposit
    glat = grid_in['lat'].values
    glon = grid_in['lon'].values
    min_dep_dist = np.array([
        haversine_km(glat[i], glon[i], dlat, dlon).min()
        for i in range(len(glat))
    ])
    grid_ok = grid_in[min_dep_dist > buffer_km][['lat', 'lon']].copy()

    # ── candidates must ALSO fall inside the A∧B sampleable domain ─────────
    n_before_domain = len(grid_ok)
    cand_in_dom = sampleable_lookup(grid_ok['lat'].values, grid_ok['lon'].values, name)
    candidates = grid_ok[cand_in_dom].copy()
    candidates['label'] = -1  # -1 = unselected candidate
    print(f'  Candidate grid: {n_before_domain} (>BUFFER_KM from any deposit) -> '
          f'{len(candidates)} after A∧B sampleable-domain filter '
          f'({len(candidates) / n_before_domain * 100:.1f}% kept)')

    pts = pd.concat([pos[['lat', 'lon', 'label']], candidates], ignore_index=True)
    print(f'  Positive: {len(pos)}  |  Candidates: {len(candidates)}')
    return pts


# ── Section 3f: negative-selection strategies ───────────────────────────────
def select_negatives_cosine(pts, feat_cols, n_neg):
    """From candidates (label=-1), select the n_neg points whose feature
    vector is most cosine-dissimilar from the positive sample centroid.
    Falls back to random if too few candidates. (Superseded by
    select_negatives_random for the current shared-background design --
    kept for reference/reuse.)"""
    pos_mask  = (pts['label'] == 1).values
    cand_mask = (pts['label'] == -1).values

    X_raw = pts[feat_cols].astype(float).values
    X_imp = SimpleImputer(strategy='median').fit_transform(X_raw)
    X_std = StandardScaler().fit_transform(X_imp)

    pos_vec  = X_std[pos_mask].mean(axis=0)
    pos_norm = pos_vec / (np.linalg.norm(pos_vec) + 1e-9)

    cand_X   = X_std[cand_mask]
    norms    = np.linalg.norm(cand_X, axis=1, keepdims=True) + 1e-9
    cos_sim  = (cand_X / norms) @ pos_norm   # shape (n_cands,)

    cand_idx = np.where(cand_mask)[0]
    n_select = min(n_neg, len(cand_idx))
    top      = np.argsort(cos_sim)[:n_select]   # ascending -> most dissimilar first
    selected = cand_idx[top]

    keep = np.concatenate([np.where(pos_mask)[0], selected])
    pts_out = pts.iloc[keep].copy().reset_index(drop=True)
    pts_out.loc[pts_out['label'] == -1, 'label'] = 0
    print(f'  cosine selection: kept {n_select} / {len(cand_idx)} candidates')
    return pts_out


def select_negatives_mixed(pts, feat_cols, n_neg, cosine_frac=0.5, seed=42):
    """cosine_frac of n_neg from most cosine-dissimilar candidates;
    remainder drawn uniformly at random. (Kept for reference/reuse.)"""
    pos_mask  = (pts['label'] == 1).values
    cand_mask = (pts['label'] == -1).values
    cand_idx  = np.where(cand_mask)[0]
    X_raw = pts[feat_cols].astype(float).values
    X_imp = SimpleImputer(strategy='median').fit_transform(X_raw)
    X_std = StandardScaler().fit_transform(X_imp)
    pos_vec  = X_std[pos_mask].mean(axis=0)
    pos_norm = pos_vec / (np.linalg.norm(pos_vec) + 1e-9)
    cand_X   = X_std[cand_mask]
    norms    = np.linalg.norm(cand_X, axis=1, keepdims=True) + 1e-9
    cos_sim  = (cand_X / norms) @ pos_norm
    n_cos    = int(n_neg * cosine_frac)
    n_rand   = n_neg - n_cos
    cos_sel  = cand_idx[np.argsort(cos_sim)[:n_cos]]
    remain   = np.setdiff1d(cand_idx, cos_sel)
    rand_sel = np.random.RandomState(seed).choice(
                   remain, size=min(n_rand, len(remain)), replace=False)
    keep = np.concatenate([np.where(pos_mask)[0], cos_sel, rand_sel])
    out  = pts.iloc[keep].copy().reset_index(drop=True)
    out.loc[out['label'] == -1, 'label'] = 0
    print(f'  mixed selection: {n_cos} cosine + {len(rand_sel)} random / {len(cand_idx)} candidates')
    return out


def select_negatives_random(pts, n_neg, seed=42):
    """Uniform random selection from buffer-outside candidates. This is the
    negative-selection strategy the current shared-background design
    actually uses (see src/features/build_region.py)."""
    pos_mask  = (pts['label'] == 1).values
    cand_mask = (pts['label'] == -1).values
    cand_idx  = np.where(cand_mask)[0]
    selected  = np.random.RandomState(seed).choice(
                    cand_idx, size=min(n_neg, len(cand_idx)), replace=False)
    keep = np.concatenate([np.where(pos_mask)[0], selected])
    out  = pts.iloc[keep].copy().reset_index(drop=True)
    out.loc[out['label'] == -1, 'label'] = 0
    print(f'  random selection: kept {len(selected)} / {len(cand_idx)} candidates')
    return out
