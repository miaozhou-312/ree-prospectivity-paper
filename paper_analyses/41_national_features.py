"""
Stage 41 -- national feature table (feature set 3) built with the regional
rules, so that the national and regional analyses differ only in scale.

Points
- Positives: the Stage-40 national label table, dropped if outside the
  national A∧B domain (the regional rule).
- Background: the 0.1 degree lattice inside the national outline, more than
  BUFFER_KM from every national occurrence and inside the A∧B domain, as in
  make_points. The regional pools hold P_KEEP = 8000 / (Superior lattice
  candidates) of their lattice, more in Cordillera and Grenville (smaller
  candidate sets); the national table uses Superior's share everywhere:
    * Superior: its regional pools as they are;
    * Cordillera, Grenville: random subsets of their regional pools of size
      P_KEEP x candidates, split 3 : 5 between evaluation and training as
      in build_region;
    * elsewhere: new pools of P_KEEP x candidates, evaluation drawn first.
  Every national background point inside a study province is therefore
  also a regional one, in the same pool. National points must also lie on
  the H3 geology map (H3_IN_KM): the set-3 geology columns are undefined
  off the map, and nationally the province polygons take in Hudson Bay and
  Arctic channels. Points of the regional pools that fail the national
  filters stay in the file (`in_national` False) for Stage 44.
- Features are computed for every point (national table and all regional
  pool points), then: set-1 pruning on the national table, terrane columns
  removed, Stage-24 and Stage-26 columns added (H3 classes with >= 10
  national training-background points), re-pruned at |rho| > 0.7 =
  feature set 3; without the Stage-26 columns = "without point-level map
  classes".

Output ../work/national/national_table.pkl (Stages 42-45 read it).

    python -m paper_analyses.41_national_features
"""
import importlib
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from paper_analyses.common import REGION_ORDER, config, load_region, save_pickle, seed_everything
from paper_analyses.national import (EXT_COLS, H3_IN_KM, LABELS, OUTLINE, STUDY, TERRANE, Extractor,
                                     encode_geology, nat_path, province_matrix, provinces_of, regional_rows,
                                     regional_spec)
from src.features.collinearity import prune_collinear
from src.features.sampling import haversine_km

EVAL_SHARE = config.N_EVAL_BG / (config.N_EVAL_BG + config.N_TRAIN_BG)


def min_dist_km(lat, lon, dlat, dlon, chunk=20000):
    out = np.empty(len(lat))
    for s in range(0, len(lat), chunk):
        a, b = lat[s:s + chunk, None], lon[s:s + chunk, None]
        out[s:s + chunk] = haversine_km(a, b, dlat[None, :], dlon[None, :]).min(axis=1)
    return out


def lattice(outline):
    x0, y0, x1, y1 = gpd.GeoSeries([outline], crs='EPSG:4326').total_bounds
    xx, yy = np.meshgrid(np.arange(x0, x1, config.GRID_DEG), np.arange(y0, y1, config.GRID_DEG))
    xx, yy = xx.ravel(), yy.ravel()
    shapely.prepare(outline)
    m = shapely.contains_xy(outline, xx, yy)
    return pd.DataFrame({'lat': yy[m], 'lon': xx[m]})


def draw(idx_eval, idx_train, n, rng):
    """n points split 3 : 5, from the given evaluation and training pools."""
    n_eval = int(round(n * EVAL_SHARE))
    return (rng.choice(idx_eval, min(n_eval, len(idx_eval)), replace=False),
            rng.choice(idx_train, min(n - n_eval, len(idx_train)), replace=False))


def check_regional(pts):
    """Inside each study province the national table must give the regional
    feature-set-3 values (up to the affine min-max scaling of proximity columns)."""
    from scipy.stats import spearmanr
    s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
    for region in REGION_ORDER:
        rec, cols, XA_lab, XA_unl, _ = s25.extended_matrices(region, 'reencoded')
        lab, unl = regional_rows({'pts': pts}, region)
        bad = []
        for rows, XA in [(rec['unl'], XA_unl), (rec['lab'], XA_lab)]:
            df = lab if rows is rec['lab'] else unl
            key = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_arrays([df.lat.round(5), df.lon.round(5)]))
            j = key.reindex(pd.MultiIndex.from_arrays([rows.lat.round(5), rows.lon.round(5)])).values.astype(int)
            XN = province_matrix(df.iloc[j], regional_spec(region))
            for k, c in enumerate(cols):
                a, b = XA[:, k], XN[:, k]
                if c.endswith('_proximity'):
                    m = np.isfinite(a) & np.isfinite(b)
                    ok = spearmanr(a[m], b[m]).correlation > 0.99999
                else:
                    ok = np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-9)
                if not ok:
                    bad.append(c)
        print(f'[{region}] national values vs regional feature set 3: '
              + ('identical' if not bad else f'differ in {sorted(set(bad))}'), flush=True)


def main():
    t0 = time.time()
    seed_everything()
    ex = Extractor()
    rng = np.random.default_rng(config.RANDOM_SEED)
    lab = pd.read_csv(LABELS)
    dlat, dlon = lab.Latitude.values, lab.Longitude.values

    # -- lattice candidates ------------------------------------------------------
    outline = gpd.read_file(OUTLINE).geometry.iloc[0]
    grid = lattice(outline)
    grid['province'] = provinces_of(grid.lat.values, grid.lon.values)
    n0 = len(grid)
    grid = grid[min_dist_km(grid.lat.values, grid.lon.values, dlat, dlon) > config.BUFFER_KM]
    grid = grid[ex.in_domain(grid.lat.values, grid.lon.values)].reset_index(drop=True)
    grid['on_map'] = ex.h3_distance(grid.lat.values, grid.lon.values) <= H3_IN_KM
    print(f'lattice: {n0:,} nodes in Canada -> {len(grid):,} candidates (>{config.BUFFER_KM:.0f} km, A∧B); '
          f'{grid.on_map.mean():.1%} on the H3 geology map', flush=True)
    cand = grid.groupby('province').agg(n=('lat', 'size'), on_map=('on_map', 'sum'))
    p_keep = 8000.0 / cand.loc['Superior', 'n']
    print(cand.assign(share_on_map=(cand.on_map / cand.n).round(3)).to_string())
    print(f'P_KEEP = 8000 / {cand.loc["Superior", "n"]} Superior candidates = {p_keep:.4f}', flush=True)

    # -- regional pools and positives -----------------------------------------------
    rows = []
    for region in REGION_ORDER:
        rec = load_region(region)
        pts = rec['pts']
        rows.append(pd.DataFrame({'lat': pts.lat.values, 'lon': pts.lon.values, 'label': pts.label.values,
                                  'pool': pts.pool.values, 'source': region}))
    reg = pd.concat(rows, ignore_index=True)
    reg = reg[reg.pool != 'positive'].reset_index(drop=True)
    pos = pd.DataFrame({'lat': dlat, 'lon': dlon, 'label': 1, 'pool': 'positive', 'source': 'national',
                        'name': lab.Name.values, 'province': lab.province.values})
    # inside the study provinces the regional caches decide (their A∧B grids differ from the national one
    # at coverage edges); elsewhere the national A∧B domain does
    cached = set()
    for region in REGION_ORDER:
        rp = load_region(region)['pts']
        cached |= set(zip(rp.lat[rp.pool == 'positive'].round(5), rp.lon[rp.pool == 'positive'].round(5)))
    in_cache = np.array([k in cached for k in zip(pos.lat.round(5), pos.lon.round(5))])
    dom = ex.in_domain(pos.lat.values, pos.lon.values)
    keep = np.where(pos.province.isin(STUDY), in_cache, dom)
    for _, r in pos[~keep].iterrows():
        why = 'not in the regional cache' if r.province in STUDY else 'outside the national A∧B domain'
        print(f'  positive dropped ({why}): {r["name"]!r} ({r.province})')
    pos = pos[keep].reset_index(drop=True)

    reg['province'] = reg.source
    ok_dom = ex.in_domain(reg.lat.values, reg.lon.values)
    ok_buf = min_dist_km(reg.lat.values, reg.lon.values, dlat, dlon) > config.BUFFER_KM
    ok_map = ex.h3_distance(reg.lat.values, reg.lon.values) <= H3_IN_KM
    reg['eligible'] = ok_dom & ok_buf & ok_map
    print('regional pool points failing a national filter (A∧B / 10 km of a national occurrence / off the map):')
    for region in REGION_ORDER:
        m = (reg.source == region).values
        print(f'  [{region}] {int((~ok_dom[m]).sum())} / {int((~ok_buf[m]).sum())} / {int((~ok_map[m]).sum())}')

    reg['in_national'] = False
    for region in REGION_ORDER:
        m = (reg.source == region) & reg.eligible
        ie = np.where(m & (reg.pool == 'eval_neg'))[0]
        it = np.where(m & (reg.pool == 'train_unlabeled'))[0]
        if region == 'Superior':
            keep = np.r_[ie, it]
        else:
            n = int(round(p_keep * cand.loc[region, 'n']))
            keep = np.concatenate(draw(ie, it, n, rng))
        reg.loc[keep, 'in_national'] = True
        print(f'  [{region}] national subset: {int(reg.loc[keep].pool.eq("eval_neg").sum())} evaluation + '
              f'{int(reg.loc[keep].pool.eq("train_unlabeled").sum())} training (of {int(m.sum())} eligible)')

    rest = grid[~grid.province.isin(STUDY) & grid.on_map].reset_index(drop=True)
    n_rest = int(round(p_keep * len(rest)))
    n_eval = int(round(n_rest * EVAL_SHARE))
    ie = rng.choice(len(rest), n_eval, replace=False)
    it = rng.choice(np.setdiff1d(np.arange(len(rest)), ie), n_rest - n_eval, replace=False)
    new = pd.concat([rest.iloc[ie].assign(pool='eval_neg'), rest.iloc[it].assign(pool='train_unlabeled')],
                    ignore_index=True)
    new = new.assign(label=0, source='national', in_national=True, eligible=True).drop(columns='on_map')
    print(f'  [elsewhere] {n_eval} evaluation + {n_rest - n_eval} training of {len(rest):,} candidates', flush=True)

    pos['in_national'] = True
    pos['eligible'] = True
    pts = pd.concat([pos, reg, new], ignore_index=True)
    print(f'feature extraction for {len(pts):,} points ({int(pts.in_national.sum()):,} in the national table)',
          flush=True)
    sample = pts.sample(300, random_state=0)
    print('  line distances, tree vs src union (max abs diff, m):', ex.check_lines(sample.lat.values,
                                                                                 sample.lon.values), flush=True)
    raw = ex.raw(pts, pts.province.values)
    feats = raw.drop(columns=['lat', 'lon'])
    raster_cols = list(ex.rasters)
    has_raster = feats[raster_cols].notna().any(axis=1).values
    print(f'  points with no raster value (dropped from the national table): {int((~has_raster).sum())}')
    pts['in_national'] &= has_raster
    pts = pd.concat([pts.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    # -- feature sets ----------------------------------------------------------
    nat = pts[pts.in_national].reset_index(drop=True)
    set1_cols = [c for c in raw.columns if c not in ('lat', 'lon') and not c.endswith('_dist_m')
                 and not c.startswith('h3_') and c not in EXT_COLS]
    set1_pruned = prune_collinear(nat, set1_cols, thresh=config.COLLINEARITY_THRESH)
    base = [c for c in set1_pruned if c not in TERRANE]
    unl = nat[nat.pool == 'train_unlabeled']
    lc = unl.h3_LithMajor.fillna('Unknown').value_counts()
    ec = unl.h3_EraMaxMjr.fillna('Unknown').value_counts()
    lith_classes = sorted(lc[lc >= 10].index)
    era_classes = sorted(ec[ec >= 10].index)
    geo = encode_geology(pts, lith_classes, era_classes)
    pts = pd.concat([pts, geo], axis=1)
    nat = pts[pts.in_national].reset_index(drop=True)
    print(f'H3 classes kept nationally: {len(lith_classes)} lithology (of {len(lc)}), {len(era_classes)} era '
          f'(of {len(ec)})')
    cols2 = prune_collinear(nat, base + EXT_COLS, thresh=config.COLLINEARITY_THRESH)
    cols3 = prune_collinear(nat, base + EXT_COLS + list(geo.columns), thresh=config.COLLINEARITY_THRESH)
    const = [c for c in cols3 if nat[c].nunique() < 2]
    cols2, cols3 = [c for c in cols2 if c not in const], [c for c in cols3 if c not in const]
    print(f'constant over the national table, removed: {const}')
    print(f'set 1: {len(set1_cols)} -> {len(set1_pruned)}; without map classes: {len(cols2)}; '
          f'feature set 3: {len(cols3)} features')
    print('  set 3:', cols3)
    y = nat.label.values
    print(f'national table: {int(y.sum())} positives, {int((nat.pool == "eval_neg").sum())} evaluation and '
          f'{int((nat.pool == "train_unlabeled").sum())} training background points')
    print(nat.groupby(['province', 'pool']).size().unstack(fill_value=0).to_string())
    check_regional(pts)
    save_pickle({'pts': pts, 'set1_cols': set1_cols, 'set1_pruned': set1_pruned, 'cols2': cols2, 'cols3': cols3,
                 'lith_classes': lith_classes, 'era_classes': era_classes, 'p_keep': p_keep,
                 'candidates': cand, 'raster_cols': raster_cols},
                nat_path('national_table.pkl'))
    print(f'Done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
