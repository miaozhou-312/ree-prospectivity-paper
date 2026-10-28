"""
Stage 31 -- geochemical validation in all three provinces, with a
compositional (clr-PCA) carbonatite-affinity index.

Whole-rock geochemistry (the Gard et al., 2019 compilation clipped per region,
dataset/geochem/Geochemistry_<Region>_Clip.csv) is far too sparse to be a
model input, but it never enters training, so it can test the maps. Stage 30
did this for Superior with grid-node percentiles and La+Ce+Nd only; this
stage scores every sample location directly, covers all three provinces and
adds a multi-element index. Choices fixed before running:

  * samples: ore, vein and xenolith material excluded; samples sharing a
    location (coordinates rounded to 0.001 degree) are merged by the median
    of each element, so heavily sampled complexes do not dominate;
  * indicators
      LREE       La+Ce+Nd (ppm) and its percentile within rock type (types
                 with < 30 locations in the province pooled), as Stage 30;
      affinity   clr-PCA on ten elements (La, Ce, Nd, Sm, Yb, Y, Nb, Sr, Ba,
                 Zr; locations with all ten > 0), fitted on the pooled
                 locations of the three provinces. The affinity is the score
                 on the principal component, among the first three, that
                 best separates carbonatite/foid-named locations from all
                 others (largest |AUC - 0.5|), signed so that carbonatites
                 score high. Rock names choose the axis; the maps play no
                 part in it;
      rock name  carbonatite and silica-undersaturated alkaline rocks
                 (rock names matching CARBONATITE_PATTERN);
  * map value at each location: features extracted at the location exactly
    as for Clay-Howells (Stages 12 and 29) and scored by the production
    models of feature set 3 (DevNet, MLP-BCE, RF) and set 1 (DevNet, RF); the
    percentile is taken among the region's 8,000 background points, a random
    sample of the domain, so it is a province-wide percentile. Locations
    outside the sampleable domain are dropped;
  * evaluated locations: La, Ce and Nd measured (as Stage 30); the
    rock-name test and every sampling baseline use the same locations;
  * locations within 10 km (main) or 25 km (sensitivity) of a training
    occurrence are excluded;
  * sampling is clustered (a map's top 10% of area can hold far more than
    10% of the sampled locations), so every comparison conditions on where
    samples were taken: Spearman rho between map percentile and indicator
    (within-type percentile for LREE), with a 95% interval from a 100 km
    block bootstrap (1,000 resamples); "high" locations (top 5% of the
    indicator within the province; top 1% of LREE as a sensitivity) -- their
    share in the map's top 10% divided by the share of all evaluated
    locations there (enrichment; 1 = no more than sampling explains), with a
    block-bootstrap interval and a one-sided Mann-Whitney test of their map
    percentiles against all other locations; the same share and test for the
    carbonatite/foid-named locations, with a block-bootstrap interval for
    their enrichment (they cluster at a few complexes).

Added after the first run (post hoc, labelled as such in the outputs): the
affinity as a percentile within rock type, because the raw index turned out
to track the granitoid-versus-tholeiite contrast of the host rocks.
Map percentiles are cached in geochem3_locations.csv; --rescore recomputes
them (about 20 min).

Outputs (git-ignored out/): geochem3_locations.csv, geochem3_validation.csv,
geochem3_rocknames.csv, geochem3_pca.csv, geochem3_pca_scores.csv.

    nice -n 10 python -m paper_analyses.31_geochem_pca
"""
import importlib
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

from paper_analyses.common import (GLCM_RASTERS, REGION_ORDER, config, out_path, seed_everything,
                                   setup_regions)
from src.features import raster_preprocessing, sampleable_domain
from src.features.proximity import build_worm_trees

_p12 = importlib.import_module('paper_analyses.12_final_production')
_p29 = importlib.import_module('paper_analyses.29_production_fs3')
_s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
_s30 = importlib.import_module('paper_analyses.30_geochem_validation')

GEOCHEM_DIR = os.path.dirname(_s30.GEOCHEM)
FILE_TAG = {'Superior': 'Superior', 'Cordillera': 'Cordilleran', 'Grenville': 'Grenville'}
ELEMENTS = ['la', 'ce', 'nd', 'sm', 'yb', 'y', 'nb', 'sr', 'ba', 'zr']
MODELS = [('3', 'DevNet'), ('3', 'MLP-BCE'), ('3', 'RF'), ('1', 'DevNet'), ('1', 'RF')]
HIGH_Q = {'LREE': 0.95, 'LREE (top 1%)': 0.99, 'affinity': 0.95, 'affinity within type (post hoc)': 0.95}
COL = {'LREE': 'lree', 'LREE (top 1%)': 'lree', 'affinity': 'affinity', 'affinity within type (post hoc)': 'aff_in_type'}
IND = {'LREE': 'lree_pct_in_type', 'LREE (top 1%)': 'lree_pct_in_type', 'affinity': 'affinity',
       'affinity within type (post hoc)': 'aff_in_type'}
EXCLUSIONS = (10.0, 25.0)


def load_locations(region):
    d = pd.read_csv(os.path.join(GEOCHEM_DIR, f'Geochemistry_{FILE_TAG[region]}_Clip.csv'), low_memory=False)
    d = d[~d['material'].isin(['ore', 'vein', 'xenolith'])].copy()
    cols = [f'{e}_ppm' for e in ELEMENTS]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors='coerce')
        d.loc[d[c] <= 0, c] = np.nan
    d['lree'] = d.la_ppm + d.ce_ppm + d.nd_ppm            # NaN unless all three measured
    d['loc'] = d.latitude.round(3).astype(str) + '_' + d.longitude.round(3).astype(str)
    agg = {'lat': ('latitude', 'median'), 'lon': ('longitude', 'median'), 'lree': ('lree', 'median'),
           'n_samples': ('latitude', 'size'),
           'rock_type': ('rock_type', lambda s: s.mode().iloc[0] if s.notna().any() else 'unknown')}
    agg.update({c: (c, 'median') for c in cols})
    g = d.groupby('loc').agg(**agg).reset_index(drop=True)
    g['rock_type'] = g.rock_type.fillna('unknown')
    g['carb_named'] = g.rock_type.str.contains(_s30.CARBONATITE_PATTERN, case=False, na=False)
    has = g.lree.notna()
    counts = g.loc[has, 'rock_type'].value_counts()
    g['rock_group'] = np.where(g.rock_type.map(counts).fillna(0) >= 30, g.rock_type, 'other')
    g.loc[has, 'lree_pct_in_type'] = g[has].groupby('rock_group').lree.rank(pct=True)
    g['region'] = region
    return g


def clr_pca(loc):
    cols = [f'{e}_ppm' for e in ELEMENTS]
    ok = loc[cols].notna().all(axis=1).values
    L = np.log(loc.loc[ok, cols].values)
    clr = L - L.mean(axis=1, keepdims=True)
    mu = clr.mean(axis=0)
    U, S, Vt = np.linalg.svd(clr - mu, full_matrices=False)
    scores = (clr - mu) @ Vt[:3].T
    carb = loc.loc[ok, 'carb_named'].values
    aucs = [roc_auc_score(carb, scores[:, k]) for k in range(3)]
    k = int(np.argmax(np.abs(np.array(aucs) - 0.5)))
    sign = 1.0 if aucs[k] >= 0.5 else -1.0
    aff = np.full(len(loc), np.nan)
    aff[ok] = sign * scores[:, k]
    info = pd.DataFrame({'element': [e.capitalize() for e in ELEMENTS], **{f'PC{j + 1}_loading': Vt[j] for j in range(3)}})
    info.attrs = {'explained': (S ** 2 / (S ** 2).sum())[:3], 'aucs': aucs, 'chosen': k, 'sign': sign, 'n': int(ok.sum())}
    sc = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3']).assign(
        region=loc.loc[ok, 'region'].values, rock_type=loc.loc[ok, 'rock_type'].values, carb_named=carb)
    return aff, info, sc


def map_percentiles(region, loc):
    """Province-wide percentiles of the production maps at each location."""
    regions = setup_regions()
    cfg = regions[region]
    clipped_r, clipped_v = raster_preprocessing.preprocess_all({region: cfg})
    sampleable_domain.build_all({region: cfg}, clipped_r, config.PROJECTED_CRS, GLCM_RASTERS)
    worm_trees = build_worm_trees(cfg['points'], config.PROJECTED_CRS)
    inside = sampleable_domain.sampleable_lookup(loc.lat.values, loc.lon.values, region)
    pts = loc.loc[inside, ['lat', 'lon']].reset_index(drop=True).assign(label=0)
    rec, cols, X_lab, X_unl, _ = _s25.extended_matrices(region, 'reencoded')
    y = rec['pruned']['y']
    pruned = rec['pruned']['cols']
    _, X1 = _p12.features_at(pts.copy(), cfg, clipped_r[region], clipped_v[region], worm_trees, pruned)
    base = [c for c in cols if c in pruned]
    X3 = _p29.set3_rows(region, rec, cols, {'pts': (pd.DataFrame(X1, columns=pruned)[base], pts.lon.values,
                                                     pts.lat.values)})['pts']
    ok = np.isfinite(X1).all(axis=1) & np.isfinite(X3).all(axis=1)
    pct = lambda bg, s: np.searchsorted(np.sort(bg), s, side='left') / len(bg)
    out = pd.DataFrame(index=np.where(inside)[0][ok])
    m3 = _p29.fit_models(X_lab[y == 1], X_unl, region)
    bg3 = np.vstack([X_lab[y == 0], X_unl])
    for key, which in [('3|DevNet', 'devnet'), ('3|MLP-BCE', 'mlp'), ('3|RF', 'rf')]:
        out[key] = pct(_p29.score(m3, bg3, which), _p29.score(m3, X3[ok], which))
    d = rec['pruned']
    t, t_fin, tf = _p12.fit_all(region)
    bg1 = np.vstack([d['X_lab'][d['y'] == 0], d['X_unl']])
    rf_bg, _, dn_bg = _p12.scores(t, t_fin, tf, bg1)
    rf_s, _, dn_s = _p12.scores(t, t_fin, tf, X1[ok])
    out['1|DevNet'], out['1|RF'] = pct(dn_bg, dn_s), pct(rf_bg, rf_s)
    pos = rec['lab'][rec['lab']['pool'] == 'positive']
    occ_km = cKDTree(_s30.xy(pos.lon, pos.lat)).query(_s30.xy(loc.lon, loc.lat))[0]
    print(f'[{region}] {len(loc)} locations, {inside.sum()} inside the domain, {ok.sum()} with complete features',
          flush=True)
    return out, occ_km


def clusters(v, km=10.0):
    if len(v) == 0:
        return 0
    pairs = cKDTree(_s30.xy(v.lon, v.lat)).query_pairs(km, output_type='ndarray')
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(v), len(v)))
    return int(connected_components(A, directed=False)[0])


def statistics(v, region, fs, model, excl, rng):
    key = f'{fs}|{model}'
    rows = []
    top = v[key] >= 0.90
    base = top.mean()
    for ind, q in HIGH_Q.items():
        col = COL[ind]
        w = v[v[col].notna()].copy()
        if len(w) < 30:
            continue
        w['mp'] = w[key]
        w['ind'] = w[IND[ind]]
        thr = w[col].quantile(q)
        w['high'] = w[col] >= thr
        topw = w.mp >= 0.90
        enrich = lambda x: (((x.mp >= 0.9) & x.high).sum() / max(x.high.sum(), 1)) / max((x.mp >= 0.9).mean(), 1e-9)
        rho = spearmanr(w.mp, w.ind)[0]
        ci_rho = _s30.block_bootstrap(lambda x: spearmanr(x.mp, x.ind)[0], w, rng)
        ci_en = _s30.block_bootstrap(enrich, w, rng)
        rows.append({'region': region, 'featureset': fs, 'model': model, 'exclusion_km': excl, 'indicator': ind,
                     'n_locations': len(w), 'share_locations_top10': float(topw.mean()), 'rho': rho,
                     'rho_lo': ci_rho[0], 'rho_hi': ci_rho[1], 'n_high': int(w.high.sum()),
                     'high_in_top10': int((topw & w.high).sum()), 'enrichment': enrich(w),
                     'enrich_lo': ci_en[0], 'enrich_hi': ci_en[1],
                     'p_high': mannwhitneyu(w.mp[w.high], w.mp[~w.high], alternative='greater').pvalue,
                     'median_ind_top10': float(w[col][topw].median()), 'median_ind_rest': float(w[col][~topw].median())})
    c = v.carb_named
    w = v.assign(mp=v[key])
    en_rn = lambda x: ((x.mp[x.carb_named] >= 0.9).mean() / max((x.mp >= 0.9).mean(), 1e-9)) if x.carb_named.any() else np.nan
    ci_rn = _s30.block_bootstrap(en_rn, w, rng) if c.sum() >= 3 else (np.nan, np.nan)
    rn = {'region': region, 'featureset': fs, 'model': model, 'exclusion_km': excl, 'n_locations': int(c.sum()),
          'n_clusters_10km': clusters(v[c]), 'enrichment': en_rn(w), 'enrich_lo': ci_rn[0], 'enrich_hi': ci_rn[1],
          'share_top10': float((v[key][c] >= 0.9).mean()) if c.any() else np.nan, 'base_share_top10': float(base),
          'median_pct': float(v[key][c].median()) if c.any() else np.nan, 'median_pct_all': float(v[key].median()),
          'p': mannwhitneyu(v[key][c], v[key][~c], alternative='greater').pvalue if c.sum() >= 3 else np.nan}
    return rows, rn


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--rescore', action='store_true')
    args = ap.parse_args()
    seed_everything()
    loc = pd.concat([load_locations(r) for r in REGION_ORDER], ignore_index=True)
    aff, info, sc = clr_pca(loc)
    loc['affinity'] = aff
    a = info.attrs
    print(f"clr-PCA on {a['n']} locations: explained variance {np.round(a['explained'], 3)}; AUC (carbonatite/foid "
          f"vs rest) per PC {np.round(a['aucs'], 3)} -> affinity = {'+' if a['sign'] > 0 else '-'}PC{a['chosen'] + 1}",
          flush=True)
    print(info.round(3).to_string(index=False), flush=True)
    info.assign(explained=list(a['explained'][:3]) + [np.nan] * (len(info) - 3)).to_csv(out_path('geochem3_pca.csv'),
                                                                                         index=False)
    pd.DataFrame({'pc': [1, 2, 3], 'explained': a['explained'], 'auc_carbonatite': a['aucs'],
                  'chosen': [k == a['chosen'] for k in range(3)], 'sign': a['sign']}).to_csv(
        out_path('geochem3_pca_summary.csv'), index=False)
    sc.to_csv(out_path('geochem3_pca_scores.csv'), index=False)
    cache = out_path('geochem3_locations.csv')
    if os.path.exists(cache) and not args.rescore:
        evald = pd.read_csv(cache).drop(columns=['aff_in_type'], errors='ignore')
        print(f'map percentiles read from {cache}', flush=True)
    else:
        keep = []
        for region in REGION_ORDER:
            # evaluated locations: La, Ce and Nd measured (the affinity needs them too), as in Stage 30
            sub = loc[(loc.region == region) & loc.lree.notna()].reset_index(drop=True)
            mp, occ_km = map_percentiles(region, sub)
            sub['occ_km'] = occ_km
            keep.append(sub.join(mp, how='inner'))
        evald = pd.concat(keep, ignore_index=True)
    parts = []
    for region in REGION_ORDER:
        sub = evald[evald.region == region].copy()
        h = sub.affinity.notna()
        cnt = sub.loc[h, 'rock_type'].value_counts()
        grp = np.where(sub.loc[h, 'rock_type'].map(cnt) >= 30, sub.loc[h, 'rock_type'], 'other')
        sub.loc[h, 'aff_in_type'] = sub.loc[h, 'affinity'].groupby(grp).rank(pct=True).values
        parts.append(sub)
    evald = pd.concat(parts, ignore_index=True)
    evald.to_csv(cache, index=False)
    rows, rn_rows = [], []
    rng = np.random.default_rng(config.RANDOM_SEED)
    for region in REGION_ORDER:
        sub = evald[evald.region == region]
        for fs, model in MODELS:
            for excl in EXCLUSIONS:
                v = sub[sub.occ_km > excl]
                r, rn = statistics(v, region, fs, model, excl, rng)
                rows += r
                rn_rows.append(rn)
                for x in r:
                    print(f"[{region} | set {fs} {model} | >{excl:.0f} km | {x['indicator']}] n={x['n_locations']} "
                          f"({x['share_locations_top10']:.0%} in top 10%) rho={x['rho']:+.3f} "
                          f"[{x['rho_lo']:+.2f},{x['rho_hi']:+.2f}] high in top10 {x['high_in_top10']}/{x['n_high']} "
                          f"x{x['enrichment']:.2f} [{x['enrich_lo']:.2f},{x['enrich_hi']:.2f}] p={x['p_high']:.2g}",
                          flush=True)
                print(f"[{region} | set {fs} {model} | >{excl:.0f} km | carbonatite/foid] n={rn['n_locations']} "
                      f"({rn['n_clusters_10km']} clusters) top10 {rn['share_top10']:.0%} vs {rn['base_share_top10']:.0%} "
                      f"x{rn['enrichment']:.2f} [{rn['enrich_lo']:.2f},{rn['enrich_hi']:.2f}] p={rn['p']:.2g}", flush=True)
    pd.DataFrame(rows).to_csv(out_path('geochem3_validation.csv'), index=False)
    pd.DataFrame(rn_rows).to_csv(out_path('geochem3_rocknames.csv'), index=False)


if __name__ == '__main__':
    main()
