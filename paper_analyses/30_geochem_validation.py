"""
Stage 30 -- independent geochemical validation of the Superior maps.

SUPERSEDED by Stage 31, which scores every sample location directly: the
nearest 0.1 degree grid node used here lies a median 3.5 km from the sample,
and the set-3 maps vary at the scale of single H3 cells. Kept for the record.

Whole-rock geochemistry (dataset/geochem/Geochemistry_Superior_Clip.csv, a
literature compilation) never enters training, so it can test whether
ground ranked as prospective is REE-enriched. Choices fixed before running:

  * samples with La, Ce and Nd all measured; ore, vein and xenolith
    material excluded; samples sharing a location (coordinates rounded to
    0.001 degree) are merged by their median, so that heavily sampled
    complexes do not dominate;
  * indicator: La+Ce+Nd (ppm) and its percentile within the sample's rock
    type (rock types with < 30 locations pooled), which removes the
    first-order dependence of REE content on lithology;
  * circularity: locations within 10 km (main) or 25 km (sensitivity) of a
    training occurrence are excluded;
  * each location takes the province-wide percentile of the nearest 0.1
    degree grid node (within 10 km) of each production map;
  * statistics: Spearman rho between map percentile and the within-type
    REE percentile; median La+Ce+Nd in the map's top 10% against the rest
    (one-sided Mann-Whitney); enrichment of strong anomalies (top 1% of
    La+Ce+Nd among all locations) in the top 10% of the map; 95% intervals
    from a 100 km block bootstrap (1,000 resamples);
  * unlabelled carbonatite and silica-undersaturated alkaline rocks (rock
    names matching CARBONATITE_PATTERN) beyond the exclusion radius: their
    map percentiles, as an independent occurrence test.

Sampling is far from uniform: a map's top 10% of area can hold a much larger
share of the sampled locations (e.g. where it covers greenstone belts). Every
share is therefore compared with the share of ALL evaluated locations in the
same class (the sampling baseline), not with the 10% of area: enrichment =
P(top 10% | strong anomaly) / P(top 10% | sampled), and the alkaline-rock
share is set against the same baseline (one-sided Mann-Whitney of their map
percentiles against all other locations). Only statistics that condition on
where samples were taken are reported. (A first version used the 10% of
area as baseline; that inflates enrichment for maps whose top class
coincides with densely sampled ground.)

    python -m paper_analyses.30_geochem_validation
"""
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr

from paper_analyses.common import config, load_region, out_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOCHEM = os.path.join(os.path.dirname(REPO), 'dataset', 'geochem', 'Geochemistry_Superior_Clip.csv')
T = Transformer.from_crs('EPSG:4326', 'EPSG:3978', always_xy=True)
MAPS = {('3 re-encoded', 'DevNet'): ('superior_grid_scores_fs3.csv', 'devnet'),
        ('3 re-encoded', 'MLP-BCE'): ('superior_grid_scores_fs3.csv', 'mlp'),
        ('3 re-encoded', 'RF'): ('superior_grid_scores_fs3.csv', 'rf'),
        ('1 thesis', 'DevNet'): ('superior_grid_scores.csv', 'devnet'),
        ('1 thesis', 'RF'): ('superior_grid_scores.csv', 'rf')}
N_BOOT, BLOCK_KM = 1000, 100.0
CARBONATITE_PATTERN = 'carbonatite|foid|ijolite|melilit|nephelin|urtite|melteigite'


def xy(lon, lat):
    x, y = T.transform(np.asarray(lon), np.asarray(lat))
    return np.c_[x, y] / 1000.0


def load_samples():
    d = pd.read_csv(GEOCHEM, low_memory=False)
    for c in ('la_ppm', 'ce_ppm', 'nd_ppm'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[(d[['la_ppm', 'ce_ppm', 'nd_ppm']] > 0).all(axis=1)]
    d = d[~d['material'].isin(['ore', 'vein', 'xenolith'])]
    d['lree'] = d.la_ppm + d.ce_ppm + d.nd_ppm
    d['loc'] = d.latitude.round(3).astype(str) + '_' + d.longitude.round(3).astype(str)
    g = d.groupby('loc').agg(lat=('latitude', 'median'), lon=('longitude', 'median'), lree=('lree', 'median'),
                             rock_type=('rock_type', lambda s: s.mode().iloc[0] if s.notna().any() else 'unknown'))
    g['rock_type'] = g.rock_type.fillna('unknown')
    counts = g.rock_type.value_counts()
    g['rock_group'] = np.where(g.rock_type.map(counts) >= 30, g.rock_type, 'other')
    g['lree_pct_in_type'] = g.groupby('rock_group').lree.rank(pct=True)
    return g.reset_index(drop=True), len(d)


def block_bootstrap(stat, df, rng):
    b = np.floor(xy(df.lon, df.lat) / BLOCK_KM).astype(int)
    ids = b[:, 0] * 100003 + b[:, 1]
    groups = [np.where(ids == u)[0] for u in np.unique(ids)]
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        vals.append(stat(df.iloc[idx]))
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    samples, n_raw = load_samples()
    rec = load_region('Superior')
    pos = rec['lab'][rec['lab']['pool'] == 'positive']
    d_occ = cKDTree(xy(pos.lon, pos.lat)).query(xy(samples.lon, samples.lat))[0]
    thr_strong = samples.lree.quantile(0.99)
    print(f'{n_raw} whole-rock samples -> {len(samples)} locations; strong-anomaly threshold (top 1%) '
          f'La+Ce+Nd >= {thr_strong:.0f} ppm', flush=True)
    rows, per_loc = [], {}
    rng = np.random.default_rng(config.RANDOM_SEED)
    for (fs, model), (fname, col) in MAPS.items():
        path = out_path(fname)
        if not os.path.exists(path):
            print(f'skip {fs} {model}: {fname} missing')
            continue
        grid = pd.read_csv(path)
        gpct = grid[col].rank(pct=True).values
        dist, i = cKDTree(xy(grid.lon, grid.lat)).query(xy(samples.lon, samples.lat))
        s = samples.assign(map_pct=gpct[i], grid_dist=dist, occ_km=d_occ)
        s = s[s.grid_dist <= 10.0]
        per_loc[(fs, model)] = s
        for excl in (10.0, 25.0):
            v = s[s.occ_km > excl]
            top = v.map_pct >= 0.90
            rho = spearmanr(v.map_pct, v.lree_pct_in_type)[0]
            p_mw = mannwhitneyu(v.lree[top], v.lree[~top], alternative='greater').pvalue
            strong = v.lree >= thr_strong
            enrichment = lambda x: (((x.map_pct >= 0.9) & (x.lree >= thr_strong)).sum()
                                    / max((x.lree >= thr_strong).sum(), 1) / max((x.map_pct >= 0.9).mean(), 1e-9))
            enrich = enrichment(v)
            p_strong = mannwhitneyu(v.map_pct[strong], v.map_pct[~strong], alternative='greater').pvalue
            ci_rho = block_bootstrap(lambda x: spearmanr(x.map_pct, x.lree_pct_in_type)[0], v, rng)
            ci_en = block_bootstrap(enrichment, v, rng)
            rows.append({'featureset': fs, 'model': model, 'exclusion_km': excl, 'n_locations': len(v),
                         'n_in_top10': int(top.sum()), 'share_locations_top10': float(top.mean()),
                         'rho': rho, 'rho_lo': ci_rho[0], 'rho_hi': ci_rho[1],
                         'median_lree_top10': float(v.lree[top].median()), 'median_lree_rest': float(v.lree[~top].median()),
                         'p_mannwhitney': p_mw, 'n_strong': int(strong.sum()), 'strong_in_top10': int((top & strong).sum()),
                         'enrichment': enrich, 'enrich_lo': ci_en[0], 'enrich_hi': ci_en[1], 'p_strong_pct': p_strong})
            print(f"[{fs} | {model} | >{excl:.0f} km] n={len(v)} ({top.mean():.0%} of locations in top 10%) "
                  f"rho={rho:+.3f} [{ci_rho[0]:+.3f},{ci_rho[1]:+.3f}] "
                  f"median LREE top10 {rows[-1]['median_lree_top10']:.0f} vs {rows[-1]['median_lree_rest']:.0f} ppm "
                  f"(p={p_mw:.2g}); strong anomalies in top10: {rows[-1]['strong_in_top10']}/{rows[-1]['n_strong']} "
                  f"(x{enrich:.2f} [{ci_en[0]:.2f},{ci_en[1]:.2f}] vs sampling; p={p_strong:.2g})", flush=True)
    pd.DataFrame(rows).to_csv(out_path('geochem_validation.csv'), index=False)
    # Unlabelled carbonatite and silica-undersaturated alkaline rocks (fixed rock-name rule): an
    # independent occurrence test analogous to Clay-Howells.
    rows2 = []
    for (fs, model), s in per_loc.items():
        for excl in (10.0, 25.0):
            e = s[s.occ_km > excl]
            alk = e.rock_type.str.contains(CARBONATITE_PATTERN, case=False, na=False)
            v, rest = e[alk], e[~alk]
            rows2.append({'featureset': fs, 'model': model, 'exclusion_km': excl, 'n_locations': len(v),
                          'median_map_pct': float(v.map_pct.median()), 'median_map_pct_all': float(e.map_pct.median()),
                          'share_top10': float((v.map_pct >= 0.9).mean()),
                          'base_share_top10': float((e.map_pct >= 0.9).mean()),
                          'share_top5': float((v.map_pct >= 0.95).mean()),
                          'base_share_top5': float((e.map_pct >= 0.95).mean()),
                          'p_vs_other_locations': mannwhitneyu(v.map_pct, rest.map_pct, alternative='greater').pvalue})
            x = rows2[-1]
            print(f"[carbonatite/foid rocks | {fs} | {model} | >{excl:.0f} km] n={len(v)} median percentile "
                  f"{x['median_map_pct']:.2f} (all locations {x['median_map_pct_all']:.2f}), in top 10% "
                  f"{x['share_top10']:.0%} (sampling baseline {x['base_share_top10']:.0%}), top 5% {x['share_top5']:.0%} "
                  f"({x['base_share_top5']:.0%}); p={x['p_vs_other_locations']:.2g}", flush=True)
    pd.DataFrame(rows2).to_csv(out_path('geochem_validation_alkaline_rocks.csv'), index=False)
    pd.concat([v.assign(featureset=k[0], model=k[1]) for k, v in per_loc.items()]).to_csv(
        out_path('geochem_validation_locations.csv'), index=False)


if __name__ == '__main__':
    main()
