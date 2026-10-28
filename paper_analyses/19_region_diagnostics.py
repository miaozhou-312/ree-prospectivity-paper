"""
Stage 19 -- why the provinces differ: model-free diagnostics of the label
sets and of the signal available to any model.

Per region:
  * label structure: positives, independent sites (single-linkage at 10 km),
    spatial clusters at 50 km, share of positives sharing a site, median
    distance to the nearest other positive, development classes, number of
    host-era classes mapped at the positives;
  * univariate signal: |AUC - 0.5| of every pruned feature, positives vs the
    evaluation background (strongest and median feature);
  * transferable similarity ("far-neighbour index"): features as percentile
    ranks (0-1, fitted on positives + training background); for each
    positive, the share of background points more than 100 km away that are
    closer in feature space than the nearest positive more than 100 km away.
    0 means distant occurrences resemble each other more than they resemble
    any background, 0.5 means no more than random ground. Computed on all
    pruned features and on geophysical/structural features only;
  * missing values before imputation.

    python -m paper_analyses.19_region_diagnostics
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

from paper_analyses.common import REGION_ORDER, config, load_region, out_path
from paper_analyses.figstyle import pretty
from paper_analyses.protocol import project_xy

FAR_KM = 100.0
TERRANE = ('geo_era_', 'lith_')


def sites(xy_km, cut_km):
    if len(xy_km) < 2:
        return np.ones(len(xy_km), int)
    return fcluster(linkage(xy_km, method='single'), t=cut_km, criterion='distance')


def far_neighbour_index(F_pos, F_bg, xy_pos, xy_bg):
    out = []
    for i in range(len(F_pos)):
        far_p = np.linalg.norm(xy_pos - xy_pos[i], axis=1) > FAR_KM
        far_b = np.linalg.norm(xy_bg - xy_pos[i], axis=1) > FAR_KM
        if not far_p.any():
            out.append(np.nan)
            continue
        d_pos = np.linalg.norm(F_pos[far_p] - F_pos[i], axis=1).min()
        d_bg = np.linalg.norm(F_bg[far_b] - F_pos[i], axis=1)
        out.append(float((d_bg < d_pos).mean()))
    return np.array(out)


def main():
    rows, per_pos, uni = [], [], []
    for region in REGION_ORDER:
        rec = load_region(region)
        d = rec['pruned']
        cols, y = d['cols'], d['y']
        lab = rec['lab'].reset_index(drop=True)
        xy_lab = project_xy(lab['lat'], lab['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        pos = np.where(y == 1)[0]
        xy_pos = xy_lab[pos]
        names = pd.read_csv(f'data/labels/{region}_REE_deposits_deduped_2.0km.csv')
        # match label rows to named occurrences by coordinates
        nm = [names.loc[np.argmin((names.Latitude - la) ** 2 + (names.Longitude - lo) ** 2)]
              for la, lo in zip(lab.loc[pos, 'lat'], lab.loc[pos, 'lon'])]
        s10, s50 = sites(xy_pos, 10.0), sites(xy_pos, 50.0)
        nn = cdist(xy_pos, xy_pos)
        np.fill_diagonal(nn, np.inf)
        eras = [c for c in cols if c.startswith('geo_era_')]
        era_at_pos = lab.loc[pos, [c for c in lab.columns if c.startswith('geo_era_')]].idxmax(axis=1)

        X_all = np.vstack([d['X_lab'], d['X_unl']])
        missing = float(np.isnan(X_all).mean())
        imp = SimpleImputer(strategy='median').fit(np.vstack([d['X_lab'][pos], d['X_unl']]))
        Xl, Xu = imp.transform(d['X_lab']), imp.transform(d['X_unl'])
        ev = y == 0
        aucs = {}
        for j, c in enumerate(cols):
            if np.unique(Xl[:, j]).size < 2:
                continue
            a = roc_auc_score(y, Xl[:, j])
            aucs[c] = a
            uni.append({'region': region, 'feature': pretty(c), 'auc': a, 'abs_dev': abs(a - 0.5)})
        dev = np.array([abs(a - 0.5) for a in aucs.values()])

        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000,
                                 random_state=config.RANDOM_SEED).fit(np.vstack([Xl[pos], Xu]))
        Fl, Fu = qt.transform(Xl), qt.transform(Xu)
        geo = [j for j, c in enumerate(cols) if not c.startswith(TERRANE)]
        fni_all = far_neighbour_index(Fl[pos], Fu, xy_pos, xy_bg)
        fni_geo = far_neighbour_index(Fl[pos][:, geo], Fu[:, geo], xy_pos, xy_bg)
        # same index for random background points posing as "occurrences" (reference distribution)
        rng = np.random.default_rng(config.RANDOM_SEED)
        ref = []
        for _ in range(200):
            k = rng.choice(len(Xu), len(pos), replace=False)
            rest = np.setdiff1d(np.arange(len(Xu)), k)
            ref.append(np.nanmedian(far_neighbour_index(Fu[k][:, geo], Fu[rest][:, geo], xy_bg[k], xy_bg[rest])))
        for i, p in enumerate(pos):
            per_pos.append({'region': region, 'name': nm[i]['Name'], 'class': nm[i]['dep_type'],
                            'site_10km': int(s10[i]), 'cluster_50km': int(s50[i]),
                            'nearest_positive_km': float(nn[i].min()), 'host_era': era_at_pos.iloc[i].replace('geo_era_', ''),
                            'far_neighbour_index_all': fni_all[i], 'far_neighbour_index_geo': fni_geo[i]})
        cls = pd.Series([n['dep_type'] for n in nm]).value_counts()
        rows.append({'region': region, 'n_pos': len(pos), 'sites_10km': len(np.unique(s10)),
                     'clusters_50km': len(np.unique(s50)),
                     'share_in_multi_site': float(np.mean(pd.Series(s10).map(pd.Series(s10).value_counts()) > 1)),
                     'median_nn_km': float(np.median(nn.min(1))),
                     'deposit_prospect': int(cls.get('Deposit/Prospect', 0)), 'showing': int(cls.get('Showing', 0)),
                     'occurrence': int(cls.get('Occurrence', 0)), 'host_era_classes': int(era_at_pos.nunique()),
                     'max_abs_auc_dev': float(dev.max()), 'median_abs_auc_dev': float(np.median(dev)),
                     'n_features_abs_dev_ge_0.25': int((dev >= 0.25).sum()),
                     'far_neighbour_index_all': float(np.nanmedian(fni_all)),
                     'far_neighbour_index_geo': float(np.nanmedian(fni_geo)),
                     'fni_geo_random_ref_median': float(np.median(ref)),
                     'fni_geo_random_ref_p05': float(np.percentile(ref, 5)),
                     'positives_without_far_positive': int(np.isnan(fni_geo).sum()), 'missing_rate': missing})
        print(pd.Series(rows[-1]).to_string(), flush=True)
    pd.DataFrame(rows).to_csv(out_path('region_diagnostics.csv'), index=False)
    pd.DataFrame(per_pos).to_csv(out_path('region_diagnostics_positives.csv'), index=False)
    pd.DataFrame(uni).sort_values(['region', 'abs_dev'], ascending=[True, False]).to_csv(
        out_path('region_diagnostics_univariate.csv'), index=False)


if __name__ == '__main__':
    main()
