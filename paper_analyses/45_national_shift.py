"""
Stage 45 (N4) -- how different are the provinces? Model-free and simple
model diagnostics on the national table (feature set 3), for the groups
Superior, Cordillera, Grenville and the rest of Canada.

1. Domain classifier: RF (300 trees, 5-fold CV) telling the training
   background of one group from another's; AUC 0.5 = same feature
   distribution, 1.0 = fully separable. The same for the occurrences of
   two provinces (5-fold CV repeated 20 times; few points).
2. Cross-province far-neighbour index (Stage 19, across provinces): for
   each occurrence of province B, the share of province A's background
   that is closer in feature space (percentile ranks fitted on the national
   training background) than A's nearest occurrence. 0 = B's occurrence
   resembles A's occurrences more than any of A's ground; 0.5 = no more
   than random ground. Diagonal: within A, occurrences more than 100 km away.
   On all set-3 features and on the geophysical/structural ones only (no
   map classes, no distances to the generalised geology), as in Stage 19.
3. Univariate signal: AUC of each feature alone for a province's
   occurrences against its evaluation background (> 0.5: occurrences have
   higher values); features whose direction flips between provinces with
   |AUC - 0.5| >= 0.15 in both.

    python -m paper_analyses.45_national_shift
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import QuantileTransformer

from paper_analyses.common import config
from paper_analyses.national import STUDY, load_national, regional_rows

_b = __import__('importlib').import_module('paper_analyses.42_national_benchmark')
GROUPS = STUDY + ['Other']
FAR_KM = 100.0
STRONG = 0.15


def group_rows(d, g):
    if g in STUDY:
        return regional_rows(d['meta'], g)
    nat = d['pts'][d['pts'].in_national & (d['pts'].group == 'Other')]
    return (nat[nat.pool.isin(['positive', 'eval_neg'])].reset_index(drop=True),
            nat[nat.pool == 'train_unlabeled'].reset_index(drop=True))


def main():
    d = load_national('set3')
    cols = d['cols']
    data = {g: group_rows(d, g) for g in GROUPS}
    imp = SimpleImputer(strategy='median').fit(d['X_unl'])
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=0).fit(
        imp.transform(d['X_unl']))
    F = lambda df: qt.transform(imp.transform(df[cols].values.astype(float)))  # noqa: E731
    geo = [j for j, c in enumerate(cols) if not c.startswith(('lith_', 'era_', 'dist_')) and
           c not in ('lith_contact', 'age_contact')]
    rows = []
    for i, a in enumerate(GROUPS):                                     # 1b. occurrences of two provinces
        for b in GROUPS[i + 1:]:
            Pa = data[a][0].loc[data[a][0].label == 1, cols].values.astype(float)
            Pb = data[b][0].loc[data[b][0].label == 1, cols].values.astype(float)
            X = SimpleImputer(strategy='median').fit_transform(np.vstack([Pa, Pb]))
            y = np.r_[np.zeros(len(Pa)), np.ones(len(Pb))]
            aucs = [roc_auc_score(y, cross_val_predict(
                RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=r), X, y,
                cv=StratifiedKFold(5, shuffle=True, random_state=r), method='predict_proba')[:, 1]) for r in range(20)]
            rows.append({'test': 'domain classifier (occurrences)', 'a': a, 'b': b, 'value': float(np.median(aucs)),
                         'n': len(y)})
            print(f'[N4] occurrences {a} ({len(Pa)}) vs {b} ({len(Pb)}): AUC {rows[-1]["value"]:.3f}', flush=True)
    for i, a in enumerate(GROUPS):                                     # 1. domain classifier
        for b in GROUPS[i + 1:]:
            Xa, Xb = data[a][1][cols].values.astype(float), data[b][1][cols].values.astype(float)
            rng = np.random.default_rng(0)
            n = min(len(Xa), len(Xb), 5000)
            X = np.vstack([Xa[rng.choice(len(Xa), n, replace=False)], Xb[rng.choice(len(Xb), n, replace=False)]])
            y = np.r_[np.zeros(n), np.ones(n)]
            X = SimpleImputer(strategy='median').fit_transform(X)
            p = cross_val_predict(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0), X, y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
            rows.append({'test': 'domain classifier (background)', 'a': a, 'b': b, 'value': roc_auc_score(y, p)})
            print(f'[N4] domain classifier {a} vs {b}: AUC {rows[-1]["value"]:.3f}', flush=True)
    for a in GROUPS:                                                   # 2. cross-province far-neighbour index
        lab_a, unl_a = data[a]
        Pa, Ba = F(lab_a[lab_a.label == 1]), F(unl_a)
        xy_pa = lab_a.loc[lab_a.label == 1, ['xy_x', 'xy_y']].values
        for b in GROUPS:
            lab_b = data[b][0]
            Pb = F(lab_b[lab_b.label == 1])
            xy_pb = lab_b.loc[lab_b.label == 1, ['xy_x', 'xy_y']].values
            for name, j in [('all features', slice(None)), ('geophysical/structural', geo)]:
                vals = []
                for i in range(len(Pb)):
                    far = np.linalg.norm(xy_pa - xy_pb[i], axis=1) > FAR_KM if a == b else np.ones(len(Pa), bool)
                    if not far.any():
                        continue
                    d_pos = np.linalg.norm(Pa[far][:, j] - Pb[i, j], axis=1).min()
                    vals.append(float((np.linalg.norm(Ba[:, j] - Pb[i, j], axis=1) < d_pos).mean()))
                rows.append({'test': f'far-neighbour index ({name})', 'a': a, 'b': b, 'value': float(np.median(vals)),
                             'n': len(vals)})
                print(f'[N4] far-neighbour index ({name}), occurrences of {b} against {a}: '
                      f'{rows[-1]["value"]:.3f}', flush=True)
    uni = {}
    for g in GROUPS:                                                   # 3. univariate directions
        lab = data[g][0]
        y = lab.label.values.astype(int)
        X = SimpleImputer(strategy='median').fit_transform(lab[cols].values.astype(float))
        uni[g] = pd.Series([roc_auc_score(y, X[:, j]) if np.ptp(X[:, j]) > 0 else 0.5 for j in range(len(cols))],
                           index=cols)
    U = pd.DataFrame(uni)
    for i, a in enumerate(GROUPS):
        for b in GROUPS[i + 1:]:
            sa, sb = U[a] - 0.5, U[b] - 0.5
            strong = (sa.abs() >= STRONG) & (sb.abs() >= STRONG)
            flip = strong & (np.sign(sa) != np.sign(sb))
            rows.append({'test': 'univariate direction flips', 'a': a, 'b': b, 'value': int(flip.sum()),
                         'n': int(strong.sum()), 'features': '; '.join(U.index[flip])})
            print(f'[N4] {a} vs {b}: {int(flip.sum())} of {int(strong.sum())} strong features flip direction '
                  f'({", ".join(U.index[flip][:6])})', flush=True)
    pd.DataFrame(rows).to_csv(_b.out('shift.csv'), index=False)
    U.to_csv(_b.out('shift_univariate_auc.csv'))


if __name__ == '__main__':
    main()
