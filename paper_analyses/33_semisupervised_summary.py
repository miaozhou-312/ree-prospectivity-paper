"""
Stage 33 -- summary statistics for the three tests of Stage 32, exactly as
fixed in out/semisup_predictions.txt before the runs.

Pairs: DevNet vs DevNet-BCE (MLP-BCE; the loss alone differs), bagging-PU vs
RF (the tree-based pair) and DevNet vs nnPU. For a pair (a, b) and each of
the 20 repeats (identical folds for both models and all conditions):

  E1  change_m = AP_m(n = 50) - AP_m(n = 0); effect = change_a - change_b
      (> 0: a loses less when mislabelled ground near occurrences is added).
      stat_a, stat_b: median change of a and of b.
  E2  gap = AP_a - AP_b; effect = gap(5,000) - gap(625) (> 0: a gains more
      from a larger unlabelled pool). stat_a, stat_b: median gap at 625 and
      at 5,000.
  E3  AP averaged over k = 1, 2, 3; effect = mean AP_a - mean AP_b.
      stat_a, stat_b: median of the averaged AP of a and of b.

effect: median over repeats; wins: repeats with effect > 0; p: one-sided
exact sign test. Also writes the curves behind the figure (median AP per
condition) and the per-condition gaps. Every output carries a region column
(Superior, and Cordillera and Grenville once their runs exist).

    python -m paper_analyses.33_semisupervised_summary
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from paper_analyses.common import out_path

PAIRS = [('DevNet', 'MLP-BCE', 'DevNet vs DevNet-BCE'), ('BaggingPU', 'RF', 'Bagging-PU vs RF'),
         ('DevNet', 'nnPU', 'DevNet vs nnPU')]
ORDER = {'E1': ['near=0', 'near=20', 'near=50'],
         'E2': ['pool=625', 'pool=1250', 'pool=2500', 'pool=5000'],
         'E3': ['k=1', 'k=2', 'k=3', 'k=5', 'k=8', 'k=all']}


def wide(t, cond, model):
    return t[(t.condition == cond) & (t.model == model)].sort_values('repeat').ap.values


def effect_row(region, exp, fs, proto, label, eff, a_stat, b_stat):
    w = int((eff > 0).sum())
    return {'region': region, 'experiment': exp, 'featureset': fs, 'protocol': proto, 'pair': label, 'stat_a': float(np.median(a_stat)),
            'stat_b': float(np.median(b_stat)), 'effect': float(np.median(eff)), 'wins': w, 'n': len(eff),
            'p_sign': binomtest(w, len(eff), 0.5, alternative='greater').pvalue}


def main():
    files = sorted(glob.glob(out_path('semisup_E*_*.csv')))
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    t['region'] = t['region'].fillna('Superior') if 'region' in t else 'Superior'
    rows, curves, gaps = [], [], []
    for (region, exp, fs, proto), d in t.groupby(['region', 'experiment', 'featureset', 'protocol']):
        conds = [c for c in ORDER[exp] if c in set(d.condition)]
        for i, c in enumerate(conds):
            for m, q in d[d.condition == c].groupby('model'):
                curves.append({'region': region, 'experiment': exp, 'featureset': fs, 'protocol': proto, 'model': m, 'x': i,
                               'x_label': c.split('=')[1], 'ap_med': q.ap.median()})
            for a, b, label in PAIRS:
                if {a, b} <= set(d[d.condition == c].model):
                    gap = wide(d, c, a) - wide(d, c, b)
                    gaps.append({'region': region, 'experiment': exp, 'featureset': fs, 'protocol': proto, 'condition': c, 'pair': label,
                                 'gap_median': float(np.median(gap)), 'wins': int((gap > 0).sum()), 'n': len(gap)})
        for a, b, label in PAIRS:
            have = lambda c: {a, b} <= set(d[d.condition == c].model)
            if exp == 'E1' and have('near=0') and have('near=50'):
                ca = wide(d, 'near=50', a) - wide(d, 'near=0', a)
                cb = wide(d, 'near=50', b) - wide(d, 'near=0', b)
                rows.append(effect_row(region, exp, fs, proto, label, ca - cb, ca, cb))
            if exp == 'E2' and have('pool=625') and have('pool=5000'):
                g0 = wide(d, 'pool=625', a) - wide(d, 'pool=625', b)
                g1 = wide(d, 'pool=5000', a) - wide(d, 'pool=5000', b)
                rows.append(effect_row(region, exp, fs, proto, label, g1 - g0, g0, g1))
            if exp == 'E3' and all(have(f'k={k}') for k in (1, 2, 3)):
                ma = np.mean([wide(d, f'k={k}', a) for k in (1, 2, 3)], axis=0)
                mb = np.mean([wide(d, f'k={k}', b) for k in (1, 2, 3)], axis=0)
                rows.append(effect_row(region, exp, fs, proto, label, ma - mb, ma, mb))
    s = pd.DataFrame(rows)
    s.to_csv(out_path('semisup_summary.csv'), index=False)
    pd.DataFrame(curves).to_csv(out_path('semisup_curves.csv'), index=False)
    pd.DataFrame(gaps).to_csv(out_path('semisup_gaps.csv'), index=False)
    pd.set_option('display.width', 200)
    print(s.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
