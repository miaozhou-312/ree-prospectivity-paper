"""
Check that key numbers typed into the third revision (DevNet-BCE as an
ablation, tests of the semi-supervised premise) match the analysis outputs.
Same idea as check_claims.py:
each claim is a manuscript phrase with its numbers as format fields,
recomputed from paper_analyses/out/; the formatted phrase must occur in
main_src_revised3.tex (whitespace-normalised).

    python paper/check_claims_revised3.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'paper_analyses', 'out')
TEX = os.path.join(REPO, 'paper', 'manuscript', 'main_src_revised3.tex')


def csv(name):
    return pd.read_csv(os.path.join(OUT, name))


T = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'reencoded_benchmark_*.csv'))], ignore_index=True)
T['keep'] = T['keep'].astype(str)
C = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'controls_fs3_*.csv'))], ignore_index=True)
X2 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'extended_benchmark_*.csv'))], ignore_index=True)
X2['keep'] = X2['keep'].astype(str)
HOLM = csv('reencoded_spatial_holm_Superior.csv')
OB = csv('occurrence_breakdown_fs3.csv')
EXT = csv('external_check_fs3.csv').iloc[0]
JK = csv('superior_grid_uncertainty_fs3.csv')
GRID = csv('superior_grid_scores_fs3.csv')
LA = csv('loss_ablation_Superior.csv')
SS = csv('semisup_summary.csv')
SSA = SS.copy()
SS = SS[SS.region == 'Superior']
CV = csv('semisup_curves.csv')
CV = CV[CV.region == 'Superior']


def ss(exp, fs, proto, pair, col='effect'):
    return SS[(SS.experiment == exp) & (SS.featureset == fs) & (SS.protocol == proto) & (SS.pair == pair)].iloc[0][col]


def sr(region, exp, fs, pair, col='effect'):
    return SSA[(SSA.region == region) & (SSA.experiment == exp) & (SSA.featureset == fs) & (SSA.protocol == 'random')
               & (SSA.pair == pair)].iloc[0][col]


def cv(exp, fs, proto, model, x):
    return CV[(CV.experiment == exp) & (CV.featureset == fs) & (CV.protocol == proto) & (CV.model == model)
              & (CV.x_label.astype(str) == str(x))].iloc[0].ap_med


def t3(model, block=0, metric='ap_med', region='Superior', keep='0', table=T):
    return table[(table.region == region) & (table.model == model) & (table.block_km == block)
                 & (table['keep'] == keep)].iloc[0][metric]


def tuned(model, block):
    return C[(C.region == 'Superior') & (C.model == model) & (C.block_km == block)].iloc[0].ap_med


def jk(model):
    top = JK[f'{model}_pct'] >= 0.95
    f = JK.loc[top, f'{model}_top5_freq']
    return 100 * (f >= 0.8).mean(), 100 * (f < 0.5).mean()


def ob(model, proto, group='Abitibi cluster'):
    return OB[(OB.model == model) & (OB.protocol == proto) & (OB.group == group)].iloc[0].median_pct


B = [50, 100, 200]
CLAIMS = [
    ('DevNet has the highest median AP ({:.3f}) and RF the highest AUC ({:.3f})',
     lambda: [t3('DevNet'), t3('RF', metric='auc_med')]),
    ('nnPU, which shares DevNet\'s network and inputs, reaches an AP of only {:.3f} and an AUC of {:.3f}.',
     lambda: [t3('nnPU'), t3('nnPU', metric='auc_med')]),
    ("DevNet's AP is {:.3f}, {:.3f} and {:.3f} with 50, 100 and 200~km blocks.", lambda: [t3('DevNet', b) for b in B]),
    ("RF's falls from {:.3f} under random splits to {:.3f}, {:.3f} and {:.3f}, and bagging-PU's to {:.3f}, {:.3f} and {:.3f}.",
     lambda: [t3('RF')] + [t3('RF', b) for b in B] + [t3('BaggingPU', b) for b in B]),
    ('where RF ({:.3f}) and bagging-PU ({:.3f}) exceed DevNet ({:.3f};',
     lambda: [t3(m, 200, 'auc_med') for m in ('RF', 'BaggingPU', 'DevNet')]),
    ('whereas those from RF do not ({:.3f} at each block size)',
     lambda: [HOLM[HOLM.comparison == 'DevNet - RF'].holm.max()]),
    ("Removing the alkalic-intrusion class lowers DevNet's AP to {:.3f}, {:.3f}, {:.3f} and {:.3f} and RF's to {:.3f}, {:.3f}, {:.3f} and {:.3f}",
     lambda: [t3('DevNet', b, keep='no-alkalic') for b in [0] + B] + [t3('RF', b, keep='no-alkalic') for b in [0] + B]),
    ('DevNet the highest under every block size ({:.3f}, {:.3f} and {:.3f}, narrowly ahead of bagging-PU\'s {:.3f}, {:.3f} and {:.3f})',
     lambda: [t3('DevNet', b, table=X2) for b in B] + [t3('BaggingPU', b, table=X2) for b in B]),
    ('RF-tuned reached an AP of {:.3f}, {:.3f}, {:.3f} and {:.3f} under random splits and 50, 100 and 200~km blocks, and HGB-tuned {:.3f}, {:.3f}, {:.3f} and {:.3f}',
     lambda: [tuned('RF-tuned', b) for b in [0] + B] + [tuned('HGB-tuned', b) for b in [0] + B]),
    ('(RF median percentile {:.3f} under random splits and {:.3f} under 100~km blocks) than DevNet ({:.3f} and {:.3f})',
     lambda: [ob('RF', 'random'), ob('RF', '100 km'), ob('DevNet', 'random'), ob('DevNet', '100 km')]),
    ('at the {:.1f}th (DevNet) and {:.1f}th (RF) percentile', lambda: [100 * EXT.devnet_percentile, 100 * EXT.rf_percentile]),
    ("{:.0f}\\% of the grid nodes in the full DevNet model's top 5\\%", lambda: [jk('devnet')[0]]),
    ('for RF the shares are {:.0f}\\% and {:.0f}\\%', lambda: list(jk('rf'))),
    ('(Spearman correlation over the grid {:.2f})', lambda: [spearmanr(GRID.devnet, GRID.rf)[0]]),
    ('RF assigns a probability of exactly zero to {:.0f}\\% of the grid', lambda: [100 * (GRID.rf == 0).mean()]),
    ('(paired differences ${:.3f}$ to ${:.3f}$; DevNet higher in at most three of six repeats)',
     lambda: [LA[LA.featureset.astype(str) != '2'].d_ap.min(), LA[LA.featureset.astype(str) != '2'].d_ap.max()]),
    ('but by small margins (${:.3f}$ to ${:+.3f}$, $p \\geq {:.2f}$)',
     lambda: [LA[LA.featureset.astype(str) == '2'].d_ap.min(), LA[LA.featureset.astype(str) == '2'].d_ap.max(),
              np.floor(100 * LA[LA.featureset.astype(str) == '2'].p.min()) / 100]),
    ('Averaged over $k = 1$--3 the gap was ${:+.3f}$ under random splits and ${:+.3f}$ under 100~km blocks',
     lambda: [ss('E3', 'extended', p, 'DevNet vs DevNet-BCE') for p in ('random', '100 km')]),
    ('bagging-PU was likewise ahead of RF (${:+.3f}$ and ${:+.3f}$;',
     lambda: [ss('E3', 'extended', p, 'Bagging-PU vs RF') for p in ('random', '100 km')]),
    ('the two losses were level for $k \\leq 3$ (${:+.3f}$ and ${:+.3f}$)',
     lambda: [ss('E3', 'reencoded', p, 'DevNet vs DevNet-BCE') for p in ('random', '100 km')]),
    ("lowered RF's AP from {:.3f} to {:.3f} (median change ${:.3f}$), but bagging-PU's only from {:.3f} to {:.3f} (${:.3f}$)",
     lambda: [cv('E1', 'reencoded', 'random', 'RF', 0), cv('E1', 'reencoded', 'random', 'RF', 50),
              ss('E1', 'reencoded', 'random', 'Bagging-PU vs RF', 'stat_b'),
              cv('E1', 'reencoded', 'random', 'BaggingPU', 0), cv('E1', 'reencoded', 'random', 'BaggingPU', 50),
              ss('E1', 'reencoded', 'random', 'Bagging-PU vs RF', 'stat_a')]),
    ('DevNet-BCE lost {:.3f} and DevNet {:.3f} under random splits',
     lambda: [-ss('E1', 'reencoded', 'random', 'DevNet vs DevNet-BCE', 'stat_b'),
              -ss('E1', 'reencoded', 'random', 'DevNet vs DevNet-BCE', 'stat_a')]),
    ('in Cordillera on both feature sets (effects ${:+.3f}$ and ${:+.3f}$; {:d} and {:d} of 20 repeats',
     lambda: [sr('Cordillera', 'E1', 'reencoded', 'DevNet vs DevNet-BCE'), sr('Cordillera', 'E1', 'extended', 'DevNet vs DevNet-BCE'),
              int(sr('Cordillera', 'E1', 'reencoded', 'DevNet vs DevNet-BCE', 'wins')),
              int(sr('Cordillera', 'E1', 'extended', 'DevNet vs DevNet-BCE', 'wins'))]),
    ('and in Grenville on feature set 3 (${:+.3f}$; {:d} of 20',
     lambda: [sr('Grenville', 'E1', 'reencoded', 'DevNet vs DevNet-BCE'), int(sr('Grenville', 'E1', 'reencoded', 'DevNet vs DevNet-BCE', 'wins'))]),
    ('cost RF more than bagging-PU in Grenville on both feature sets (${:+.3f}$ and ${:+.3f}$; {:d} and {:d} of 20',
     lambda: [sr('Grenville', 'E1', 'extended', 'Bagging-PU vs RF'), sr('Grenville', 'E1', 'reencoded', 'Bagging-PU vs RF'),
              int(sr('Grenville', 'E1', 'extended', 'Bagging-PU vs RF', 'wins')), int(sr('Grenville', 'E1', 'reencoded', 'Bagging-PU vs RF', 'wins'))]),
    ('in Grenville on feature set 2 (${:+.3f}$; {:d} of 20', lambda: [sr('Grenville', 'E3', 'extended', 'DevNet vs DevNet-BCE'),
                                                               int(sr('Grenville', 'E3', 'extended', 'DevNet vs DevNet-BCE', 'wins'))]),
    ('in Cordillera on feature set 3 (${:+.3f}$; {:d} of 20', lambda: [sr('Cordillera', 'E3', 'reencoded', 'DevNet vs DevNet-BCE'),
                                                                int(sr('Cordillera', 'E3', 'reencoded', 'DevNet vs DevNet-BCE', 'wins'))]),
]


def norm(t):
    return re.sub(r'\s+', ' ', t)


def main():
    tex = norm(open(TEX).read())
    bad = 0
    for template, f in CLAIMS:
        phrase = norm(template.format(*f()))
        ok = phrase in tex
        bad += not ok
        print(('OK   ' if ok else 'FAIL ') + phrase)
    print(f'\n{len(CLAIMS) - bad}/{len(CLAIMS)} claims match')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
