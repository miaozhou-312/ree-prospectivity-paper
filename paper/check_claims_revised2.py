"""
Check that key numbers typed into the second revision (feature set 3 as the
main analysis) match the analysis outputs. Same idea as check_claims.py:
each claim is a manuscript phrase with its numbers as format fields,
recomputed from paper_analyses/out/; the formatted phrase must occur in
main_src_revised2.tex (whitespace-normalised).

    python paper/check_claims_revised2.py
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
TEX = os.path.join(REPO, 'paper', 'manuscript', 'main_src_revised2.tex')


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
    ('MLP-BCE is level with it (AP {:.3f};', lambda: [t3('MLP-BCE')]),
    ('nnPU, which shares DevNet\'s network and inputs, reaches an AP of only {:.3f} and an AUC of {:.3f}.',
     lambda: [t3('nnPU'), t3('nnPU', metric='auc_med')]),
    ("DevNet's AP is {:.3f}, {:.3f} and {:.3f} with 50, 100 and 200~km blocks, and MLP-BCE's is similar ({:.3f}, {:.3f} and {:.3f}).",
     lambda: [t3('DevNet', b) for b in B] + [t3('MLP-BCE', b) for b in B]),
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
    ("DevNet's AP is {:.3f} under random splits and {:.3f} under 100~km blocks, against {:.3f} and {:.3f} for MLP-BCE",
     lambda: [t3('DevNet', 0, keep='3', table=X2), t3('DevNet', 100, keep='3', table=X2),
              t3('MLP-BCE', 0, keep='3', table=X2), t3('MLP-BCE', 100, keep='3', table=X2)]),
    ('RF-tuned reached an AP of {:.3f}, {:.3f}, {:.3f} and {:.3f} under random splits and 50, 100 and 200~km blocks, and HGB-tuned {:.3f}, {:.3f}, {:.3f} and {:.3f}',
     lambda: [tuned('RF-tuned', b) for b in [0] + B] + [tuned('HGB-tuned', b) for b in [0] + B]),
    ('(RF median percentile {:.3f} under random splits and {:.3f} under 100~km blocks) than DevNet ({:.3f} and {:.3f})',
     lambda: [ob('RF', 'random'), ob('RF', '100 km'), ob('DevNet', 'random'), ob('DevNet', '100 km')]),
    ('at the {:.1f}th (DevNet), {:.1f}th (MLP-BCE) and {:.1f}th (RF) percentile',
     lambda: [100 * EXT.devnet_percentile, 100 * EXT.mlp_percentile, 100 * EXT.rf_percentile]),
    ("{:.0f}\\% of the grid nodes in the full DevNet model's top 5\\%", lambda: [jk('devnet')[0]]),
    ('for RF the shares are {:.0f}\\% and {:.0f}\\%', lambda: list(jk('rf'))),
    ('(Spearman correlation over the grid {:.2f}; DevNet and MLP-BCE {:.2f})',
     lambda: [spearmanr(GRID.devnet, GRID.rf)[0], spearmanr(GRID.devnet, GRID.mlp)[0]]),
    ('RF assigns a probability of exactly zero to {:.0f}\\% of the grid', lambda: [100 * (GRID.rf == 0).mean()]),
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
