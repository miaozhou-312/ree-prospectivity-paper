"""
Check that key numbers typed into the fourth revision (simplified; feature
set 3 only, earlier set-1 analyses rerun in out/fs3/) match the outputs.
Same idea as check_claims.py:
each claim is a manuscript phrase with its numbers as format fields,
recomputed from paper_analyses/out/; the formatted phrase must occur in
main_src_revised4.tex (whitespace-normalised).

    python paper/check_claims_revised4.py
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
TEX = os.path.join(REPO, 'paper', 'manuscript', 'main_src_revised4.tex')


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
F3 = lambda name: pd.read_csv(os.path.join(OUT, 'fs3', name))
DIAG = pd.concat([F3(f'devnet_diagnostics_{r}.csv') for r in ('Superior', 'Cordillera', 'Grenville')])
NEST = F3('nested_selection_Superior.csv')
AUG = F3('augmentation_Superior.csv')
AUD = F3('protocol_audit.csv')
AGD = F3('augmentation_diagnostics_Superior.csv').set_index('source')
FS = F3('final_suite.csv')
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
    ('DevNet has the highest median AP ({:.3f}) and RF the highest AUC ({:.3f})', lambda: [t3('DevNet'), t3('RF', metric='auc_med')]),
    ("DevNet's AP is {:.3f}, {:.3f} and {:.3f} with 50, 100 and 200~km blocks.", lambda: [t3('DevNet', b) for b in B]),
    ("RF's falls from {:.3f} under random splits to {:.3f}, {:.3f} and {:.3f}, and bagging-PU's to {:.3f}, {:.3f} and {:.3f}.",
     lambda: [t3('RF')] + [t3('RF', b) for b in B] + [t3('BaggingPU', b) for b in B]),
    ('where RF ({:.3f}) and bagging-PU ({:.3f}) exceed DevNet ({:.3f};', lambda: [t3(m, 200, 'auc_med') for m in ('RF', 'BaggingPU', 'DevNet')]),
    ("Removing the alkalic-intrusion column lowers DevNet's AP to {:.3f}, {:.3f}, {:.3f} and {:.3f}",
     lambda: [t3('DevNet', b, keep='no-alkalic') for b in [0] + B]),
    ("and RF's to {:.3f}, {:.3f}, {:.3f} and {:.3f}: this one class", lambda: [t3('RF', b, keep='no-alkalic') for b in [0] + B]),
    ("DevNet's falls to {:.3f} under random splits and {:.3f}, {:.3f} and {:.3f} under blocks",
     lambda: [t3('DevNet', b, table=X2) for b in [0] + B]),
    ('RF-tuned reached an AP of {:.3f}, {:.3f}, {:.3f} and {:.3f} under random splits and 50, 100 and 200~km blocks, and HGB-tuned {:.3f}, {:.3f}, {:.3f} and {:.3f}',
     lambda: [tuned('RF-tuned', b) for b in [0] + B] + [tuned('HGB-tuned', b) for b in [0] + B]),
    ('gave Superior APs of {:.3f}, {:.3f}, {:.3f} and {:.3f}, against {:.3f}, {:.3f}, {:.3f} and {:.3f} for the final configuration',
     lambda: [NEST.set_index('block_km').loc[b].ap_med for b in [0, 50, 100, 200]] +
             [NEST.set_index('block_km').loc[b].ref_ap_final for b in [0, 50, 100, 200]]),
    ('random splits: {:.3f}, {:.3f} and {:.3f} against {:.3f}, {:.3f} and {:.3f}; 100~km blocks in Superior: {:.3f} against {:.3f}',
     lambda: [t3('DevNet', 0, region=r) for r in ('Superior', 'Cordillera', 'Grenville')] +
             [DIAG[(DIAG.region == r) & (DIAG.model == 'LogReg') & (DIAG.protocol == 'random')].iloc[0].ap_med for r in ('Superior', 'Cordillera', 'Grenville')] +
             [t3('DevNet', 100), DIAG[(DIAG.region == 'Superior') & (DIAG.model == 'LogReg') & (DIAG.protocol != 'random')].iloc[0].ap_med]),
    ('The best augmented RF (jitter $10\\times$, AP {:.3f})', lambda: [AUG[(AUG.model == 'RF') & (AUG.method == 'jitter') & (AUG['mult'] == 10)].iloc[0].ap_med]),
    ('but for {:.0f}\\% of them the nearest labelled point is a background point', lambda: [100 * (1 - AGD.loc['wgan'].share_nn_is_positive)]),
    ('It gives a mean AUC of {:.3f} in Superior, {:.3f} in Cordillera and {:.3f} in Grenville',
     lambda: [AUD[(AUD.region == r) & (AUD.negatives == 'random') & (AUD.split == 'no augmentation')].iloc[0].auc_mean for r in ('Superior', 'Cordillera', 'Grenville')]),
    ('(Superior {:.3f}, Cordillera {:.3f}, Grenville {:.3f} with random negatives)',
     lambda: [AUD[(AUD.region == r) & (AUD.negatives == 'random') & (AUD.split == 'grouped by parent')].iloc[0].auc_mean for r in ('Superior', 'Cordillera', 'Grenville')]),
    ('(Superior {:.3f} with $a = 2$ against {:.3f} with $a = 5$)',
     lambda: [FS[(FS.region == 'Superior') & (FS.model == 'DevNet(a=2)')].iloc[0].ap_med, t3('DevNet')]),
    ('at the {:.1f}th (DevNet) and {:.1f}th (RF) percentile', lambda: [100 * EXT.devnet_percentile, 100 * EXT.rf_percentile]),
    ("{:.0f}\\% of the grid nodes in the full DevNet model's top 5\\%", lambda: [jk('devnet')[0]]),
    ('whereas for RF the shares are {:.0f}\\% and {:.0f}\\%', lambda: list(jk('rf'))),
    ('lowered RF\'s AP from {:.3f} to {:.3f} but bagging-PU\'s only from {:.3f} to {:.3f}',
     lambda: [cv('E1', 'reencoded', 'random', 'RF', 0), cv('E1', 'reencoded', 'random', 'RF', 50),
              cv('E1', 'reencoded', 'random', 'BaggingPU', 0), cv('E1', 'reencoded', 'random', 'BaggingPU', 50)]),
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
