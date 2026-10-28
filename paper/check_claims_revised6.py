"""
Check that numbers typed into the sixth revision match the outputs: the
claims of the fifth revision that the sixth keeps, plus the sentences the
sixth rewrote (national design and transfer, synthetic positives as plain
APs, the discussion of DevNet's design).

    python paper/check_claims_revised6.py
"""
import importlib.util
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'paper_analyses', 'out')
FS3 = os.path.join(OUT, 'fs3')
TEX = os.path.join(REPO, 'paper', 'manuscript', 'main_src_revised6.tex')

spec = importlib.util.spec_from_file_location('c5', os.path.join(REPO, 'paper', 'check_claims_revised5.py'))
c5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c5)
c4 = c5.c4

# claims of the fifth revision about analyses the sixth leaves out or sentences it rewrote
DROPPED = ('gave Superior APs of', "% of the grid nodes in the full DevNet", 'whereas for RF the shares are',
           "lowered RF\\'s AP from", "lowered RF's AP from", 'It gives a mean AUC of', '(Superior {:.3f}, Cordillera',
           'In {:.0f}\\% of the test positives', '(R1) the AUC stays', '(R2) it falls', '(R2c, R3c)',
           'the R3c models reach', 'the AUC of the R3 models', '(R5; AP', 'with 100~km blocks the AP is',
           'the AUC falls to', 'RF and bagging-PU reach an AP', 'the best AP is', '(AP {:.3f} under random splits and',
           'DevNet {:.3f} against {:.3f}, RF', 'in {} of the {} model--province pairs', '(AUC {:.2f} for DevNet and',
           'held-out Superior reaches an AP of at most {:.3f} and', 'lowered the AP in {} of 15', 'raised the AP in',
           "in the province's own specification", '(under 100~km blocks DevNet-BCE', '(RF {:.3f} with three',
           'DevNet stays between', 'gives an AP of {:.3f} under random splits and {:.3f} under 100~km',
           "does not raise DevNet's AP", 'place {:.0f}', "DevNet's AP doubles", 'leave DevNet unchanged',
           'DevNet stays below DevNet-BCE', "lowered Superior DevNet's blocked AP")

LAD = c5.LAD
L = c5.L
TR, lo = c5.TR, c5.lo
off5 = TR[(TR.design == 'transfer') & TR.model.isin(['DevNet', 'RF', 'XGBoost', 'BaggingPU', 'nnPU'])]
c2s = off5[(off5.train == 'Cordillera') & (off5.test == 'Superior')].set_index('model')
prev = lo.groupby('test').prevalence.first()
lo5 = lo[lo.model.isin(['DevNet', 'RF', 'XGBoost', 'BaggingPU', 'nnPU'])]
AUG = pd.read_csv(os.path.join(FS3, 'augmentation_Superior.csv'))           # RF
AUGF = pd.read_csv(os.path.join(FS3, 'augmentation_final_Superior.csv'))    # final DevNet
AUGS = pd.read_csv(os.path.join(FS3, 'augmentation_spatial_Superior.csv'))  # RF, blocks
AUGFS = pd.read_csv(os.path.join(FS3, 'augmentation_final_spatial_Superior.csv'))  # final DevNet, blocks
REG = c5.XGB
t3 = c4.t3


def aug(table, model, method=None, block=None):
    z = table[table.model == model]
    if block is not None:
        z = z[z.block_km == block]
    z = z[z.method == method] if method else z[z.method != 'none']
    return z.ap_med


def r7(held, col='ap_mean'):
    return L.loc[('R7', held), col]


def loss(model):
    base = t3(model) if model != 'XGBoost' else c5.xgb('Superior', 0)
    blocked = [t3(model, b) if model != 'XGBoost' else c5.xgb('Superior', b) for b in (50, 100, 200)]
    return [1 - v / base for v in blocked]


ABL = open(os.path.join(REPO, 'paper', 'manuscript', 'generated_revised6', 'tab_ablation.tex')).read()
z16 = float(re.search(r'& z-score & 16\\,\+\\,16 & 300 & ([0-9.]+)', ABL).group(1))

NEW = [
    ('reaches an AUC of {:.3f} and an AP of {:.3f}', lambda: [c5.lad('R5', 'auc_mean'), c5.lad('R5', 'ap_mean')]),
    ('leaves a sibling of {:.0f}\\% of the test positives in the training set, and with random negatives the AUC stays at {:.3f}; cosine-dissimilar negatives without any augmentation give {:.3f}',
     lambda: [100 * c5.lad('R0', 'sibling_in_train_mean'), c5.lad('R1', 'auc_balanced_mean'), c5.lad('R3c', 'auc_balanced_mean')]),
    ("the forest's AP at realistic prevalence falls from {:.3f} to {:.3f}", lambda: [c5.lad('R4', 'ap_mean'), c5.lad('R4c', 'ap_mean')]),
    ('in {} of the {} model--province pairs the AUC is below 0.5', lambda: [int((off5.auc_med < 0.5).sum()), len(off5)]),
    ('The exception is DevNet trained in Cordillera and applied to Superior (AUC {:.2f}, AP {:.3f} against {:.3f})',
     lambda: [c2s.loc['DevNet', 'auc_med'], c2s.loc['DevNet', 'ap_med'], c2s.loc['DevNet', 'prevalence']]),
    ('held-out Grenville is ranked at or below chance by every model (AUC {:.2f}--{:.2f}, AP $\\leq {:.3f}$, against {:.3f} for a random ranking), and held-out Superior reaches an AP of at most {:.3f} (random ranking {:.3f})',
     lambda: [lo5[lo5.test == 'Grenville'].auc_med.min(), lo5[lo5.test == 'Grenville'].auc_med.max(),
              lo5[lo5.test == 'Grenville'].ap_med.max(), prev['Grenville'], lo5[lo5.test == 'Superior'].ap_med.max(),
              prev['Superior']]),
    ('with augmentation its AP lies between {:.3f} and {:.3f}, against {:.3f} without',
     lambda: [aug(AUGF, 'DevNet').min(), aug(AUGF, 'DevNet').max(), aug(AUGF, 'DevNet', 'none').iloc[0]]),
    ('jitter raises the AP from {:.3f} to {:.3f}--{:.3f}, SMOTE lowers it ({:.3f}--{:.3f}) and WGAN-GP leaves it at or below the unaugmented value ({:.3f}--{:.3f})',
     lambda: [aug(AUG, 'RF', 'none').iloc[0], aug(AUG, 'RF', 'jitter').min(), aug(AUG, 'RF', 'jitter').max(),
              aug(AUG, 'RF', 'smote').min(), aug(AUG, 'RF', 'smote').max(), aug(AUG, 'RF', 'wgan').min(),
              aug(AUG, 'RF', 'wgan').max()]),
    ("raises RF's AP with 100~km blocks from {:.3f} to {:.3f}", lambda: [aug(AUGS, 'RF', 'none', 100).iloc[0],
                                                                         aug(AUGS, 'RF', 'wgan', 100).max()]),
    ("every generator lowers RF's AP ({:.3f}--{:.3f} against {:.3f})",
     lambda: [aug(AUGS, 'RF', block=200).min(), aug(AUGS, 'RF', block=200).max(), aug(AUGS, 'RF', 'none', 200).iloc[0]]),
    ("DevNet's AP under blocking changes little ({:.3f}--{:.3f} with 100~km and {:.3f}--{:.3f} with 200~km blocks)",
     lambda: [aug(AUGFS, 'DevNet', block=100).min(), aug(AUGFS, 'DevNet', block=100).max(),
              aug(AUGFS, 'DevNet', block=200).min(), aug(AUGFS, 'DevNet', block=200).max()]),
    ('lose {:.0f}--{:.0f}\\% of their AP', lambda: [100 * min(min(loss(m)) for m in ('RF', 'XGBoost', 'BaggingPU')),
                                                     100 * max(max(loss(m)) for m in ('RF', 'XGBoost', 'BaggingPU'))]),
    ("Together they raise DevNet's random-split AP in Superior from {:.3f} to {:.3f}", lambda: [z16, t3('DevNet')]),
]
CLAIMS = [c for c in c5.CLAIMS if not any(c[0].startswith(d) or d in c[0] for d in DROPPED)] + NEW


def main():
    tex = c4.norm(open(TEX).read())
    bad = 0
    for template, f in CLAIMS:
        phrase = c4.norm(template.format(*f()))
        ok = phrase in tex
        bad += not ok
        print(('OK   ' if ok else 'FAIL ') + phrase)
    print(f'\n{len(CLAIMS) - bad}/{len(CLAIMS)} claims match')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
