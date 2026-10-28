"""
Check that numbers typed into the seventh revision match the outputs of the
public-data rerun (Stages 50-51 onwards; provincial results in
paper_analyses/out/public/, national ones in out/public/national/). The
claims are those of the sixth revision, rewritten against the new outputs,
plus the sentences the seventh revision added.

    python paper/check_claims_revised7.py
"""
import glob
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'paper_analyses', 'out', 'public')
NAT = os.path.join(OUT, 'national')
TEX = os.path.join(REPO, 'paper', 'manuscript', 'main_src_revised7.tex')
GEN = os.path.join(REPO, 'paper', 'manuscript', 'generated_revised7')
REGIONS = ('Superior', 'Cordillera', 'Grenville')
B = [50, 100, 200]


def csv(name, base=OUT):
    path = os.path.join(base, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame({'missing': [name]})


def cat(pattern):
    files = glob.glob(os.path.join(OUT, pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame({'missing': [pattern]})


T = cat('reencoded_benchmark_*.csv')
T['keep'] = T['keep'].astype(str)
X2 = cat('extended_benchmark_*.csv')
X2['keep'] = X2['keep'].astype(str)
C = cat('controls_fs3_*.csv')
XGB = csv('xgboost_regional.csv')
XGB = XGB[XGB.featureset == 'reencoded'] if 'featureset' in XGB else XGB
EXT = csv('external_check_fs3.csv').iloc[0]
EXO = csv('external_check_fs3_others.csv').set_index('model')  # Stage 53
GRID = csv('superior_grid_scores_fs3.csv')  # Stage 29
AUG = csv('augmentation_Superior.csv')
AUGF = csv('augmentation_final_Superior.csv')
AUGS = csv('augmentation_spatial_Superior.csv')
AUGFS = csv('augmentation_final_spatial_Superior.csv')
AGD = csv('augmentation_diagnostics_Superior.csv')
AGD = AGD.set_index('source') if 'source' in AGD else AGD
FS = csv('final_suite.csv') if os.path.exists(os.path.join(OUT, 'final_suite.csv')) else cat('partial14_*.csv')
LAD = csv('tab_ladder.csv', NAT)
L = {m: LAD[LAD.model == m].drop_duplicates('rung').set_index('rung') for m in ('RF-Parsa', 'CNN-Parsa')}
TR = csv('tab_transfer.csv', NAT)
TABLE = pd.read_pickle(os.path.join(REPO, '..', 'work_public', 'national', 'national_table.pkl'))
NT = TABLE['pts'][TABLE['pts'].in_national]
_dg = os.path.join(REPO, '..', 'work_public', 'gis_export', 'national_s2', 'national_s2_l12_diagnostics.csv')
DG = pd.read_csv(_dg).set_index('metric')['value'] if os.path.exists(_dg) else pd.Series(dtype=float)  # Stage 52
M5 = ['DevNet', 'RF', 'XGBoost', 'BaggingPU', 'nnPU']
off5 = TR[(TR.design == 'transfer') & TR.model.isin(M5)]
c2s = off5[(off5.train == 'Cordillera') & (off5.test == 'Superior')].set_index('model')
lo5 = TR[(TR.design == 'leave-one-province-out') & TR.model.isin(M5)]
prev = lo5.groupby('test').prevalence.first()


def ordinal(v):
    """One decimal with the English ordinal suffix of its last digit (96.6th, 71.3rd)."""
    s = f'{v:.1f}'
    return s + {'1': 'st', '2': 'nd', '3': 'rd'}.get(s[-1], 'th')


def t3(model, block=0, metric='ap_med', region='Superior', keep='0', table=T):
    z = table[(table.model == model) & (table.block_km == block) & (table.region == region) & (table['keep'] == keep)]
    return z.iloc[0][metric]


def xgb(region, block, col='ap_med'):
    return XGB[(XGB.region == region) & (XGB.block_km == block)].iloc[0][col]


def tuned(model, block, region='Superior'):
    return C[(C.region == region) & (C.model == model) & (C.block_km == block)].iloc[0].ap_med


def lad(r, col='auc_balanced_mean', model='RF-Parsa'):
    return L[model].loc[r, col]


def cnn(r, col='auc_balanced_mean'):
    return lad(r, col, 'CNN-Parsa')


def both(r, col='auc_balanced_mean'):
    """The value shared by RF and CNN at three decimals (NaN if they differ)."""
    a, b = round(lad(r, col), 3), round(cnn(r, col), 3)
    return a if a == b else float('nan')


def aug(table, model, method=None, block=None):
    z = table[table.model == model]
    if block is not None:
        z = z[z.block_km == block]
    z = z[z.method == method] if method else z[z.method != 'none']
    return z.ap_med


def loss(model):
    base = t3(model) if model != 'XGBoost' else xgb('Superior', 0)
    blocked = [t3(model, b) if model != 'XGBoost' else xgb('Superior', b) for b in B]
    return [1 - v / base for v in blocked]


VARIANTS = ['z-score & 16\\,+\\,240 & 300', 'quantile & 16\\,+\\,16 & 300', 'z-score & 16\\,+\\,16 & 300',
            'z-score & 16\\,+\\,16 & 3,000', 'z-score & 16\\,+\\,16 & nested ES']


def abl(config, col, region='Superior'):
    """AP of an ablation row (Superior block of tab_ablation.tex); col 0-3 = random, 50, 100, 200 km."""
    block = ABL.split('\\midrule')[0] if region == 'Superior' else ''
    for line in block.splitlines():
        if config in line:
            cells = [c.strip().strip('\\\\ ').replace('\\textbf{', '').replace('}', '') for c in line.split('&')[-4:]]
            return float(cells[col])
    raise KeyError(config)


ABL = open(os.path.join(GEN, 'tab_ablation.tex')).read() if os.path.exists(os.path.join(GEN, 'tab_ablation.tex')) else ''
_z = re.search(r'& z-score & 16\\,\+\\,16 & 300 & ([0-9.]+)', ABL)
z16 = float(_z.group(1)) if _z else float('nan')

CLAIMS = [
    ('DevNet has the highest median AP ({:.3f}) and RF the highest AUC ({:.3f})',
     lambda: [t3('DevNet'), t3('RF', metric='auc_med')]),
    ("DevNet's AP is {:.3f}, {:.3f} and {:.3f} with 50, 100 and 200~km blocks.", lambda: [t3('DevNet', b) for b in B]),
    ("Removing the alkalic-intrusion column lowers DevNet's AP to {:.3f}, {:.3f}, {:.3f} and {:.3f}",
     lambda: [t3('DevNet', b, keep='no-alkalic') for b in [0] + B]),
    ("and RF's to {:.3f}, {:.3f}, {:.3f} and {:.3f}", lambda: [t3('RF', b, keep='no-alkalic') for b in [0] + B]),
    ("DevNet's falls to {:.3f} under random splits and {:.3f}, {:.3f} and {:.3f} under blocks",
     lambda: [t3('DevNet', b, table=X2) for b in [0] + B]),
    ("RF's to {:.3f} and {:.3f}, {:.3f} and {:.3f}", lambda: [t3('RF', b, table=X2) for b in [0] + B]),
    ('The best augmented RF (WGAN-GP $10\\times$, AP {:.3f}) remains below DevNet without augmentation ({:.3f})',
     lambda: [AUG[(AUG.model == 'RF') & (AUG.method == 'wgan') & (AUG['mult'] == 10)].iloc[0].ap_med, t3('DevNet')]),
    ('For RF, jitter changes the AP from {:.3f} to {:.3f}--{:.3f}, SMOTE lowers it ({:.3f}--{:.3f}) and WGAN-GP raises it '
     '({:.3f}--{:.3f})',
     lambda: [aug(AUG, 'RF', 'none').iloc[0], aug(AUG, 'RF', 'jitter').min(), aug(AUG, 'RF', 'jitter').max(),
              aug(AUG, 'RF', 'smote').min(), aug(AUG, 'RF', 'smote').max(), aug(AUG, 'RF', 'wgan').min(),
              aug(AUG, 'RF', 'wgan').max()]),
    ('but for {:.0f}\\% of them the nearest labelled point is a background point',
     lambda: [100 * (1 - AGD.loc['wgan'].share_nn_is_positive)]),
    ('(Superior {:.3f} with $a = 2$ against {:.3f} with $a = 5$)',
     lambda: [FS[(FS.region == 'Superior') & (FS.model == 'DevNet(a=2)') & (FS.block_km == 0)].iloc[0].ap_med,
              t3('DevNet')]),
    ('at the {} (DevNet), {} (RF), {} (XGBoost), {} (bagging-PU) and {} (nnPU) percentile',
     lambda: [ordinal(100 * v) for v in [EXT.devnet_percentile, EXT.rf_percentile]
              + [EXO.loc[m, 'clay_howells_percentile'] for m in ('XGBoost', 'bagging-PU', 'nnPU')]]),
    ('(Spearman correlation with DevNet over the grid {:.2f} and {:.2f}), whereas nnPU',
     lambda: [EXO.loc[m, 'spearman_with_devnet'] for m in ('XGBoost', 'bagging-PU')]),
    ('away from the occurrences ({:.2f})', lambda: [EXO.loc['nnPU', 'spearman_with_devnet']]),
    ('RF assigns a probability of exactly zero to {:.0f}\\% of the grid', lambda: [100 * (GRID.rf == 0).mean()]),
    ('gives a probability of exactly zero to {:.0f}\\% of the prediction grid', lambda: [100 * (GRID.rf == 0).mean()]),
    ("RF's falls from {:.3f} under random splits to {:.3f}, {:.3f} and {:.3f}, XGBoost's from {:.3f} to {:.3f}, {:.3f} "
     "and {:.3f}, and bagging-PU's to {:.3f}, {:.3f} and {:.3f}.",
     lambda: [t3('RF')] + [t3('RF', b) for b in B] + [xgb('Superior', 0)] + [xgb('Superior', b) for b in B]
     + [t3('BaggingPU', b) for b in B]),
    ("DevNet's AP exceeds RF's, XGBoost's ({:.3f}),", lambda: [xgb('Superior', 0)]),
    ('The national table holds {} occurrences, {:,} evaluation-background and {:,} training-background points '
     '(prevalence {:.2f}\\%',
     lambda: [int((NT.pool == 'positive').sum()), int((NT.pool == 'eval_neg').sum()),
              int((NT.pool == 'train_unlabeled').sum()),
              100 * (NT.pool == 'positive').sum() / NT.pool.isin(['positive', 'eval_neg']).sum()]),
    ('national models use the {} inputs pruned on the national table', lambda: [len(TABLE['cols3'])]),
    ('the design of \\citet{{Parsa2024}} gives a test AUC of {:.3f} with both its random forest and its CNN (Table',
     lambda: [both('R0')]),
    ('the earlier design yields a test AUC of {:.3f} with both of its classifiers, which falls to {:.3f} (AP {:.3f}) '
     'for its random forest', lambda: [both('R0'), lad('R5', 'auc_mean'), lad('R5', 'ap_mean')]),
    ('the tree ensembles reach {:.3f}--{:.3f}',
     lambda: list(off5[off5.model.isin(['RF', 'XGBoost', 'BaggingPU'])].ap_med.agg(['min', 'max']))),
    ('the forest reaches an AUC of {:.3f} and an AP of {:.3f}, and the CNN an AUC of {:.3f} and an AP of {:.3f}',
     lambda: [lad('R5', 'auc_mean'), lad('R5', 'ap_mean'), cnn('R5', 'auc_mean'), cnn('R5', 'ap_mean')]),
    ('without any augmentation they give {:.3f} (forest) and {:.3f} (CNN)', lambda: [lad('R3c'), cnn('R3c')]),
    ('leaves a sibling of {:.0f}\\% of the test positives in the training set; with random negatives this gives the '
     'forest {:.3f} and the CNN {:.3f}, against {:.3f} and {:.3f} when the four children',
     lambda: [100 * lad('R0', 'sibling_in_train_mean'), lad('R1'), cnn('R1'), lad('R2'), cnn('R2')]),
    ('the AP at realistic prevalence falls from {:.3f} to {:.3f} for the forest and from {:.3f} to {:.3f} for the CNN',
     lambda: [lad('R4', 'ap_mean'), lad('R4c', 'ap_mean'), cnn('R4', 'ap_mean'), cnn('R4c', 'ap_mean')]),
    ('which suffices on its own ({:.3f} and {:.3f} without augmentation); and splitting', lambda: [lad('R3c'), cnn('R3c')]),
    ('on the training side ({:.3f} for the forest and {:.3f} for the CNN with random negatives, against {:.3f} and '
     '{:.3f} with siblings kept together)', lambda: [lad('R1'), cnn('R1'), lad('R2'), cnn('R2')]),
    ('augmentation raises the balanced test AUC of the CNN from {:.3f} to {:.3f} under a random split but only to '
     '{:.3f} when siblings are kept together (forest: {:.3f}, {:.3f} and {:.3f})',
     lambda: [cnn('R3'), cnn('R1'), cnn('R2'), lad('R3'), lad('R1'), lad('R2')]),
    ('the published design produces a test AUC of {:.3f} with either classifier, whereas with one label per '
     'occurrence, random background and a realistic prevalence the forest reaches an AUC of {:.3f} and an AP of {:.3f} '
     'and the CNN an AUC of {:.3f} and an AP of {:.3f}',
     lambda: [both('R0'), lad('R5', 'auc_mean'), lad('R5', 'ap_mean'), cnn('R5', 'auc_mean'), cnn('R5', 'ap_mean')]),
    ('gives a test AUC of {:.3f} with both its random forest and its CNN; negatives drawn from the least similar '
     'ground suffice for it on their own ({:.3f} and {:.3f} without augmentation), and random splitting of augmented '
     'child cells alone lifts the forest to {:.3f}. With one label per occurrence, random background and a realistic '
     'prevalence, the forest reaches an AUC of {:.3f} and an AP of {:.3f} and the CNN an AUC of {:.3f} and an AP of '
     '{:.3f}', lambda: [both('R0'), lad('R3c'), cnn('R3c'), lad('R1'), lad('R5', 'auc_mean'), lad('R5', 'ap_mean'),
                       cnn('R5', 'auc_mean'), cnn('R5', 'ap_mean')]),
    ('in {} of the {} model--province pairs the AUC is below 0.5', lambda: [int((off5.auc_med < 0.5).sum()), len(off5)]),
    ('held-out Grenville is ranked close to chance (AUC {:.2f}--{:.2f}, AP $\\leq {:.3f}$, against '
     '{:.3f} for a random ranking), and held-out Superior reaches an AP of at most {:.3f} (random ranking {:.3f})',
     lambda: [lo5[lo5.test == 'Grenville'].auc_med.min(), lo5[lo5.test == 'Grenville'].auc_med.max(),
              lo5[lo5.test == 'Grenville'].ap_med.max(), prev['Grenville'], lo5[lo5.test == 'Superior'].ap_med.max(),
              prev['Superior']]),
    ('with augmentation its AP lies between {:.3f} and {:.3f}, against {:.3f} without',
     lambda: [aug(AUGF, 'DevNet').min(), aug(AUGF, 'DevNet').max(), aug(AUGF, 'DevNet', 'none').iloc[0]]),
    ("WGAN-GP raises RF's AP with 100~km blocks from {:.3f} to {:.3f}--{:.3f} and with 200~km blocks from {:.3f} to "
     "{:.3f}--{:.3f}, still well below unaugmented DevNet ({:.3f} and {:.3f})",
     lambda: [aug(AUGS, 'RF', 'none', 100).iloc[0], aug(AUGS, 'RF', 'wgan', 100).min(), aug(AUGS, 'RF', 'wgan', 100).max(),
              aug(AUGS, 'RF', 'none', 200).iloc[0], aug(AUGS, 'RF', 'wgan', 200).min(), aug(AUGS, 'RF', 'wgan', 200).max(),
              t3('DevNet', 100), t3('DevNet', 200)]),
    ("DevNet's AP changes little with 100~km blocks ({:.3f}--{:.3f} against {:.3f}) and varies between {:.3f} and {:.3f} "
     "with 200~km blocks (against {:.3f})",
     lambda: [aug(AUGFS, 'DevNet', block=100).min(), aug(AUGFS, 'DevNet', block=100).max(),
              aug(AUGFS, 'DevNet', 'none', 100).iloc[0], aug(AUGFS, 'DevNet', block=200).min(),
              aug(AUGFS, 'DevNet', block=200).max(), aug(AUGFS, 'DevNet', 'none', 200).iloc[0]]),
    ('lose {:.0f}--{:.0f}\\% of their AP', lambda: [100 * min(min(loss(m)) for m in ('RF', 'XGBoost', 'BaggingPU')),
                                                     100 * max(max(loss(m)) for m in ('RF', 'XGBoost', 'BaggingPU'))]),
    ('with z-scored inputs, ten times more steps raised random-split AP ({:.3f} against {:.3f})',
     lambda: [abl('z-score & 16\\,+\\,16 & 3,000', 0), abl('z-score & 16\\,+\\,16 & 300', 0)]),
    ('under random splits z-scored inputs with background-heavy batches give the highest AP ({:.3f} against {:.3f} for the '
     'final configuration)', lambda: [abl('z-score & 16\\,+\\,240 & 300', 0), t3('DevNet')]),
    ('(AP {:.3f}--{:.3f} under random splits and {:.3f}--{:.3f} with 100~km blocks, against {:.3f} and {:.3f} for RF)',
     lambda: [min(abl(c, 0) for c in VARIANTS), max(abl(c, 0) for c in VARIANTS),
              min(abl(c, 2) for c in VARIANTS), max(abl(c, 2) for c in VARIANTS), t3('RF'), t3('RF', 100)]),
    # national maps (Stage 52 diagnostics)
    ('the forest assigns a probability of 0.5 or more to {:.1f}\\% of the cells and the CNN to {:.1f}\\%',
     lambda: [100 * DG['parsa_RF: share of cells >= 0.5'], 100 * DG['parsa_CNN: share of cells >= 0.5']]),
    ('a median {:.0f}~km from the nearest occurrence against {:.0f}~km for all cells',
     lambda: [DG['parsa: median distance of the negatives to the nearest occurrence (km)'],
              DG['all cells: median distance to the nearest occurrence (km)']]),
    ('for {:.1f}\\% of 50,000 random cells the nearest training row is a positive',
     lambda: [100 * DG['parsa: share of 50,000 random cells whose nearest training row is a positive']]),
    ('gives 0.5 or more to {:.0f} cells (less than 0.1\\%), {:.0f} of them the occurrence cells themselves and the rest '
     'within {:.0f}~km of an occurrence, and {:.1f}\\% of the cells receive no positive vote',
     lambda: [DG['ours_RF: cells >= 0.5'], DG['ours_RF: occurrence cells among them'],
              -(-DG['ours_RF: largest distance of the others to an occurrence (km)'] // 1), 100 * DG['ours_RF: share of cells = 0']]),
    ('the nearest training row is a positive for only {:.1f}\\% of the random cells',
     lambda: [100 * DG['ours: share of 50,000 random cells whose nearest training row is a positive']]),
    ('the {:.1f}\\% of cells at 0.01 or more lie a median {:.0f}~km from an occurrence, {:.0f}\\% of them in',
     lambda: [100 * DG['ours_RF: share of cells >= 0.01'],
              DG['ours_RF: median distance of the cells >= 0.01 to an occurrence (km)'],
              100 * DG['ours_RF: share of the cells >= 0.01 in Superior, Cordillera or Grenville']]),
    ('{:.1f}\\% of the cells reach 0.5 and {:.1f}\\% reach 0.9, against {:.1f}\\% at 0.9',
     lambda: [100 * DG['ours_CNN: share of cells >= 0.5'], 100 * DG['ours_CNN: share of cells >= 0.9'],
              100 * DG['parsa_CNN: share of cells >= 0.9']]),
    ('like its ranking of held-out occurrences (AP {:.3f})', lambda: [cnn('R5', 'ap_mean')]),
]


def norm(s):
    return re.sub(r'\s+', ' ', s.replace('~', ' ')).strip()


def main():
    tex = norm(open(TEX).read())
    bad = 0
    for template, f in CLAIMS:
        try:
            phrase = norm(template.format(*f()))
        except Exception as e:  # noqa: BLE001
            print(f'ERR  {template[:80]}: {e}')
            bad += 1
            continue
        ok = phrase in tex
        bad += not ok
        print(('OK   ' if ok else 'FAIL ') + phrase)
    print(f'\n{len(CLAIMS) - bad}/{len(CLAIMS)} claims match')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
