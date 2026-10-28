"""
Check that numbers typed into the fifth revision match the outputs: the
claims of the fourth revision that the fifth keeps (two sentences changed
for XGBoost), plus the national part (paper_analyses/out/national/) and
the provincial XGBoost runs (out/xgboost_regional.csv).

    python paper/check_claims_revised5.py
"""
import importlib.util
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'paper_analyses', 'out')
NAT = os.path.join(OUT, 'national')
TEX = os.path.join(REPO, 'paper', 'manuscript', 'main_src_revised5.tex')

spec = importlib.util.spec_from_file_location('c4', os.path.join(REPO, 'paper', 'check_claims_revised4.py'))
c4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c4)

nat = lambda n: pd.read_csv(os.path.join(NAT, n))  # noqa: E731
LAD = nat('tab_ladder.csv')
LAD['held'] = LAD.design.str.replace('leave one province out: ', '', regex=False)
L = LAD[LAD.model == 'RF-Parsa'].drop_duplicates(['rung', 'held']).set_index(['rung', 'held'])
BEN = nat('tab_benchmark.csv')
TR = nat('tab_transfer.csv')
POOL = nat('tab_pooling.csv')
CUR = nat('tab_curve.csv')
DV = nat('tab_devnet_variants.csv')
DV = DV[DV.scope == 'Canada'].set_index(['model', 'variant', 'protocol'])
OFF = nat('tab_devnet_offsets.csv').set_index(['model', 'protocol'])
AUD = nat('tab_feature_audit.csv')
XGB = pd.read_csv(os.path.join(OUT, 'xgboost_regional.csv'))
XGB = XGB[XGB.featureset == 'reencoded']
TABLE = pd.read_pickle(os.path.join(REPO, '..', 'work', 'national', 'national_table.pkl'))
NT = TABLE['pts'][TABLE['pts'].in_national]
M5 = ['DevNet', 'DevNet-BCE', 'RF', 'XGBoost', 'BaggingPU']


def lad(r, col, held=None):
    held = held if held is not None else (L.loc[r].index[0] if r != 'R7' else 'Superior')
    return L.loc[(r, held), col]


def ben(model, proto, scope='Canada', col='ap_med'):
    return BEN[(BEN.scope == scope) & (BEN.model == model) & (BEN.protocol == proto)].iloc[0][col]


def xgb(region, block, col='ap_med'):
    return XGB[(XGB.region == region) & (XGB.block_km == block)].iloc[0][col]


def pool_count(proto, variant, direction):
    z = POOL[(POOL.protocol == proto) & (POOL.variant == variant) & POOL.model.isin(M5)]
    return int((z.d_ap < 0).sum()) if direction == 'down' else int((z.d_ap > 0).sum())


def pool_ap(region, model, variant, proto='100 km'):
    return POOL[(POOL.region == region) & (POOL.model == model) & (POOL.protocol == proto)
                & (POOL.variant == variant)].iloc[0].ap


def audit(table):
    return AUD[(AUD.region == 'Superior') & (AUD.model == 'DevNet') & (AUD.protocol == '100 km')].iloc[0][table]


off = TR[TR.design == 'transfer']
lo = TR[TR.design == 'leave-one-province-out']
c2s = off[(off.train == 'Cordillera') & (off.test == 'Superior')].set_index('model')
sp_pooled = POOL[(POOL.region == 'Superior') & (POOL.protocol == '100 km') & (POOL.variant != 'regional')
                 & POOL.model.isin(M5)].ap.max()
B = [50, 100, 200]
t3 = c4.t3
REPLACED = ("RF's falls from {:.3f} under random splits", 'where RF ({:.3f}) and bagging-PU ({:.3f}) exceed DevNet')
CLAIMS = [c for c in c4.CLAIMS if not c[0].startswith(REPLACED)] + [
    # provincial, with XGBoost
    ("RF's falls from {:.3f} under random splits to {:.3f}, {:.3f} and {:.3f}, XGBoost's from {:.3f} to {:.3f}, {:.3f} and {:.3f}, and bagging-PU's to {:.3f}, {:.3f} and {:.3f}.",
     lambda: [t3('RF')] + [t3('RF', b) for b in B] + [xgb('Superior', 0)] + [xgb('Superior', b) for b in B]
             + [t3('BaggingPU', b) for b in B]),
    ('where RF ({:.3f}), bagging-PU ({:.3f}) and XGBoost ({:.3f}) exceed DevNet ({:.3f};',
     lambda: [t3('RF', 200, 'auc_med'), t3('BaggingPU', 200, 'auc_med'), xgb('Superior', 200, 'auc_med'),
              t3('DevNet', 200, 'auc_med')]),
    ("DevNet's AP exceeds RF's, XGBoost's ({:.3f}),", lambda: [xgb('Superior', 0)]),
    ('XGBoost leads ({:.3f}), followed by RF ({:.3f})', lambda: [xgb('Grenville', 0), t3('RF', region='Grenville')]),
    ('and XGBoost ({:.3f}) trail', lambda: [xgb('Cordillera', 0)]),
    ('XGBoost, with parameters fixed in advance, behaves like these baselines ({:.3f}, {:.3f}, {:.3f} and {:.3f};',
     lambda: [xgb('Superior', b) for b in [0] + B]),
    # national table
    ('The national table holds {} occurrences, {:,} evaluation-background and {:,} training-background points (prevalence {:.2f}\\%',
     lambda: [int((NT.pool == 'positive').sum()), int((NT.pool == 'eval_neg').sum()),
              int((NT.pool == 'train_unlabeled').sum()),
              100 * (NT.pool == 'positive').sum() / NT.pool.isin(['positive', 'eval_neg']).sum()]),
    ('national models use the {} inputs pruned on the national table', lambda: [len(TABLE['cols3'])]),
    # ladder
    ('the design of \\citet{{Parsa2024}} gives a test AUC of {:.3f} (Table', lambda: [lad('R0', 'auc_balanced_mean')]),
    ('In {:.0f}\\% of the test positives at least one sibling', lambda: [100 * lad('R0', 'sibling_in_train_mean')]),
    ('(R1) the AUC stays at {:.3f}', lambda: [lad('R1', 'auc_balanced_mean')]),
    ('(R2) it falls to {:.3f}, and with one label per occurrence cell (R3) to {:.3f}',
     lambda: [lad('R2', 'auc_balanced_mean'), lad('R3', 'auc_balanced_mean')]),
    ('(R2c, R3c) give {:.3f} and {:.3f}', lambda: [lad('R2c', 'auc_balanced_mean'), lad('R3c', 'auc_balanced_mean')]),
    ('the R3c models reach an AP of {:.3f}, the R3 models trained against random negatives {:.3f}',
     lambda: [lad('R4c', 'ap_mean'), lad('R4', 'ap_mean')]),
    ('the AUC of the R3 models is unchanged ({:.3f}), but their AP is {:.3f}', lambda: [lad('R4', 'auc_mean'), lad('R4', 'ap_mean')]),
    ('(R5; AP {:.3f} for the 1,000-tree forest)', lambda: [lad('R5', 'ap_mean')]),
    ('with 100~km blocks the AP is {:.3f} and the AUC {:.3f}', lambda: [lad('R6', 'ap_mean'), lad('R6', 'auc_mean')]),
    ('the AUC falls to {:.3f}--{:.3f} and the AP for Superior to {:.3f}, about the random-ranking value of {:.3f}',
     lambda: [L.loc['R7'].auc_mean.min(), L.loc['R7'].auc_mean.max(), lad('R7', 'ap_mean', 'Superior'),
              lo.groupby('test').prevalence.first()['Superior']]),
    # national benchmark
    ('RF and bagging-PU reach an AP of {:.3f} and {:.3f} at a prevalence of {:.2f}\\%',
     lambda: [ben('RF', 'random'), ben('BaggingPU', 'random'), 100 * ben('RF', 'random', col='prevalence')]),
    ('the best AP is {:.3f} with 50~km blocks, {:.3f} with 100~km and {:.3f} with 200~km blocks',
     lambda: [BEN[(BEN.scope == 'Canada') & (BEN.protocol == p)].ap_med.max()
              for p in ('50 km', '100 km', '200 km')]),
    ('(AP {:.3f} under random splits and {:.3f} under 100~km blocks), well below its own cross-entropy ablation DevNet-BCE ({:.3f} and {:.3f})',
     lambda: [ben('DevNet', 'random'), ben('DevNet', '100 km'), ben('DevNet-BCE', 'random'), ben('DevNet-BCE', '100 km')]),
    ('DevNet {:.3f} against {:.3f}, RF {:.3f} against {:.3f} and XGBoost {:.3f} against {:.3f}',
     lambda: [ben('DevNet', '100 km', 'Superior'), t3('DevNet', 100), ben('RF', '100 km', 'Superior'), t3('RF', 100),
              ben('XGBoost', '100 km', 'Superior'), xgb('Superior', 100)]),
    # transfer
    ('the tree ensembles reach {:.3f}--{:.3f}', lambda: list(off[off.model.isin(['RF', 'XGBoost', 'BaggingPU'])].ap_med.agg(['min', 'max']))),
    ('in {} of the {} model--province pairs the AUC is below 0.5', lambda: [int((off.auc_med < 0.5).sum()), len(off)]),
    ('(AUC {:.2f} for DevNet and {:.2f} for DevNet-BCE; AP {:.3f} and {:.3f} against {:.3f})',
     lambda: [c2s.loc['DevNet', 'auc_med'], c2s.loc['DevNet-BCE', 'auc_med'], c2s.loc['DevNet', 'ap_med'],
              c2s.loc['DevNet-BCE', 'ap_med'], c2s.loc['DevNet', 'prevalence']]),
    ('held-out Superior reaches an AP of at most {:.3f} and held-out Cordillera of at most {:.3f} (RF)',
     lambda: [lo[lo.test == 'Superior'].ap_med.max(), lo[lo.test == 'Cordillera'].ap_med.max()]),
    # pooling
    ('lowered the AP in {} of 15 model--province combinations, and under random cross-validation in {} of 15; DevNet lost most (Superior {:.3f} to {:.3f})',
     lambda: [pool_count('100 km', '+background', 'down'), pool_count('random', '+background', 'down'),
              pool_ap('Superior', 'DevNet', 'regional'), pool_ap('Superior', 'DevNet', '+background')]),
    ('raised the AP in {} of 15 combinations under blocks', lambda: [pool_count('100 km', '+positives', 'up')]),
    ("in the province's own specification, lowered the AP in {} of 15", lambda: [pool_count('100 km', '+both', 'down')]),
    ('(under 100~km blocks DevNet-BCE {:.3f} and DevNet {:.3f}; best pooled {:.3f})',
     lambda: [pool_ap('Superior', 'DevNet-BCE', 'regional'), pool_ap('Superior', 'DevNet', 'regional'), sp_pooled]),
    # curve
    ('(RF {:.3f} with three, {:.3f} with ten and {:.3f} with all;',
     lambda: [CUR[(CUR.model == 'RF') & (CUR.k_train_pos == k)].iloc[0].ap_med for k in (3, 10, CUR.k_train_pos.max())]),
    ('DevNet stays between {:.3f} and {:.3f} at every count', lambda: list(CUR[CUR.model == 'DevNet'].ap_med.agg(['min', 'max']))),
    # why DevNet is weak
    ('gives an AP of {:.3f} under random splits and {:.3f} under 100~km blocks, against {:.3f} and {:.3f}',
     lambda: [ben('DevNet-scaled', 'random'), ben('DevNet-scaled', '100 km'), ben('DevNet', 'random'), ben('DevNet', '100 km')]),
    ("does not raise DevNet's AP ({:.3f})", lambda: [OFF.loc[('DevNet', 'random'), 'ap_within_group']]),
    ('place {:.0f}\\% of its national top 10\\% of background in Grenville and Cordillera, against {:.0f}\\% for RF',
     lambda: [100 * (OFF.loc[(m, 'random'), 'bg_top10_Grenville'] + OFF.loc[(m, 'random'), 'bg_top10_Cordillera'])
              for m in ('DevNet', 'RF')]),
    ("DevNet's AP doubles ({:.3f} and {:.3f}), a larger relative gain than RF's ({:.3f} and {:.3f}",
     lambda: [DV.loc[(m, 'local reference', p), 'ap_med'] for m in ('DevNet', 'RF') for p in ('random', '100 km')]),
    ('leave DevNet unchanged ({:.3f} and {:.3f}) while they help DevNet-BCE ({:.3f} and {:.3f})',
     lambda: [DV.loc[(m, 'province one-hot', p), 'ap_med'] for m in ('DevNet', 'DevNet-BCE') for p in ('random', '100 km')]),
    ('DevNet stays below DevNet-BCE ({:.3f} and {:.3f})',
     lambda: [DV.loc[('DevNet-BCE', 'local reference', p), 'ap_med'] for p in ('random', '100 km')]),
    # feature audit
    ("lowered Superior DevNet's blocked AP from {:.3f} to {:.3f} before any model was changed", lambda: [audit('A'), audit('B')]),
]


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
