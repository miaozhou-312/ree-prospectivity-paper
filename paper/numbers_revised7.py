"""
Every provincial number the seventh revision types into its text, printed
from the outputs (`old`: the sixth revision's out/ and out/fs3/, to check this
script against the sixth revision's text; default: the public-data rerun in
out/public/).

    python paper/numbers_revised7.py [old]
"""
import ast
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = len(sys.argv) > 1 and sys.argv[1] == 'old'
BASE = os.path.join(REPO, 'paper_analyses', 'out', *([] if OLD else ['public']))
FS3 = os.path.join(BASE, 'fs3') if OLD else BASE
R = ['Superior', 'Cordillera', 'Grenville']
B = [50, 100, 200]
P = [0] + B


def cat(d, pattern):
    files = glob.glob(os.path.join(d, pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


T = cat(BASE, 'reencoded_benchmark_*.csv')
T['keep'] = T['keep'].astype(str)
X2 = cat(BASE, 'extended_benchmark_*.csv')
X2['keep'] = X2['keep'].astype(str)
XGB = pd.read_csv(os.path.join(BASE, 'xgboost_regional.csv'))
C = cat(BASE, 'controls_fs3_*.csv')
BOOT = cat(BASE, 'reencoded_bootstrap_*.csv')


def t(model, block=0, region='Superior', metric='ap_med', keep='0', table=T):
    if model == 'XGBoost' and table is T:
        z = XGB[(XGB.featureset == 'reencoded') & (XGB.region == region) & (XGB.block_km == block)]
        return float(z.iloc[0][metric])
    if model == 'XGBoost':
        z = XGB[(XGB.featureset == 'extended') & (XGB.region == region) & (XGB.block_km == block)]
        return float(z.iloc[0][metric])
    z = table[(table.model == model) & (table.block_km == block) & (table.region == region) & (table['keep'] == keep)]
    return float(z.iloc[0][metric])


def f(v):
    return f'{v:.3f}'


def line(k, v):
    print(f'{k:60s} {v}')


M = ['DevNet', 'RF', 'XGBoost', 'BaggingPU', 'nnPU']
try:
    print('== Superior random CV')
    sup = {m: t(m) for m in M}
    line('AP by model', {m: f(v) for m, v in sup.items()})
    line('AUC by model', {m: f(t(m, metric='auc_med')) for m in M})
    line('AP range excl nnPU', (f(min(v for m, v in sup.items() if m != 'nnPU')), f(max(sup.values()))))
    line('AUC range excl nnPU', (f(min(t(m, metric='auc_med') for m in M if m != 'nnPU')),
                                 f(max(t(m, metric='auc_med') for m in M if m != 'nnPU'))))
    b = BOOT[(BOOT.region == 'Superior') & (BOOT.block_km == 0)]
    line('DevNet - X random: d_ap, wins', {r.comparison: (f(r.d_ap), r.wins) for _, r in b.iterrows()})
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
try:
    print('== Cordillera / Grenville random CV')
    for r in ['Cordillera', 'Grenville']:
        line(f'{r} AP', {m: f(t(m, region=r)) for m in M})
        line(f'{r} AUC', {m: f(t(m, region=r, metric='auc_med')) for m in M})
        bb = BOOT[(BOOT.region == r) & (BOOT.block_km == 0)]
        line(f'{r} DevNet - X wins', {x.comparison: x.wins for _, x in bb.iterrows()})
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
try:
    print('== Spatial blocks, Superior')
    for m in M:
        line(f'{m} AP random/50/100/200', [f(t(m, p)) for p in P])
    for p in B:
        bb = BOOT[(BOOT.region == 'Superior') & (BOOT.block_km == p)]
        line(f'{p} km DevNet - X d_ap [ci] wins', {x.comparison: (f(x.d_ap), tuple(round(v, 3) for v in ast.literal_eval(x.d_ap_ci)), x.wins)
                                                 for _, x in bb.iterrows()})
    line('AUC 200 km', {m: f(t(m, 200, metric='auc_med')) for m in M})
    for r in ['Cordillera', 'Grenville']:
        line(f'{r} max AP 50 / 100-200', (f(max(t(m, 50, r) for m in M)), f(max(t(m, p, r) for m in M for p in (100, 200)))))
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
try:
    print('== Loss of AP under blocking (RF, XGBoost, BaggingPU) and DevNet')
    for m in ['RF', 'XGBoost', 'BaggingPU', 'DevNet']:
        line(m, [f'{100 * (1 - t(m, p) / t(m)):.0f}%' for p in B])
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
try:
    print('== Map classes')
    line('DevNet no-alkalic', [f(t('DevNet', p, keep='no-alkalic')) for p in P])
    line('RF no-alkalic', [f(t('RF', p, keep='no-alkalic')) for p in P])
    for m in M:
        line(f'{m} without map classes', [f(t(m, p, table=X2)) for p in P])
    line('share of DevNet blocked AP lost without alkalic',
         [f'{100 * (1 - t("DevNet", p, keep="no-alkalic") / t("DevNet", p)):.0f}%' for p in B])
    line('DevNet no-alkalic vs RF full (blocked)', [(f(t('DevNet', p, keep='no-alkalic')), f(t('RF', p))) for p in B])
    pr = {'Superior': 15 / 3015}
    line('without map classes DevNet blocked / prevalence', [f'{t("DevNet", p, table=X2) / pr["Superior"]:.0f}x' for p in B])
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
try:
    print('== Tuned')
    for r in R:
        for m in ['RF-tuned', 'HGB-tuned']:
            line(f'{r} {m}', [f(C[(C.region == r) & (C.model == m) & (C.block_km == p)].iloc[0].ap_med) for p in P])
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
try:
    print('== Geography-only and logistic')
    for r in R:
        d = pd.read_csv(os.path.join(FS3, f'devnet_diagnostics_{r}.csv'))
        for m in ['RF_xy', 'LogReg']:
            z = d[d.model == m].set_index('protocol')
            line(f'{r} {m} AP/AUC random, 100 km', [f(z.loc['random', 'ap_med']), f(z.loc['random', 'auc_med']),
                                                     f(z.loc['100 km', 'ap_med']), f(z.loc['100 km', 'auc_med'])])
        line(f'{r} best evidence model random (DevNet,RF,XGB,BPU,nnPU,tuned)',
             f(max([t(m, region=r) for m in M] + [C[(C.region == r) & (C.block_km == 0)].ap_med.max()])))
        line(f'{r} max evidence AP 100 km', f(max([t(m, 100, r) for m in M])))
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
try:
    print('== Augmentation')
    AG = pd.read_csv(os.path.join(FS3, 'augmentation_Superior.csv'))
    AGF = pd.read_csv(os.path.join(FS3, 'augmentation_final_Superior.csv'))
    AGS = pd.read_csv(os.path.join(FS3, 'augmentation_spatial_Superior.csv'))
    AGFS = pd.read_csv(os.path.join(FS3, 'augmentation_final_spatial_Superior.csv'))
    AGD = pd.read_csv(os.path.join(FS3, 'augmentation_diagnostics_Superior.csv'))
    print(AGD.round(3).to_string(index=False))
    for name, d, m in [('RF', AG, 'RF'), ('DevNet', AGF, 'DevNet')]:
        z = d[d.model == m]
        line(f'{name} none', f(z[z.method == 'none'].ap_med.iloc[0]))
        for meth in ['jitter', 'smote', 'wgan']:
            q = z[z.method == meth].sort_values('mult')
            line(f'  {name} {meth} 1/3/10', [f(v) for v in q.ap_med])
    for name, d, m in [('RF', AGS, 'RF'), ('DevNet', AGFS, 'DevNet')]:
        for blk in (100.0, 200.0):
            z = d[(d.model == m) & (d.block_km == blk)]
            line(f'{name} {blk:.0f} km', {f'{x.method}{int(x.mult)}': f(x.ap_med) for _, x in z.iterrows()})
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
try:
    print('== nnPU prior, margin, ablation')
    FS = pd.read_csv(os.path.join(FS3, 'final_suite.csv'))
    s = FS[(FS.region == 'Superior') & (FS.block_km == 0) & (FS.feature_set == 'pruned')]
    line('nnPU prior APs', {x.model: f(x.ap_med) for _, x in s.iterrows() if x.model.startswith('nnPU')})
    line('margin', {x.model: f(x.ap_med) for _, x in s.iterrows() if x.model.startswith('DevNet(a=')})
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
try:
    print('== Capture rates (Superior, median over repeats)')
    st = pd.read_pickle(os.path.join(BASE, 'reencoded_benchmark_Superior.pkl'))
    sys.path.insert(0, REPO)
    y = None
    for k, v in st.items():
        if k[1] == 'DevNet':
            y = v.get('y')
            break
    if y is None:
        os.environ.setdefault('REE_PAPER_WORK', '../work' if OLD else '../work_public')
        from paper_analyses.common import load_region  # noqa: E402
        y = load_region('Superior')['pruned']['y']
    for proto in ['random', 50.0, 100.0, 200.0]:
        row = {}
        for m in ['DevNet', 'RF', 'BaggingPU']:
            caps = []
            for oof in st[('Superior', m, proto, None)]['oof']:
                order = np.argsort(-oof)
                cum = np.cumsum(y[order]) / y.sum()
                frac = np.arange(1, len(oof) + 1) / len(oof)
                caps.append([np.interp(q, frac, cum) for q in (0.01, 0.10)])
            caps = np.median(np.array(caps), 0)
            row[m] = (f'{100 * caps[0]:.0f}%', f'{100 * caps[1]:.0f}%')
        line(f'capture top1/top10 {proto}', row)
except Exception as e:  # noqa: BLE001
    print('   (skipped:', repr(e)[:120], ')')
