"""
Stage 15 -- the Stage-4 augmentation experiments repeated with the final
DevNet configuration (Stage 11) for the revised manuscript. RF results are
unaffected by the DevNet configuration and are taken from Stage 4.

  A  random CV (5x6), jitter / SMOTE / WGAN-GP at 1x, 3x, 10x, fold-safe;
  B  diagnostics: DevNet score percentile of 150 synthetic samples under
     the final production DevNet (the model-independent nearest-neighbour
     diagnostics are those of Stage 4);
  C  100 and 200 km spatial blocks for the Stage-4 configurations.

    nice -n 10 python -m paper_analyses.15_augmentation_final
"""
import importlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from paper_analyses import augment
from paper_analyses.common import config, load_region, out_path, save_pickle, seed_everything
from paper_analyses.protocol import paired_bootstrap, project_xy, repeated_random_cv, repeated_spatial_cv

_fin = importlib.import_module('paper_analyses.11_devnet_final')
_tb = importlib.import_module('paper_analyses.10_training_budget')
METHODS, MULTS = ['jitter', 'smote', 'wgan'], [1, 3, 10]
SPATIAL_CONFIGS = [('none', 0), ('jitter', 10), ('smote', 3), ('wgan', 3), ('wgan', 10)]
REGION = 'Superior'


def main():
    rec = load_region(REGION)
    d = rec['pruned']
    base = _fin.final_factory(REGION)
    rows, store = [], {}

    seed_everything()
    ref = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], base, config.CV_N_SPLITS, config.CV_N_REPEATS,
                             config.RANDOM_SEED)
    store[('DevNet', 'none', 0)] = ref
    rows.append({'model': 'DevNet', 'method': 'none', 'mult': 0, 'auc_med': ref['auc_med'], 'auc_lo': ref['auc_ci'][0],
                 'auc_hi': ref['auc_ci'][1], 'ap_med': ref['ap_med'], 'ap_lo': ref['ap_ci'][0], 'ap_hi': ref['ap_ci'][1]})
    print(f"[A] DevNet none: AUC {ref['auc_med']:.3f} AP {ref['ap_med']:.3f}", flush=True)
    for method in METHODS:
        for mult in MULTS:
            seed_everything()
            res = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'],
                                     augment.augmented_factory(base, method, mult, d['cols']),
                                     config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
            store[('DevNet', method, mult)] = res
            dd = res['aps'] - ref['aps']
            rows.append({'model': 'DevNet', 'method': method, 'mult': mult, 'auc_med': res['auc_med'],
                         'auc_lo': res['auc_ci'][0], 'auc_hi': res['auc_ci'][1], 'ap_med': res['ap_med'],
                         'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1],
                         'ap_wins_vs_none': f'{int((dd > 0).sum())}/{len(dd)}'})
            print(f"[A] DevNet + {method} x{mult}: AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f} "
                  f"dAP/repeat median {np.median(dd):+.3f} wins {rows[-1]['ap_wins_vs_none']}", flush=True)
    pd.DataFrame(rows).to_csv(out_path('augmentation_final_Superior.csv'), index=False)
    save_pickle(store, out_path('augmentation_final_Superior.pkl'))

    # B: DevNet percentile of synthetic samples under the final production DevNet
    X_pos, X_bg = d['X_lab'][d['y'] == 1], d['X_unl']
    tf, models = _fin.fit_final(X_pos, X_bg, REGION, config.DN_SEED)
    diag = []
    for name, X in [('real positives (LOO)', X_pos)] + \
            [(m, augment.synthesize(m, X_pos, X_bg, d['cols'], 150, seed=2024)) for m in METHODS]:
        s = _tb._ensemble(models, tf(np.vstack([X_bg, X])))
        pct = (rankdata(s) / len(s))[len(X_bg):]
        diag.append({'source': name, 'devnet_pct_med_final': float(np.median(pct))})
        print(f'[B] {name}: median DevNet percentile {np.median(pct):.3f}', flush=True)
    pd.DataFrame(diag).to_csv(out_path('augmentation_final_diag_Superior.csv'), index=False)

    # C: spatial blocks
    xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
    xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
    srows, sstore = [], {}
    for block in [100.0, 200.0]:
        ref = None
        for method, mult in SPATIAL_CONFIGS:
            seed_everything()
            fac = base if method == 'none' else augment.augmented_factory(base, method, mult, d['cols'])
            res = repeated_spatial_cv(d['X_lab'], d['y'], xy_lab, d['X_unl'], xy_bg, fac, block_km=block,
                                      n_splits=config.CV_N_SPLITS, n_repeats=config.CV_N_REPEATS,
                                      seed=config.RANDOM_SEED)
            sstore[(block, 'DevNet', method, mult)] = res
            row = {'block_km': block, 'model': 'DevNet', 'method': method, 'mult': mult, 'auc_med': res['auc_med'],
                   'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1]}
            if method == 'none':
                ref = res
            else:
                bt = paired_bootstrap(d['y'], res['oof_mean'], ref['oof_mean'], seed=config.RANDOM_SEED)
                row.update({'d_ap_vs_none': bt['d_ap'], 'd_ap_lo': bt['d_ap_ci'][0], 'd_ap_hi': bt['d_ap_ci'][1],
                            'p_ap_gain': bt['p_ap'],
                            'ap_wins_vs_none': f"{int((res['aps'] > ref['aps']).sum())}/{len(res['aps'])}"})
            srows.append(row)
            print(f"[C | {block:.0f} km] DevNet {method} x{mult}: AP {res['ap_med']:.3f}", flush=True)
    pd.DataFrame(srows).to_csv(out_path('augmentation_final_spatial_Superior.csv'), index=False)
    save_pickle(sstore, out_path('augmentation_final_spatial_Superior.pkl'))


if __name__ == '__main__':
    main()
