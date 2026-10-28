"""
Stage 1 -- benchmark under the thesis protocol (random repeated stratified
CV, 5 folds x 6 repeats), now with
  * AP as well as AUC reported as median and 95% interval over repeats,
  * two PU-learning baselines (bagging-PU, nnPU) next to RF and DevNet,
  * stratified paired bootstrap tests of DevNet against every other model
    on identical folds,
  * an nnPU class-prior sensitivity check (Superior).

Self-check: RF and DevNet repeated-CV AUC medians must match the stored
thesis run (notebook Section 9) to within seed/BLAS noise.

    python -m paper_analyses.01_benchmark [--regions Superior ...]
"""
import argparse
import time

import pandas as pd

from paper_analyses.common import (REGION_ORDER, config, load_region, out_path, save_pickle,
                                   seed_everything)
from paper_analyses.models import MODEL_FACTORIES, nnpu_factory
from paper_analyses.protocol import paired_bootstrap, repeated_random_cv

# Thesis/notebook Section 9 repeated-CV AUC medians (pruned set) for the self-check.
THESIS_REP_MEDIAN_AUC = {
    ('Cordillera', 'RF_fixedseed'): 0.916, ('Cordillera', 'DevNet'): 0.872,
    ('Grenville', 'RF_fixedseed'): 0.840, ('Grenville', 'DevNet'): 0.676,
    ('Superior', 'RF_fixedseed'): 0.969, ('Superior', 'DevNet'): 0.956,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    ap.add_argument('--models', nargs='+', default=list(MODEL_FACTORIES))
    ap.add_argument('--feature-sets', nargs='+', default=['pruned', 'full'])
    ap.add_argument('--priors', nargs='+', type=float, default=[0.005, 0.02, 0.05])
    args = ap.parse_args()

    rows, boot_rows, store = [], [], {}
    for region in args.regions:
        rec = load_region(region)
        for fs in args.feature_sets:
            d = rec[fs]
            for m in args.models:
                seed_everything()
                t0 = time.time()
                res = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], MODEL_FACTORIES[m](region),
                                         config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
                store[(region, fs, m)] = res
                rows.append({'region': region, 'feature_set': fs, 'model': m,
                             'n_pos': int(d['y'].sum()), 'n_features': len(d['cols']),
                             'auc_med': res['auc_med'], 'auc_lo': res['auc_ci'][0], 'auc_hi': res['auc_ci'][1],
                             'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1],
                             'minutes': (time.time() - t0) / 60})
                print(f"[{region}/{fs}] {m:9s} AUC {res['auc_med']:.3f} [{res['auc_ci'][0]:.3f},"
                      f"{res['auc_ci'][1]:.3f}]  AP {res['ap_med']:.3f} [{res['ap_ci'][0]:.3f},"
                      f"{res['ap_ci'][1]:.3f}]  ({rows[-1]['minutes']:.1f} min)", flush=True)
                ref = THESIS_REP_MEDIAN_AUC.get((region, m))
                if fs == 'pruned' and ref is not None:
                    print(f'    self-check vs thesis repeated-CV median AUC {ref:.3f}: '
                          f"diff {res['auc_med'] - ref:+.3f}", flush=True)
            if 'DevNet' in args.models:
                for m in args.models:
                    if m == 'DevNet':
                        continue
                    a, b = store[(region, fs, 'DevNet')], store[(region, fs, m)]
                    bt = paired_bootstrap(d['y'], a['oof_mean'], b['oof_mean'], seed=config.RANDOM_SEED)
                    wins = int((a['aps'] > b['aps']).sum())
                    boot_rows.append({'region': region, 'feature_set': fs, 'comparison': f'DevNet - {m}',
                                      'ap_wins_of_repeats': f'{wins}/{len(a["aps"])}', **bt})
                    print(f"    DevNet - {m}: dAP {bt['d_ap']:+.3f} [{bt['d_ap_ci'][0]:+.3f},"
                          f"{bt['d_ap_ci'][1]:+.3f}] p={bt['p_ap']:.3f} | dAUC {bt['d_auc']:+.3f} "
                          f"[{bt['d_auc_ci'][0]:+.3f},{bt['d_auc_ci'][1]:+.3f}] p={bt['p_auc']:.3f} | "
                          f'AP wins {wins}/{len(a["aps"])}', flush=True)

    if 'Superior' in args.regions and 'nnPU' in args.models:
        d = load_region('Superior')['pruned']
        for prior in args.priors:
            seed_everything()
            res = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], nnpu_factory('Superior', prior=prior),
                                     config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
            store[('Superior', 'pruned', f'nnPU(pi={prior})')] = res
            rows.append({'region': 'Superior', 'feature_set': 'pruned', 'model': f'nnPU(pi={prior})',
                         'n_pos': int(d['y'].sum()), 'n_features': len(d['cols']),
                         'auc_med': res['auc_med'], 'auc_lo': res['auc_ci'][0], 'auc_hi': res['auc_ci'][1],
                         'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1]})
            print(f"[Superior/pruned] nnPU pi={prior}: AUC {res['auc_med']:.3f}  AP {res['ap_med']:.3f}",
                  flush=True)

    tag = '_'.join(args.regions)
    pd.DataFrame(rows).to_csv(out_path(f'benchmark_{tag}.csv'), index=False)
    pd.DataFrame(boot_rows).to_csv(out_path(f'benchmark_bootstrap_{tag}.csv'), index=False)
    save_pickle(store, out_path(f'benchmark_{tag}.pkl'))
    print('saved', out_path(f'benchmark_{tag}.csv'))


if __name__ == '__main__':
    main()
