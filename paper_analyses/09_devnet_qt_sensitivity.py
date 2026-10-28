"""
Stage 9 -- sensitivity analysis: DevNet with quantile-normal inputs.

Added AFTER the Stage-8 diagnostics, and reported as such: the main results
use the original DevNet (z-scored inputs). The variant changes one thing --
median imputation followed by QuantileTransformer(output_distribution=
'normal') instead of StandardScaler, fit on each training fold -- with the
network, loss, 300 steps and five restarts unchanged. No other setting was
tried or tuned.

It is evaluated under the full protocol of Stages 1-2 (random 5x6 CV and
50 / 100 / 200 km blocks, identical folds and seeds), with paired bootstrap
comparisons against the original DevNet and against RF, read from the
Stage-1/2 stores so the folds are identical.

    nice -n 10 python -m paper_analyses.09_devnet_qt_sensitivity
"""
import importlib

import numpy as np
import pandas as pd

from paper_analyses.common import (REGION_ORDER, config, load_pickle, load_region, out_path, save_pickle,
                                   seed_everything)
from paper_analyses.protocol import paired_bootstrap, project_xy, repeated_random_cv, repeated_spatial_cv

devnet_variant_factory = importlib.import_module('paper_analyses.08_devnet_diagnostics').devnet_variant_factory
BLOCKS = [50.0, 100.0, 200.0]


def main():
    bench = load_pickle(out_path('benchmark_Superior_Cordillera_Grenville.pkl'))
    spat = load_pickle(out_path('spatial_cv_Superior_Cordillera_Grenville.pkl'))
    rows, store = [], {}
    for region in REGION_ORDER:
        rec = load_region(region)
        d = rec['pruned']
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        fac = devnet_variant_factory(region, config.DN_EPOCHS, 'quantile')
        for proto in ['random'] + BLOCKS:
            seed_everything()
            if proto == 'random':
                res = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], fac, config.CV_N_SPLITS,
                                         config.CV_N_REPEATS, config.RANDOM_SEED)
                ref_dn, ref_rf = bench[(region, 'pruned', 'DevNet')], bench[(region, 'pruned', 'RF')]
            else:
                res = repeated_spatial_cv(d['X_lab'], d['y'], xy_lab, d['X_unl'], xy_bg, fac, block_km=proto,
                                          n_splits=config.CV_N_SPLITS, n_repeats=config.CV_N_REPEATS,
                                          seed=config.RANDOM_SEED)
                ref_dn, ref_rf = spat[(region, proto, 'DevNet')], spat[(region, proto, 'RF')]
            store[(region, proto)] = res
            row = {'region': region, 'protocol': 'random' if proto == 'random' else f'{proto:.0f} km',
                   'auc_med': res['auc_med'], 'ap_med': res['ap_med'],
                   'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1],
                   'devnet_ap_med': ref_dn['ap_med'], 'rf_ap_med': ref_rf['ap_med']}
            for tag, ref in [('dn', ref_dn), ('rf', ref_rf)]:
                bt = paired_bootstrap(d['y'], res['oof_mean'], ref['oof_mean'], seed=config.RANDOM_SEED)
                dd = res['aps'] - ref['aps']
                row.update({f'd_ap_vs_{tag}': bt['d_ap'], f'd_ap_vs_{tag}_lo': bt['d_ap_ci'][0],
                            f'd_ap_vs_{tag}_hi': bt['d_ap_ci'][1], f'p_vs_{tag}': bt['p_ap'],
                            f'wins_vs_{tag}': f'{int((dd > 0).sum())}/{len(dd)}'})
            rows.append(row)
            print(f"[{region} | {row['protocol']}] DevNet-QT AP {res['ap_med']:.3f} (DevNet {ref_dn['ap_med']:.3f}, "
                  f"RF {ref_rf['ap_med']:.3f}) | vs DevNet {row['d_ap_vs_dn']:+.3f} p={row['p_vs_dn']:.3f} "
                  f"wins {row['wins_vs_dn']} | vs RF {row['d_ap_vs_rf']:+.3f} p={row['p_vs_rf']:.3f} "
                  f"wins {row['wins_vs_rf']}", flush=True)
    pd.DataFrame(rows).to_csv(out_path('devnet_qt_sensitivity.csv'), index=False)
    save_pickle(store, out_path('devnet_qt_sensitivity.pkl'))


if __name__ == '__main__':
    main()
