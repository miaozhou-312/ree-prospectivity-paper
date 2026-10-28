"""
Stage 47 -- XGBoost in the three provinces, on the folds of the main
results: feature set 3 (and "without point-level map classes", Stage 25's
'extended' set), random 5x6 CV and 50 / 100 / 200 km blocks. Parameters are
fixed (paper_analyses.models.XGB_PARAMS, from J. Fang's national study);
the national runs use the same scorer (Stages 42-44).

Output out/xgboost_regional.csv (columns as the Stage-25 benchmark) and
out/xgboost_regional.pkl (out-of-fold scores).

    python -m paper_analyses.47_xgboost_regional
"""
import argparse
import importlib

import pandas as pd

from paper_analyses.common import REGION_ORDER, config, out_path, save_pickle, seed_everything
from paper_analyses.models import xgb_factory
from paper_analyses.protocol import project_xy, run_protocol

_s25 = importlib.import_module('paper_analyses.25_extended_benchmark')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    ap.add_argument('--featuresets', nargs='+', default=['reencoded', 'extended'])
    args = ap.parse_args()
    rows, store = [], {}
    for fs in args.featuresets:
        for region in args.regions:
            rec, cols, X_lab, X_unl, _ = _s25.extended_matrices(region, fs)
            y = rec['pruned']['y']
            xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
            xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
            for proto in _s25.PROTOCOLS:
                seed_everything()
                res = run_protocol(proto, X_lab, y, xy_lab, X_unl, xy_bg, xgb_factory(region), config.CV_N_SPLITS,
                                   config.CV_N_REPEATS, config.RANDOM_SEED)
                pname = 'random' if proto == 'random' else f'{proto:.0f} km'
                store[(fs, region, pname)] = res
                rows.append({'region': region, 'model': 'XGBoost', 'protocol': pname,
                             'block_km': 0 if proto == 'random' else proto, 'keep': 0, 'featureset': fs,
                             'n_features': len(cols), 'auc_med': res['auc_med'], 'auc_lo': res['auc_ci'][0],
                             'auc_hi': res['auc_ci'][1], 'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0],
                             'ap_hi': res['ap_ci'][1]})
                print(f"[{fs} | {region} | {pname}] XGBoost AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f}",
                      flush=True)
    pd.DataFrame(rows).to_csv(out_path('xgboost_regional.csv'), index=False)
    save_pickle(store, out_path('xgboost_regional.pkl'))


if __name__ == '__main__':
    main()
