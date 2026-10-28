"""
Stage 28 -- tuned tree controls (Stage 17) on feature set 3.

RF-tuned and HGB-tuned choose their hyperparameters inside every outer
training fold by inner 100 km spatial CV (paper_analyses/nested.py), exactly
as in Stage 17, but on feature set 3 (re-encoded H3 geology + extended
layers, Stage 25 --featureset reencoded). MLP-BCE on set 3 is part of
Stage 25.

    nice -n 10 python -m paper_analyses.28_controls_fs3 [--regions ...]
"""
import argparse
import importlib
from collections import Counter

import numpy as np
import pandas as pd

from paper_analyses.common import REGION_ORDER, config, load_pickle, out_path, save_pickle, seed_everything
from paper_analyses.nested import nested_factory
from paper_analyses.protocol import paired_bootstrap, project_xy, run_protocol

_c17 = importlib.import_module('paper_analyses.17_controls')
_s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
PROTOCOLS = ['random', 50.0, 100.0, 200.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    args = ap.parse_args()
    rows, store = [], {}
    for region in args.regions:
        rec, cols, X_lab, X_unl, _ = _s25.extended_matrices(region, 'reencoded')
        y = rec['pruned']['y']
        n = X_lab.shape[1]
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        XL, XB = np.hstack([X_lab, xy_lab]), np.hstack([X_unl, xy_bg])
        dn = load_pickle(out_path(f'reencoded_benchmark_{region}.pkl'))
        for model, kind, grid in [('RF-tuned', 'RF', _c17.RF_GRID), ('HGB-tuned', 'HGB', _c17.HGB_GRID)]:
            for proto in PROTOCOLS:
                log = []
                fs = _c17.tree_fit_score(kind)
                seed_everything()
                res = run_protocol(proto, XL, y, xy_lab, XB, xy_bg, nested_factory(grid, fs, fs, n, log),
                                   config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
                store[(region, model, proto)] = {**res, 'log': log}
                ref = dn[(region, 'DevNet', proto, None)]
                bt = paired_bootstrap(y, ref['oof_mean'], res['oof_mean'], seed=config.RANDOM_SEED)
                dd = ref['aps'] - res['aps']
                pname = 'random' if proto == 'random' else f'{proto:.0f} km'
                rows.append({'region': region, 'model': model, 'protocol': pname,
                             'block_km': 0 if proto == 'random' else proto, 'auc_med': res['auc_med'],
                             'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1],
                             'devnet_ap_med': ref['ap_med'], 'd_ap_devnet_minus': bt['d_ap'],
                             'd_ap_lo': bt['d_ap_ci'][0], 'd_ap_hi': bt['d_ap_ci'][1], 'p_ap': bt['p_ap'],
                             'devnet_wins': f'{int((dd > 0).sum())}/{len(dd)}',
                             'chosen': dict(Counter(str(e['chosen']) for e in log))})
                print(f"[{region} | {pname}] {model} AP {res['ap_med']:.3f} | DevNet {ref['ap_med']:.3f}: "
                      f"dAP {bt['d_ap']:+.3f} p={bt['p_ap']:.3f} wins {rows[-1]['devnet_wins']}", flush=True)
    tag = '_'.join(args.regions)
    pd.DataFrame(rows).to_csv(out_path(f'controls_fs3_{tag}.csv'), index=False)
    save_pickle(store, out_path(f'controls_fs3_{tag}.pkl'))


if __name__ == '__main__':
    main()
