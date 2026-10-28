"""
Stage 22 -- does the semi-supervised formulation help when labels are
scarce or the unlabelled background hides deposits?

The premise of PU and semi-supervised anomaly detection is that the
unlabelled background contains undiscovered occurrences. In every outer
training fold, all but k training positives are withheld (k = 8, 5, 3;
Superior folds hold about 12), in one of two ways:

  drop  the withheld positives are removed: fewer labels only (label
        scarcity, a learning curve);
  hide  the withheld positives are added to the unlabelled training
        background: fewer labels AND a background that contains deposits.

The same positives are withheld for every model (seeded by the fold), held-
out positives are always all real occurrences, and folds are those of the
main results. AP(hide) - AP(drop) is the cost of background contamination at
equal label count; a semi-supervised loss should pay less of it than a
supervised one. Models: DevNet, MLP-BCE (same network, inputs and batches,
supervised loss), nnPU (same network and inputs, PU risk), RF and
bagging-PU. Feature sets: all pruned features, and without the era and
lithology dummies (which mark known intrusions in Superior).

    nice -n 10 python -m paper_analyses.22_label_contamination --features all|no-terrane --protocol random|100
"""
import argparse
import importlib

import numpy as np
import pandas as pd

from paper_analyses.common import config, load_region, out_path, save_pickle, seed_everything
from paper_analyses.models import bagging_pu_factory, rf_factory
from paper_analyses.protocol import project_xy, run_protocol

_fin = importlib.import_module('paper_analyses.11_devnet_final')
_c17 = importlib.import_module('paper_analyses.17_controls')
_s14 = importlib.import_module('paper_analyses.14_final_suite')
REGION = 'Superior'
KEEP = [8, 5, 3]
FACTORIES = {'DevNet': _fin.final_factory, 'MLP-BCE': _c17.mlp_bce_factory, 'nnPU': _s14.nnpu_qt_factory,
             'RF': rf_factory, 'BaggingPU': bagging_pu_factory}


def withheld(n_pos, k, seed_counter):
    rng = np.random.default_rng(config.RANDOM_SEED + 7919 * seed_counter + k)
    return rng.permutation(n_pos)[:max(n_pos - k, 0)]


def withholding_factory(base, k, mode):
    def make(X_bg):
        def scorer(X_tr, y_tr, X_va, seed_counter=0):
            pos = np.where(y_tr == 1)[0]
            out = pos[withheld(len(pos), k, seed_counter)]
            keep = np.ones(len(y_tr), bool)
            keep[out] = False
            bg = np.vstack([X_bg, X_tr[out]]) if mode == 'hide' else X_bg
            return base(bg)(X_tr[keep], y_tr[keep], X_va, seed_counter=seed_counter)
        return scorer
    return make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', choices=['all', 'no-terrane'], default='all')
    ap.add_argument('--protocol', choices=['random', '100'], default='random')
    args = ap.parse_args()
    proto = 'random' if args.protocol == 'random' else 100.0
    rec = load_region(REGION)
    d = rec['pruned']
    j = [i for i, c in enumerate(d['cols']) if args.features == 'all' or not c.startswith(('geo_era_', 'lith_'))]
    X_lab, X_bg = d['X_lab'][:, j], d['X_unl'][:, j]
    xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
    xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
    rows, store = [], {}
    conditions = [('all', None)] + [(m, k) for k in KEEP for m in ('drop', 'hide')]
    for model, base in FACTORIES.items():
        for mode, k in conditions:
            fac = base(REGION) if mode == 'all' else withholding_factory(base(REGION), k, mode)
            seed_everything()
            res = run_protocol(proto, X_lab, d['y'], xy_lab, X_bg, xy_bg, fac,
                               config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
            store[(model, mode, k)] = res
            rows.append({'region': REGION, 'features': args.features, 'protocol': args.protocol, 'model': model,
                         'mode': mode, 'keep': k if k else 0, 'auc_med': res['auc_med'], 'ap_med': res['ap_med'],
                         'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1],
                         'aps': ' '.join(f'{a:.5f}' for a in res['aps'])})
            print(f"[{args.features} | {args.protocol}] {model} {mode} k={k}: AP {res['ap_med']:.3f} "
                  f"AUC {res['auc_med']:.3f}", flush=True)
    tag = f'{args.features}_{args.protocol}'
    pd.DataFrame(rows).to_csv(out_path(f'contamination_{tag}.csv'), index=False)
    save_pickle(store, out_path(f'contamination_{tag}.pkl'))


if __name__ == '__main__':
    main()
