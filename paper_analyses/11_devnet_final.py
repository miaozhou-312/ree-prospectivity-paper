"""
Stage 11 -- the final DevNet configuration ("DevNet" in the revised
manuscript; the z-scored configuration of Stages 01-02 is its ablation
baseline).

DEFINITION (fixed before this script was first run, after the Stage 8-10
diagnostics, and not changed afterwards):
  * inputs: median imputation, then QuantileTransformer to standard-normal
    marginals (1,000 quantiles), fit on each training fold;
  * training: background-heavy batches of 16 positives + 240 background
    rows, 300 steps, cosine schedule over 300 steps;
  * everything else as the z-scored DevNet of Stages 01-02 (one hidden
    layer, region capacity, deviation loss with margin 5, Adam, five-restart
    rank ensemble).

Evaluated once under the full protocol (random 5x6 CV, 50 / 100 / 200 km
blocks; identical folds), with paired comparisons against RF, bagging-PU,
nnPU and the z-scored DevNet from the Stage-1/2 stores (nnPU with the same
quantile inputs is compared in Stage 14).

    nice -n 10 python -m paper_analyses.11_devnet_final
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

from paper_analyses.common import (REGION_ORDER, config, load_pickle, load_region, model_cfg, out_path,
                                   save_pickle, seed_everything)
from paper_analyses.protocol import paired_bootstrap, project_xy, repeated_random_cv, repeated_spatial_cv

_tb = importlib.import_module('paper_analyses.10_training_budget')
N_POS_BATCH, N_BG_BATCH = 16, 240
BLOCKS = [50.0, 100.0, 200.0]


class QuantileNormal:
    """Median imputation + quantile-normal marginals, fit on training rows."""
    def __init__(self, X_train):
        self.imp = SimpleImputer(strategy='median').fit(X_train)
        self.qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000,
                                      random_state=config.RANDOM_SEED).fit(self.imp.transform(X_train))

    def __call__(self, X):
        return self.qt.transform(self.imp.transform(X))


def fit_final(X_pos_raw, X_bg_raw, region, seed_base, margin=None):
    """Train the final ensemble on raw positives + background; returns
    (transform, models)."""
    X_train = np.vstack([X_pos_raw, X_bg_raw])
    y_train = np.r_[np.ones(len(X_pos_raw)), np.zeros(len(X_bg_raw))]
    tf = QuantileNormal(X_train)
    Xt = tf(X_train)
    models = [_tb.train(Xt, y_train, seed_base + k, model_cfg(region), N_POS_BATCH, N_BG_BATCH,
                        t_max=config.DN_EPOCHS, stop_at=config.DN_EPOCHS, margin=margin)[0]
              for k in range(config.DN_RESTARTS)]
    return tf, models


def final_factory(region, margin=None):
    def make(X_bg_raw):
        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            tf, models = fit_final(X_tr_raw[y_tr == 1], X_bg_raw, region, config.DN_SEED + seed_counter * 1000,
                                margin=margin)
            return _tb._ensemble(models, tf(X_va_raw))
        return scorer
    return make


def main():
    bench = load_pickle(out_path('benchmark_Superior_Cordillera_Grenville.pkl'))
    spat = load_pickle(out_path('spatial_cv_Superior_Cordillera_Grenville.pkl'))
    rows, store = [], {}
    for region in REGION_ORDER:
        rec = load_region(region)
        d = rec['pruned']
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        for proto in ['random'] + BLOCKS:
            seed_everything()
            if proto == 'random':
                res = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], final_factory(region),
                                         config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
                refs = {m: bench[(region, 'pruned', m)] for m in ['RF', 'DevNet', 'BaggingPU', 'nnPU']}
            else:
                res = repeated_spatial_cv(d['X_lab'], d['y'], xy_lab, d['X_unl'], xy_bg, final_factory(region),
                                          block_km=proto, n_splits=config.CV_N_SPLITS,
                                          n_repeats=config.CV_N_REPEATS, seed=config.RANDOM_SEED)
                refs = {m: spat[(region, proto, m)] for m in ['RF', 'DevNet', 'BaggingPU', 'nnPU']}
            store[(region, proto)] = res
            row = {'region': region, 'protocol': 'random' if proto == 'random' else f'{proto:.0f} km',
                   'block_km': 0 if proto == 'random' else proto,
                   'auc_med': res['auc_med'], 'auc_lo': res['auc_ci'][0], 'auc_hi': res['auc_ci'][1],
                   'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1]}
            msg = []
            for m, ref in refs.items():
                bt = paired_bootstrap(d['y'], res['oof_mean'], ref['oof_mean'], seed=config.RANDOM_SEED)
                dd = res['aps'] - ref['aps']
                row.update({f'd_ap_vs_{m}': bt['d_ap'], f'd_ap_vs_{m}_lo': bt['d_ap_ci'][0],
                            f'd_ap_vs_{m}_hi': bt['d_ap_ci'][1], f'p_vs_{m}': bt['p_ap'],
                            f'd_auc_vs_{m}': bt['d_auc'], f'p_auc_vs_{m}': bt['p_auc'],
                            f'wins_vs_{m}': f'{int((dd > 0).sum())}/{len(dd)}'})
                msg.append(f"{m} {bt['d_ap']:+.3f} p={bt['p_ap']:.3f} {row[f'wins_vs_{m}']}")
            rows.append(row)
            print(f"[{region} | {row['protocol']}] DevNet AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f} "
                  f"[{res['ap_ci'][0]:.3f},{res['ap_ci'][1]:.3f}] | vs " + ' | '.join(msg), flush=True)
    pd.DataFrame(rows).to_csv(out_path('devnet_final.csv'), index=False)
    save_pickle(store, out_path('devnet_final.pkl'))


if __name__ == '__main__':
    main()
