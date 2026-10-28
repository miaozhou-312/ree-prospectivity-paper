"""
Stage 16 -- nested selection of DevNet's design.

The final DevNet (Stage 11) was settled with diagnostics on the same three
provinces, so its scores may carry selection optimism. Here the choice
between the four designs of the ablation,

    (z-scored | quantile-normal inputs) x (16 + 16 | 16 + 240 rows per step),

all with 300 steps, is made inside every outer training fold by inner 100 km
spatial CV (paper_analyses/nested.py; 2 inner restarts per candidate), and
the chosen design (5 restarts) scores the outer test fold. The outer AP is
therefore an honest estimate of the procedure "pick one of these designs by
spatial CV", and its gap to the final DevNet measures the selection
optimism. When a fold chooses the final design its outer model is identical
to the final DevNet's (same seeds).

    nice -n 10 python -m paper_analyses.16_nested_selection [--regions ...]
"""
import argparse
import importlib
from collections import Counter

import numpy as np
import pandas as pd

from paper_analyses.common import (REGION_ORDER, config, load_pickle, load_region, model_cfg, out_path,
                                   save_pickle, seed_everything)
from paper_analyses.nested import nested_factory
from paper_analyses.protocol import paired_bootstrap, project_xy, run_protocol

_fin = importlib.import_module('paper_analyses.11_devnet_final')
_tb = importlib.import_module('paper_analyses.10_training_budget')
CANDIDATES = [('z', 16), ('z', 240), ('q', 16), ('q', 240)]
NAMES = {('z', 16): 'z-score, 16+16', ('z', 240): 'z-score, 16+240', ('q', 16): 'quantile, 16+16',
         ('q', 240): 'quantile, 16+240'}
INNER_RESTARTS = 2
PROTOCOLS = ['random', 50.0, 100.0, 200.0]


def devnet_fit_score(region, restarts, seed_of):
    cfg = model_cfg(region)

    def fit_score(cand, X_pos, X_bg, X_eval, seed):
        inputs, n_bg = cand
        X_train = np.vstack([X_pos, X_bg])
        y_train = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg))]
        if inputs == 'q':
            tf = _fin.QuantileNormal(X_train)
            Xt, Xe = tf(X_train), tf(X_eval)
        else:
            Xt, Xe = _tb._fit_scale(X_train, X_eval)
        models = [_tb.train(Xt, y_train, seed_of(seed, k), cfg, 16, n_bg, t_max=config.DN_EPOCHS,
                            stop_at=config.DN_EPOCHS)[0] for k in range(restarts)]
        return _tb._ensemble(models, Xe)
    return fit_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    args = ap.parse_args()
    final = load_pickle(out_path('devnet_final.pkl'))
    bench = load_pickle(out_path('benchmark_Superior_Cordillera_Grenville.pkl'))
    spat = load_pickle(out_path('spatial_cv_Superior_Cordillera_Grenville.pkl'))
    rows, store = [], {}
    for region in args.regions:
        rec = load_region(region)
        d = rec['pruned']
        n_feat = d['X_lab'].shape[1]
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        X_lab, X_bg = np.hstack([d['X_lab'], xy_lab]), np.hstack([d['X_unl'], xy_bg])
        inner = devnet_fit_score(region, INNER_RESTARTS, lambda s, k: config.DN_SEED + s + k)
        outer = devnet_fit_score(region, config.DN_RESTARTS, lambda s, k: config.DN_SEED + s * 1000 + k)
        for proto in PROTOCOLS:
            log = []
            seed_everything()
            res = run_protocol(proto, X_lab, d['y'], xy_lab, X_bg, xy_bg,
                               nested_factory(CANDIDATES, inner, outer, n_feat, log),
                               config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
            store[(region, proto)] = {**res, 'log': log}
            counts = Counter(NAMES[e['chosen']] for e in log)
            pname = 'random' if proto == 'random' else f'{proto:.0f} km'
            row = {'region': region, 'protocol': pname, 'block_km': 0 if proto == 'random' else proto,
                   'auc_med': res['auc_med'], 'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0],
                   'ap_hi': res['ap_ci'][1], 'n_outer_folds': len(log),
                   **{f'chosen: {NAMES[c]}': counts.get(NAMES[c], 0) for c in CANDIDATES}}
            ref_rf = bench[(region, 'pruned', 'RF')] if proto == 'random' else spat[(region, proto, 'RF')]
            for tag, ref in [('final', final[(region, proto)]), ('RF', ref_rf)]:
                bt = paired_bootstrap(d['y'], res['oof_mean'], ref['oof_mean'], seed=config.RANDOM_SEED)
                dd = res['aps'] - ref['aps']
                row.update({f'ref_ap_{tag}': ref['ap_med'], f'd_ap_vs_{tag}': bt['d_ap'],
                            f'd_ap_vs_{tag}_lo': bt['d_ap_ci'][0], f'd_ap_vs_{tag}_hi': bt['d_ap_ci'][1],
                            f'p_vs_{tag}': bt['p_ap'], f'wins_vs_{tag}': f'{int((dd > 0).sum())}/{len(dd)}'})
            rows.append(row)
            print(f"[{region} | {pname}] nested AP {res['ap_med']:.3f} (final {row['ref_ap_final']:.3f}, "
                  f"RF {row['ref_ap_RF']:.3f}) | vs final {row['d_ap_vs_final']:+.3f} | vs RF "
                  f"{row['d_ap_vs_RF']:+.3f} p={row['p_vs_RF']:.3f} {row['wins_vs_RF']} | chosen {dict(counts)}",
                  flush=True)
    tag = '_'.join(args.regions)
    pd.DataFrame(rows).to_csv(out_path(f'nested_selection_{tag}.csv'), index=False)
    save_pickle(store, out_path(f'nested_selection_{tag}.pkl'))


if __name__ == '__main__':
    main()
