"""
Stage 25 -- benchmark on the extended, map-lookup-free feature set.

Feature set "extended": the pruned geophysical and structural features
(terrane dummies removed, Stage 18) plus the Stage-24 layers (distances to
the generalised regional geology, neighbourhood statistics, and
radiometrics when present), re-pruned with the same label-blind Spearman
rule (|rho| > 0.7). Models: DevNet, MLP-BCE, nnPU (quantile inputs), RF,
bagging-PU; random 5x6 CV and 50 / 100 / 200 km blocks on the folds of the
main results; plus the label-scarcity condition of Stage 22 (k = 3 kept
training positives) for DevNet and MLP-BCE.

--featureset reencoded (feature set 3) adds the raw-H3 geology encoding of
Stage 26 (one-hot lithology and era, contact flags) to the extended set; it
also reruns DevNet and RF without the alkalic-intrusion column, to show how
much of the score that single class carries.

    nice -n 10 python -m paper_analyses.25_extended_benchmark [--regions ...] [--featureset extended|reencoded]
"""
import argparse
import importlib

import numpy as np
import pandas as pd

from paper_analyses.common import (REGION_ORDER, config, load_pickle, load_region, out_path, save_pickle,
                                   seed_everything)
from paper_analyses.models import bagging_pu_factory, rf_factory
from paper_analyses.protocol import paired_bootstrap, project_xy, run_protocol
from src.features.collinearity import prune_collinear

_fin = importlib.import_module('paper_analyses.11_devnet_final')
_c17 = importlib.import_module('paper_analyses.17_controls')
_s14 = importlib.import_module('paper_analyses.14_final_suite')
_s22 = importlib.import_module('paper_analyses.22_label_contamination')
PROTOCOLS = ['random', 50.0, 100.0, 200.0]
FACTORIES = {'DevNet': _fin.final_factory, 'MLP-BCE': _c17.mlp_bce_factory, 'nnPU': _s14.nnpu_qt_factory,
             'RF': rf_factory, 'BaggingPU': bagging_pu_factory}


def extended_matrices(region, featureset='extended'):
    rec = load_region(region)
    ext = load_pickle(out_path(f'extended_{region}.pkl'))
    if featureset == 'reencoded':
        geo = load_pickle(out_path(f'geology_raw_{region}.pkl'))
        ext = {k: pd.concat([ext[k], geo[k]], axis=1) for k in ext}
    base = [c for c in rec['pruned_cols'] if not c.startswith(('geo_era_', 'lith_'))]
    lab = pd.concat([rec['lab'][base].reset_index(drop=True), ext['lab']], axis=1)
    unl = pd.concat([rec['unl'][base].reset_index(drop=True), ext['unl']], axis=1)
    cand = base + list(ext['lab'].columns)
    cols = prune_collinear(pd.concat([lab, unl], ignore_index=True), cand, thresh=config.COLLINEARITY_THRESH)
    return rec, cols, lab[cols].values.astype(float), unl[cols].values.astype(float), ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    ap.add_argument('--featureset', choices=['extended', 'reencoded'], default='extended')
    args = ap.parse_args()
    rows, boot, store = [], [], {}
    for region in args.regions:
        rec, cols, X_lab, X_unl, ext = extended_matrices(region, args.featureset)
        y = rec['pruned']['y']
        new = [c for c in cols if c in ext['lab'].columns]
        print(f'[{region}] extended set: {len(cols)} features ({len(new)} new: {new})', flush=True)
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        conditions = [(m, p, None) for m in FACTORIES for p in PROTOCOLS] + \
                     [(m, p, 3) for m in ('DevNet', 'MLP-BCE') for p in ('random', 100.0)]
        alk = 'lith_Igneous_Intrusive_Alkalic'
        if args.featureset == 'reencoded' and alk in cols:
            conditions += [(m, p, 'no-alkalic') for m in ('DevNet', 'RF') for p in PROTOCOLS]
        for model, proto, k in conditions:
            base = FACTORIES[model](region)
            Xl, Xu = X_lab, X_unl
            if k == 'no-alkalic':
                keep = [i for i, c in enumerate(cols) if c != alk]
                Xl, Xu, fac = X_lab[:, keep], X_unl[:, keep], base
            else:
                fac = base if k is None else _s22.withholding_factory(base, k, 'drop')
            seed_everything()
            res = run_protocol(proto, Xl, y, xy_lab, Xu, xy_bg, fac, config.CV_N_SPLITS, config.CV_N_REPEATS,
                               config.RANDOM_SEED)
            store[(region, model, proto, k)] = res
            pname = 'random' if proto == 'random' else f'{proto:.0f} km'
            rows.append({'region': region, 'model': model, 'protocol': pname,
                         'block_km': 0 if proto == 'random' else proto, 'keep': k if k else 0,
                         'featureset': args.featureset,
                         'n_features': len(cols), 'auc_med': res['auc_med'], 'ap_med': res['ap_med'],
                         'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1]})
            print(f"[{region} | {pname}{'' if k is None else f' | k={k}'}] {model} AUC {res['auc_med']:.3f} "
                  f"AP {res['ap_med']:.3f}", flush=True)
        for proto in PROTOCOLS:
            a = store[(region, 'DevNet', proto, None)]
            for m in ('MLP-BCE', 'nnPU', 'RF', 'BaggingPU'):
                b = store[(region, m, proto, None)]
                bt = paired_bootstrap(y, a['oof_mean'], b['oof_mean'], seed=config.RANDOM_SEED)
                boot.append({'region': region, 'protocol': 'random' if proto == 'random' else f'{proto:.0f} km',
                             'block_km': 0 if proto == 'random' else proto, 'comparison': f'DevNet - {m}', **bt,
                             'wins': f"{int((a['aps'] > b['aps']).sum())}/{len(a['aps'])}"})
        for proto in ('random', 100.0):
            a, b = store[(region, 'DevNet', proto, 3)], store[(region, 'MLP-BCE', proto, 3)]
            bt = paired_bootstrap(y, a['oof_mean'], b['oof_mean'], seed=config.RANDOM_SEED)
            boot.append({'region': region, 'protocol': ('random' if proto == 'random' else '100 km') + ', k=3',
                         'block_km': 0 if proto == 'random' else proto, 'comparison': 'DevNet - MLP-BCE', **bt,
                         'wins': f"{int((a['aps'] > b['aps']).sum())}/{len(a['aps'])}"})
        store[(region, 'cols')] = cols
    tag = '_'.join(args.regions)
    pre = 'extended' if args.featureset == 'extended' else 'reencoded'
    pd.DataFrame(rows).to_csv(out_path(f'{pre}_benchmark_{tag}.csv'), index=False)
    pd.DataFrame(boot).to_csv(out_path(f'{pre}_bootstrap_{tag}.csv'), index=False)
    save_pickle(store, out_path(f'{pre}_benchmark_{tag}.pkl'))


if __name__ == '__main__':
    main()
