"""
Stage 42 -- national (Canada) experiments on feature set 3, with the
regional models, folds and metrics.

--part benchmark (N1): random 5x6 CV and 50 / 100 / 200 km blocks on the
    national table; pooled out-of-fold AUC, AP, AP lift (AP / prevalence)
    and the share of occurrences in the top 10% of scored points, overall
    and for each study province's points within the national out-of-fold
    scores.
--part transfer (N2): leave-one-province-out (train on the national table
    outside a province, test on that province: for a study province its
    positives and full regional evaluation pool) and province-to-province
    transfer between the three study provinces: province A's own model
    (its feature-set-3 columns, H3 classes and network capacity, trained on
    its positives and regional training pool) tested on province B's
    positives and regional evaluation pool; the diagonal is random 5x6 CV
    within A, i.e. the main regional result. Leave-one-province-out uses the
    national columns. 6 seeds per fit.
--part curve (N5): national random CV keeping k training positives per
    fold (Stage 22's withholding, 'drop'), k = 3, 5, 10, 20, 40 and all.

Outputs paper_analyses/out/national/<part>_<model>[_<featureset>].csv/.pkl

    python -m paper_analyses.42_national_benchmark --part benchmark --models RF [--protocols random 100]
"""
import argparse
import importlib
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from paper_analyses.common import config, model_cfg, save_pickle, seed_everything
from paper_analyses.models import bagging_pu_factory, rf_factory, xgb_factory
from paper_analyses.national import STUDY, load_national, province_matrix, regional_rows, regional_spec, summarise
from paper_analyses.protocol import repeated_random_cv, run_protocol

_fin = importlib.import_module('paper_analyses.11_devnet_final')
_c17 = importlib.import_module('paper_analyses.17_controls')
_s14 = importlib.import_module('paper_analyses.14_final_suite')
_s22 = importlib.import_module('paper_analyses.22_label_contamination')


def devnet_scaled_factory(region):
    """Sensitivity: the final DevNet with its 300 training steps scaled by
    the background size (300 x n_background / 5,000), i.e. as many passes
    over the background as in a region."""
    def make(X_bg):
        steps = int(round(config.DN_EPOCHS * len(X_bg) / config.N_TRAIN_BG))

        def scorer(X_tr, y_tr, X_va, seed_counter=0):
            X_pos = X_tr[y_tr == 1]
            X_train = np.vstack([X_pos, X_bg])
            y_train = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg))]
            tf = _fin.QuantileNormal(X_train)
            Xt = tf(X_train)
            models = [_fin._tb.train(Xt, y_train, config.DN_SEED + seed_counter * 1000 + k, model_cfg(region),
                                     _fin.N_POS_BATCH, _fin.N_BG_BATCH, t_max=steps, stop_at=steps)[0]
                      for k in range(config.DN_RESTARTS)]
            return _fin._tb._ensemble(models, tf(X_va))
        return scorer
    return make


FACTORIES = {'DevNet': _fin.final_factory, 'DevNet-BCE': _c17.mlp_bce_factory, 'nnPU': _s14.nnpu_qt_factory,
             'RF': rf_factory, 'XGBoost': xgb_factory, 'BaggingPU': bagging_pu_factory,
             'DevNet-scaled': devnet_scaled_factory}
NAME = 'Canada'
GROUPS = STUDY + ['Other']
REPS = 6
CURVE_K = [3, 5, 10, 20, 40]
OUT = os.environ.get('REE_NAT_OUT') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'national')


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def pname(proto):
    return 'random' if proto == 'random' else f'{proto:.0f} km'


def fit_reps(fac, X_pos, X_bg, X_va, reps=REPS):
    """Rank scores of `reps` fits (seed_counter 0..reps-1) trained on
    positives X_pos and background X_bg."""
    y_tr = np.ones(len(X_pos))
    return np.array([rankdata(s) / len(s) for s in
                     (fac(X_bg)(X_pos, y_tr, X_va, seed_counter=r) for r in range(reps))])


def benchmark(d, model, protocols, tag):
    rows, store = [], {}
    lab, y = d['lab'], d['y']
    for proto in protocols:
        seed_everything()
        res = run_protocol(proto, d['X_lab'], y, d['xy_lab'], d['X_unl'], d['xy_unl'], FACTORIES[model](NAME),
                           config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
        store[pname(proto)] = res
        meta = dict(model=model, protocol=pname(proto), block_km=0 if proto == 'random' else proto,
                    featureset=tag, n_features=len(d['cols']))
        rows.append(summarise(y, res['oof'], scope='Canada', **meta))
        for g in GROUPS:
            m = (lab.group == g).values
            rows.append(summarise(y[m], res['oof'][:, m], scope=g, **meta))
        r = rows[-len(GROUPS) - 1]
        print(f"[N1 | {model} | {pname(proto)}] AUC {r['auc_med']:.3f} AP {r['ap_med']:.3f} "
              f"lift {r['lift_med']:.0f} top-10% {r['cap10_med']:.2f}", flush=True)
    return rows, store


def transfer(d, model, tag):
    fac = FACTORIES[model](NAME)
    pts, cols = d['pts'], d['cols']
    rows, store = [], {}
    regional = {g: regional_rows(d['meta'], g) for g in STUDY}
    nat = pts[pts.in_national]
    for g in GROUPS:                                           # leave one province out
        if g == 'Other':
            test = nat[(nat.group == 'Other') & nat.pool.isin(['positive', 'eval_neg'])]
        else:
            test = regional[g][0]
        train_pos = nat[(nat.group != g) & (nat.pool == 'positive')]
        train_bg = nat[(nat.group != g) & (nat.pool == 'train_unlabeled')]
        S = fit_reps(fac, train_pos[cols].values.astype(float), train_bg[cols].values.astype(float),
                     test[cols].values.astype(float))
        yt = test.label.values.astype(int)
        store[('lopo', g)] = (yt, S)
        rows.append(summarise(yt, S, design='leave-one-province-out', train='Canada without ' + g, test=g,
                              model=model, featureset=tag, n_train_pos=len(train_pos)))
        r = rows[-1]
        print(f"[N2 | {model} | LOPO {g}] AUC {r['auc_med']:.3f} AP {r['ap_med']:.3f} "
              f"(prevalence {r['prevalence']:.4f}, {len(train_pos)} training positives)", flush=True)
    for a in STUDY:                                            # province-to-province transfer
        spec = regional_spec(a)
        lab_a, unl_a = regional[a]
        Xp = province_matrix(lab_a[lab_a.label == 1], spec)
        Xb = province_matrix(unl_a, spec)
        fac_a = FACTORIES[model](a)
        for b in STUDY:
            lab_b = regional[b][0]
            yb = lab_b.label.values.astype(int)
            if a == b:
                seed_everything()
                res = repeated_random_cv(province_matrix(lab_b, spec), yb, Xb, FACTORIES[model](a),
                                         config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
                S = res['oof']
            else:
                S = fit_reps(fac_a, Xp, Xb, province_matrix(lab_b, spec))
            store[('transfer', a, b)] = (yb, S)
            rows.append(summarise(yb, S, design='within (random CV)' if a == b else 'transfer', train=a, test=b,
                                  model=model, featureset=tag, n_train_pos=len(Xp)))
            r = rows[-1]
            print(f"[N2 | {model} | {a} -> {b}] AUC {r['auc_med']:.3f} AP {r['ap_med']:.3f}", flush=True)
    return rows, store


def curve(d, model, tag):
    rows, store = [], {}
    y = d['y']
    n_tr = int(y.sum() * (config.CV_N_SPLITS - 1) / config.CV_N_SPLITS)
    for k in CURVE_K + [None]:
        base = FACTORIES[model](NAME)
        fac = base if k is None else _s22.withholding_factory(base, k, 'drop')
        seed_everything()
        res = repeated_random_cv(d['X_lab'], y, d['X_unl'], fac, config.CV_N_SPLITS, config.CV_N_REPEATS,
                                 config.RANDOM_SEED)
        kk = n_tr if k is None else k
        store[kk] = res
        rows.append(summarise(y, res['oof'], scope='Canada', model=model, featureset=tag, k_train_pos=kk))
        r = rows[-1]
        print(f"[N5 | {model} | k={kk}] AUC {r['auc_med']:.3f} AP {r['ap_med']:.3f} lift {r['lift_med']:.0f}",
              flush=True)
    return rows, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['benchmark', 'transfer', 'curve'], required=True)
    ap.add_argument('--models', nargs='+', default=[m for m in FACTORIES if m != 'DevNet-scaled'])
    ap.add_argument('--protocols', nargs='+', default=['random', '50', '100', '200'])
    ap.add_argument('--featureset', choices=['set3', 'nomap'], default='set3')
    args = ap.parse_args()
    d = load_national(args.featureset)
    print(f"national table ({args.featureset}): {int(d['y'].sum())} positives, {int((d['y'] == 0).sum())} "
          f"evaluation and {len(d['X_unl'])} training background points, {len(d['cols'])} features", flush=True)
    protocols = ['random' if p == 'random' else float(p) for p in args.protocols]
    suffix = '' if args.featureset == 'set3' else '_nomap'
    for model in args.models:
        if args.part == 'benchmark':
            rows, store = benchmark(d, model, protocols, args.featureset)
            name = f"benchmark_{model}_{'_'.join(args.protocols)}{suffix}"
        elif args.part == 'transfer':
            rows, store = transfer(d, model, args.featureset)
            name = f'transfer_{model}{suffix}'
        else:
            rows, store = curve(d, model, args.featureset)
            name = f'curve_{model}{suffix}'
        pd.DataFrame(rows).to_csv(out(name + '.csv'), index=False)
        save_pickle(store, out(name + '.pkl'))


if __name__ == '__main__':
    main()
