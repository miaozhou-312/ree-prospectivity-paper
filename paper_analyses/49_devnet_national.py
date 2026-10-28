"""
Stage 49 -- why is DevNet weak at the national scale?

Hypothesis (written before the runs): DevNet scores a point by how far it
deviates from the training background, its reference of "normal". The
national background mixes provinces whose feature distributions are fully
separable (Stage 45), so the reference is not concentrated and DevNet's
scores mostly track between-province differences rather than the
occurrence signature. Scaling the training steps did not help (Stage 42,
DevNet-scaled), so under-training is ruled out.

Predictions, each against RF as the control:
  P1 offsets: DevNet's national out-of-fold scores differ systematically
     between provinces; ranking within each province (Superior, Cordillera,
     Grenville, rest) raises DevNet's national AP clearly more than RF's.
  P2 local reference: one DevNet per province group, each trained on all
     training occurrences against that group's background only and scoring
     that group's points, raises the national AP; for RF the same change
     helps less.
  P3 province indicator: adding the province group as one-hot inputs helps
     DevNet more than RF.

P1 re-scores the Stage-42 out-of-fold scores; P2 and P3 rerun the national
protocol (random 5x6 CV and 100 km blocks, the Stage-42 folds). The province
group travels as an extra last column that the scorers strip off.

    python -m paper_analyses.49_devnet_national --part offsets
    python -m paper_analyses.49_devnet_national --part models --models DevNet RF
"""
import argparse
import importlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from paper_analyses.common import config, load_pickle, seed_everything
from paper_analyses.national import STUDY, load_national, summarise
from paper_analyses.protocol import run_protocol

_b = importlib.import_module('paper_analyses.42_national_benchmark')
GROUPS = STUDY + ['Other']


def within_group_ranks(oof, groups):
    out = np.empty_like(oof)
    for g in np.unique(groups):
        m = groups == g
        out[:, m] = np.array([rankdata(o) / m.sum() for o in oof[:, m]])
    return out


def offsets():
    d = load_national('set3')
    y, grp = d['y'], d['lab'].group.values
    rows = []
    for model in ['DevNet', 'DevNet-BCE', 'RF', 'XGBoost', 'BaggingPU']:
        name = 'benchmark_RF_random' if model == 'RF' else f'benchmark_{model}_random_50_100_200'
        st = load_pickle(_b.out(f'{name}.pkl'))
        st100 = load_pickle(_b.out('benchmark_RF_100.pkl')) if model == 'RF' else st
        for proto, res in [('random', st['random']), ('100 km', st100['100 km'])]:
            oof = res['oof']
            a = summarise(y, oof)
            b = summarise(y, within_group_ranks(oof, grp))
            bg = (y == 0)
            top = np.mean([[np.mean(o[bg & (grp == g)] >= np.quantile(o[bg], 0.9)) for g in GROUPS] for o in oof], 0)
            rows.append({'model': model, 'protocol': proto, 'ap_national': a['ap_med'], 'ap_within_group': b['ap_med'],
                         'auc_national': a['auc_med'], 'auc_within_group': b['auc_med'],
                         **{f'bg_top10_{g}': float(t) for g, t in zip(GROUPS, top)}})
            r = rows[-1]
            print(f"[P1 | {model} | {proto}] AP national {r['ap_national']:.3f} -> within provinces "
                  f"{r['ap_within_group']:.3f}; share of each group's background in the national top 10%: "
                  + ', '.join(f'{g} {t:.2f}' for g, t in zip(GROUPS, top)), flush=True)
    pd.DataFrame(rows).to_csv(_b.out('devnet_national_offsets.csv'), index=False)


def local_factory(base_name):
    """One model per province group: all training positives against that
    group's background, scoring that group's points."""
    def make(X_bg_g):
        X_bg, g_bg = X_bg_g[:, :-1], X_bg_g[:, -1]

        def scorer(X_tr_g, y_tr, X_va_g, seed_counter=0):
            X_tr, X_va, g_va = X_tr_g[:, :-1], X_va_g[:, :-1], X_va_g[:, -1]
            s = np.zeros(len(X_va))
            for g in np.unique(g_va):
                fac = _b.FACTORIES[base_name]('Canada')
                s[g_va == g] = fac(X_bg[g_bg == g])(X_tr, y_tr, X_va[g_va == g], seed_counter=seed_counter)
            return s
        return scorer
    return make


def indicator_factory(base_name):
    """The province group as one-hot inputs."""
    def onehot(X_g):
        return np.hstack([X_g[:, :-1], (X_g[:, -1:] == np.arange(len(GROUPS))[None, :]).astype(float)])

    def make(X_bg_g):
        inner = _b.FACTORIES[base_name]('Canada')(onehot(X_bg_g))

        def scorer(X_tr_g, y_tr, X_va_g, seed_counter=0):
            return inner(onehot(X_tr_g), y_tr, onehot(X_va_g), seed_counter=seed_counter)
        return scorer
    return make


def models(names, protocols):
    d = load_national('set3')
    code = {g: k for k, g in enumerate(GROUPS)}
    gl = d['lab'].group.map(code).values[:, None].astype(float)
    gu = d['unl'].group.map(code).values[:, None].astype(float)
    X_lab, X_unl, y = np.hstack([d['X_lab'], gl]), np.hstack([d['X_unl'], gu]), d['y']
    rows = []
    for base in names:
        for variant, fac in [('local reference', local_factory(base)), ('province one-hot', indicator_factory(base))]:
            for p in protocols:
                proto = 'random' if p == 'random' else float(p)
                seed_everything()
                res = run_protocol(proto, X_lab, y, d['xy_lab'], X_unl, d['xy_unl'], fac, config.CV_N_SPLITS,
                                   config.CV_N_REPEATS, config.RANDOM_SEED)
                rows.append(summarise(y, res['oof'], model=base, variant=variant, protocol=_b.pname(proto),
                                      scope='Canada'))
                for g in GROUPS:
                    m = (d['lab'].group == g).values
                    rows.append(summarise(y[m], res['oof'][:, m], model=base, variant=variant,
                                          protocol=_b.pname(proto), scope=g))
                r = rows[-len(GROUPS) - 1]
                print(f"[{base} | {variant} | {_b.pname(proto)}] AP {r['ap_med']:.3f} AUC {r['auc_med']:.3f}",
                      flush=True)
    pd.DataFrame(rows).to_csv(_b.out(f"devnet_national_variants_{'_'.join(names)}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['offsets', 'models'], required=True)
    ap.add_argument('--models', nargs='+', default=['DevNet', 'RF'])
    ap.add_argument('--protocols', nargs='+', default=['random', '100'])
    args = ap.parse_args()
    offsets() if args.part == 'offsets' else models(args.models, args.protocols)


if __name__ == '__main__':
    main()
