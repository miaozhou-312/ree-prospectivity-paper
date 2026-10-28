"""
Stage 44 (N3) -- does pooling the country help a province? For each study
province, the same held-out occurrences and the same regional evaluation
pool are scored by models trained on

    regional      the province's training positives + its training pool
    +background   ... + the national training background outside it
    +positives    ... + the national occurrences outside it
    +both         ... + both (the national model)

on identical folds (random 5x6 CV and 100 km blocks with the Stage-02 seeds);
under blocks, added points that fall in a held-out block are purged as the
province's own background is. Every variant uses the province's own model:
its feature-set-3 columns and H3 classes (Stages 25-26) and its network
capacity, with values from the national table (identical to the regional
ones inside the province), so the variants differ only in training data
and 'regional' reproduces the main results.

    python -m paper_analyses.44_national_pooling --regions Superior --models DevNet RF
"""
import argparse
import importlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

from paper_analyses.common import config, save_pickle, seed_everything
from paper_analyses.national import STUDY, load_national, province_matrix, regional_rows, regional_spec, summarise
from paper_analyses.protocol import block_ids

_b = importlib.import_module('paper_analyses.42_national_benchmark')
VARIANTS = ['regional', '+background', '+positives', '+both']


def pooled_cv(proto, X_lab, y, xy_lab, X_bg, xy_bg, X_e, xy_e, make_scorer, n_splits, n_repeats, seed):
    """paper_analyses.protocol.run_protocol with extra training positives
    X_e (outside every test fold). With no extra rows it is run_protocol."""
    oof = np.full((n_repeats, len(y)), np.nan)
    if proto == 'random':
        scorer = make_scorer(X_bg)
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        for k, (tr, va) in enumerate(rskf.split(X_lab, y)):
            s = scorer(np.vstack([X_lab[tr], X_e]), np.r_[y[tr], np.ones(len(X_e))], X_lab[va], seed_counter=k)
            oof[k // n_splits, va] = rankdata(s) / len(s)
        return oof
    rng = np.random.default_rng(seed)
    counter = 0
    for r in range(n_repeats):
        offset = rng.uniform(0, proto, size=2)
        g_lab, g_bg = block_ids(xy_lab, proto, offset), block_ids(xy_bg, proto, offset)
        g_e = block_ids(xy_e, proto, offset) if len(X_e) else np.zeros(0, np.int64)
        k = min(n_splits, len(np.unique(g_lab[y == 1])))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        for tr, va in sgkf.split(X_lab, y, groups=g_lab):
            held = np.unique(g_lab[va])
            keep_bg, keep_e = ~np.isin(g_bg, held), ~np.isin(g_e, held)
            s = make_scorer(X_bg[keep_bg])(np.vstack([X_lab[tr], X_e[keep_e]]),
                                           np.r_[y[tr], np.ones(int(keep_e.sum()))], X_lab[va], seed_counter=counter)
            oof[r, va] = rankdata(s) / len(s)
            counter += 1
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=STUDY)
    ap.add_argument('--models', nargs='+', default=['DevNet', 'RF', 'BaggingPU', 'DevNet-BCE', 'nnPU'])
    ap.add_argument('--protocols', nargs='+', default=['random', '100'])
    ap.add_argument('--featureset', choices=['set3', 'nomap'], default='set3')
    args = ap.parse_args()
    d = load_national(args.featureset)
    pts = d['pts']
    nat = pts[pts.in_national]
    rows, store = [], {}
    for g in args.regions:
        spec = regional_spec(g)
        cols = spec[0]
        lab, unl = regional_rows(d['meta'], g)
        X_lab, y = province_matrix(lab, spec), lab.label.values.astype(int)
        X_bg, xy_lab, xy_bg = province_matrix(unl, spec), lab[['xy_x', 'xy_y']].values, unl[['xy_x', 'xy_y']].values
        out_pos = nat[(nat.group != g) & (nat.pool == 'positive')]
        out_bg = nat[(nat.group != g) & (nat.pool == 'train_unlabeled')]
        Xp_o, xyp_o = province_matrix(out_pos, spec), out_pos[['xy_x', 'xy_y']].values
        Xb_o, xyb_o = province_matrix(out_bg, spec), out_bg[['xy_x', 'xy_y']].values
        print(f'[{g}] {int(y.sum())} positives, {int((y == 0).sum())} evaluation, {len(X_bg)} training background; '
              f'added: {len(Xp_o)} positives, {len(Xb_o)} background points', flush=True)
        for model in args.models:
            for p in args.protocols:
                proto = 'random' if p == 'random' else float(p)
                for v in VARIANTS:
                    add_bg, add_pos = v in ('+background', '+both'), v in ('+positives', '+both')
                    Xb = np.vstack([X_bg, Xb_o]) if add_bg else X_bg
                    xyb = np.vstack([xy_bg, xyb_o]) if add_bg else xy_bg
                    Xe = Xp_o if add_pos else np.zeros((0, len(cols)))
                    xye = xyp_o if add_pos else np.zeros((0, 2))
                    seed_everything()
                    oof = pooled_cv(proto, X_lab, y, xy_lab, Xb, xyb, Xe, xye, _b.FACTORIES[model](g),
                                    config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
                    store[(g, model, _b.pname(proto), v)] = oof
                    rows.append(summarise(y, oof, region=g, model=model, protocol=_b.pname(proto), variant=v,
                                          featureset=args.featureset))
                    r = rows[-1]
                    print(f"[N3 | {g} | {model} | {_b.pname(proto)} | {v}] AUC {r['auc_med']:.3f} "
                          f"AP {r['ap_med']:.3f} top-10% {r['cap10_med']:.2f}", flush=True)
    tag = f"{'_'.join(args.regions)}_{'_'.join(args.models)}{'' if args.featureset == 'set3' else '_nomap'}"
    pd.DataFrame(rows).to_csv(_b.out(f'pooling_{tag}.csv'), index=False)
    save_pickle(store, _b.out(f'pooling_{tag}.pkl'))


if __name__ == '__main__':
    main()
