"""
Stage 10 -- sensitivity analyses: two fixes for DevNet's training budget.

Added AFTER the Stage-8 diagnostics and reported as such; the main results
use the original DevNet. In the original configuration each of the 300
gradient steps uses 16 positives and 16 background rows, so a background
row is seen about once while each positive is seen about 400 times.
Simply training 10x longer raised random-split AP but lowered spatial
transfer (Stage 8). Two principled alternatives, fixed before running:

  DevNet-BG  background-heavy batches: 16 positives + 240 background rows
             per step, 300 steps, everything else unchanged. Each
             background row is seen ~14 times while positive exposure is
             unchanged.
  DevNet-ES  spatially nested early stopping: inside every outer training
             fold, labelled rows are split into 100 km blocks (3 inner
             folds, background purged from inner-validation blocks); the
             number of steps is chosen from checkpoints between 100 and
             3,000 (cosine schedule over 3,000 steps) by pooled inner AP,
             averaged over 2 inner restarts; the outer model (5 restarts)
             is then trained on the whole outer training fold and stopped
             at the chosen step. The test fold is never used for the choice.
             Inner folds whose training part holds no positive are skipped;
             if no inner fold remains, the original 300 steps are kept.

Both are evaluated under the full protocol (random 5x6 CV and 50 / 100 /
200 km blocks, identical folds) with paired comparisons against the
original DevNet and RF from the Stage-1/2 stores.

    nice -n 10 python -m paper_analyses.10_training_budget --variant bg|es [--regions ...]
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from paper_analyses.common import (REGION_ORDER, config, load_pickle, load_region, model_cfg, out_path,
                                   save_pickle, seed_everything)
from paper_analyses.models import DEVICE
from paper_analyses.protocol import (block_ids, paired_bootstrap, project_xy, repeated_random_cv,
                                     repeated_spatial_cv)
from src.models.devnet import DevNet, deviation_loss

BLOCKS = [50.0, 100.0, 200.0]
BG_BATCH = 240
ES_MAX_STEPS = 3000
ES_CHECKPOINTS = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
ES_INNER_BLOCK_KM, ES_INNER_FOLDS, ES_INNER_RESTARTS = 100.0, 3, 2


def train(X, y, seed, cfg, n_pos, n_bg, t_max, stop_at, X_eval=None, checkpoints=()):
    """DevNet training loop of src.models.devnet.train_devnet with separate
    positive / background batch sizes, a cosine schedule over `t_max` steps,
    and optional scoring of X_eval at checkpoints."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    model = DevNet(Xt.shape[1], hidden=cfg['hidden'], dropout=cfg['dropout']).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=config.DN_LR, weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-5)
    pos_idx, bg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    lab = torch.tensor(np.r_[np.ones(n_pos), np.zeros(n_bg)], dtype=torch.float32, device=DEVICE)
    Xe = None if X_eval is None else torch.tensor(X_eval, dtype=torch.float32, device=DEVICE)
    scores = {}
    for step in range(1, stop_at + 1):
        idx = np.r_[rng.choice(pos_idx, n_pos, replace=True),
                    rng.choice(bg_idx, n_bg, replace=len(bg_idx) < n_bg)]
        model.train()
        opt.zero_grad()
        loss = deviation_loss(model(Xt[idx]), lab, config.DN_MARGIN)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        if step in checkpoints:
            model.eval()
            with torch.no_grad():
                scores[step] = model(Xe).cpu().numpy()
    model.eval()
    return model, scores


def _fit_scale(X_train, *others):
    imp = SimpleImputer(strategy='median').fit(X_train)
    sc = StandardScaler().fit(imp.transform(X_train))
    return [sc.transform(imp.transform(X_train))] + [sc.transform(imp.transform(o)) for o in others]


def _ensemble(models, X):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    acc = np.zeros(len(X))
    for m in models:
        with torch.no_grad():
            s = m(Xt).cpu().numpy()
        acc += rankdata(s) / len(s)
    return acc / len(models)


def bg_heavy_factory(region):
    cfg = model_cfg(region)

    def make(X_bg_raw):
        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            X_pos = X_tr_raw[y_tr == 1]
            X_train = np.vstack([X_pos, X_bg_raw])
            y_train = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg_raw))]
            Xt, Xv = _fit_scale(X_train, X_va_raw)
            models = [train(Xt, y_train, config.DN_SEED + seed_counter * 1000 + k, cfg, 16, BG_BATCH,
                            t_max=config.DN_EPOCHS, stop_at=config.DN_EPOCHS)[0]
                      for k in range(config.DN_RESTARTS)]
            return _ensemble(models, Xv)
        return scorer
    return make


CHOSEN = []   # chosen steps per outer fold (reported)


def nested_es_factory(region, n_feat):
    """Scorer whose input rows carry projected x, y (km) as the last two
    columns; they are used only for the inner spatial split."""
    cfg = model_cfg(region)

    def make(X_bg_aug):
        X_bg_raw, xy_bg = X_bg_aug[:, :n_feat], X_bg_aug[:, n_feat:]

        def scorer(X_tr_aug, y_tr, X_va_aug, seed_counter=0):
            X_tr, xy_tr = X_tr_aug[:, :n_feat], X_tr_aug[:, n_feat:]
            X_va = X_va_aug[:, :n_feat]
            rng = np.random.default_rng(config.RANDOM_SEED + seed_counter)
            offset = rng.uniform(0, ES_INNER_BLOCK_KM, size=2)
            g_tr, g_bg = block_ids(xy_tr, ES_INNER_BLOCK_KM, offset), block_ids(xy_bg, ES_INNER_BLOCK_KM, offset)
            k = min(ES_INNER_FOLDS, len(np.unique(g_tr[y_tr == 1])))
            chosen = config.DN_EPOCHS
            if k >= 2:
                inner = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_SEED + seed_counter)
                ranks = {c: np.full(len(y_tr), np.nan) for c in ES_CHECKPOINTS}
                for i, (itr, iva) in enumerate(inner.split(X_tr, y_tr, groups=g_tr)):
                    if not (y_tr[itr] == 1).any():          # all positive blocks held out: skip
                        continue
                    keep = ~np.isin(g_bg, np.unique(g_tr[iva]))
                    X_pos = X_tr[itr][y_tr[itr] == 1]
                    X_train = np.vstack([X_pos, X_bg_raw[keep]])
                    y_train = np.r_[np.ones(len(X_pos)), np.zeros(int(keep.sum()))]
                    Xt, Xv = _fit_scale(X_train, X_tr[iva])
                    acc = {c: np.zeros(len(iva)) for c in ES_CHECKPOINTS}
                    for r in range(ES_INNER_RESTARTS):
                        _, sc = train(Xt, y_train, config.DN_SEED + 7919 * (seed_counter + 1) + 100 * i + r, cfg, 16, 16,
                                      t_max=ES_MAX_STEPS, stop_at=ES_MAX_STEPS, X_eval=Xv, checkpoints=ES_CHECKPOINTS)
                        for c in ES_CHECKPOINTS:
                            acc[c] += rankdata(sc[c]) / len(sc[c])
                    for c in ES_CHECKPOINTS:
                        ranks[c][iva] = acc[c]
                scored = ~np.isnan(ranks[ES_CHECKPOINTS[0]])
                if (y_tr[scored] == 1).any():                # else keep the original 300 steps
                    ap = {c: average_precision_score(y_tr[scored], ranks[c][scored]) for c in ES_CHECKPOINTS}
                    chosen = max(ES_CHECKPOINTS, key=lambda c: (ap[c], -c))  # ties -> fewer steps
            CHOSEN.append(chosen)
            X_pos = X_tr[y_tr == 1]
            X_train = np.vstack([X_pos, X_bg_raw])
            y_train = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg_raw))]
            Xt, Xv = _fit_scale(X_train, X_va)
            models = [train(Xt, y_train, config.DN_SEED + seed_counter * 1000 + kk, cfg, 16, 16,
                            t_max=ES_MAX_STEPS, stop_at=chosen)[0] for kk in range(config.DN_RESTARTS)]
            return _ensemble(models, Xv)
        return scorer
    return make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['bg', 'es'], required=True)
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    args = ap.parse_args()
    name = {'bg': 'DevNet-BG', 'es': 'DevNet-ES'}[args.variant]
    bench = load_pickle(out_path('benchmark_Superior_Cordillera_Grenville.pkl'))
    spat = load_pickle(out_path('spatial_cv_Superior_Cordillera_Grenville.pkl'))
    rows, store = [], {}
    for region in args.regions:
        rec = load_region(region)
        d = rec['pruned']
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        if args.variant == 'bg':
            fac, X_lab, X_bg = bg_heavy_factory(region), d['X_lab'], d['X_unl']
        else:
            fac = nested_es_factory(region, d['X_lab'].shape[1])
            X_lab, X_bg = np.hstack([d['X_lab'], xy_lab]), np.hstack([d['X_unl'], xy_bg])
        for proto in ['random'] + BLOCKS:
            seed_everything()
            CHOSEN.clear()
            if proto == 'random':
                res = repeated_random_cv(X_lab, d['y'], X_bg, fac, config.CV_N_SPLITS, config.CV_N_REPEATS,
                                         config.RANDOM_SEED)
                ref_dn, ref_rf = bench[(region, 'pruned', 'DevNet')], bench[(region, 'pruned', 'RF')]
            else:
                res = repeated_spatial_cv(X_lab, d['y'], xy_lab, X_bg, xy_bg, fac, block_km=proto,
                                          n_splits=config.CV_N_SPLITS, n_repeats=config.CV_N_REPEATS,
                                          seed=config.RANDOM_SEED)
                ref_dn, ref_rf = spat[(region, proto, 'DevNet')], spat[(region, proto, 'RF')]
            store[(region, proto)] = {**res, 'chosen_steps': list(CHOSEN)}
            row = {'region': region, 'variant': name, 'protocol': 'random' if proto == 'random' else f'{proto:.0f} km',
                   'auc_med': res['auc_med'], 'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0],
                   'ap_hi': res['ap_ci'][1], 'devnet_ap_med': ref_dn['ap_med'], 'rf_ap_med': ref_rf['ap_med'],
                   'chosen_steps_median': float(np.median(CHOSEN)) if CHOSEN else np.nan}
            for tag, ref in [('dn', ref_dn), ('rf', ref_rf)]:
                bt = paired_bootstrap(d['y'], res['oof_mean'], ref['oof_mean'], seed=config.RANDOM_SEED)
                dd = res['aps'] - ref['aps']
                row.update({f'd_ap_vs_{tag}': bt['d_ap'], f'd_ap_vs_{tag}_lo': bt['d_ap_ci'][0],
                            f'd_ap_vs_{tag}_hi': bt['d_ap_ci'][1], f'p_vs_{tag}': bt['p_ap'],
                            f'wins_vs_{tag}': f'{int((dd > 0).sum())}/{len(dd)}'})
            rows.append(row)
            extra = f" | steps median {row['chosen_steps_median']:.0f}" if CHOSEN else ''
            print(f"[{region} | {row['protocol']}] {name} AP {res['ap_med']:.3f} (DevNet {ref_dn['ap_med']:.3f}, "
                  f"RF {ref_rf['ap_med']:.3f}) | vs DevNet {row['d_ap_vs_dn']:+.3f} p={row['p_vs_dn']:.3f} "
                  f"wins {row['wins_vs_dn']} | vs RF {row['d_ap_vs_rf']:+.3f} p={row['p_vs_rf']:.3f} "
                  f"wins {row['wins_vs_rf']}{extra}", flush=True)
    tag = '_'.join(args.regions)
    pd.DataFrame(rows).to_csv(out_path(f'training_budget_{args.variant}_{tag}.csv'), index=False)
    save_pickle(store, out_path(f'training_budget_{args.variant}_{tag}.pkl'))


if __name__ == '__main__':
    main()
