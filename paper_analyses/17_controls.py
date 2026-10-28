"""
Stage 17 -- controls for the DevNet comparison.

  MLP-BCE    DevNet's network, quantile-normal inputs, 16 positive + 240
             background rows per step, 300 steps and five-restart ensemble,
             trained with binary cross-entropy (background = 0) instead of
             the deviation loss: the plain supervised counterpart, so DevNet
             vs MLP-BCE isolates the loss.
  RF-tuned   RF whose hyperparameters are chosen inside every outer
             training fold by inner 100 km spatial CV (paper_analyses/
             nested.py), from min_samples_leaf {1, 10} x max_features
             {sqrt, 0.33}; 300 trees, balanced class weights.
  HGB-tuned  gradient-boosted trees (sklearn HistGradientBoosting, balanced
             class weights, 200 iterations, learning rate 0.05, L2 1.0),
             tuned the same way over max_leaf_nodes {8, 31} x
             min_samples_leaf {5, 40}.

Tuning gives the tree baselines the same kind of design selection that
DevNet received, but without ever using the outer test fold.

    nice -n 10 python -m paper_analyses.17_controls [--models ...] [--regions ...]
"""
import argparse
import importlib
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer

from paper_analyses.common import (REGION_ORDER, config, load_pickle, load_region, model_cfg, out_path,
                                   save_pickle, seed_everything)
from paper_analyses.models import DEVICE
from paper_analyses.nested import nested_factory
from paper_analyses.protocol import paired_bootstrap, project_xy, run_protocol
from src.models.devnet import DevNet

_fin = importlib.import_module('paper_analyses.11_devnet_final')
_tb = importlib.import_module('paper_analyses.10_training_budget')
PROTOCOLS = ['random', 50.0, 100.0, 200.0]
RF_GRID = [(1, 'sqrt'), (1, 0.33), (10, 'sqrt'), (10, 0.33)]           # (min_samples_leaf, max_features)
HGB_GRID = [(8, 5), (8, 40), (31, 5), (31, 40)]                         # (max_leaf_nodes, min_samples_leaf)


def train_bce(X, y, seed, cfg, n_pos=16, n_bg=240, steps=None):
    """_tb.train with binary cross-entropy on the network output (logit)."""
    steps = config.DN_EPOCHS if steps is None else steps
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    model = DevNet(Xt.shape[1], hidden=cfg['hidden'], dropout=cfg['dropout']).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=config.DN_LR, weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    pos_idx, bg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    lab = torch.tensor(np.r_[np.ones(n_pos), np.zeros(n_bg)], dtype=torch.float32, device=DEVICE)
    for _ in range(steps):
        idx = np.r_[rng.choice(pos_idx, n_pos, replace=True), rng.choice(bg_idx, n_bg, replace=len(bg_idx) < n_bg)]
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xt[idx]), lab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
    model.eval()
    return model


def mlp_bce_factory(region):
    cfg = model_cfg(region)

    def make(X_bg):
        def scorer(X_tr, y_tr, X_va, seed_counter=0):
            X_pos = X_tr[y_tr == 1]
            X_train = np.vstack([X_pos, X_bg])
            y_train = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg))]
            tf = _fin.QuantileNormal(X_train)
            models = [train_bce(tf(X_train), y_train, config.DN_SEED + seed_counter * 1000 + k, cfg)
                      for k in range(config.DN_RESTARTS)]
            return _tb._ensemble(models, tf(X_va))
        return scorer
    return make


def tree_fit_score(kind):
    def fit_score(cand, X_pos, X_bg, X_eval, seed):
        X_train = np.vstack([X_pos, X_bg])
        y_train = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg))]
        imp = SimpleImputer(strategy='median').fit(X_train)
        if kind == 'RF':
            leaf, feat = cand
            m = RandomForestClassifier(n_estimators=300, max_features=feat, min_samples_leaf=leaf,
                                       class_weight='balanced', random_state=config.RANDOM_SEED + seed, n_jobs=4)
        else:
            leaves, leaf = cand
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=leaves,
                                               min_samples_leaf=leaf, l2_regularization=1.0, early_stopping=False,
                                               class_weight='balanced', random_state=config.RANDOM_SEED + seed)
        m.fit(imp.transform(X_train), y_train)
        return m.predict_proba(imp.transform(X_eval))[:, 1]
    return fit_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['MLP-BCE', 'RF-tuned', 'HGB-tuned'])
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    args = ap.parse_args()
    final = load_pickle(out_path('devnet_final.pkl'))
    rows, store = [], {}
    for region in args.regions:
        rec = load_region(region)
        d = rec['pruned']
        n_feat = d['X_lab'].shape[1]
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        for model in args.models:
            for proto in PROTOCOLS:
                log = []
                seed_everything()
                if model == 'MLP-BCE':
                    fac, X_lab, X_bg = mlp_bce_factory(region), d['X_lab'], d['X_unl']
                else:
                    kind, grid = ('RF', RF_GRID) if model == 'RF-tuned' else ('HGB', HGB_GRID)
                    fs = tree_fit_score(kind)
                    fac = nested_factory(grid, fs, fs, n_feat, log)
                    X_lab, X_bg = np.hstack([d['X_lab'], xy_lab]), np.hstack([d['X_unl'], xy_bg])
                res = run_protocol(proto, X_lab, d['y'], xy_lab, X_bg, xy_bg, fac,
                                   config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
                store[(region, model, proto)] = {**res, 'log': log}
                pname = 'random' if proto == 'random' else f'{proto:.0f} km'
                ref = final[(region, proto)]
                bt = paired_bootstrap(d['y'], ref['oof_mean'], res['oof_mean'], seed=config.RANDOM_SEED)
                dd = ref['aps'] - res['aps']
                counts = Counter(str(e['chosen']) for e in log)
                rows.append({'region': region, 'model': model, 'protocol': pname,
                             'block_km': 0 if proto == 'random' else proto, 'auc_med': res['auc_med'],
                             'auc_lo': res['auc_ci'][0], 'auc_hi': res['auc_ci'][1], 'ap_med': res['ap_med'],
                             'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1], 'devnet_ap_med': ref['ap_med'],
                             'd_ap_devnet_minus': bt['d_ap'], 'd_ap_lo': bt['d_ap_ci'][0], 'd_ap_hi': bt['d_ap_ci'][1],
                             'p_ap': bt['p_ap'], 'd_auc_devnet_minus': bt['d_auc'], 'p_auc': bt['p_auc'],
                             'devnet_wins': f'{int((dd > 0).sum())}/{len(dd)}', 'chosen': dict(counts)})
                print(f"[{region} | {pname}] {model} AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f} | DevNet "
                      f"{ref['ap_med']:.3f}: dAP {bt['d_ap']:+.3f} p={bt['p_ap']:.3f} wins {rows[-1]['devnet_wins']}"
                      f"{' | chosen ' + str(dict(counts)) if log else ''}", flush=True)
    tag = '_'.join(args.models + args.regions).replace('-', '')
    pd.DataFrame(rows).to_csv(out_path(f'controls_{tag}.csv'), index=False)
    save_pickle(store, out_path(f'controls_{tag}.pkl'))


if __name__ == '__main__':
    main()
