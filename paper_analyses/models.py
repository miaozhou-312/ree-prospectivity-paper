"""
Model zoo for the paper. Every entry is a *scorer factory*:

    make_scorer(X_bg_raw) -> scorer(X_tr_raw, y_tr, X_va_raw, seed_counter) -> scores

with the same interface as src/models/*'s fold scorers, so the pooled-OOF
protocol in src/evaluation/cv_protocol.py (and the spatial variant in
paper_analyses/protocol.py) treats every model identically. `X_bg_raw` is
the shared training background (train_unlabeled): RF labels it 0, DevNet,
bagging-PU and nnPU treat it as unlabelled.

RF and DevNet are the packaged src/ scorers, unchanged. The two PU
baselines are added for the journal paper:

- **Bagging-PU** (Mordelet & Vert, 2014): T bags, each a decision tree fit
  on all positives vs a random subsample of the unlabelled pool of the same
  size; scores are the mean over bags. It is the standard tree-based PU
  learner and the closest PU analogue of the RF baseline.
- **nnPU** (Kiryo et al., 2017): non-negative PU risk with the sigmoid loss,
  trained on the *same* one-hidden-layer MLP, capacity, optimiser and
  restart ensemble as DevNet, so DevNet vs nnPU isolates the loss (deviation
  vs PU risk) from the architecture. Needs a class prior pi = P(y=1) in the
  unlabelled population; deposits are rare, so pi = 0.01 by default
  (sensitivity reported in the paper).

Elkan & Noto (2008) is deliberately not included: under the SCAR
assumption its calibrated output is a monotone transform of the
"non-traditional" classifier (positives vs unlabelled), so its ranking --
and therefore every rank metric reported here (AUC, AP, capture rate) -- is
identical to the RF baseline's.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from paper_analyses.common import config, dn_train_kwargs, model_cfg
from src.models.devnet import DevNet, devnet_semisup_fold_scorer_factory, get_device
from src.models.random_forest import make_rf_params, rf_shared_bg_fold_scorer_factory

DEVICE = get_device(config.FORCE_CPU)


def _fit_transform(X_train_raw, *others):
    imp = SimpleImputer(strategy='median').fit(X_train_raw)
    sc = StandardScaler().fit(imp.transform(X_train_raw))
    return [sc.transform(imp.transform(X_train_raw))] + [sc.transform(imp.transform(o)) for o in others]


# ── packaged models ─────────────────────────────────────────────────────────
def rf_fixed_factory(region):
    """The thesis RF: one fixed random_state for every fold."""
    params = make_rf_params(config.RANDOM_SEED)
    return lambda X_bg: rf_shared_bg_fold_scorer_factory(X_bg, params)


def rf_factory(region):
    """RF with random_state = RANDOM_SEED + fold counter. With ~5,000
    background rows against ~12 positives most validation points receive
    P = 0 exactly, so pooled AUC depends strongly on which trees happen to
    vote for a held-out positive; varying the seed per fold folds that
    variance into the repeated-CV interval (as DevNet's restarts do) instead
    of reporting one arbitrary seed."""
    def make(X_bg):
        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            params = make_rf_params(config.RANDOM_SEED + seed_counter)
            return rf_shared_bg_fold_scorer_factory(X_bg, params)(X_tr_raw, y_tr, X_va_raw, seed_counter)
        return scorer
    return make


def devnet_factory(region, restarts=None):
    restarts = config.DN_RESTARTS if restarts is None else restarts
    return lambda X_bg: devnet_semisup_fold_scorer_factory(
        X_bg, model_cfg=model_cfg(region), device=DEVICE, train_kwargs=dn_train_kwargs(),
        restarts=restarts, base_seed=config.DN_SEED)


# ── Bagging-PU ──────────────────────────────────────────────────────────────
def bagging_pu_factory(region, n_bags=200):
    def make(X_bg_raw):
        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            X_pos_raw = X_tr_raw[y_tr == 1]
            X_all, X_va = _fit_transform(np.vstack([X_pos_raw, X_bg_raw]), X_va_raw)
            X_pos, X_u = X_all[:len(X_pos_raw)], X_all[len(X_pos_raw):]
            rng = np.random.default_rng(config.RANDOM_SEED + seed_counter)
            k = len(X_pos)
            acc = np.zeros(len(X_va))
            for b in range(n_bags):
                ui = rng.choice(len(X_u), k, replace=True)
                Xb = np.vstack([X_pos, X_u[ui]])
                yb = np.r_[np.ones(k), np.zeros(k)]
                tree = DecisionTreeClassifier(max_features='sqrt',
                                              random_state=int(rng.integers(1 << 31)))
                acc += tree.fit(Xb, yb).predict_proba(X_va)[:, 1]
            return acc / n_bags
        return scorer
    return make


# ── nnPU (same MLP as DevNet) ───────────────────────────────────────────────
def _train_nnpu(X_pos, X_u, prior, seed, cfg, epochs, lr, batch_u=256, beta=0.0, gamma=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xp = torch.tensor(X_pos, dtype=torch.float32, device=DEVICE)
    Xu = torch.tensor(X_u, dtype=torch.float32, device=DEVICE)
    model = DevNet(Xp.shape[1], hidden=cfg['hidden'], dropout=cfg['dropout']).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    sig = lambda z: torch.sigmoid(-z)          # sigmoid loss l(z, +1); l(z, -1) = sig(-z)
    model.train()
    for _ in range(epochs):
        ub = Xu[torch.as_tensor(rng.choice(len(X_u), batch_u, replace=False), device=DEVICE)]
        gp, gu = model(Xp), model(ub)
        risk_p_pos = prior * sig(gp).mean()
        risk_neg = sig(-gu).mean() - prior * sig(-gp).mean()
        opt.zero_grad()
        if risk_neg.item() < -beta:
            (-gamma * risk_neg).backward()       # Kiryo et al. (2017) defence step
        else:
            (risk_p_pos + risk_neg).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
    model.eval()
    return model


def nnpu_factory(region, prior=0.01, restarts=None):
    restarts = config.DN_RESTARTS if restarts is None else restarts
    cfg = model_cfg(region)

    def make(X_bg_raw):
        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            X_pos_raw = X_tr_raw[y_tr == 1]
            X_all, X_va = _fit_transform(np.vstack([X_pos_raw, X_bg_raw]), X_va_raw)
            X_pos, X_u = X_all[:len(X_pos_raw)], X_all[len(X_pos_raw):]
            Xv = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
            acc = np.zeros(len(X_va))
            for k in range(restarts):
                m = _train_nnpu(X_pos, X_u, prior, seed=config.DN_SEED + seed_counter * 1000 + k,
                                cfg=cfg, epochs=config.DN_EPOCHS, lr=config.DN_LR)
                with torch.no_grad():
                    s = m(Xv).cpu().numpy()
                acc += rankdata(s) / len(s)
            return acc / restarts
        return scorer
    return make


# ── XGBoost (gradient-boosted trees) ────────────────────────────────────────
# Fixed parameters of the national study of J. Fang (irp-jf1522: XGB_BASE plus
# the "unified clean-mode" setting), not tuned here. Trained like RF on the
# fold's positives against the shared training background labelled 0, with
# the positives up-weighted to balance the classes (scale_pos_weight, the
# analogue of RF's class_weight='balanced') and the same median imputation.
XGB_PARAMS = dict(n_estimators=200, max_depth=2, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=3, reg_lambda=2.0, reg_alpha=0.5, gamma=0, objective='binary:logistic',
                  tree_method='hist', n_jobs=2)


def xgb_factory(region):
    """Each fit runs in paper_analyses.xgb_worker (a process without PyTorch;
    see that module for why)."""
    import os
    import pickle
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def make(X_bg_raw):
        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            X_pos = X_tr_raw[y_tr == 1]
            X = np.vstack([X_pos, X_bg_raw])
            y = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg_raw))]
            imp = SimpleImputer(strategy='median').fit(X)
            params = dict(XGB_PARAMS, scale_pos_weight=len(X_bg_raw) / max(len(X_pos), 1),
                          random_state=config.RANDOM_SEED + seed_counter)
            job = pickle.dumps({'X': imp.transform(X), 'y': y, 'X_va': imp.transform(X_va_raw), 'params': params})
            env = dict(os.environ, PYTHONPATH=repo)
            out = subprocess.run([sys.executable, '-m', 'paper_analyses.xgb_worker'], input=job, capture_output=True,
                                 cwd=repo, env=env)
            if out.returncode != 0:
                raise RuntimeError(f'xgb_worker failed: {out.stderr.decode()[-2000:]}')
            return pickle.loads(out.stdout)
        return scorer
    return make


MODEL_FACTORIES = {
    'RF': rf_factory,
    'XGBoost': xgb_factory,
    'RF_fixedseed': rf_fixed_factory,
    'DevNet': devnet_factory,
    'BaggingPU': bagging_pu_factory,
    'nnPU': nnpu_factory,
}
