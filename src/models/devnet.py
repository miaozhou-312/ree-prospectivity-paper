"""
Section 8 -- DevNet: corrected deviation loss + ranking-loss alternative +
per-region capacity.

Two interchangeable losses:

- `deviation_loss` -- standard DevNet: negatives pulled to z~=0, positives
  pushed past z>margin. Batches are label-balanced 50/50, with positives
  oversampled with replacement up to the negative count. Uses a FIXED
  50,000-sample reference pool (`_REF`, built once at import time) so every
  batch computes a stable z-score against a fixed mean/std, instead of
  resampling `torch.randn_like(scores)` fresh each batch (which would inject
  a second source of gradient noise on top of the actual training signal).
- `ranking_loss` -- pairwise hinge over every positive/negative pair in the
  batch. Not used by the main (semi-supervised) experiment -- see
  `devnet_semisup_fold_scorer_factory` below -- kept for reference/reuse.

`DevNet`, `train_devnet`, and the fold-scorer factories take an explicit
`model_cfg` (hidden/dropout/weight_decay) so each region can use its own
capacity (see `REGION_MODEL_CFG` in src/config.py) instead of one fixed
architecture for all regions.

`devnet_semisup_fold_scorer_factory` is the scorer the main experiment
actually uses: within a CV fold, DevNet trains on that fold's positives plus
the region's ENTIRE unlabeled pool (`train_unlabeled`), and is scored on
that fold's held-out rows from the labeled set (positives + `eval_neg`).
The fold's own `eval_neg` rows are deliberately never used for training --
only the fold's positives are extracted from `X_tr_raw`/`y_tr`; the
unlabeled pool supplies every "background" example instead.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def get_device(force_cpu):
    return torch.device('cpu') if force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Fixed reference pool (build once, reuse for every batch/fold/restart) ──
# Drawn from its own seeded generator: this runs at import time, before any
# caller can seed torch, and torch's start-up seed differs per process
# (KNOWN ISSUE #5, fixed).
_REF = torch.randn(50000, generator=torch.Generator().manual_seed(42))


def deviation_loss(scores, labels, margin, ref=_REF):
    ref = ref.to(scores.device)
    dev = (scores - ref.mean()) / ref.std()          # stable z-score
    inlier  = dev[labels == 0].abs().mean()          # negatives -> dev ~ 0
    outlier = torch.clamp(margin - dev[labels == 1], min=0).mean()  # positives -> dev > margin
    return inlier + outlier


def ranking_loss(scores, labels, margin):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.sum() * 0.0
    diff = margin - (pos.unsqueeze(1) - neg.unsqueeze(0))  # (P, N) all pairs
    return torch.clamp(diff, min=0).mean()


_LOSS_FNS = {'deviation': deviation_loss, 'ranking': ranking_loss}


class DevNet(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _make_batch(X_t, y_t, pos_idx, neg_idx, rng, loss_type, dn_batch, rank_neg_ratio):
    if loss_type == 'deviation':
        n_half = dn_batch // 2
        pi = rng.choice(pos_idx, n_half, replace=True)                       # oversample positives
        ni = rng.choice(neg_idx, n_half, replace=len(neg_idx) < n_half)
        idx = np.concatenate([pi, ni])
    else:  # ranking: all true positives, negatives ~rank_neg_ratio:1, no oversampling
        n_neg_b = min(len(neg_idx), len(pos_idx) * rank_neg_ratio)
        ni = rng.choice(neg_idx, n_neg_b, replace=False)
        idx = np.concatenate([pos_idx, ni])
    rng.shuffle(idx)
    return X_t[idx], y_t[idx]


def train_devnet(X_tr, y_tr, loss_type, seed, model_cfg, device,
                  dn_lr, dn_batch, dn_epochs, rank_neg_ratio,
                  loss_margin, record_losses=False, margin=None):
    """Train one DevNet instance on already-imputed/scaled data, sized and
    regularized according to model_cfg (hidden/dropout/weight_decay) so each
    region can use its own capacity.

    `loss_margin` is the dict {'deviation': DN_MARGIN, 'ranking': RANK_MARGIN}
    (the default margin per loss type). `margin=None` (the default, used by
    every main-experiment call site) reproduces the original behaviour
    exactly -- falls back to loss_margin[loss_type]. Passing an explicit
    margin overrides it for that one call only -- used by the DN_MARGIN
    sensitivity ablation (see src/evaluation/ablation_margin.py) so it can
    sweep margins without touching any shared default.
    """
    torch.manual_seed(seed)
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    model = DevNet(X_t.shape[1], hidden=model_cfg['hidden'],
                   dropout=model_cfg['dropout']).to(device)
    opt   = optim.Adam(model.parameters(), lr=dn_lr,
                        weight_decay=model_cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=dn_epochs, eta_min=1e-5)

    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    rng     = np.random.default_rng(seed)
    loss_fn = _LOSS_FNS[loss_type]
    margin  = loss_margin[loss_type] if margin is None else margin

    losses = []
    model.train()
    for _ in range(dn_epochs):
        xb, yb = _make_batch(X_t, y_t, pos_idx, neg_idx, rng, loss_type, dn_batch, rank_neg_ratio)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb, margin)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        if record_losses:
            losses.append(loss.item())
    return model, losses


def ensemble_score(models, X, device):
    """Rank-average scores from a list of trained DevNets -- the same
    combination rule the fold-scorer factories use -- so a production
    ensemble can be scored identically to how it was evaluated."""
    acc = np.zeros(len(X))
    for m in models:
        m.eval()
        with torch.no_grad():
            s = m(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
        acc += rankdata(s) / len(s)
    return acc / len(models)


def devnet_fold_scorer_factory(loss_type, model_cfg, device, train_kwargs,
                                restarts, base_seed):
    """Plain discriminative fold scorer (trains on this fold's own labeled
    rows only). Superseded by devnet_semisup_fold_scorer_factory for the
    main (semi-supervised) experiment -- kept for reference/reuse.

    `train_kwargs` is forwarded to train_devnet (dn_lr, dn_batch, dn_epochs,
    rank_neg_ratio, loss_margin)."""
    def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
        imp = SimpleImputer(strategy='median').fit(X_tr_raw)
        sc  = StandardScaler().fit(imp.transform(X_tr_raw))
        X_tr = sc.transform(imp.transform(X_tr_raw))
        X_va = sc.transform(imp.transform(X_va_raw))

        rank_accum = np.zeros(len(X_va))
        for k in range(restarts):
            seed = base_seed + seed_counter * 1000 + k
            model, _ = train_devnet(X_tr, y_tr, loss_type, seed=seed, model_cfg=model_cfg,
                                     device=device, **train_kwargs)
            model.eval()
            with torch.no_grad():
                va_scores = model(
                    torch.tensor(X_va, dtype=torch.float32, device=device)
                ).cpu().numpy()
            rank_accum += rankdata(va_scores) / len(va_scores)
        return rank_accum / restarts
    return scorer


def devnet_semisup_fold_scorer_factory(X_unlabeled_raw, model_cfg, device, train_kwargs,
                                        restarts, base_seed, margin=None):
    """Semi-supervised DevNet fold scorer -- the one the main experiment
    actually uses. TRAINS on (this fold's positives + the region's full
    unlabeled background pool) with the deviation loss, but is SCORED on
    X_va_raw as passed in by pooled_oof_eval/repeated_pooled_auc (that
    fold's held-out positives + eval_neg). The fold's own eval_neg rows in
    X_tr_raw are intentionally ignored for training -- eval_neg exists only
    to be scored against, never to teach the model what "negative" looks
    like, so it stays a fair, never-trained-on evaluation set.

    `margin=None` (default, every main-experiment call site) reproduces the
    original behaviour exactly -- train_devnet falls back to
    train_kwargs['loss_margin']['deviation']. Passing an explicit margin
    here threads it through to every train_devnet call this scorer makes,
    for the DN_MARGIN sensitivity ablation -- no shared default is ever
    mutated.
    """
    def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
        X_pos_tr = X_tr_raw[y_tr == 1]
        X_train_raw = np.vstack([X_pos_tr, X_unlabeled_raw])
        y_train     = np.concatenate([np.ones(len(X_pos_tr)), np.zeros(len(X_unlabeled_raw))])

        imp = SimpleImputer(strategy='median').fit(X_train_raw)
        sc  = StandardScaler().fit(imp.transform(X_train_raw))
        X_tr = sc.transform(imp.transform(X_train_raw))
        X_va = sc.transform(imp.transform(X_va_raw))

        rank_accum = np.zeros(len(X_va))
        for k in range(restarts):
            seed = base_seed + seed_counter * 1000 + k
            model, _ = train_devnet(X_tr, y_train, 'deviation', seed=seed, model_cfg=model_cfg,
                                     device=device, margin=margin, **train_kwargs)
            model.eval()
            with torch.no_grad():
                va_scores = model(
                    torch.tensor(X_va, dtype=torch.float32, device=device)
                ).cpu().numpy()
            rank_accum += rankdata(va_scores) / len(va_scores)
        return rank_accum / restarts
    return scorer
