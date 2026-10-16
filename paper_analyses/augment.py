"""
Positive-sample augmentation for the label-scarce regime: Gaussian jitter,
SMOTE-style interpolation, and a tabular WGAN-GP (Gulrajani et al., 2017).

Everything is fold-safe by construction: an augmenter only ever sees the
positives of the current training fold (plus, for preprocessing, the shared
training background), and the synthetic points are only ever added to the
training side. Held-out positives are always real.

Augmentation happens in standardised feature space (median imputation and
scaling fit on training positives + background), samples are mapped back to
raw units, and binary terrane dummies are snapped back to valid 0/1 one-hot
values so synthetic points look like real feature rows to the downstream
(unchanged) RF / DevNet pipelines.
"""
import hashlib

import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

JITTER_SIGMA = 0.1     # in units of the (positive + background) standard deviation
SMOTE_K = 3
GAN_Z = 8
GAN_HIDDEN = 64
GAN_STEPS = 2000       # generator updates
GAN_CRITIC = 5
GAN_GP = 10.0
GAN_LR = 1e-4


class _Space:
    """Standardised space + binary-column snapping, fit on training rows."""
    def __init__(self, X_pos_raw, X_bg_raw, cols):
        X = np.vstack([X_pos_raw, X_bg_raw])
        self.imp = SimpleImputer(strategy='median').fit(X)
        Xi = self.imp.transform(X)
        self.sc = StandardScaler().fit(Xi)
        self.binary = [j for j in range(X.shape[1]) if set(np.unique(Xi[:, j])).issubset({0.0, 1.0})]
        groups = {}
        for j in self.binary:
            prefix = 'geo_era_' if cols[j].startswith('geo_era_') else (
                'lith_' if cols[j].startswith('lith_') else cols[j])
            groups.setdefault(prefix, []).append(j)
        self.groups = list(groups.values())

    def to_z(self, X_raw):
        return self.sc.transform(self.imp.transform(X_raw))

    def to_raw(self, Z):
        X = self.sc.inverse_transform(Z)
        for g in self.groups:
            v = X[:, g]
            if len(g) == 1:
                X[:, g[0]] = (v[:, 0] >= 0.5).astype(float)
            else:
                hot = np.zeros_like(v)
                am = v.argmax(1)
                on = v.max(1) >= 0.5
                hot[np.arange(len(v))[on], am[on]] = 1.0
                X[:, g] = hot
        return X


def _jitter(Zp, n, rng):
    base = Zp[rng.integers(len(Zp), size=n)]
    return base + rng.normal(0, JITTER_SIGMA, size=base.shape)


def _smote(Zp, n, rng):
    k = min(SMOTE_K, len(Zp) - 1)
    nn_idx = NearestNeighbors(n_neighbors=k + 1).fit(Zp).kneighbors(Zp, return_distance=False)[:, 1:]
    i = rng.integers(len(Zp), size=n)
    j = nn_idx[i, rng.integers(k, size=n)]
    lam = rng.uniform(size=(n, 1))
    return Zp[i] + lam * (Zp[j] - Zp[i])


class _G(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(GAN_Z, GAN_HIDDEN), nn.LeakyReLU(0.2),
                                 nn.Linear(GAN_HIDDEN, GAN_HIDDEN), nn.LeakyReLU(0.2),
                                 nn.Linear(GAN_HIDDEN, d))

    def forward(self, z):
        return self.net(z)


class _D(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, GAN_HIDDEN), nn.LeakyReLU(0.2),
                                 nn.Linear(GAN_HIDDEN, GAN_HIDDEN), nn.LeakyReLU(0.2),
                                 nn.Linear(GAN_HIDDEN, 1))

    def forward(self, x):
        return self.net(x)


def train_wgan_gp(Zp, seed, steps=GAN_STEPS, return_history=False):
    torch.manual_seed(seed)
    real = torch.tensor(Zp, dtype=torch.float32)
    n, d = real.shape
    G, D = _G(d), _D(d)
    opt_g = torch.optim.Adam(G.parameters(), lr=GAN_LR, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=GAN_LR, betas=(0.5, 0.9))
    hist = []
    for step in range(steps):
        for _ in range(GAN_CRITIC):
            fake = G(torch.randn(n, GAN_Z)).detach()
            eps = torch.rand(n, 1)
            mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
            grad = torch.autograd.grad(D(mix).sum(), mix, create_graph=True)[0]
            gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
            loss_d = D(fake).mean() - D(real).mean() + GAN_GP * gp
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
        loss_g = -D(G(torch.randn(n, GAN_Z))).mean()
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()
        if return_history and step % 20 == 0:
            hist.append((step, -loss_d.item() + GAN_GP * gp.item(), loss_g.item()))
    G.eval()
    return (G, hist) if return_history else G


def _wgan(Zp, n, rng, seed):
    G = train_wgan_gp(Zp, seed)
    with torch.no_grad():
        torch.manual_seed(seed + 1)
        return G(torch.randn(n, GAN_Z)).numpy()


_CACHE = {}


def synthesize(method, X_pos_raw, X_bg_raw, cols, n, seed):
    """n synthetic positives (raw feature units) from this fold's positives.
    Samples are generated once per (method, fold positives, seed) at the
    largest requested size and prefix-sliced, so different multipliers and
    both downstream models share the same synthetic draws."""
    if n <= 0 or method == 'none':
        return np.empty((0, X_pos_raw.shape[1]))
    key = (method, hashlib.md5(np.ascontiguousarray(X_pos_raw).tobytes()).hexdigest(), seed)
    if key in _CACHE and len(_CACHE[key]) >= n:
        return _CACHE[key][:n]
    space = _Space(X_pos_raw, X_bg_raw, cols)
    Zp = space.to_z(X_pos_raw)
    rng = np.random.default_rng(seed)
    n_gen = max(n, 10 * len(X_pos_raw))
    if method == 'jitter':
        Zs = _jitter(Zp, n_gen, rng)
    elif method == 'smote':
        Zs = _smote(Zp, n_gen, rng)
    elif method == 'wgan':
        Zs = _wgan(Zp, n_gen, rng, seed)
    else:
        raise ValueError(method)
    _CACHE[key] = space.to_raw(Zs)
    return _CACHE[key][:n]


def augmented_factory(base_factory, method, mult, cols, base_seed=1234):
    """Wrap a paper_analyses.models factory so each fold's training
    positives are augmented with mult x n_pos synthetic positives."""
    def make(X_bg_raw):
        base = base_factory(X_bg_raw)

        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            X_pos = X_tr_raw[y_tr == 1]
            X_syn = synthesize(method, X_pos, X_bg_raw, cols, int(round(mult * len(X_pos))),
                               seed=base_seed + seed_counter)
            X2 = np.vstack([X_tr_raw, X_syn])
            y2 = np.r_[y_tr, np.ones(len(X_syn))]
            return base(X2, y2, X_va_raw, seed_counter=seed_counter)
        return scorer
    return make


def nn_distance_ratio(Z_syn, Z_real):
    """Per synthetic sample: distance to the nearest real positive divided by
    the median nearest-neighbour distance among real positives. ~0 means the
    sample copies a training point (memorisation); ~1 means it sits as far
    from the data as real positives sit from each other."""
    nn_real = NearestNeighbors(n_neighbors=2).fit(Z_real)
    d_rr = nn_real.kneighbors(Z_real)[0][:, 1]
    d_sr = NearestNeighbors(n_neighbors=1).fit(Z_real).kneighbors(Z_syn)[0][:, 0]
    return d_sr / np.median(d_rr)
