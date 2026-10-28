"""
Stage 55 -- SHAP for the other three models of the Superior benchmark
(XGBoost, bagging-PU, nnPU), on the explain set of Stage 29 (all positives
plus evaluation-background points, 200 in total; 100 training-background
points as the DeepSHAP reference), each model fitted as in Stage 53:

  XGBoost     TreeSHAP of the boosted trees (log-odds), via xgb_worker
  bagging-PU  TreeSHAP averaged over the 200 bagged trees (probability)
  nnPU        DeepSHAP on the quantile-normal inputs, mean over the five
              ensemble members

Directions follow the Spearman rule of Stage 3. Writes out/shap_fs3_others.pkl
and out/shap_fs3_others_directions.csv, and draws figures_r7/fig_shap_superior_r7.png
with all five models (DevNet and RF from Stage 29).

    source ../work_public/env.sh
    (unset REE_FEATURESET; python -m paper_analyses.55_shap_others [--draw])
"""
import importlib
import os
import pickle
import subprocess
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from paper_analyses.common import config, load_pickle, model_cfg, out_path, save_pickle, seed_everything  # noqa: E402
from paper_analyses.figstyle import pretty  # noqa: E402
from paper_analyses.models import DEVICE, XGB_PARAMS, _fit_transform, _train_nnpu  # noqa: E402
from src.interpretation.shap_explainers import explain_devnet  # noqa: E402

_s3 = importlib.import_module('paper_analyses.03_shap_directions')
_s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
_fin = importlib.import_module('paper_analyses.11_devnet_final')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, 'paper', 'manuscript', 'figures_r7')
REGION = 'Superior'


def explain_set(rec, X_lab, X_unl):
    """The rows Stage 29 explains, drawn with the same generator calls."""
    y = rec['pruned']['y']
    rng = np.random.default_rng(config.RANDOM_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = np.r_[pos, rng.choice(neg, 200 - len(pos), replace=False)]
    return X_lab[idx], X_unl[rng.choice(len(X_unl), 100, replace=False)]


def shap_xgboost(X_pos, X_bg, X_ex):
    X = np.vstack([X_pos, X_bg])
    y = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg))]
    imp = SimpleImputer(strategy='median').fit(X)
    params = dict(XGB_PARAMS, scale_pos_weight=len(X_bg) / len(X_pos), random_state=config.RANDOM_SEED)
    job = pickle.dumps({'X': imp.transform(X), 'y': y, 'X_va': imp.transform(X_ex), 'params': params, 'shap': True})
    out = subprocess.run([sys.executable, '-m', 'paper_analyses.xgb_worker'], input=job, capture_output=True, cwd=REPO,
                         env=dict(os.environ, PYTHONPATH=REPO))
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode()[-2000:])
    return pickle.loads(out.stdout)


def shap_bagging_pu(X_pos_raw, X_bg_raw, X_ex_raw, n_bags=200):
    """bagging_pu_factory's production fit (seed_counter 0), tree by tree."""
    X_all, X_ex = _fit_transform(np.vstack([X_pos_raw, X_bg_raw]), X_ex_raw)
    X_pos, X_u = X_all[:len(X_pos_raw)], X_all[len(X_pos_raw):]
    rng = np.random.default_rng(config.RANDOM_SEED)
    k = len(X_pos)
    acc = np.zeros(X_ex.shape)
    for _ in range(n_bags):
        ui = rng.choice(len(X_u), k, replace=True)
        tree = DecisionTreeClassifier(max_features='sqrt', random_state=int(rng.integers(1 << 31)))
        tree.fit(np.vstack([X_pos, X_u[ui]]), np.r_[np.ones(k), np.zeros(k)])
        sv = shap.TreeExplainer(tree).shap_values(X_ex)
        acc += sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
    return acc / n_bags


def shap_nnpu(X_pos_raw, X_bg_raw, X_ex_raw, X_ref_raw, prior=0.01):
    """nnpu_qt_factory's production fit (seed_counter 0): five members."""
    tf = _fin.QuantileNormal(np.vstack([X_pos_raw, X_bg_raw]))
    sv = []
    for k in range(config.DN_RESTARTS):
        m = _train_nnpu(tf(X_pos_raw), tf(X_bg_raw), prior, seed=config.DN_SEED + k, cfg=model_cfg(REGION),
                        epochs=config.DN_EPOCHS, lr=config.DN_LR)
        sv.append(explain_devnet(m, tf(X_ref_raw), tf(X_ex_raw), DEVICE, config.DN_SEED)[0])
    return np.mean(sv, axis=0)


def main():
    seed_everything()
    rec, cols, X_lab, X_unl, _ = _s25.extended_matrices(REGION, 'reencoded')
    y = rec['pruned']['y']
    X_pos = X_lab[y == 1]
    X_ex, X_ref = explain_set(rec, X_lab, X_unl)
    X_raw = SimpleImputer(strategy='median').fit(np.vstack([X_pos, X_unl])).transform(X_ex)
    store = {'cols': cols, 'X_raw': X_raw,
             'shap_xgb': shap_xgboost(X_pos, X_unl, X_ex),
             'shap_bpu': shap_bagging_pu(X_pos, X_unl, X_ex),
             'shap_nnpu': shap_nnpu(X_pos, X_unl, X_ex, X_ref)}
    save_pickle(store, out_path('shap_fs3_others.pkl'))
    rows = []
    for key, name in (('shap_xgb', 'XGBoost'), ('shap_bpu', 'bagging-PU'), ('shap_nnpu', 'nnPU')):
        imp = np.abs(store[key]).mean(0)
        rank = pd.Series(imp).rank(ascending=False).astype(int).values
        for j, c in enumerate(cols):
            rho, direction = _s3.direction(X_raw[:, j], store[key][:, j])
            rows.append({'model': name, 'feature': c, 'label': pretty(c), 'mean_abs_shap': imp[j], 'rank': rank[j],
                         'rho': rho, 'direction': direction})
    pd.DataFrame(rows).to_csv(out_path('shap_fs3_others_directions.csv'), index=False)
    draw()


def draw(max_display=10):
    sr = load_pickle(out_path('shap_fs3_results.pkl'))[REGION]
    so = load_pickle(out_path('shap_fs3_others.pkl'))
    assert list(so['cols']) == list(sr['cols'])
    names = [pretty(c) for c in sr['cols']]
    panels = [(sr['shap_dn'], sr['X_raw'], '(a) DevNet (DeepSHAP)'), (sr['shap_rf'], sr['X_raw'], '(b) RF (TreeSHAP)'),
              (so['shap_xgb'], so['X_raw'], '(c) XGBoost (TreeSHAP, log-odds)'),
              (so['shap_bpu'], so['X_raw'], '(d) Bagging-PU (TreeSHAP)'), (so['shap_nnpu'], so['X_raw'], '(e) nnPU (DeepSHAP)')]
    fig = plt.figure(figsize=(21, 11))
    for i, (sv, X, title) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 1)
        plt.sca(ax)
        np.random.seed(0)
        shap.summary_plot(sv, X, feature_names=names, max_display=max_display, show=False, plot_size=None,
                          color_bar=(i == len(panels) - 1))
        ax.set_title(title, fontsize=13, loc='left')
        ax.tick_params(labelsize=9.5)
        ax.xaxis.label.set_size(10)
    plt.tight_layout(w_pad=2.5, h_pad=2.0)
    fig.savefig(os.path.join(FIG, 'fig_shap_superior_r7.png'), dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved fig_shap_superior_r7.png', flush=True)


if __name__ == '__main__':
    draw() if '--draw' in sys.argv[1:] else main()
