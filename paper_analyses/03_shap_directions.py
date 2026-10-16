"""
Stage 3 -- SHAP interpretation with the direction rule fixed
(KNOWN_ISSUES.md #1).

Production models are refit exactly as src.evaluation.run_experiment does
(pruned features; RF on positives + train_unlabeled labelled 0; a
DN_RESTARTS DevNet ensemble on positives + train_unlabeled), SHAP values are
computed with src.interpretation.shap_pipeline unchanged (explain set = all
positives + eval_neg sample, 200 points; DevNet background = 100 unlabelled
points), and then:

* direction = Spearman rho between a feature's value and its SHAP value over
  the explain set (> 0: higher values raise the score), |rho| < 0.1 labelled
  "no clear direction"; for 0/1 dummies the effect is reported as
  mean(SHAP | present) - mean(SHAP | absent);
* the old sign(mean SHAP) rule is reported next to it so every changed
  statement in the thesis text can be traced;
* DevNet SHAP is additionally averaged over all ensemble members (the thesis
  explained member 0 only) to check that rankings are not a single-seed
  artefact.

    python -m paper_analyses.03_shap_directions
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from paper_analyses.common import (REGION_ORDER, config, dn_train_kwargs, load_region,  # noqa: E402
                                   model_cfg, out_path, save_pickle, seed_everything)
from paper_analyses.figstyle import pretty  # noqa: E402
from paper_analyses.models import DEVICE  # noqa: E402
from src.interpretation.shap_explainers import explain_devnet  # noqa: E402
from src.interpretation.shap_pipeline import compute_shap_results  # noqa: E402
from src.models.devnet import train_devnet  # noqa: E402
from src.models.random_forest import make_rf_params, make_rf_pipeline  # noqa: E402

SHAP_BG_N, SHAP_EXPLAIN_N, TOP_K = 100, 200, 10


def fit_production(region):
    rec = load_region(region)
    d = rec['pruned']
    X_pos = d['X_lab'][d['y'] == 1]
    X_tr = np.vstack([X_pos, d['X_unl']])
    y_tr = np.r_[np.ones(len(X_pos)), np.zeros(len(d['X_unl']))]
    rf = make_rf_pipeline(make_rf_params(config.RANDOM_SEED)).fit(X_tr, y_tr)
    imp = SimpleImputer(strategy='median').fit(X_tr)
    sc = StandardScaler().fit(imp.transform(X_tr))
    Xs = sc.transform(imp.transform(X_tr))
    ens = [train_devnet(Xs, y_tr, 'deviation', seed=config.DN_SEED + k, model_cfg=model_cfg(region),
                        device=DEVICE, **dn_train_kwargs())[0] for k in range(config.DN_RESTARTS)]
    return {'pts': rec['pts'], 'feat_cols': d['cols'],
            'rf_model': rf.named_steps['rf'], 'rf_scaler': rf.named_steps['scaler'],
            'rf_imp': rf.named_steps['imputer'], 'dn_ensemble': ens, 'dn_scaler': sc, 'dn_imp': imp}


def _is_binary(x):
    return set(np.unique(x[~np.isnan(x)])).issubset({0.0, 1.0})


def direction(x, s):
    if np.nanstd(x) == 0:
        return np.nan, 'constant'
    if _is_binary(x):
        eff = s[x == 1].mean() - s[x == 0].mean() if (x == 1).any() and (x == 0).any() else np.nan
        rho = spearmanr(x, s)[0]
        lab = 'present -> more' if eff > 0 else 'present -> less'
        return rho, lab if abs(rho) >= 0.1 else 'no clear direction'
    rho = spearmanr(x, s, nan_policy='omit')[0]
    if abs(rho) < 0.1:
        return rho, 'no clear direction'
    return rho, 'higher -> more' if rho > 0 else 'higher -> less'


def main():
    seed_everything()
    trained = {r: fit_production(r) for r in REGION_ORDER}
    shap_res = compute_shap_results(config, trained, REGION_ORDER, SHAP_BG_N, SHAP_EXPLAIN_N)

    rows = []
    for r in REGION_ORDER:
        sr, t = shap_res[r], trained[r]
        # DevNet SHAP averaged over every ensemble member (thesis: member 0 only).
        sv_all = [explain_devnet(m, sr['X_bg_dn'], sr['X_explain_dn'], DEVICE, config.DN_SEED)[0]
                  for m in t['dn_ensemble']]
        sr['shap_dn_ens'] = np.mean(sv_all, axis=0)
        cols = sr['feat_cols']
        imp_rf = np.abs(sr['shap_rf']).mean(0)
        imp_dn = np.abs(sr['shap_dn']).mean(0)
        imp_dn_ens = np.abs(sr['shap_dn_ens']).mean(0)
        rank = lambda v: pd.Series(v).rank(ascending=False).astype(int).values
        r_rf, r_dn, r_dn_ens = rank(imp_rf), rank(imp_dn), rank(imp_dn_ens)
        for j, c in enumerate(cols):
            rho_rf, dir_rf = direction(sr['X_explain_rf'][:, j], sr['shap_rf'][:, j])
            rho_dn, dir_dn = direction(sr['X_explain_dn'][:, j], sr['shap_dn_ens'][:, j])
            rows.append({'region': r, 'feature': c, 'label': pretty(c),
                         'rf_mean_abs_shap': imp_rf[j], 'rf_rank': r_rf[j], 'rf_rho': rho_rf, 'rf_direction': dir_rf,
                         'rf_old_rule': 'more' if sr['shap_rf'][:, j].mean() > 0 else 'less',
                         'dn_mean_abs_shap': imp_dn_ens[j], 'dn_rank': r_dn_ens[j],
                         'dn_rank_member0': r_dn[j], 'dn_rho': rho_dn, 'dn_direction': dir_dn,
                         'dn_old_rule': 'more' if sr['shap_dn'][:, j].mean() > 0 else 'less'})
    df = pd.DataFrame(rows)
    df.to_csv(out_path('shap_directions_all.csv'), index=False)

    for r in REGION_ORDER:
        sub = df[df.region == r].copy()
        sub['joint'] = sub['rf_mean_abs_shap'] / sub['rf_mean_abs_shap'].max() + \
            sub['dn_mean_abs_shap'] / sub['dn_mean_abs_shap'].max()
        top = sub.sort_values('joint', ascending=False).head(TOP_K)
        agree = sum(a.split('->')[-1].strip() == b.split('->')[-1].strip() and 'clear' not in a
                    for a, b in zip(top.rf_direction, top.dn_direction))
        print(f'\n[{r}] top {TOP_K} jointly important features; directions agree on {agree}/{TOP_K}')
        print(top[['label', 'rf_rank', 'rf_rho', 'rf_direction', 'dn_rank', 'dn_rank_member0', 'dn_rho',
                   'dn_direction']].to_string(index=False, float_format=lambda v: f'{v:.2f}'))
        from scipy.stats import spearmanr as _sp
        print(f'  DevNet rank stability member0 vs ensemble: rho={_sp(sub.dn_rank_member0, sub.dn_rank)[0]:.2f}; '
              f'RF vs DevNet importance rank rho={_sp(sub.rf_rank, sub.dn_rank)[0]:.2f}')
        top.to_csv(out_path(f'shap_top{TOP_K}_{r}.csv'), index=False)

    # Beeswarm figure for the focus region (RF | DevNet ensemble).
    r = 'Superior'
    sr = shap_res[r]
    names = [pretty(c) for c in sr['feat_cols']]
    fig = plt.figure(figsize=(14, 6.5))
    for i, (key, X, title) in enumerate([('shap_dn_ens', sr['X_explain_dn'], '(a) DevNet (ensemble-mean SHAP)'),
                                         ('shap_rf', sr['X_explain_rf'], '(b) Random Forest (TreeSHAP)')]):
        ax = fig.add_subplot(1, 2, i + 1)
        plt.sca(ax)
        shap.summary_plot(sr[key], X, feature_names=names, max_display=12, show=False,
                          plot_size=None, color_bar=(i == 1))
        ax.set_title(title, fontsize=12, loc='left')
        ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(out_path('fig_shap_superior.png'), dpi=300, bbox_inches='tight')
    save_pickle({'shap': shap_res}, out_path('shap_results.pkl'))
    print('saved', out_path('fig_shap_superior.png'))


if __name__ == '__main__':
    main()
