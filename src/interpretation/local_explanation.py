"""
Section 13f -- Local explanation for DevNet's top-scoring candidate targets.

Waterfall plots for DevNet's top-scoring unlabeled points (capped at 5 here
regardless of TOP_N_TARGETS) -- which features pushed each specific
candidate target above the background baseline. This is the evidence for
"these targets are geologically plausible, not noise": if the features
driving a high score are the same ones the global analysis says matter (and
in the expected direction), that's a much stronger argument than the score
alone. Figures are saved at SHAP_FIG_DPI into SHAP_FIG_DIR.
"""
import os

import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap'])
    import shap

from src.features.feature_matrix import get_raw_features
from src.interpretation.shap_explainers import explain_devnet
from src.models.devnet import get_device


def plot_local_explanations(config, trained, shap_results, shap_regions, shap_fig_dir, shap_fig_dpi=600):
    device = get_device(config.FORCE_CPU)
    os.makedirs(shap_fig_dir, exist_ok=True)

    for name in shap_regions:
        t = trained[name]
        r = shap_results[name]
        top_targets = t.get('top_targets')
        if top_targets is None or len(top_targets) == 0:
            print(f'[{name}] no top_targets to explain -- skipping.')
            continue

        top_local = top_targets.sort_values('dn_score', ascending=False).head(5).reset_index(drop=True)
        pts_all = t['pts']
        top_rows = pts_all.merge(top_local[['lat', 'lon', 'dn_score']], on=['lat', 'lon'], how='inner')
        top_rows = top_rows.sort_values('dn_score', ascending=False).reset_index(drop=True)
        if len(top_rows) == 0:
            print(f'[{name}] could not match top_targets back to feature rows -- skipping.')
            continue

        X_top_raw, _, _ = get_raw_features(top_rows, feat_subset=r['feat_cols'])
        X_top_dn = t['dn_scaler'].transform(t['dn_imp'].transform(X_top_raw))

        shap_top, ev_top, method_top = explain_devnet(t['dn_ensemble'][0], r['X_bg_dn'], X_top_dn,
                                                        device, config.DN_SEED)
        print(f'[{name}] local explanation for top {len(top_rows)} candidate targets ({method_top}):')

        for i in range(len(top_rows)):
            exp = shap.Explanation(
                values=shap_top[i], base_values=ev_top,
                data=X_top_dn[i], feature_names=r['feat_cols'],
            )
            plt.figure()  # shap.plots.waterfall draws on the current axes rather
                           # than creating its own figure -- without this, it
                           # lands on top of whatever plot preceded it.
            shap.plots.waterfall(exp, show=False)
            plt.title(f"{name}: target #{i+1} (lat={top_rows.loc[i,'lat']:.3f}, "
                      f"lon={top_rows.loc[i,'lon']:.3f}, dn_score={top_rows.loc[i,'dn_score']:.3f})",
                      fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_fig_dir, f'{name}_waterfall_target{i+1}.png'),
                        dpi=shap_fig_dpi, bbox_inches='tight')
            plt.show()
