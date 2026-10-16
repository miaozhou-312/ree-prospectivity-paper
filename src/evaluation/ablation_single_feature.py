"""
Section 13g -- Single-feature ablation (partial predictive power).

Question: how much predictive power does each input feature carry on its
own, independent of every other feature? SHAP measures a feature's
contribution *within* the full model, including interactions. Single-feature
ablation asks the complementary, simpler question: train a model on ONLY
that one feature (plus the mandatory imputer) and see how far it gets alone
-- the nonlinear-model analogue of a linear model's partial R². The two are
NOT substitutes for each other; disagreement between the two rankings is
itself informative, not a bug.

Design: for each region, for every feature in the region's pruned set,
train RF and DevNet on that feature ALONE, using the exact same shared-
background pools, CV folds, and pooled_oof_eval protocol as the main
experiment -- only the feature-matrix width changes. Single 5-fold
pooled_oof_eval only (not repeated CV), and DevNet's restart count is
halved to keep this affordable across every feature in every region.
"""
import pandas as pd

from src.evaluation.cv_protocol import pooled_oof_eval
from src.evaluation.run_experiment import _dn_train_kwargs
from src.features.feature_matrix import get_raw_features
from src.models.devnet import devnet_semisup_fold_scorer_factory, get_device
from src.models.random_forest import make_rf_params, rf_shared_bg_fold_scorer_factory


def run_single_feature_ablation(config, trained, region_feats, results, run_regions=None):
    run_regions = list(config.RUN_REGIONS) if run_regions is None else list(run_regions)
    device = get_device(config.FORCE_CPU)
    dn_train_kwargs = _dn_train_kwargs(config)
    rf_params = make_rf_params(config.RANDOM_SEED)
    single_feat_dn_restarts = max(1, config.DN_RESTARTS // 2)
    print(f'Single-feature ablation: single 5-fold pooled_oof_eval only (no repeated CV), '
          f'DevNet restarts={single_feat_dn_restarts} (main experiment uses {config.DN_RESTARTS})')

    single_feat_results = {}
    for name in run_regions:
        pts         = trained[name]['pts']
        feat_cols   = region_feats[name]
        model_cfg   = trained[name]['model_cfg']

        pts_labeled   = pts[pts['pool'].isin(['positive', 'eval_neg'])].reset_index(drop=True)
        pts_unlabeled = pts[pts['pool'] == 'train_unlabeled'].reset_index(drop=True)

        print(f'\n[{name}] single-feature ablation over {len(feat_cols)} pruned features ...')
        rows = []
        for feat in feat_cols:
            X_lab,   y, _ = get_raw_features(pts_labeled,   feat_subset=[feat])
            X_unlab, _, _ = get_raw_features(pts_unlabeled, feat_subset=[feat])

            rf_scorer = rf_shared_bg_fold_scorer_factory(X_unlab, rf_params)
            dn_scorer = devnet_semisup_fold_scorer_factory(
                X_unlab, model_cfg=model_cfg, device=device, train_kwargs=dn_train_kwargs,
                restarts=single_feat_dn_restarts, base_seed=config.DN_SEED)

            rf_auc, rf_ap, _ = pooled_oof_eval(X_lab, y, rf_scorer,
                                                n_splits=config.CV_N_SPLITS, seed=config.RANDOM_SEED)
            dn_auc, dn_ap, _ = pooled_oof_eval(X_lab, y, dn_scorer,
                                                n_splits=config.CV_N_SPLITS, seed=config.RANDOM_SEED)

            rows.append({'feature': feat, 'RF_auc': rf_auc, 'RF_ap': rf_ap,
                         'DevNet_auc': dn_auc, 'DevNet_ap': dn_ap})

        df = pd.DataFrame(rows).sort_values('RF_ap', ascending=False).reset_index(drop=True)
        assert len(df) == len(feat_cols), \
            f'[{name}] expected {len(feat_cols)} rows, got {len(df)} -- some feature was skipped'
        single_feat_results[name] = df

        ref_rf_auc = results[name]['RF']['pruned']['oof_auc']
        ref_rf_ap  = results[name]['RF']['pruned']['oof_ap']
        ref_dn_auc = results[name]['DevNet(deviation)']['pruned']['oof_auc']
        ref_dn_ap  = results[name]['DevNet(deviation)']['pruned']['oof_ap']

        print(f'\n=== [{name}] Single-feature ablation ({len(feat_cols)} features, sorted by RF_ap desc) ===')
        print(f'  Full-pruned-model baseline (main experiment): '
              f'RF AUC={ref_rf_auc:.3f} AP={ref_rf_ap:.3f}  |  DevNet AUC={ref_dn_auc:.3f} AP={ref_dn_ap:.3f}')
        pd.set_option('display.width', 160)
        print(df.round(4).to_string(index=False))

        close_or_exceeds = df[(df['RF_ap'] >= 0.9 * ref_rf_ap) | (df['DevNet_ap'] >= 0.9 * ref_dn_ap)]
        if len(close_or_exceeds):
            print(f'\n  FLAG -- feature(s) whose single-feature AP is within 90% of (or exceeds) the '
                  f'full-pruned-model baseline (i.e. carrying most of the signal alone):')
            for _, r in close_or_exceeds.iterrows():
                flags = []
                if r['RF_ap'] >= ref_rf_ap:
                    flags.append(f"RF_ap={r['RF_ap']:.3f} >= baseline {ref_rf_ap:.3f}")
                elif r['RF_ap'] >= 0.9 * ref_rf_ap:
                    flags.append(f"RF_ap={r['RF_ap']:.3f} within 90% of baseline {ref_rf_ap:.3f}")
                if r['DevNet_ap'] >= ref_dn_ap:
                    flags.append(f"DevNet_ap={r['DevNet_ap']:.3f} >= baseline {ref_dn_ap:.3f}")
                elif r['DevNet_ap'] >= 0.9 * ref_dn_ap:
                    flags.append(f"DevNet_ap={r['DevNet_ap']:.3f} within 90% of baseline {ref_dn_ap:.3f}")
                print(f"    {r['feature']}: " + '; '.join(flags))
        else:
            print(f'\n  No single feature approaches the full-model baseline -- consistent with the '
                  f'model relying on combinations/interactions of features, not one dominant signal.')

    print('\n=== Reminder: single-feature ablation vs SHAP -- complementary, not interchangeable ===')
    print('Single-feature ablation evaluates each feature in total isolation (no interactions), so its')
    print('ranking need NOT match SHAP importance. Disagreement between the two is expected and itself')
    print('informative -- it is exactly the interaction signal a linear partial-R^2 analogue cannot see.')
    return single_feat_results


def plot_single_feature_ap(name, df):
    """Horizontal bar chart, one figure per region -- RF vs DevNet
    single-feature AP side by side."""
    import numpy as np
    import matplotlib.pyplot as plt

    df_sorted = df.sort_values('RF_ap', ascending=True)  # ascending -> highest ends up at top after invert
    y = np.arange(len(df_sorted))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_sorted) * 0.28)))
    ax.barh(y + h / 2, df_sorted['RF_ap'],     height=h, color='#2a78d6', label='RF')
    ax.barh(y - h / 2, df_sorted['DevNet_ap'], height=h, color='#d03b3b', label='DevNet')
    ax.set_yticks(y); ax.set_yticklabels(df_sorted['feature'], fontsize=8)
    ax.set_xlabel('Single-feature pooled OOF AP')
    ax.set_title(f'{name}: single-feature ablation (AP) -- RF vs DevNet', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.25); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
