"""
Section 11b -- DN_MARGIN sensitivity ablation.

DN_MARGIN=5 (the deviation loss's margin) was carried over from Pang et al.
(2019)'s default and never actually validated on this data. This checks
whether the main conclusions are sensitive to that choice.

Scope (deliberately narrow to control compute): DN_MARGIN in [2, 3, 5, 8];
DevNet(deviation) only (RF is never retrained -- margin doesn't affect it);
pruned feature set only; a single pooled_oof_eval (5-fold), not repeated
CV.

Mandatory self-check: the margin=5 row must reproduce Section 9's
DevNet(deviation)/pruned row EXACTLY (same seeds, same CV splits, same
data) -- `run_margin_ablation` raises if it doesn't, since that would mean
the ablation's call path has drifted from the main experiment and the other
margin values can't be trusted.
"""
import numpy as np
import pandas as pd

from src.evaluation.cv_protocol import pooled_oof_eval
from src.evaluation.run_experiment import _dn_train_kwargs
from src.features.feature_matrix import get_raw_features
from src.models.devnet import devnet_semisup_fold_scorer_factory, get_device

DN_MARGIN_ABLATION_VALUES = [2, 3, 5, 8]


def run_margin_ablation(config, region_pts, region_feats, trained, results,
                         margin_values=DN_MARGIN_ABLATION_VALUES, run_regions=None):
    run_regions = list(config.RUN_REGIONS) if run_regions is None else list(run_regions)
    device = get_device(config.FORCE_CPU)
    dn_train_kwargs = _dn_train_kwargs(config)

    rows = []
    for name in run_regions:
        pts         = region_pts[name]
        feat_cols   = region_feats[name]
        model_cfg   = trained[name]['model_cfg']

        pts_labeled   = pts[pts['pool'].isin(['positive', 'eval_neg'])].reset_index(drop=True)
        pts_unlabeled = pts[pts['pool'] == 'train_unlabeled'].reset_index(drop=True)
        X_lab,   y, _ = get_raw_features(pts_labeled,   feat_subset=feat_cols)
        X_unlab, _, _ = get_raw_features(pts_unlabeled, feat_subset=feat_cols)

        print(f'[{name}] DN_MARGIN sensitivity ({len(feat_cols)} pruned features, model_cfg={model_cfg}) ...')
        for margin in margin_values:
            scorer = devnet_semisup_fold_scorer_factory(
                X_unlab, model_cfg=model_cfg, device=device, train_kwargs=dn_train_kwargs,
                restarts=config.DN_RESTARTS, base_seed=config.DN_SEED, margin=margin)
            auc_oof, ap_oof, _ = pooled_oof_eval(
                X_lab, y, scorer, n_splits=config.CV_N_SPLITS, seed=config.RANDOM_SEED)
            print(f'    margin={margin}: pooled OOF AUC={auc_oof:.3f}  AP={ap_oof:.3f}')
            rows.append({'region': name, 'margin': margin, 'oof_auc': auc_oof, 'oof_ap': ap_oof})

    margin_ablation_df = pd.DataFrame(rows)
    pd.set_option('display.width', 140)
    print('\n=== DN_MARGIN ablation table (pooled OOF, pruned, DevNet(deviation) only) ===')
    print(margin_ablation_df.round(4).to_string(index=False))

    _self_check(margin_ablation_df, results, run_regions)
    return margin_ablation_df


def _self_check(margin_ablation_df, results, run_regions):
    print("\n=== Self-check: margin=5 must exactly match Section 9's DevNet(deviation)/pruned row ===")
    self_check_ok = True
    for name in run_regions:
        row = margin_ablation_df[(margin_ablation_df['region'] == name) &
                                  (margin_ablation_df['margin'] == 5)].iloc[0]
        ref = results[name]['DevNet(deviation)']['pruned']
        match_auc = bool(np.isclose(row['oof_auc'], ref['oof_auc'], atol=1e-6))
        match_ap  = bool(np.isclose(row['oof_ap'],  ref['oof_ap'],  atol=1e-6))
        ok = match_auc and match_ap
        self_check_ok &= ok
        print(f"  [{name}] ablation(margin=5): AUC={row['oof_auc']:.4f} AP={row['oof_ap']:.4f}  |  "
              f"main experiment: AUC={ref['oof_auc']:.4f} AP={ref['oof_ap']:.4f}  -> {'OK' if ok else 'MISMATCH'}")

    if not self_check_ok:
        raise RuntimeError(
            'DN_MARGIN ablation self-check FAILED: margin=5 does not reproduce the main '
            "experiment's DevNet(deviation)/pruned row exactly. The ablation's call path (data, "
            'model_cfg, seeds, or CV split) has drifted from the main experiment -- fix that '
            'before trusting any of the other margin values above.')
    print('\nSelf-check PASSED: the ablation call path is verified identical to the main experiment.')


def plot_margin_ablation(margin_ablation_df, run_regions, margin_values=DN_MARGIN_ABLATION_VALUES):
    import matplotlib.pyplot as plt

    region_colors = {'Cordillera': '#1a6faf', 'Superior': '#2ad633', 'Grenville': '#d73027'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in run_regions:
        sub = margin_ablation_df[margin_ablation_df['region'] == name].sort_values('margin')
        c = region_colors.get(name)
        axes[0].plot(sub['margin'], sub['oof_ap'],  marker='o', color=c, lw=2, label=name)
        axes[1].plot(sub['margin'], sub['oof_auc'], marker='o', color=c, lw=2, label=name)

    axes[0].axvline(5, color='grey', ls=':', lw=1, alpha=0.6, label='DN_MARGIN default (5)')
    axes[0].set_xlabel('DN_MARGIN'); axes[0].set_ylabel('Pooled OOF AP')
    axes[0].set_title('DevNet(deviation): AP vs. DN_MARGIN', fontsize=10)
    axes[0].set_xticks(margin_values)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25); axes[0].set_axisbelow(True)

    axes[1].axvline(5, color='grey', ls=':', lw=1, alpha=0.6, label='DN_MARGIN default (5)')
    axes[1].set_xlabel('DN_MARGIN'); axes[1].set_ylabel('Pooled OOF AUC')
    axes[1].set_title('DevNet(deviation): AUC vs. DN_MARGIN', fontsize=10)
    axes[1].set_xticks(margin_values)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25); axes[1].set_axisbelow(True)

    plt.tight_layout(); plt.show()

    print('=== AP sensitivity to DN_MARGIN (max - min per region) ===')
    for name in run_regions:
        sub = margin_ablation_df[margin_ablation_df['region'] == name]
        ap_min, ap_max = sub['oof_ap'].min(), sub['oof_ap'].max()
        best_margin  = int(sub.loc[sub['oof_ap'].idxmax(), 'margin'])
        worst_margin = int(sub.loc[sub['oof_ap'].idxmin(), 'margin'])
        print(f'  [{name}] AP range: {ap_min:.3f} (margin={worst_margin})  -  '
              f'{ap_max:.3f} (margin={best_margin})   Δ={ap_max - ap_min:.3f}')
