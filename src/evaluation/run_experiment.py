"""
Section 9 -- Run the full evaluation for every region x model: build
features, prune collinear features, evaluate RF and DevNet(deviation) under
the shared-background-pool protocol, then fit each region's production
models and export DevNet's top candidate targets.

Shared-background design: `eval_neg` and `train_unlabeled` are drawn once
per region (build_region, disjoint by construction) and used IDENTICALLY by
both models -- RF and DevNet are trained on the exact same set of
background locations, differing only in whether those locations are
labelled negative (RF) or left unlabelled (DevNet). This isolates "modeling
paradigm" as the only variable between the two models.

- RF uses `rf_shared_bg_fold_scorer_factory(train_unlabeled)`.
- DevNet(deviation) uses `devnet_semisup_fold_scorer_factory`.

Both go through the same `pooled_oof_eval` / `repeated_pooled_auc` protocol,
scored against the identical `eval_neg` pool.

Production models are then fit on ALL of a region's data, both on the same
shared background pool: RF on positive + train_unlabeled (labeled 0);
DevNet as a DN_RESTARTS-model ensemble on positive + train_unlabeled
(unlabeled). Finally, the production DevNet ensemble scores every point in
train_unlabeled and the top TOP_N_TARGETS are saved as candidate new
targets.
"""
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.evaluation.cv_protocol import pooled_oof_eval, repeated_pooled_auc
from src.features.build_region import build_region
from src.features.collinearity import plot_collinearity_heatmap, prune_collinear
from src.features.feature_matrix import feature_cols, get_raw_features
from src.features.proximity import build_worm_trees
from src.models.devnet import (devnet_semisup_fold_scorer_factory, ensemble_score,
                                get_device, train_devnet)
from src.models.random_forest import make_rf_params, make_rf_pipeline, rf_shared_bg_fold_scorer_factory


def _dn_train_kwargs(config):
    return dict(dn_lr=config.DN_LR, dn_batch=config.DN_BATCH, dn_epochs=config.DN_EPOCHS,
                rank_neg_ratio=config.RANK_NEG_RATIO,
                loss_margin={'deviation': config.DN_MARGIN, 'ranking': config.RANK_MARGIN})


def run_all_regions(config, clipped_rasters, clipped_vectors, glcm_rasters, run_regions=None, plot=True):
    """Run the full evaluation + production-model fit for every region in
    `run_regions` (defaults to config.RUN_REGIONS).

    Returns (region_pts, region_feats, results, trained, worm_trees):
      region_pts[name]   -- full (positive + eval_neg + train_unlabeled) table
      region_feats[name] -- pruned (collinearity-reduced) feature list
      results[name][model][feature_set] -- evaluation metrics dict
      trained[name]      -- production models + top candidate targets
      worm_trees          -- built once, shared across all regions
    """
    run_regions = list(config.RUN_REGIONS) if run_regions is None else list(run_regions)
    device = get_device(config.FORCE_CPU)
    dn_train_kwargs = _dn_train_kwargs(config)
    rf_params = make_rf_params(config.RANDOM_SEED)

    print('Building point-feature KDTrees ...')
    point_cfg = list(config.REGIONS.values())[0]['points']
    worm_trees = build_worm_trees(point_cfg, config.PROJECTED_CRS)

    region_pts, region_feats, results, trained = {}, {}, {}, {}

    for name in run_regions:
        cfg = config.REGIONS[name]
        print(f'\n{"="*70}\n[{name}]\n{"="*70}')
        pts = build_region(name, cfg, worm_trees, clipped_rasters[name], clipped_vectors[name],
                            config, glcm_rasters)
        region_pts[name] = pts

        pts_labeled   = pts[pts['pool'].isin(['positive', 'eval_neg'])].reset_index(drop=True)
        pts_unlabeled = pts[pts['pool'] == 'train_unlabeled'].reset_index(drop=True)

        feat_cols_full = feature_cols(pts)
        pruned_cols = prune_collinear(pts, feat_cols_full, thresh=config.COLLINEARITY_THRESH)
        region_feats[name] = pruned_cols
        if plot:
            plot_collinearity_heatmap(pts, feat_cols_full, pruned_cols, name, thresh=config.COLLINEARITY_THRESH)
        print(f'  [{name}] collinearity pruning: {len(feat_cols_full)} -> {len(pruned_cols)} features')

        X_full_labeled,   y, feat_cols_full_out = get_raw_features(pts_labeled)
        X_pruned_labeled, _, feat_cols_pruned   = get_raw_features(pts_labeled, feat_subset=pruned_cols)
        X_full_unlabeled,   _, _ = get_raw_features(pts_unlabeled)
        X_pruned_unlabeled, _, _ = get_raw_features(pts_unlabeled, feat_subset=pruned_cols)

        n_pos, n_eval_neg = int((y == 1).sum()), int((y == 0).sum())
        print(f'  positives={n_pos}  eval_near_negatives={n_eval_neg}  '
              f'unlabeled_pool={len(pts_unlabeled)}  full_features={len(feat_cols_full_out)}')

        model_cfg = config.REGION_MODEL_CFG.get(name, config.DEFAULT_MODEL_CFG)
        print(f'  model_cfg = {model_cfg}')

        results[name] = {}
        feature_sets = {
            'full':   (X_full_labeled,   feat_cols_full_out, X_full_unlabeled),
            'pruned': (X_pruned_labeled, feat_cols_pruned,   X_pruned_unlabeled),
        }
        for model_name in ['RF'] + [f'DevNet({lt})' for lt in config.RUN_LOSSES]:
            results[name][model_name] = {}
            for fs_name, (X_lab, cols_fs, X_unlab) in feature_sets.items():
                if model_name == 'RF':
                    scorer = rf_shared_bg_fold_scorer_factory(X_unlab, rf_params)
                    paradigm = 'discriminative (positive vs. shared background, labeled negative)'
                else:
                    scorer = devnet_semisup_fold_scorer_factory(
                        X_unlab, model_cfg=model_cfg, device=device, train_kwargs=dn_train_kwargs,
                        restarts=config.DN_RESTARTS, base_seed=config.DN_SEED)
                    paradigm = 'semi-supervised (positive vs. shared background, left unlabeled)'

                print(f'  -- {model_name} [{fs_name}, {len(cols_fs)} feats, {paradigm}] --')
                auc_oof, ap_oof, oof_rank = pooled_oof_eval(
                    X_lab, y, scorer, n_splits=config.CV_N_SPLITS, seed=config.RANDOM_SEED)
                print(f'    pooled OOF (5-fold):        AUC={auc_oof:.3f}  AP={ap_oof:.3f}')
                rep_aucs = repeated_pooled_auc(
                    X_lab, y, scorer, n_splits=config.CV_N_SPLITS,
                    n_repeats=config.CV_N_REPEATS, seed=config.RANDOM_SEED)
                med, lo, hi = np.median(rep_aucs), *np.percentile(rep_aucs, [2.5, 97.5])
                print(f'    repeated pooled OOF (5x{config.CV_N_REPEATS}): median AUC={med:.3f}  '
                      f'95% CI=[{lo:.3f}, {hi:.3f}]')
                results[name][model_name][fs_name] = {
                    'n_features': len(cols_fs),
                    'data_paradigm': paradigm,
                    'oof_auc': auc_oof, 'oof_ap': ap_oof,
                    'rep_aucs': rep_aucs, 'rep_median': med, 'rep_ci': (lo, hi),
                    'oof_rank': oof_rank,
                    'oof_y':    y,
                }

        # ── Production models: PRUNED feature set only, BOTH trained on the
        # SAME shared background locations (X_pruned_unlabeled).
        X_pos_pruned_rf = X_pruned_labeled[y == 1]
        X_rf_train_raw  = np.vstack([X_pos_pruned_rf, X_pruned_unlabeled])
        y_rf_train      = np.concatenate([np.ones(len(X_pos_pruned_rf)), np.zeros(len(X_pruned_unlabeled))])
        rf_pipeline = make_rf_pipeline(rf_params)
        rf_pipeline.fit(X_rf_train_raw, y_rf_train)

        X_pos_pruned   = X_pruned_labeled[y == 1]
        X_dn_train_raw = np.vstack([X_pos_pruned, X_pruned_unlabeled])
        y_dn_train     = np.concatenate([np.ones(len(X_pos_pruned)), np.zeros(len(X_pruned_unlabeled))])
        dn_imputer = SimpleImputer(strategy='median').fit(X_dn_train_raw)
        dn_scaler  = StandardScaler().fit(dn_imputer.transform(X_dn_train_raw))
        X_dn_train = dn_scaler.transform(dn_imputer.transform(X_dn_train_raw))

        dn_ensemble, dn_losses = [], []
        for k in range(config.DN_RESTARTS):
            record = (k == 0)
            m, losses = train_devnet(X_dn_train, y_dn_train, 'deviation', seed=config.DN_SEED + k,
                                      model_cfg=model_cfg, device=device, record_losses=record,
                                      **dn_train_kwargs)
            dn_ensemble.append(m)
            if record:
                dn_losses = losses

        X_unlab_scaled = dn_scaler.transform(dn_imputer.transform(X_pruned_unlabeled))
        unlab_scores = ensemble_score(dn_ensemble, X_unlab_scaled, device)
        top_idx = np.argsort(unlab_scores)[::-1][:config.TOP_N_TARGETS]
        top_targets = pts_unlabeled.iloc[top_idx][['lat', 'lon']].copy().reset_index(drop=True)
        top_targets['dn_score'] = unlab_scores[top_idx]
        table_dir = getattr(config, 'TABLE_DIR', '.')
        os.makedirs(table_dir, exist_ok=True)
        top_csv = os.path.join(
            table_dir, f'{name}_top{config.TOP_N_TARGETS}_potential_targets_combined_dedup.csv')
        top_targets.to_csv(top_csv, index=False)
        print(f'  [{name}] top {config.TOP_N_TARGETS} potential targets (highest DevNet score on the '
              f'unlabeled pool) -> {top_csv}')

        trained[name] = {
            'pts': pts, 'feat_cols': pruned_cols,
            'rf_model':  rf_pipeline.named_steps['rf'],
            'rf_scaler': rf_pipeline.named_steps['scaler'],
            'rf_imp':    rf_pipeline.named_steps['imputer'],
            'dn_ensemble': dn_ensemble, 'dn_scaler': dn_scaler, 'dn_imp': dn_imputer,
            'dn_losses': dn_losses, 'model_cfg': model_cfg, 'top_targets': top_targets,
        }
        print(f'[{name}] evaluation + production fit complete '
              f'(RF and DevNet trained on the IDENTICAL shared background locations, '
              f'differing only in label -- 0 for RF, unlabeled for DevNet; both use pruned features).')

    print(f'\nRegions evaluated: {run_regions}  |  Losses: {config.RUN_LOSSES}')
    return region_pts, region_feats, results, trained, worm_trees


def plot_devnet_loss_curves(trained, loss_type):
    """Section 10 -- training-loss curve for the first ensemble member of
    each region's production DevNet (LOSS_TYPE, seed=DN_SEED)."""
    import matplotlib.pyplot as plt

    for name, t in trained.items():
        losses = t['dn_losses']
        alpha = 2 / (50 + 1)
        ema = [losses[0]]
        for l in losses[1:]:
            ema.append(alpha * l + (1 - alpha) * ema[-1])
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(losses, lw=0.5, alpha=0.3, color='#8b1a1a', label='raw')
        ax.plot(ema,    lw=1.5, color='#8b1a1a', label='EMA-50')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title(f'{name}: DevNet({loss_type}) training loss — production model')
        ax.legend(fontsize=8); plt.tight_layout(); plt.show()
