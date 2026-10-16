"""
Section 11c -- Background-sampling sensitivity analysis.

Theoretical basis: Zuo, R. and Wang, J. (2020) identify the selection of
negative/unlabeled samples itself as a distinct source of uncertainty in
mineral prospectivity mapping, separate from model choice. Xiong, Y. and
Zuo, R. (2021) tested this with 10 random unlabeled-sampling repeats.

What varies, what stays fixed: only the background/unlabeled sampling seed
changes across the draws below. The positive set is completely fixed (same
deposits, same dedup). Each draw independently redraws N_BACKGROUND
(eval_neg, affects RF) and N_UNLABELED (train_unlabeled, affects DevNet) as
two still-disjoint pools. Pruned feature set only, reusing region_feats[name]
as-is (not recomputed per resample). A single pooled_oof_eval (5-fold) is
used per draw, not repeated CV.

Performance note: feature extraction for the full A∧B-domain candidate pool
is done ONCE per region and cached (`build_full_candidate_pools`) -- every
resample below only selects different rows from that cached table.

Mandatory reporting policy: all draws are reported -- table, per-region/
model median/min/max summary, and a figure -- with no cherry-picking of the
best run. This deliberately departs from Xiong & Zuo (2021), who report
only the single highest-AUC run out of 10 repeats.
"""
import numpy as np
import pandas as pd

from src.evaluation.cv_protocol import pooled_oof_eval
from src.evaluation.run_experiment import _dn_train_kwargs
from src.features.feature_matrix import get_raw_features
from src.features.glcm import add_glcm_features
from src.features.proximity import add_proximity_line_layers, add_proximity_worms, add_rasters
from src.features.sampling import make_points, select_negatives_random
from src.features.terrane import add_terrane_h3
from src.models.devnet import devnet_semisup_fold_scorer_factory, get_device
from src.models.random_forest import make_rf_params, rf_fold_scorer_factory

BACKGROUND_SEEDS = [101, 102, 103, 104, 105]   # distinct from RANDOM_SEED=42 used everywhere else


def build_full_candidate_pool_features(name, cfg, worm_trees, clipped_rasters, clipped_vectors, config,
                                        glcm_rasters):
    """Runs make_points + the SAME feature-extraction steps as build_region,
    but WITHOUT build_region's N_TRAIN_BG/N_EVAL_BG subsampling -- returns
    ALL A∧B-domain candidates (label==-1) plus the positives (label==1),
    features already attached. Called ONCE per region; every seed in the
    sensitivity analysis then only resamples ROWS from this cached table --
    add_glcm_features (the expensive step) is never called a second time
    for the same region."""
    rng = np.random.default_rng(config.RANDOM_SEED)   # unused inside make_points, kept for signature parity
    pts = make_points(cfg, rng, name, config.POSITIVE_DEDUP_RADIUS_KM,
                       buffer_km=config.BUFFER_KM, grid_deg=config.GRID_DEG)
    n_pos = int((pts['label'] == 1).sum())
    n_cand = int((pts['label'] == -1).sum())
    print(f'  [{name}] full A∧B candidate pool: {n_pos} positive, {n_cand} candidates')

    pts = add_terrane_h3(pts, clipped_vectors.get('h3_grid'))
    pts = add_proximity_line_layers(pts, cfg['vectors'], config.PROJECTED_CRS)
    pts = add_proximity_worms(pts, worm_trees, config.PROJECTED_CRS)
    pts = add_rasters(pts, clipped_rasters)
    print(f'  [{name}] extracting GLCM for the full candidate pool (one-time; every '
          f'resample below only selects rows from this table) ...')
    pts = add_glcm_features(pts, clipped_rasters, glcm_rasters)

    raster_cols = list(clipped_rasters.keys())
    pts = pts[pts[raster_cols].notna().any(axis=1)].reset_index(drop=True)
    return pts


def build_full_candidate_pools(config, clipped_rasters, clipped_vectors, worm_trees, glcm_rasters,
                                run_regions=None):
    run_regions = list(config.RUN_REGIONS) if run_regions is None else list(run_regions)
    pools = {}
    for name in run_regions:
        print(f'\n{"="*70}\n[{name}] one-time full-candidate-pool feature extraction\n{"="*70}')
        pools[name] = build_full_candidate_pool_features(
            name, config.REGIONS[name], worm_trees, clipped_rasters[name], clipped_vectors[name],
            config, glcm_rasters)
    return pools


def resample_background_pools(pts_full, seed, config):
    """Redraws N_BACKGROUND eval_neg + N_UNLABELED train_unlabeled (still
    disjoint) from the ALREADY-FEATURED full candidate pool `pts_full` --
    mirrors build_region's own disjoint-pool-construction block exactly,
    just parameterized by `seed` instead of RANDOM_SEED, and reading
    features from the cache instead of recomputing them. Only the sampling
    seed varies -- the positive set is completely fixed."""
    eval_tbl = select_negatives_random(pts_full, config.N_BACKGROUND, seed=seed)
    eval_neg_tbl = eval_tbl[eval_tbl['label'] == 0].reset_index(drop=True)

    cand_key = pd.MultiIndex.from_arrays([pts_full['lat'].values, pts_full['lon'].values])
    eval_key = pd.MultiIndex.from_arrays([eval_neg_tbl['lat'].values, eval_neg_tbl['lon'].values])
    is_eval_neg = pd.Series(cand_key.isin(eval_key), index=pts_full.index) & (pts_full['label'] == -1)
    pts_remaining = pts_full[(pts_full['label'] == 1) | ((pts_full['label'] == -1) & ~is_eval_neg)].reset_index(drop=True)

    unlabeled_tbl = select_negatives_random(pts_remaining, config.N_UNLABELED, seed=seed)
    unlabeled_neg_tbl = unlabeled_tbl[unlabeled_tbl['label'] == 0].reset_index(drop=True)

    pos_tbl = pts_full[pts_full['label'] == 1].reset_index(drop=True)
    pos_tbl = pos_tbl.copy();           pos_tbl['pool']           = 'positive'
    eval_neg_tbl = eval_neg_tbl.copy(); eval_neg_tbl['pool']      = 'eval_neg'
    unlabeled_neg_tbl = unlabeled_neg_tbl.copy(); unlabeled_neg_tbl['pool'] = 'train_unlabeled'
    return pd.concat([pos_tbl, eval_neg_tbl, unlabeled_neg_tbl], ignore_index=True)


def run_background_sensitivity(config, full_candidate_pools, region_feats, trained,
                                background_seeds=BACKGROUND_SEEDS, run_regions=None):
    run_regions = list(config.RUN_REGIONS) if run_regions is None else list(run_regions)
    device = get_device(config.FORCE_CPU)
    dn_train_kwargs = _dn_train_kwargs(config)
    rf_params = make_rf_params(config.RANDOM_SEED)
    rf_fold_scorer = rf_fold_scorer_factory(rf_params)

    rows = []
    for name in run_regions:
        pts_full = full_candidate_pools[name]
        feat_cols = region_feats[name]   # fixed -- reused from the main experiment
        model_cfg = trained[name]['model_cfg']

        print(f'\n{"="*70}\n[{name}] background-sampling sensitivity ({len(background_seeds)} reseeds)\n{"="*70}')
        for seed in background_seeds:
            pts_seed = resample_background_pools(pts_full, seed, config)
            pts_labeled   = pts_seed[pts_seed['pool'].isin(['positive', 'eval_neg'])].reset_index(drop=True)
            pts_unlabeled = pts_seed[pts_seed['pool'] == 'train_unlabeled'].reset_index(drop=True)

            X_lab,   y, _ = get_raw_features(pts_labeled,   feat_subset=feat_cols)
            X_unlab, _, _ = get_raw_features(pts_unlabeled, feat_subset=feat_cols)
            n_pos, n_eval = int((y == 1).sum()), int((y == 0).sum())
            print(f'  seed={seed}: positive={n_pos} eval_neg={n_eval} unlabeled={len(pts_unlabeled)}')

            rf_auc, rf_ap, _ = pooled_oof_eval(X_lab, y, rf_fold_scorer,
                                                n_splits=config.CV_N_SPLITS, seed=config.RANDOM_SEED)
            dn_scorer = devnet_semisup_fold_scorer_factory(
                X_unlab, model_cfg=model_cfg, device=device, train_kwargs=dn_train_kwargs,
                restarts=config.DN_RESTARTS, base_seed=config.DN_SEED)
            dn_auc, dn_ap, _ = pooled_oof_eval(X_lab, y, dn_scorer,
                                                n_splits=config.CV_N_SPLITS, seed=config.RANDOM_SEED)
            print(f'    RF:                 AUC={rf_auc:.3f}  AP={rf_ap:.3f}')
            print(f'    DevNet(deviation):  AUC={dn_auc:.3f}  AP={dn_ap:.3f}')

            rows.append({'region': name, 'model': 'RF', 'seed': seed, 'oof_auc': rf_auc, 'oof_ap': rf_ap})
            rows.append({'region': name, 'model': 'DevNet(deviation)', 'seed': seed,
                          'oof_auc': dn_auc, 'oof_ap': dn_ap})

    bg_sensitivity_df = pd.DataFrame(rows)
    print('\n=== Background-sampling sensitivity: all individual results (no cherry-picking) ===')
    pd.set_option('display.width', 160)
    print(bg_sensitivity_df.round(3).to_string(index=False))

    bg_summary = bg_sensitivity_df.groupby(['region', 'model'])['oof_ap'].agg(
        median='median', min='min', max='max').reset_index()
    print('\n=== Per-region/model AP summary across the resamples ===')
    print(bg_summary.round(3).to_string(index=False))
    return bg_sensitivity_df, bg_summary


def plot_bg_sensitivity(bg_sensitivity_df, results, run_regions):
    """One panel per region: RF vs DevNet(deviation) AP distribution across
    the background-sampling resamples (boxplot + individual seed points),
    with the main experiment's own pooled-OOF AP overlaid as a gold-star
    reference."""
    import matplotlib.pyplot as plt

    colors = {'RF': '#2a78d6', 'DevNet(deviation)': '#d03b3b'}
    models = ['RF', 'DevNet(deviation)']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    rng_j = np.random.RandomState(0)

    for ax, name in zip(axes, run_regions):
        data = [bg_sensitivity_df[(bg_sensitivity_df.region == name) &
                                   (bg_sensitivity_df.model == m)]['oof_ap'].values
                for m in models]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                         showmeans=True, zorder=2)
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=colors[models[i]], alpha=0.35, edgecolor=colors[models[i]])

        for i, (m, vals) in enumerate(zip(models, data)):
            jitter = rng_j.uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=colors[m], s=32,
                       zorder=4, edgecolors='black', linewidths=0.4)
            main_ap = results[name][m]['pruned']['oof_ap']
            ax.scatter([i], [main_ap], marker='*', s=220, color='gold',
                       edgecolors='black', linewidths=0.9, zorder=5,
                       label='Main experiment (seed=RANDOM_SEED)' if i == 0 and name == run_regions[0] else None)

        ax.set_xticks([0, 1]); ax.set_xticklabels(models, fontsize=8)
        ax.set_title(name, fontweight='bold', fontsize=10)
        ax.grid(axis='y', alpha=0.25); ax.set_axisbelow(True)
    axes[0].set_ylabel('Pooled OOF AP (single 5-fold pass)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=1, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f'Background-sampling sensitivity: AP across {len(bg_sensitivity_df["seed"].unique())} '
                 f'independent resamples (all shown, none cherry-picked)', fontsize=11, y=1.14)
    plt.tight_layout()
    plt.show()

    print('\n=== Mandatory reporting policy ===')
    print('All resamples are reported above (table + summary + figure) -- none are cherry-picked.')
    print("This deliberately departs from Xiong & Zuo (2021)'s approach of reporting only the "
          "single highest-AUC run out of 10 repeats, which would inflate the reported performance.")
