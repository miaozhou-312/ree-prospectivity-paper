"""
Section 4 -- Build one region's full feature table.

Orchestrates sampling (make_points) + every feature-extraction step
(terrane, fault/margin proximity, worm proximity, raster sampling, GLCM
texture), drops candidates outside raster coverage, then draws the two
disjoint shared background pools (`eval_neg`, `train_unlabeled`) that RF
and DevNet are both trained/evaluated against -- see the module docstring
in src/models/random_forest.py and src/models/devnet.py for how each model
consumes them.
"""
import os

import numpy as np
import pandas as pd

from src.features.glcm import add_glcm_features
from src.features.proximity import add_proximity_line_layers, add_proximity_worms, add_rasters
from src.features.sampling import make_points, select_negatives_random
from src.features.terrane import add_terrane_h3


def build_region(name, cfg, worm_trees, clipped_rasters, clipped_vectors, config, glcm_rasters,
                  output_csv=True):
    """Build the full (positive + eval_neg + train_unlabeled) feature table
    for one region.

    Parameters
    ----------
    name : str
        Region name (key into `config.REGIONS`).
    cfg : dict
        `config.REGIONS[name]` -- `cfg['vectors']`/`cfg['points']` are the
        discovered line-/point-feature layers (src/features/data_discovery.py),
        shared identically across every region.
    worm_trees : dict
        Output of `build_worm_trees(cfg['points'], ...)` (src/features/proximity.py).
    clipped_rasters, clipped_vectors : dict
        `CLIPPED_RASTERS[name]` / `CLIPPED_VECTORS[name]` from
        src/features/raster_preprocessing.py (preprocess_all must have run).
    config : module
        src.config (or an object exposing the same attributes) -- passed
        explicitly rather than imported at call time so callers can swap in
        an ablation-specific config without monkeypatching.
    glcm_rasters : list of str
        Raster keys to compute GLCM texture features for (see
        src/features/data_discovery.py's choose_glcm_rasters).
    output_csv : bool
        If True, writes `{name}_features_semisup_combined_dedup.csv` to
        `config.TABLE_DIR` (default '.', the invocation directory, matching
        the original notebook).
    """
    print(f'\n[{name}] building features ...')
    rng = np.random.default_rng(config.RANDOM_SEED)

    pts = make_points(cfg, rng, name, config.POSITIVE_DEDUP_RADIUS_KM,
                       buffer_km=config.BUFFER_KM, grid_deg=config.GRID_DEG)

    print('  extracting terrane attributes (H3 Grid) ...')
    pts = add_terrane_h3(pts, clipped_vectors.get('h3_grid'))

    print(f"  computing line-feature proximity ({', '.join(sorted(cfg['vectors']))}) ...")
    pts = add_proximity_line_layers(pts, cfg['vectors'], config.PROJECTED_CRS)

    print('  computing point-feature proximity ...')
    pts = add_proximity_worms(pts, worm_trees, config.PROJECTED_CRS)

    print('  sampling rasters ...')
    pts = add_rasters(pts, clipped_rasters)

    print('  extracting GLCM texture features ...')
    pts = add_glcm_features(pts, clipped_rasters, glcm_rasters)

    # Drop candidates where ALL rasters are NaN (outside raster coverage)
    raster_cols = list(clipped_rasters.keys())
    pts = pts[pts[raster_cols].notna().any(axis=1)].reset_index(drop=True)

    # ── Shared background pools (RF and DevNet see IDENTICAL background
    # locations; only the label differs -- 0 for RF, unlabeled for DevNet).
    # Draw the EVALUATION pool first, then the TRAINING pool from what's
    # left, guaranteeing disjointness.
    n_candidates = int((pts['label'] == -1).sum())
    n_needed = config.N_TRAIN_BG + config.N_EVAL_BG
    if n_needed > n_candidates:
        shrink = n_candidates / n_needed
        n_train_bg = max(1, int(round(config.N_TRAIN_BG * shrink)))
        n_eval_bg  = max(1, int(round(config.N_EVAL_BG * shrink)))
        print(f'  WARNING: N_TRAIN_BG+N_EVAL_BG={n_needed} exceeds {n_candidates} available '
              f'A∧B candidates -- scaling BOTH pools down proportionally (ratio preserved) to '
              f'{n_train_bg}+{n_eval_bg}={n_train_bg + n_eval_bg}.')
    else:
        n_train_bg, n_eval_bg = config.N_TRAIN_BG, config.N_EVAL_BG
    print(f'  [{name}] shared pools: N_TRAIN_BG={n_train_bg}  N_EVAL_BG={n_eval_bg}  '
          f'(of {n_candidates} A∧B candidates)')

    print(f'  selecting {n_eval_bg} shared EVALUATION background points (random) ...')
    eval_tbl = select_negatives_random(pts, n_eval_bg, seed=config.RANDOM_SEED)
    eval_neg_tbl = eval_tbl[eval_tbl['label'] == 0].reset_index(drop=True)

    # Remove the chosen evaluation background from the candidate pool so the
    # training-background draw below is guaranteed disjoint from it.
    cand_key    = pd.MultiIndex.from_arrays([pts['lat'].values, pts['lon'].values])
    eval_key    = pd.MultiIndex.from_arrays([eval_neg_tbl['lat'].values, eval_neg_tbl['lon'].values])
    is_eval_neg = pd.Series(cand_key.isin(eval_key), index=pts.index) & (pts['label'] == -1)
    pts_remaining = pts[(pts['label'] == 1) | ((pts['label'] == -1) & ~is_eval_neg)].reset_index(drop=True)

    print(f'  selecting {n_train_bg} shared TRAINING background points (random) ...')
    unlabeled_tbl = select_negatives_random(pts_remaining, n_train_bg, seed=config.RANDOM_SEED)
    unlabeled_neg_tbl = unlabeled_tbl[unlabeled_tbl['label'] == 0].reset_index(drop=True)

    pos_tbl = pts[pts['label'] == 1].reset_index(drop=True)

    pos_tbl           = pos_tbl.copy();           pos_tbl['pool']           = 'positive'
    eval_neg_tbl      = eval_neg_tbl.copy();      eval_neg_tbl['pool']      = 'eval_neg'
    unlabeled_neg_tbl = unlabeled_neg_tbl.copy();  unlabeled_neg_tbl['pool'] = 'train_unlabeled'

    pts_out = pd.concat([pos_tbl, eval_neg_tbl, unlabeled_neg_tbl], ignore_index=True)

    npos, n_eval, n_unlab = len(pos_tbl), len(eval_neg_tbl), len(unlabeled_neg_tbl)
    print(f'  final: {npos} positive | {n_eval} shared evaluation background | '
          f'{n_unlab} shared training background (RF label=0 / DevNet unlabeled) | '
          f'{pts_out.shape[1]} columns')

    if output_csv:
        table_dir = getattr(config, 'TABLE_DIR', '.')
        os.makedirs(table_dir, exist_ok=True)
        out_csv = os.path.join(table_dir, f'{name}_features_semisup_combined_dedup.csv')
        pts_out.to_csv(out_csv, index=False)
    return pts_out
