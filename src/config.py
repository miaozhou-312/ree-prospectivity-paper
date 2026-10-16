"""
Central configuration for the REE prospectivity-mapping pipeline.

Everything in this module is a plain constant or a dict of paths -- nothing
here has a side effect (no file I/O, no plotting). Every downstream module
imports its constants from here instead of hard-coding them, so the whole
pipeline has exactly one place to change a hyperparameter or a data path.

Data-path philosophy
---------------------
This repository ships NO raw data (rasters, shapefiles, CSVs, deposit
tables). All paths below are *relative* and default to sitting one level
above this repo, mirroring the layout the original research notebook used:

    <parent_of_repo>/
        dataset/            <- DATA_ROOT
        code/                <- this repo
            ...
            <deposit CSVs>   <- DEPOSITS_ROOT (defaults to '.', i.e. wherever
                                the pipeline is actually run from)

Every root can be overridden with an environment variable so the pipeline
can be pointed at data stored anywhere, without editing this file:

    REE_DATA_ROOT       (default: '../dataset')
    REE_DEPOSITS_ROOT     (default: '.')
    REE_TABLE_DIR         (default: '.', output CSV tables -- see TABLE_DIR)
    REE_RASTER_DIR        (default: '{DATA_ROOT}/rasters')
    REE_VECTOR_DIR        (default: '{DATA_ROOT}/vectors')
    REE_POINT_DIR         (default: '{DATA_ROOT}/points')
    REE_OUTLINE_DIR       (default: '{DATA_ROOT}/outlines')
    REE_H3_GRID_DIR       (default: '{DATA_ROOT}/h3_grid')

See data/README.md for exactly what each root is expected to contain.

Nothing region-specific is hand-listed here -- it's all DISCOVERED
--------------------------------------------------------------------
config.py promises no file I/O, and scanning folders is exactly that, so
none of it happens at import time here. Two discovery passes (both in
src/features/data_discovery.py) turn folders of files into everything
`build_regions()` below needs:

  - `discover_all()` -- the raster/vector/point FEATURE LAYERS, shared
    identically across every region (interactive: exclude stray files,
    choose which rasters get GLCM texture features).
  - `discover_regions()` -- the REGIONS themselves: each file in
    OUTLINE_DIR names one region, matched to the single shared file in
    H3_GRID_DIR and to its own deposit CSV in DEPOSITS_ROOT (found by
    filename PREFIX -- a region named 'Foo' matches any 'Foo*.csv').
    Add a new outline file, get a new region -- no code edit needed.

Usage (once, before anything else that reads config.REGIONS):

    from src import config
    from src.features.data_discovery import discover_all, discover_regions

    rasters, vectors, points, glcm_rasters = discover_all(
        config.RASTER_DIR, config.VECTOR_DIR, config.POINT_DIR)
    region_manual = discover_regions(
        config.OUTLINE_DIR, config.H3_GRID_DIR, config.DEPOSITS_ROOT)
    config.REGIONS = config.build_regions(region_manual, rasters, vectors, points)
    config.RUN_REGIONS = list(config.REGIONS)[:1] if config.FAST_MODE else list(config.REGIONS)

`glcm_rasters` is NOT stashed on `config` -- pass it explicitly (like
`model_cfg` or `raster_cols` elsewhere in this codebase) to every function
that needs it: `sampleable_domain.build_all(..., glcm_rasters=glcm_rasters)`,
`run_experiment.run_all_regions(..., glcm_rasters=glcm_rasters)`, etc.
"""
import os

# ── Data roots (override via environment variables, see module docstring) ──
DATA_ROOT      = os.environ.get('REE_DATA_ROOT', '../dataset')
DEPOSITS_ROOT  = os.environ.get('REE_DEPOSITS_ROOT', '.')
RASTER_DIR     = os.environ.get('REE_RASTER_DIR', f'{DATA_ROOT}/rasters')
VECTOR_DIR     = os.environ.get('REE_VECTOR_DIR', f'{DATA_ROOT}/vectors')
POINT_DIR      = os.environ.get('REE_POINT_DIR', f'{DATA_ROOT}/points')
OUTLINE_DIR    = os.environ.get('REE_OUTLINE_DIR', f'{DATA_ROOT}/outlines')
H3_GRID_DIR    = os.environ.get('REE_H3_GRID_DIR', f'{DATA_ROOT}/h3_grid')

# ── Output roots (override via environment variables) ──────────────────────
# Everything the pipeline writes -- cached clipped rasters, feature tables,
# prediction surfaces, SHAP figures, GIS exports -- goes under one of these,
# all relative to wherever the pipeline is invoked from, same as the
# original notebook.
CLIP_DIR         = os.environ.get('REE_CLIP_DIR', 'clipped_buffered')
CLIP_DIR_TIGHT   = os.environ.get('REE_CLIP_DIR_TIGHT', 'clipped')
GIS_EXPORT_DIR   = os.environ.get('REE_GIS_EXPORT_DIR', 'gis_export')
SHAP_FIG_DIR     = os.environ.get('REE_SHAP_FIG_DIR', 'shap_figures')

# The per-region CSV tables (feature table, top candidate targets) that
# build_region()/run_all_regions() write. Default '.' -- the invocation
# directory, exactly as the original notebook wrote them.
#
# Note that DEPOSITS_ROOT above ALSO defaults to '.', so driving src/ from
# the same directory the deposit CSVs live in puts both in one folder. That
# still works (discover_regions() skips the pipeline's own output CSVs by
# name -- see src/features/data_discovery.py), but pointing this at its own
# folder keeps inputs and outputs cleanly separated:
#     REE_TABLE_DIR=results python your_driver.py
TABLE_DIR        = os.environ.get('REE_TABLE_DIR', '.')

def build_regions(region_manual, rasters, vectors, points):
    """Assemble the full REGIONS dict: each region's outline/h3_grid/dep
    (from src.features.data_discovery.discover_regions()) plus the SAME
    shared raster/vector/point layers (from discover_all()) -- every region
    sees identical raster/vector/point layers, exactly as the old hardcoded
    _RASTERS/faults/margins/worms did, just sourced from folder scans
    instead of fixed dict literals."""
    return {
        name: {**manual, 'raster': rasters, 'vectors': vectors, 'points': points}
        for name, manual in region_manual.items()
    }

# ── Core sampling / study-area parameters ───────────────────────────────────
PROJECTED_CRS  = 'EPSG:3978'  # Canada Atlas Lambert (metres)
BUFFER_KM      = 10.0         # min distance from any REE deposit -> negative sample
N_BACKGROUND   = 4000         # legacy; see N_TRAIN_BG/N_EVAL_BG below (current design)
GRID_DEG       = 0.1          # ~10 km grid spacing for negative sampling & prediction
RANDOM_SEED    = 42
DN_SEED        = 42
DN_EPOCHS      = 300
DN_BATCH       = 32
DN_RESTARTS    = 5            # per-fold restarts averaged to reduce init variance
DN_MARGIN      = 5.0          # deviation margin; 8.0 is too demanding for few positives

# ── Buffer-then-clip raster cache ───────────────────────────────────────────
BUFFER_KM_GLCM = 8.0   # >= GLCM window half-width in ground units (see
                        # src/features/raster_preprocessing.py)

# ── Fast-iteration switch ───────────────────────────────────────────────────
# FAST_MODE=True:  quick debugging pass (one region, one loss, few repeats).
# FAST_MODE=False: full run used to produce final numbers.
#
# RUN_REGIONS is NOT set here: which regions exist is only known after
# discover_regions() runs (see module docstring's usage snippet), so the
# caller sets config.RUN_REGIONS right after config.REGIONS, the same way --
# `list(config.REGIONS)[:1] if FAST_MODE else list(config.REGIONS)`. Every
# downstream function's own `run_regions=None` default already falls back
# to reading config.RUN_REGIONS at call time, so nothing else changes.
FAST_MODE = False

CV_N_REPEATS = 6
DN_RESTARTS  = 5
RUN_LOSSES   = ['deviation']   # only loss supported (see src/models/devnet.py)

CV_N_SPLITS = 5

assert RUN_LOSSES == ['deviation'], (
    "This pipeline only supports RUN_LOSSES == ['deviation'] -- semi-supervised "
    "deviation detection only has a coherent meaning for the deviation loss. "
    "See src/models/devnet.py.")

FORCE_CPU = False   # DevNet is tiny (<=64 hidden units) on a few hundred rows --
                     # GPU transfer overhead can dominate actual compute.

# ── Loss selection ──────────────────────────────────────────────────────────
LOSS_TYPES     = ['deviation', 'ranking']
LOSS_TYPE      = 'deviation'  # loss used for the production ensemble / prediction map
RANK_MARGIN    = 1.0          # margin for the (unused-by-default) pairwise ranking loss
RANK_NEG_RATIO = 4            # ranking-loss batches: this many negatives per positive
DN_LR          = 2e-3

# ── Per-region DevNet capacity / regularization ─────────────────────────────
# Cordillera has the most positives of the three (27 in the shipped dataset,
# after dedup and outline clipping) and can afford more capacity with weaker
# regularization. Superior (15) and Grenville (12) stay small and heavily
# regularized or the network memorizes the few positive examples.
# Tuned specifically for these 3 named regions -- since regions are now
# discovered (see module docstring), any other/additional region name just
# falls back to DEFAULT_MODEL_CFG below (every call site already does
# `REGION_MODEL_CFG.get(name, DEFAULT_MODEL_CFG)`), not an error.
REGION_MODEL_CFG = {
    'Cordillera': dict(hidden=64, dropout=0.3, weight_decay=1e-4),
    'Superior':    dict(hidden=32, dropout=0.4, weight_decay=1e-3),
    'Grenville':   dict(hidden=32, dropout=0.4, weight_decay=1e-3),
}
DEFAULT_MODEL_CFG = dict(hidden=32, dropout=0.4, weight_decay=1e-3)

# ── Positive-sample deduplication ───────────────────────────────────────────
# REE deposit records within this radius of each other (chained,
# single-linkage) are collapsed to one representative point before any
# feature extraction or negative sampling happens. 0 (or None) disables this
# -- the deposit CSVs referenced in REGIONS above are already deduplicated
# externally, so this is a no-op in the default configuration.
POSITIVE_DEDUP_RADIUS_KM = 0

# ── Unsupervised collinearity pruning ───────────────────────────────────────
# Features whose |Spearman correlation| exceeds this, within a region, are
# clustered together and reduced to one representative.
COLLINEARITY_THRESH = 0.7

# ── Semi-supervised pools ───────────────────────────────────────────────────
N_UNLABELED = 4000   # legacy size of the training unlabeled background pool;
                      # superseded by N_TRAIN_BG below, still referenced by
                      # the background-sampling-sensitivity ablation.

# RF and DevNet are trained on the IDENTICAL set of background locations --
# differing only in whether those locations are labelled negative (RF) or
# left unlabelled (DevNet) -- via two disjoint pools:
#   N_TRAIN_BG -- shared TRAINING background (RF label=0 / DevNet unlabeled)
#   N_EVAL_BG  -- shared EVALUATION background (both models' pooled-OOF AUC/AP
#                 scores positives against this same pool)
N_TRAIN_BG = 5000
N_EVAL_BG  = 3000

TOP_N_TARGETS = 5   # how many top-scoring unlabeled points to report as candidate targets
