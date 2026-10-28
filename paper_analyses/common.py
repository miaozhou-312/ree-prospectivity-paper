"""
Shared setup for the journal-paper analyses (paper_analyses/*.py).

Every script here reuses the packaged pipeline in src/: the same
data discovery, buffer-then-clip preprocessing, A∧B sampleable domain,
region feature tables, collinearity pruning, fold scorers and pooled-OOF
protocol that produced the thesis numbers. The only change to src/ is the
seeded DevNet reference pool (KNOWN_ISSUES.md #5).

Run every script from the repository root, e.g.

    python -m paper_analyses.00_build_features

so that the default data roots resolve (`../dataset`, deposit CSVs in '.').
Outputs go under paper_analyses/out/ (git-ignored), heavy caches under
../work/ (outside the repository).
"""
import os
import pickle
import random

import numpy as np

WORK_DIR = os.path.abspath(os.environ.get('REE_PAPER_WORK', '../work'))
# REE_FEATURESET=reencoded reruns any stage on feature set 3 (raw H3 geology +
# extended layers, Stages 24-26): load_region() then serves the set-3
# matrices as rec['pruned'] (and rec['full']), and outputs go to out/fs3/ so
# that the original results are never overwritten.
FEATURESET = os.environ.get('REE_FEATURESET', '')
OUT_DIR = os.environ.get('REE_PAPER_OUT') or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', *(['fs3'] if FEATURESET == 'reencoded' else []))
CACHE_DIR = os.path.join(WORK_DIR, 'region_cache')

# Route src/'s output directories away from the repository before src.config
# is imported (it reads these env vars at import time).
os.environ.setdefault('REE_CLIP_DIR', os.path.join(WORK_DIR, 'clipped_buffered'))
os.environ.setdefault('REE_CLIP_DIR_TIGHT', os.path.join(WORK_DIR, 'clipped'))
os.environ.setdefault('REE_TABLE_DIR', os.path.join(WORK_DIR, 'tables'))
os.environ.setdefault('REE_SHAP_FIG_DIR', os.path.join(WORK_DIR, 'shap_figures'))
os.environ.setdefault('REE_GIS_EXPORT_DIR', os.path.join(WORK_DIR, 'gis_export'))

from src import config  # noqa: E402

# The notebook pins this GLCM subset instead of prompting (README, Section 1).
# REE_GLCM=all (Stage 51 onwards) takes the texture of every gravity and
# magnetic grid instead; pruning removes the redundant ones.
GLCM_RASTERS = ['Canada_BA', 'Canada_BA_HP40km', 'DRTP', 'TMI']
if os.environ.get('REE_GLCM') == 'all':
    GLCM_RASTERS = ['Canada_BA', 'Canada_BA_1VD', 'Canada_BA_THD', 'Canada_BA_THD_Tilt', 'Canada_BA_HP40km',
                    'Canada_BA_HP100km', 'TMI', 'DRTP', 'DRTP_1VD', 'DRTP_THD', 'DRTP_THD_Tilt']
REGION_ORDER = ['Superior', 'Cordillera', 'Grenville']   # Superior is the paper's focus


def seed_everything(seed=config.RANDOM_SEED):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def setup_regions():
    """Discovery exactly as the notebook does it: every layer kept, GLCM
    subset pinned. Populates config.REGIONS / config.RUN_REGIONS."""
    from src.features.data_discovery import discover_all, discover_regions
    rasters, vectors, points, _ = discover_all(
        config.RASTER_DIR, config.VECTOR_DIR, config.POINT_DIR, prompt_fn=lambda _: '')
    missing = [r for r in GLCM_RASTERS if r not in rasters]
    if missing:
        raise ValueError(f'GLCM rasters missing from {config.RASTER_DIR}: {missing}')
    region_manual = discover_regions(config.OUTLINE_DIR, config.H3_GRID_DIR, config.DEPOSITS_ROOT)
    config.REGIONS = config.build_regions(region_manual, rasters, vectors, points)
    config.RUN_REGIONS = list(config.REGIONS)
    return config.REGIONS


def region_cache_path(name):
    return os.path.join(CACHE_DIR, f'{name}.pkl')


_OVERRIDING = False


def load_region(name):
    """Cached output of 00_build_features: dict with pts, pruned_cols,
    full_cols and the labelled/unlabelled raw matrices for both feature sets.
    With REE_FEATURESET=reencoded, 'pruned' and 'full' hold feature set 3."""
    global _OVERRIDING
    with open(region_cache_path(name), 'rb') as f:
        rec = pickle.load(f)
    if FEATURESET == 'reencoded' and not _OVERRIDING:
        import contextlib
        import importlib
        import io
        _OVERRIDING = True
        try:
            s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
            with contextlib.redirect_stdout(io.StringIO()):
                _, cols, X_lab, X_unl, _ = s25.extended_matrices(name, 'reencoded')
        finally:
            _OVERRIDING = False
        m = {'cols': cols, 'X_lab': X_lab, 'X_unl': X_unl, 'y': rec['pruned']['y']}
        rec['pruned'], rec['full'], rec['featureset'] = m, dict(m), 'reencoded'
    return rec


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dn_train_kwargs():
    return dict(dn_lr=config.DN_LR, dn_batch=config.DN_BATCH, dn_epochs=config.DN_EPOCHS,
                rank_neg_ratio=config.RANK_NEG_RATIO,
                loss_margin={'deviation': config.DN_MARGIN, 'ranking': config.RANK_MARGIN})


def model_cfg(name):
    return config.REGION_MODEL_CFG.get(name, config.DEFAULT_MODEL_CFG)


def out_path(*parts):
    p = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
