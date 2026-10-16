"""
Section 5 -- Feature matrix assembly.

Defines two ways to turn a points table into a model-ready matrix:

- `feature_matrix` -- imputes + scales in one step (used where a single,
  whole-dataset-fit imputer/scaler is acceptable).
- `get_raw_features` -- returns the *unimputed, unscaled* matrix (Section 6),
  so RF/DevNet fold-safe scorers can fit their own imputer/scaler per CV
  fold instead of on the whole dataset (avoiding fold leakage).

Feature-column selection is a DENYLIST, not an allowlist: every column in
`pts` is a model feature EXCEPT the bookkeeping ones (`lat`/`lon`/`label`/
`pool`) and raw distance columns (`*_dist_m` -- only their normalized
`*_proximity` twin is a feature, see src/features/proximity.py). This is
deliberate given src/features/data_discovery.py: the raster/vector/point
layer set is whatever a given run's folder-scan + interactive selection
turned up, so there is no fixed, dataset-specific list of feature-column
names (the old PROXIMITY_COLS/GLCM_COLS constants, tied to hardcoded fault/
margin/worm/raster names) to keep in sync -- the feature set just tracks
whatever add_proximity_lines/add_proximity_worms/add_rasters/add_glcm_features
actually added to `pts`.
"""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features.terrane import ERA_DUMMY_GROUPS, LITH_DUMMY_GROUPS

# Not used for feature-column filtering (see module docstring) -- kept as a
# documentation constant for the terrane one-hot column names.
TERRANE_COLS = [f'geo_era_{e}' for e in ERA_DUMMY_GROUPS] + [f'lith_{g}' for g in LITH_DUMMY_GROUPS]

_NON_FEATURE_COLS = {'lat', 'lon', 'label', 'pool'}


def _is_feature_col(col):
    return col not in _NON_FEATURE_COLS and not col.endswith('_dist_m')


def feature_cols(pts):
    """All model-feature columns present in `pts`, in column order -- i.e.
    everything except lat/lon/label/pool bookkeeping and raw *_dist_m
    columns. This is "the full feature set" for whatever layers this run's
    data-discovery happened to turn up."""
    return [c for c in pts.columns if _is_feature_col(c)]


def feature_matrix(pts, scaler=None, imputer=None, fit=True, feat_subset=None):
    feat_cols = feature_cols(pts)
    if feat_subset is not None:
        feat_cols = [c for c in feat_cols if c in feat_subset]
    X = pts[feat_cols].astype(float).values
    y = pts['label'].values

    if imputer is None:
        imputer = SimpleImputer(strategy='median')
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X = imputer.fit_transform(X)
        X = scaler.fit_transform(X)
    else:
        X = imputer.transform(X)
        X = scaler.transform(X)

    return X, y, feat_cols, scaler, imputer


def get_raw_features(pts, feat_subset=None):
    """Raw (unimputed, unscaled) feature matrix -- same column selection as
    feature_matrix(), so RF/DevNet fold-safe scorers can fit their own
    imputer/scaler per fold instead of on the whole dataset. Pass feat_subset
    (e.g. a region's pruned feature list) to restrict to it; the master
    column order is preserved either way."""
    feat_cols = feature_cols(pts)
    if feat_subset is not None:
        feat_cols = [c for c in feat_cols if c in feat_subset]
    X = pts[feat_cols].astype(float).values
    y = pts['label'].values
    return X, y, feat_cols
