"""
Section 3g -- GLCM texture-feature extraction.

Computes gray-level co-occurrence matrix (GLCM) statistics from a small
neighbourhood patch around each sample point, for whichever subset of the
raster layers a run's data-discovery step selected for GLCM (see
src/features/data_discovery.py -- passed in explicitly as `glcm_rasters`,
not a fixed constant, since the raster set itself is discovered per-run
rather than a hardcoded 17-layer list). Nodata-aware: a window can contain a
handful of genuine nodata cells even after buffer-then-clip; rather than
filling them with a local median (which would invent spurious pixel-pair
transitions), nodata gets its own reserved gray level that is zeroed out of
the co-occurrence matrix before normalizing.
"""
import numpy as np
import pyproj
import rasterio
from skimage.feature import graycomatrix, graycoprops

GLCM_PATCH_HALF = 4                                       # -> 9x9 pixel window
GLCM_LEVELS     = 8                                       # quantisation bins
GLCM_PROPS      = ['contrast', 'homogeneity', 'energy', 'correlation']
GLCM_ANGLES     = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def add_glcm_features(pts, raster_layers, glcm_rasters,
                       patch_half=GLCM_PATCH_HALF, n_levels=GLCM_LEVELS):
    """Compute GLCM statistics from a neighbourhood patch around each point,
    for every key in `glcm_rasters` that's also present in `raster_layers`.

    Reads each raster once, quantises globally to n_levels using the
    2nd-98th percentile, then extracts a (2*patch_half+1)^2 window for every
    sample. Points whose window contains fewer than (patch_half+1)^2 valid
    pixels receive NaN (imputed later by the median imputer) -- this is
    exactly condition B of the A∧B sampleable domain (see
    src/features/sampleable_domain.py), so on the A∧B domain this NaN branch
    should essentially never trigger.

    Nodata-aware GLCM: a window can still contain a handful of GENUINE
    nodata cells (>= min_valid real cells, condition B's own threshold)
    even after buffer-then-clip. Rather than filling those cells with the
    local median -- which invents spurious real-to-filled-value pixel-pair
    transitions in the co-occurrence matrix -- nodata gets its own extra
    reserved gray level (index n_levels, one past the real quantised range
    [0, n_levels-1]). The co-occurrence matrix is computed including that
    level, every entry that touches it (its row AND its column) is zeroed,
    and the remaining matrix is renormalized to sum to 1 before
    graycoprops -- so texture reflects only genuine real-pixel-to-real-pixel
    transitions, never a fabricated pair involving a filled-in value.
    """
    rasters_to_use = [k for k in glcm_rasters if k in raster_layers]
    NODATA_LEVEL = n_levels   # reserved sentinel, one past [0, n_levels-1]

    for feat in rasters_to_use:
        path = raster_layers[feat]
        col_buffers = {f'{feat}_glcm_{p}': [] for p in GLCM_PROPS}

        with rasterio.open(path) as src:
            # Project sample coords to raster CRS
            if src.crs and src.crs.to_epsg() != 4326:
                tr = pyproj.Transformer.from_crs(
                    'EPSG:4326', src.crs, always_xy=True)
                xs, ys = tr.transform(pts['lon'].values, pts['lat'].values)
            else:
                xs, ys = pts['lon'].values, pts['lat'].values

            # Read full band into memory (clipped rasters are small)
            data = src.read(1).astype(float)
            nd   = src.nodata
            if nd is not None:
                data[data == nd] = np.nan

            # Global 2-98th-percentile quantisation to [0, n_levels-1]
            valid = np.isfinite(data)
            if valid.any():
                vmin = float(np.nanpercentile(data[valid], 2))
                vmax = float(np.nanpercentile(data[valid], 98))
                scaled = np.clip(
                    (data - vmin) / (vmax - vmin + 1e-9) * (n_levels - 1),
                    0, n_levels - 1,
                )
                data_q = np.where(valid, np.round(scaled).astype(np.uint8), NODATA_LEVEL)
            else:
                data_q = np.full(data.shape, NODATA_LEVEL, dtype=np.uint8)

            H, W      = data_q.shape
            min_valid = (patch_half + 1) ** 2  # minimum usable pixels -- condition B's own threshold

            for xi, yi in zip(xs, ys):
                try:
                    row, col = src.index(xi, yi)
                except Exception:
                    for p in GLCM_PROPS:
                        col_buffers[f'{feat}_glcm_{p}'].append(np.nan)
                    continue

                r0 = max(row - patch_half, 0)
                r1 = min(row + patch_half + 1, H)
                c0 = max(col - patch_half, 0)
                c1 = min(col + patch_half + 1, W)
                patch = data_q[r0:r1, c0:c1].copy()

                valid_px = patch[patch != NODATA_LEVEL]
                if len(valid_px) < min_valid:
                    for p in GLCM_PROPS:
                        col_buffers[f'{feat}_glcm_{p}'].append(np.nan)
                    continue

                # Nodata-aware co-occurrence: include NODATA_LEVEL as its own
                # gray level, zero every entry that touches it, renormalize.
                glcm = graycomatrix(
                    patch, distances=[1], angles=GLCM_ANGLES,
                    levels=n_levels + 1, symmetric=True, normed=False,
                ).astype(float)
                glcm[NODATA_LEVEL, :, :, :] = 0.0
                glcm[:, NODATA_LEVEL, :, :] = 0.0
                totals = glcm.sum(axis=(0, 1), keepdims=True)
                totals[totals == 0] = 1.0
                glcm = glcm / totals

                for p in GLCM_PROPS:
                    col_buffers[f'{feat}_glcm_{p}'].append(
                        float(graycoprops(glcm, p).mean()))

        for col_name, vals in col_buffers.items():
            pts[col_name] = vals

        print(f'    GLCM: {feat} -> {len(GLCM_PROPS)} texture features (nodata-aware)')

    return pts
