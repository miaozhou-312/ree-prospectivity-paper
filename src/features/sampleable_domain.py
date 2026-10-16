"""
Section 3h -- A∧B "sampleable domain" (buffer-then-clip).

- **A -- full input-layer coverage:** a pixel has real data in *all* 17
  modeling layers (gravity all + magnetic all + moho_depth + lab_depth)
  simultaneously. Built by reprojecting (nearest-neighbour) every layer's
  own valid-data mask, on the buffer-then-clip rasters, onto a reference
  grid (the clipped gravity_ba raster's own grid), intersecting all 17,
  then clipping back to the TRUE (unbuffered) region outline.
- **B -- GLCM computable:** a pixel's own 9x9 window, on each of the 4 GLCM
  rasters separately (own native grid/CRS/resolution, no shared reference
  grid for this step), has >= min_valid = (GLCM_PATCH_HALF+1)^2 = 25 real
  valid pixels (scipy.ndimage.uniform_filter box-sum), reprojected
  (nearest) onto the reference grid, ANDed across all 4, then also clipped
  back to the true outline.

SAMPLEABLE[name]['sampleable'] = A & B is what make_points (sampling.py)
and build_grid/predict_grid (prediction_map.py) gate sampling and the
prediction map on.
"""
import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import scipy.ndimage as ndi
from rasterio.features import geometry_mask as rio_geometry_mask
from rasterio.transform import rowcol as rio_rowcol
from rasterio.warp import reproject, Resampling

from src.features.glcm import GLCM_PATCH_HALF

MIN_VALID_GLCM = (GLCM_PATCH_HALF + 1) ** 2   # 25 -- add_glcm_features' own existing threshold

# Populated by build_all() -- name -> dict(ref_tf, ref_crs, ref_shape,
# common_final, glcm_ok_final, per_layer_buf, per_raster_ok, sampleable)
SAMPLEABLE = {}


def load_valid_mask_native(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nd = src.nodata
        valid = (data != nd) & np.isfinite(data) if nd is not None else np.isfinite(data)
        return valid, src.transform, src.crs


def reproject_mask_to_reference(valid_native, src_tf, src_crs, ref_tf, ref_crs, ref_shape):
    dst = np.zeros(ref_shape, dtype=np.uint8)
    reproject(source=valid_native.astype(np.uint8), destination=dst,
              src_transform=src_tf, src_crs=src_crs,
              dst_transform=ref_tf, dst_crs=ref_crs, resampling=Resampling.nearest)
    return dst.astype(bool)


def build_common_valid_mask(clipped_rasters, all_model_rasters, reference_raster_key):
    """Condition A. `reference_raster_key` fixes the pixel grid every other
    layer's valid-data mask gets reprojected onto (see build_all)."""
    ref_path = clipped_rasters[reference_raster_key]
    with rasterio.open(ref_path) as ref_src:
        ref_tf, ref_crs, ref_shape = ref_src.transform, ref_src.crs, ref_src.shape
    per_layer = {}
    common = np.ones(ref_shape, dtype=bool)
    for key in all_model_rasters:
        vnat, stf, scrs = load_valid_mask_native(clipped_rasters[key])
        von = reproject_mask_to_reference(vnat, stf, scrs, ref_tf, ref_crs, ref_shape)
        per_layer[key] = von
        common &= von
    return ref_tf, ref_crs, ref_shape, common, per_layer


def clip_mask_to_outline(mask, ref_tf, ref_crs, outline_path):
    boundary = gpd.read_file(outline_path).to_crs(ref_crs)
    outline_mask = rio_geometry_mask(boundary.geometry, out_shape=mask.shape,
                                      transform=ref_tf, invert=True)
    return mask & outline_mask


def build_glcm_ok_mask(clipped_rasters, ref_tf, ref_crs, ref_shape, glcm_rasters,
                        patch_half=GLCM_PATCH_HALF, min_valid=MIN_VALID_GLCM):
    """Condition B, evaluated over `glcm_rasters` (the same list passed to
    add_glcm_features -- see src/features/data_discovery.py). If
    `glcm_rasters` is empty, condition B is vacuously all-True (no GLCM
    layer to fail on)."""
    if not glcm_rasters:
        return np.ones(ref_shape, dtype=bool), {}
    window_area = (2 * patch_half + 1) ** 2
    per_raster_ok = {}
    glcm_ok = np.ones(ref_shape, dtype=bool)
    for key in glcm_rasters:
        with rasterio.open(clipped_rasters[key]) as src:
            data = src.read(1).astype(float)
            nd = src.nodata
            valid = (data != nd) & np.isfinite(data) if nd is not None else np.isfinite(data)
            src_tf, src_crs = src.transform, src.crs
        valid_count = ndi.uniform_filter(valid.astype(float), size=2 * patch_half + 1,
                                          mode='constant', cval=0.0) * window_area
        ok_native = valid_count >= min_valid
        ok_on_ref = reproject_mask_to_reference(ok_native, src_tf, src_crs, ref_tf, ref_crs, ref_shape)
        per_raster_ok[key] = ok_on_ref
        glcm_ok &= ok_on_ref
    return glcm_ok, per_raster_ok


def build_all(regions, clipped_rasters, projected_crs, glcm_rasters, reference_raster_key=None):
    """Build the A∧B sampleable domain for every region and populate the
    module-level SAMPLEABLE cache that sampleable_lookup() (and everything
    downstream that calls it) reads from.

    `reference_raster_key` fixes the pixel grid condition A/B are evaluated
    on -- defaults to the first (alphabetically) of `glcm_rasters`, falling
    back to the first of all discovered rasters if `glcm_rasters` is empty,
    printed either way so the choice is never silent. Pass it explicitly to
    pick a specific layer (e.g. the coarsest-resolution one) instead."""
    print('Building per-region A ∧ B sampleable domain ...')
    for name, cfg in regions.items():
        all_model_rasters = list(clipped_rasters[name].keys())
        if reference_raster_key is None:
            ref_key = sorted(glcm_rasters)[0] if glcm_rasters else sorted(all_model_rasters)[0]
            print(f'  [{name}] reference_raster_key not given -- defaulting to {ref_key!r} '
                  f'(pass reference_raster_key= to build_all() to choose a different layer).')
        else:
            ref_key = reference_raster_key
        ref_tf, ref_crs, ref_shape, common_buf, per_layer_buf = build_common_valid_mask(
            clipped_rasters[name], all_model_rasters, ref_key)
        common_final = clip_mask_to_outline(common_buf, ref_tf, ref_crs, cfg['outline'])
        glcm_ok, per_raster_ok = build_glcm_ok_mask(clipped_rasters[name], ref_tf, ref_crs, ref_shape,
                                                     glcm_rasters)
        glcm_ok_final = clip_mask_to_outline(glcm_ok, ref_tf, ref_crs, cfg['outline'])
        sampleable = common_final & glcm_ok_final

        px_km2 = abs(ref_tf.a * ref_tf.e) / 1e6
        outline_km2 = gpd.read_file(cfg['outline']).to_crs(projected_crs).geometry.area.sum() / 1e6
        print(f'  [{name}] outline={outline_km2:,.0f} km2  '
              f'A={common_final.sum()*px_km2:,.0f} km2 ({common_final.sum()*px_km2/outline_km2*100:.1f}%)  '
              f'A∧B={sampleable.sum()*px_km2:,.0f} km2 ({sampleable.sum()*px_km2/outline_km2*100:.1f}%)')

        SAMPLEABLE[name] = dict(ref_tf=ref_tf, ref_crs=ref_crs, ref_shape=ref_shape,
                                 common_final=common_final, glcm_ok_final=glcm_ok_final,
                                 per_layer_buf=per_layer_buf, per_raster_ok=per_raster_ok,
                                 sampleable=sampleable)

    print('\nSAMPLEABLE ready: A∧B domain per region, sampleable_lookup() for point queries.')
    return SAMPLEABLE


def sampleable_lookup(lats, lons, name):
    """Vectorized point-in-A∧B-sampleable-domain check for region `name`."""
    info = SAMPLEABLE[name]
    ref_tf, ref_crs, ref_shape, mask = info['ref_tf'], info['ref_crs'], info['ref_shape'], info['sampleable']
    tr = pyproj.Transformer.from_crs('EPSG:4326', ref_crs, always_xy=True)
    xs, ys = tr.transform(np.asarray(lons), np.asarray(lats))
    rows, cols = rio_rowcol(ref_tf, xs, ys)
    rows = np.asarray(rows); cols = np.asarray(cols)
    H, W = ref_shape
    in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    ok = np.zeros(len(lats), dtype=bool)
    ok[in_bounds] = mask[rows[in_bounds], cols[in_bounds]]
    return ok


def plot_sampleable_domain(name, regions, verify_pts, projected_crs):
    """Verification map -- A∧B sampleable domain + outline + positives, per
    region (see notebooks/ Section "Verification map")."""
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    info = SAMPLEABLE[name]
    ref_tf, shape = info['ref_tf'], info['ref_shape']
    H, W = shape
    extent = [ref_tf.c, ref_tf.c + W * ref_tf.a, ref_tf.f + H * ref_tf.e, ref_tf.f]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(info['sampleable'].astype(int), extent=extent, origin='upper',
              cmap=mcolors.ListedColormap(['#f0efe9', '#3f8f5f']), vmin=0, vmax=1, alpha=0.92)

    boundary = gpd.read_file(regions[name]['outline']).to_crs(projected_crs)
    for geom in boundary.geometry:
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polys:
            gx, gy = poly.exterior.xy
            ax.plot(gx, gy, 'k-', lw=1.1)

    tr = pyproj.Transformer.from_crs('EPSG:4326', projected_crs, always_xy=True)
    pos = verify_pts[name][verify_pts[name]['label'] == 1]
    px_, py_ = tr.transform(pos['lon'].values, pos['lat'].values)
    ax.scatter(px_, py_, s=110, marker='*', c='#1a6faf', edgecolors='black', linewidths=0.6,
               zorder=5, label=f'positive (n={len(pos)})')

    legend_extra = [
        mpatches.Patch(facecolor='#f0efe9', edgecolor='grey', label='outside A∧B sampleable domain'),
        mpatches.Patch(facecolor='#3f8f5f', edgecolor='grey', label='A∧B sampleable domain'),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=legend_extra + h, loc='lower right', fontsize=8, framealpha=0.92)
    ax.set_title(f'{name}: A∧B sampleable domain (buffer-then-clip, no erosion)', fontsize=10)
    ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)'); ax.set_aspect('equal')
    plt.tight_layout(); plt.show()
