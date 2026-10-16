"""
Section 15 -- GIS export (ArcGIS Pro, EPSG:3978).

Exports three layer types per region into GIS_EXPORT_DIR, all reprojected
to PROJECTED_CRS:

1. A∧B sampleable-domain mask -> {region}_sampleable_domain.tif, uint8
   (1=in domain, 0=out, nodata=255), written directly on the region's
   reference transform -- no resampling, this raster and the mask share the
   same pixel grid by construction.
2. Positive (deposit) points -> {region}_positives.gpkg. Rebuilt from
   cfg['dep'] + the same within-outline filter and dedup_close_positives
   call make_points uses, keeping the full original attribute set, then
   checked to have exactly as many rows as verify_pts[name]['label']==1
   before export.
3. Prediction surfaces (rf_proba, dn_proba) ->
   {region}_RF_proba.tif / {region}_DevNet_proba.tif, float32,
   nodata=NaN outside the domain. Reads the GeoJSON written by
   src/evaluation/prediction_map.py (predict_grid), reprojects it to
   EPSG:3978, and linearly interpolates onto every reference-grid pixel
   center, then masks to the sampleable domain. Requires predict_grid to
   have already run for every requested region -- regions missing that
   GeoJSON are skipped with a warning, not silently dropped.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as _rio_xy
from scipy.interpolate import griddata as _griddata_export

from src.features.sampleable_domain import SAMPLEABLE


def export_domain_mask_tif(name, gis_export_dir):
    """SAMPLEABLE[name]['sampleable'] -> uint8 GeoTIFF, 1=in domain / 0=out,
    nodata=255 (declared for GIS-tool convention; the mask array itself
    has no true no-data cells)."""
    info = SAMPLEABLE[name]
    ref_tf, ref_crs, ref_shape, mask = info['ref_tf'], info['ref_crs'], info['ref_shape'], info['sampleable']
    arr = mask.astype(np.uint8)

    out_path = os.path.join(gis_export_dir, f'{name}_sampleable_domain.tif')
    with rasterio.open(out_path, 'w', driver='GTiff', height=ref_shape[0], width=ref_shape[1],
                        count=1, dtype='uint8', crs=ref_crs, transform=ref_tf,
                        nodata=255, compress='lzw') as dst:
        dst.write(arr, 1)
    print(f'  [{name}] domain mask ({arr.sum()}/{arr.size} px in-domain) -> {out_path}')


def export_positive_points_gpkg(name, cfg, verify_pts, projected_crs, gis_export_dir):
    """verify_pts[name][label==1], WITH attributes. verify_pts only carries
    lat/lon/label, so attributes (Name, dep_type, ...) are joined back from
    cfg['dep'] by matching coordinates -- deliberately NOT by independently
    re-deriving the positive set via a second boundary.unary_union() +
    .within() call: GEOS's boundary test on a point sitting almost exactly
    on the outline edge can flip between two separate calls due to
    floating-point noise in unary_union, so re-deriving risks silently
    diverging from verify_pts. Joining back onto verify_pts's own
    already-computed, authoritative result sidesteps that entirely."""
    verify_pos = verify_pts[name][verify_pts[name]['label'] == 1].reset_index(drop=True)

    dep = pd.read_csv(cfg['dep']).dropna(subset=['Latitude', 'Longitude'])
    dep_gdf = gpd.GeoDataFrame(dep,
        geometry=gpd.points_from_xy(dep['Longitude'], dep['Latitude']), crs='EPSG:4326')

    dep_key    = list(zip(dep_gdf.geometry.y.round(6), dep_gdf.geometry.x.round(6)))
    verify_key = set(zip(verify_pos['lat'].round(6), verify_pos['lon'].round(6)))
    dep_in = dep_gdf[[k in verify_key for k in dep_key]].reset_index(drop=True)

    if len(dep_in) != len(verify_pos):
        raise RuntimeError(f'[{name}] positive-attribute join mismatch: '
                            f'{len(dep_in)} attribute rows matched vs {len(verify_pos)} in verify_pts -- '
                            f'lat/lon join failed for at least one positive.')

    dep_proj = dep_in.to_crs(projected_crs)
    out_path = os.path.join(gis_export_dir, f'{name}_positives.gpkg')
    dep_proj.to_file(out_path, driver='GPKG', layer=f'{name}_positives')
    print(f'  [{name}] positives ({len(dep_proj)}, attrs={list(dep_in.columns)}) -> {out_path}')


def export_prediction_tifs(name, gis_export_dir):
    """rf_proba / dn_proba, read back from the GeoJSON predict_grid wrote
    (an irregular GRID_DEG point cloud, A∧B-domain-restricted), linearly
    interpolated onto the reference-grid pixel grid (same method
    _prosp_panel uses for its own figures), masked to the sampleable
    domain, nodata=NaN outside it."""
    info = SAMPLEABLE[name]
    ref_tf, ref_crs, ref_shape, mask = info['ref_tf'], info['ref_crs'], info['ref_shape'], info['sampleable']
    H, W = ref_shape

    geojson_path = f'{name}_prospectivity_combined_dedup.geojson'
    if not os.path.exists(geojson_path):
        print(f'  [{name}] WARNING: {geojson_path} not found -- run predict_grid '
              f'(src/evaluation/prediction_map.py) for this region first. Skipping '
              f'prediction-surface export.')
        return

    grid = gpd.read_file(geojson_path).to_crs(ref_crs)
    xs_p = grid.geometry.x.values
    ys_p = grid.geometry.y.values

    rows_idx, cols_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    Xc, Yc = _rio_xy(ref_tf, rows_idx, cols_idx, offset='center')
    # rasterio.transform.xy flattens array inputs regardless of their shape --
    # reshape back to (H, W), matching rows_idx/cols_idx's row-major layout.
    Xc = np.asarray(Xc).reshape(H, W)
    Yc = np.asarray(Yc).reshape(H, W)

    for col, tag in [('rf_proba', 'RF'), ('dn_proba', 'DevNet')]:
        Z = _griddata_export(np.column_stack([xs_p, ys_p]), grid[col].values, (Xc, Yc), method='linear')
        Z = np.where(mask, Z, np.nan).astype(np.float32)

        out_path = os.path.join(gis_export_dir, f'{name}_{tag}_proba.tif')
        with rasterio.open(out_path, 'w', driver='GTiff', height=H, width=W, count=1,
                            dtype='float32', crs=ref_crs, transform=ref_tf,
                            nodata=np.nan, compress='lzw') as dst:
            dst.write(Z, 1)
        n_valid = int(np.isfinite(Z).sum())
        print(f'  [{name}] {tag} prospectivity ({n_valid}/{Z.size} px valid) -> {out_path}')


def export_all(config, verify_pts, gis_export_dir, run_regions=None):
    run_regions = list(config.RUN_REGIONS) if run_regions is None else list(run_regions)
    os.makedirs(gis_export_dir, exist_ok=True)
    print(f'Exporting GIS layers to {gis_export_dir}/ (all in {config.PROJECTED_CRS}) ...')
    for name in run_regions:
        print(f'\n[{name}]')
        export_domain_mask_tif(name, gis_export_dir)
        export_positive_points_gpkg(name, config.REGIONS[name], verify_pts, config.PROJECTED_CRS, gis_export_dir)
        export_prediction_tifs(name, gis_export_dir)
    print(f'\nDone -- see {gis_export_dir}/')
