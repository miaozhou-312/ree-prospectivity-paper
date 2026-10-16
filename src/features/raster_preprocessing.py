"""
Section 1.5 -- Preprocess: clip national rasters and the H3 Grid vector
layer to each study-area boundary, with a small on-disk cache so repeat
runs don't re-clip.

Buffer-then-clip (rasters): the region outline is buffered outward by
BUFFER_KM_GLCM before clipping, so GLCM windows near the true outline edge
draw on genuine national data instead of clip-edge nodata. The buffer is
computed once in PROJECTED_CRS (a metric CRS -- buffering directly in a
geographic CRS would not be an isotropic real-world distance).

Tight clip (H3 Grid vector layer): unchanged from the original single-clip
approach -- terrane attributes are only ever sampled at points already
inside the true outline and don't need the GLCM buffer margin.
"""
import os

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

from src.config import CLIP_DIR, CLIP_DIR_TIGHT, BUFFER_KM_GLCM, PROJECTED_CRS

# Populated by preprocess_all() -- name -> {feature: clipped_path}
CLIPPED_RASTERS = {}
CLIPPED_VECTORS = {}


def clip_raster(src_path, boundary_gdf, out_path):
    """Clip one raster to boundary_gdf and write it to out_path."""
    with rasterio.open(src_path) as src:
        bnd = boundary_gdf.to_crs(src.crs) if src.crs else boundary_gdf
        shapes = [g.__geo_interface__ for g in bnd.geometry]
        out_img, out_tf = rio_mask(src, shapes, crop=True)
        meta = src.meta.copy()
        meta.update(driver='GTiff', height=out_img.shape[1],
                    width=out_img.shape[2], transform=out_tf)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(out_img)


def preprocess_rasters(name, cfg):
    """Buffer-then-clip every raster listed in cfg['raster'] for region
    `name`. The region outline is buffered outward by BUFFER_KM_GLCM (in
    PROJECTED_CRS) and clip_raster reprojects that already-buffered polygon
    into each raster's own native CRS (no raster DATA is ever reprojected,
    only this one vector polygon, per raster). Results are cached under
    CLIP_DIR/<name>/<feature>.tif; already-clipped files are skipped."""
    boundary_native   = gpd.read_file(cfg['outline']).to_crs('EPSG:4326')
    boundary_buffered = gpd.GeoDataFrame(
        geometry=boundary_native.to_crs(PROJECTED_CRS).buffer(BUFFER_KM_GLCM * 1000.0),
        crs=PROJECTED_CRS).to_crs('EPSG:4326')

    out_dir = os.path.join(CLIP_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    clipped = {}
    for feat, path in cfg.get('raster', {}).items():
        out_path = os.path.join(out_dir, f'{feat}.tif')
        if not os.path.exists(out_path):
            print(f'  clipping {feat} (buffered +{BUFFER_KM_GLCM:.0f}km) ...')
            clip_raster(path, boundary_buffered, out_path)
        clipped[feat] = out_path
    return clipped


def preprocess_vectors(name, cfg):
    """Clip the H3 Grid polygon layer to the TRUE (unbuffered) study-area
    outline and cache it as a GeoPackage under CLIP_DIR_TIGHT/<name>/."""
    boundary = gpd.read_file(cfg['outline']).to_crs('EPSG:4326')
    out_dir  = os.path.join(CLIP_DIR_TIGHT, name)
    os.makedirs(out_dir, exist_ok=True)
    h3_path = cfg.get('h3_grid')
    if not h3_path:
        return {}
    out_path = os.path.join(out_dir, 'h3_grid.gpkg')
    if not os.path.exists(out_path):
        print(f'  clipping H3 Grid ({name}) ...')
        h3 = gpd.read_file(h3_path)
        gpd.clip(h3.to_crs('EPSG:4326'), boundary).to_file(out_path, driver='GPKG')
    else:
        print(f'  cached  H3 Grid ({name})')
    return {'h3_grid': out_path}


def preprocess_all(regions):
    """Run preprocess_rasters/preprocess_vectors for every region in
    `regions` (REGIONS from src.config) and populate the module-level
    CLIPPED_RASTERS / CLIPPED_VECTORS caches that the rest of the pipeline
    (sampleable_domain, build_region, prediction_map, gis_export, ...)
    reads from."""
    print('Clipping rasters (buffer-then-clip) and H3 Grid (true outline) to study area boundaries ...')
    for name, cfg in regions.items():
        print(f'\n[{name}]')
        CLIPPED_RASTERS[name] = preprocess_rasters(name, cfg)
        CLIPPED_VECTORS[name] = preprocess_vectors(name, cfg)
    print('\nDone.')
    return CLIPPED_RASTERS, CLIPPED_VECTORS
