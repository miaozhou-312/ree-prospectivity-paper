"""
Stage 0 -- build and cache every region's feature table with the packaged
pipeline (notebook Sections 1-5 / src.evaluation.run_experiment up to the
point where models are fitted). All later scripts read this cache, so the
background draws, positives and pruned feature sets are identical across
every analysis in the paper.
"""
import os
import time

from paper_analyses.common import (CACHE_DIR, GLCM_RASTERS, config, region_cache_path,
                                   save_pickle, seed_everything, setup_regions)
from src.features import raster_preprocessing, sampleable_domain
from src.features.build_region import build_region
from src.features.collinearity import prune_collinear
from src.features.feature_matrix import feature_cols, get_raw_features
from src.features.proximity import build_worm_trees


def main():
    t0 = time.time()
    seed_everything()
    regions = setup_regions()
    clipped_rasters, clipped_vectors = raster_preprocessing.preprocess_all(regions)
    sampleable_domain.build_all(regions, clipped_rasters, config.PROJECTED_CRS, GLCM_RASTERS)
    point_cfg = list(regions.values())[0]['points']
    worm_trees = build_worm_trees(point_cfg, config.PROJECTED_CRS)

    os.makedirs(CACHE_DIR, exist_ok=True)
    for name in sorted(regions):     # sorted == notebook order (Cordillera, Grenville, Superior)
        pts = build_region(name, regions[name], worm_trees, clipped_rasters[name],
                           clipped_vectors[name], config, GLCM_RASTERS)
        full_cols = feature_cols(pts)
        pruned_cols = prune_collinear(pts, full_cols, thresh=config.COLLINEARITY_THRESH)
        lab = pts[pts['pool'].isin(['positive', 'eval_neg'])].reset_index(drop=True)
        unl = pts[pts['pool'] == 'train_unlabeled'].reset_index(drop=True)
        rec = {'name': name, 'pts': pts, 'full_cols': full_cols, 'pruned_cols': pruned_cols,
               'lab': lab, 'unl': unl}
        for fs, cols in [('full', None), ('pruned', pruned_cols)]:
            X_lab, y, used = get_raw_features(lab, feat_subset=cols)
            X_unl, _, _ = get_raw_features(unl, feat_subset=cols)
            rec[fs] = {'X_lab': X_lab, 'y': y, 'X_unl': X_unl, 'cols': used}
        save_pickle(rec, region_cache_path(name))
        print(f'[{name}] cached: {int(rec["pruned"]["y"].sum())} positives, '
              f'{len(lab) - int(rec["pruned"]["y"].sum())} eval_neg, {len(unl)} unlabelled, '
              f'{len(full_cols)} -> {len(pruned_cols)} features')
    print(f'Done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
