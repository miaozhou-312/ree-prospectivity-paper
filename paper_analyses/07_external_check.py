"""
Stage 7 -- blind check at an REE resource that is not a training label.

Clay-Howells (Ontario Mineral Inventory MDI42G16SE00006; Fe-REE-Nb,
"Developed Prospect With Reported Reserves or Resources") lies on the
Kapuskasing Structural Zone about 80 km from the nearest Superior training
occurrence. Features are extracted at its recorded location with the src/
pipeline, and the production RF and DevNet (fit exactly as in Stage 3) score
it; its percentile among the 8,000 random background points of the
sampleable domain approximates the share of the province ranked above it.

    python -m paper_analyses.07_external_check
"""
import importlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from paper_analyses.common import GLCM_RASTERS, config, load_region, out_path, setup_regions
from paper_analyses.models import DEVICE
from src.features import raster_preprocessing
from src.features.glcm import add_glcm_features
from src.features.proximity import (add_proximity_line_layers, add_proximity_worms, add_rasters,
                                    build_worm_trees)
from src.features.sampling import haversine_km
from src.features.terrane import add_terrane_h3
from src.models.devnet import ensemble_score

TARGETS = [('Clay-Howells', 49 + 49 / 60 + 10.61 / 3600, -(82 + 3 / 60 + 10.37 / 3600))]


def main():
    fit_production = importlib.import_module('paper_analyses.03_shap_directions').fit_production
    regions = setup_regions()
    r = 'Superior'
    clipped_r, clipped_v = raster_preprocessing.preprocess_all({r: regions[r]})
    worm_trees = build_worm_trees(regions[r]['points'], config.PROJECTED_CRS)
    rec = load_region(r)
    cols = rec['pruned']['cols']

    pts = pd.DataFrame([{'lat': la, 'lon': lo, 'label': 1} for _, la, lo in TARGETS])
    pts = add_terrane_h3(pts, clipped_v[r].get('h3_grid'))
    pts = add_proximity_line_layers(pts, regions[r]['vectors'], config.PROJECTED_CRS)
    pts = add_proximity_worms(pts, worm_trees, config.PROJECTED_CRS)
    pts = add_rasters(pts, clipped_r[r])
    pts = add_glcm_features(pts, clipped_r[r], GLCM_RASTERS)
    for c in cols:
        if c not in pts:
            pts[c] = 0.0
    X_t = pts[cols].astype(float).values

    t = fit_production(r)
    X_bg = np.vstack([rec['pruned']['X_lab'][rec['pruned']['y'] == 0], rec['pruned']['X_unl']])
    pos = rec['lab'][rec['lab']['pool'] == 'positive']
    rows = []
    for k, (name, la, lo) in enumerate(TARGETS):
        d_near = haversine_km(la, lo, pos['lat'].values, pos['lon'].values).min()
        Xall = np.vstack([X_bg, X_t[k:k + 1]])
        rf_s = t['rf_model'].predict_proba(t['rf_scaler'].transform(t['rf_imp'].transform(Xall)))[:, 1]
        dn_s = ensemble_score(t['dn_ensemble'], t['dn_scaler'].transform(t['dn_imp'].transform(Xall)),
                              device=DEVICE)
        pct = lambda s: float(rankdata(s)[-1] / len(s))
        rows.append({'target': name, 'lat': la, 'lon': lo, 'nearest_training_occurrence_km': d_near,
                     'devnet_percentile': pct(dn_s), 'rf_percentile': pct(rf_s),
                     'rf_probability': float(rf_s[-1]),
                     'rf_share_background_with_P0': float((rf_s[:-1] == 0).mean())})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out_path('external_check.csv'), index=False)


if __name__ == '__main__':
    main()
