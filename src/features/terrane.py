"""
Section 3e -- Terrane attributes (geological era, lithology) via
point-in-polygon join against the clipped H3 Grid layer.

Geological era -> FULL one-hot, deliberately NOT a dropped-reference design
like lithology below. The goal is to read each era's own SHAP contribution
directly and rank all five against each other, so every era (including what
would otherwise be the reference level) gets its own column.

Lithology -> coarse geological group, REE-relevance-informed rather than an
arbitrary label code. See LITH_GROUP_MAP below for the full rationale.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd

ERA_DUMMY_GROUPS = ['Archean', 'Proterozoic', 'Paleozoic', 'Mesozoic', 'Cenozoic']

# LithMajor's raw categories (28 nationally) have no natural order, so
# integer label-encoding imposes a fake distance/ordering that both RF's
# threshold splits and, more acutely, DevNet's standardized-continuous MLP
# input would wrongly treat as meaningful. This national-scale lithology
# layer has no dedicated "Carbonatite" class, so 'alkaline_favorable' groups
# the best available proxies for carbonatite/alkaline-affinity REE hosts:
# Igneous_Intrusive_Alkalic and NYF-type Igneous_Intrusive_Felsic_Pegmatite
# directly, plus Igneous_Intrusive_Anorthosite as a proxy for Grenville-type
# AMCG alkaline igneous provinces (a spatial/petrogenetic association, not a
# claim that anorthosite itself is REE-bearing). 'other_metamorphic'
# (gneiss/schist/quartzite/amphibolite/marble) plus the rare Other_* codes
# is the implicit REFERENCE group: it gets no dummy column below, so a point
# with a mapped-but-baseline lithology reads as all-zero across the four
# dummies, matching a standard one-hot design with a dropped reference
# level.
LITH_GROUP_MAP = {
    # alkaline_favorable -- REE-favorable proxy lithologies
    'Igneous_Intrusive_Alkalic':          'alkaline_favorable',
    'Igneous_Intrusive_Felsic_Pegmatite': 'alkaline_favorable',
    'Igneous_Intrusive_Anorthosite':      'alkaline_favorable',
    # other_igneous -- non-alkaline intrusive + extrusive
    'Igneous_Intrusive_Felsic':           'other_igneous',
    'Igneous_Intrusive_Felsic_Tonalite':  'other_igneous',
    'Igneous_Intrusive_Felsic_Felsite':   'other_igneous',
    'Igneous_Intrusive_Intermediate':     'other_igneous',
    'Igneous_Intrusive_Mafic':            'other_igneous',
    'Igneous_Intrusive_Ultramafic':       'other_igneous',
    'Igneous_Intrusive':                  'other_igneous',
    'Igneous_Extrusive':                  'other_igneous',
    # highgrade_metamorphic -- high-grade/anatectic, spatially linked to AMCG
    # suites in Grenville-type terranes
    'Metamorphic_Charnockite':            'highgrade_metamorphic',
    'Metamorphic_Granulite':              'highgrade_metamorphic',
    'Metamorphic_Migmatite':              'highgrade_metamorphic',
    # sedimentary
    'Sedimentary_Siliciclastic':          'sedimentary',
    'Sedimentary_Chemical_Carbonate':     'sedimentary',
    'Sedimentary_Chemical':               'sedimentary',
    'Sedimentary_Chemical_Evaporite':     'sedimentary',
    # other_metamorphic -- REFERENCE group (no dummy column)
    'Metamorphic_Gneiss':                 'other_metamorphic',
    'Metamorphic_Gneiss_Orthogneiss':     'other_metamorphic',
    'Metamorphic_Gneiss_Paragneiss':      'other_metamorphic',
    'Metamorphic_Schist':                 'other_metamorphic',
    'Metamorphic_Quartzite':              'other_metamorphic',
    'Metamorphic_Amphibolite':            'other_metamorphic',
    'Metamorphic_Marble':                 'other_metamorphic',
    'Other_Unknown':                      'other_metamorphic',
    'Other_Melange':                      'other_metamorphic',
    'Other_Unconsolidated':               'other_metamorphic',
}
# Groups that get their own one-hot dummy column -- 'other_metamorphic' is
# the dropped reference level (see LITH_GROUP_MAP docstring above).
LITH_DUMMY_GROUPS = ['alkaline_favorable', 'other_igneous', 'highgrade_metamorphic', 'sedimentary']


def add_terrane_h3(pts, h3_path):
    """Point-in-polygon join with the clipped H3 Grid. Extracts full one-hot
    dummies for geological era (ERA_DUMMY_GROUPS -- every era gets its own
    column) and one-hot dummies for the coarse, REE-relevance-informed
    lithology grouping (LITH_GROUP_MAP / LITH_DUMMY_GROUPS above).

    Uses a FIXED mapping applied identically no matter what set of raw
    LithMajor values happens to appear in `pts` -- unlike
    `.astype('category').cat.codes`, which assigns codes based on whichever
    categories are present in that specific call and could therefore assign
    the same integer code to two different lithologies across two calls that
    saw different raw values (e.g. training-table build vs. prediction-grid
    build). A fixed dict has no such dependency.

    A point whose lithology is genuinely unavailable (no overlapping H3
    polygon, or a raw value absent from LITH_GROUP_MAP) gets NaN on every
    dummy -- left for the fold-safe SimpleImputer(median) to fill, same as
    any other missing feature in this pipeline -- rather than being silently
    folded into the reference group, which would wrongly assert
    "confirmed non-alkaline/non-high-grade/..." for a point where the
    lithology is actually unknown."""
    if not h3_path or not os.path.exists(h3_path):
        print('  H3 Grid not found -- skipping terrane features')
        return pts
    h3 = gpd.read_file(h3_path)[['EraMaxMjr', 'LithMajor', 'geometry']]
    g  = gpd.GeoDataFrame(pts.copy(),
             geometry=gpd.points_from_xy(pts['lon'], pts['lat']),
             crs='EPSG:4326')
    joined = gpd.sjoin(g, h3, how='left', predicate='within')
    joined = joined[~joined.index.duplicated(keep='first')]

    # Full one-hot for era -- every era gets its own column;
    # unknown/unmapped era -> NaN on every column (left for the median
    # imputer), not folded into any single era.
    era_vals = joined['EraMaxMjr'].where(joined['EraMaxMjr'].isin(ERA_DUMMY_GROUPS)).values
    for era in ERA_DUMMY_GROUPS:
        pts[f'geo_era_{era}'] = np.where(
            pd.isna(era_vals), np.nan, (era_vals == era).astype(float))

    # Coarse lithology group via the fixed LITH_GROUP_MAP (NaN where the H3
    # join found no polygon, or the raw value isn't in the map).
    lith_vals  = joined['LithMajor'].replace({'NA': None, '': None})
    lith_group = lith_vals.map(LITH_GROUP_MAP).values
    for grp in LITH_DUMMY_GROUPS:
        pts[f'lith_{grp}'] = np.where(
            pd.isna(lith_group), np.nan, (lith_group == grp).astype(float))
    return pts
