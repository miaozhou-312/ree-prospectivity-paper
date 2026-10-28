"""
Stage 40 -- national (Canada) outline and label table for the two-scale
framework: the national and regional analyses use one label table, one
feature pipeline and one evaluation, so that differences between them can
be attributed to scale.

Outline: the onshore geological provinces of GSC Map 1860A (Wheeler et al.,
1996; GEOPROV.SHP), i.e. every named province, orogen and platform except
the continental shelves and oceanic crust. The three regional outlines are
single polygons of the same map (Superior 1005, Cordilleran 133, Grenville
1011), so each province's national subset uses exactly the regional polygon.

Labels: the rule of experiments/02_label_analysis/positive_label_dedup.ipynb
-- USGS carbonatite-hosted records (REE_Canada_Carbonatite.csv) plus Parsa
et al. (2024) occurrences (columns swapped, EPSG:3978), collapsed by
single-linkage clustering at 2 km to the member nearest the cluster
centroid. Inside the three study polygons the regional tables are taken
as they are; applying the rule once to the whole country would not give
them back exactly, because in a two-member cluster both members are equally
far from the centroid (the pick then rests on rounding) and because one
Grenville cluster (Terres Rompues) has its second member just outside the
polygon. Records outside the study polygons are deduplicated with the same
rule, and any that lie within 2 km of a regional occurrence are dropped as
members of that occurrence's cluster. Each occurrence is assigned its
province.

Outputs (data/national/):
    Canada.geojson                          national outline (EPSG:4326)
    provinces.geojson                       provinces dissolved by name, study provinces as their regional polygon
    Canada_REE_deposits_deduped_2.0km.csv   national label table with a `province` column

    python -m paper_analyses.40_national_labels
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(REPO), 'dataset')
GEOPROV = os.path.join(DATA, 'extra', 'gsc_geological_provinces', 'GEOPROV.SHP')
SRC = os.path.join(REPO, 'data', 'labels', 'source')
OUT = os.path.join(REPO, 'data', 'national')
CRS = 'EPSG:3978'
DEDUP_KM = 2.0
INLIER_KM2 = 2000.0
OFFSHORE = {'oceanic crust', 'Atlantic Continental Shelf', 'Arctic Continental Shelf', 'Pacific Continental Shelf'}
RENAME = {'Province de Grenville': 'Grenville Province', 'Churchill': 'Churchill Province'}
STUDY = {'Superior': 'Superior Province', 'Cordillera': 'Cordilleran Orogen', 'Grenville': 'Grenville Province'}


def absorb_inliers(g):
    """Small windows (< INLIER_KM2) of one province inside another, such as
    the Churchill-basement windows of the Monashee and Malton complexes in
    the Cordillera, take the name of the neighbour they share most boundary
    with. Islands (no neighbour) keep their own name."""
    g = g.reset_index(drop=True)
    for i in np.where(g.area.values / 1e6 < INLIER_KM2)[0]:
        geom = g.geometry.iloc[i]
        nb = [j for j in g.sindex.query(geom, predicate='intersects') if j != i]
        shared = {j: geom.boundary.intersection(g.geometry.iloc[j].boundary).length for j in nb}
        shared = {j: s for j, s in shared.items() if s > 0 and g.province.iloc[j] != g.province.iloc[i]}
        if shared and sum(shared.values()) > 0.5 * geom.boundary.length:
            g.loc[i, 'province'] = g.province.iloc[max(shared, key=shared.get)]
    return g


def provinces():
    g = gpd.read_file(GEOPROV).to_crs(CRS)
    g = g[g.GEOLPROV.notna() & ~g.GEOLPROV.isin(OFFSHORE)].copy()
    g['province'] = g.GEOLPROV.replace(RENAME)
    g = absorb_inliers(g)
    parts = []
    for region, prov in STUDY.items():
        poly = gpd.read_file(os.path.join(REPO, 'data', 'outlines', f'{region}.geojson')).to_crs(CRS)
        parts.append(gpd.GeoDataFrame({'province': [region]}, geometry=[poly.geometry.union_all()], crs=CRS))
    study = pd.concat(parts).geometry.union_all()
    rest = g.copy()
    rest['geometry'] = rest.geometry.difference(study)
    rest = rest[~rest.geometry.is_empty]
    # fragments of a study province outside its regional polygon keep the map name with a suffix
    rest.loc[rest.province.isin(STUDY.values()), 'province'] += ' (outside study polygon)'
    rest = rest.dissolve('province').reset_index()[['province', 'geometry']]
    prov = pd.concat(parts + [rest], ignore_index=True)
    return gpd.GeoDataFrame(prov, geometry='geometry', crs=CRS)


def classify(v):
    v = str(v).strip().lower().replace('(?)', '').strip()
    return ('Deposit/Prospect' if 'deposit' in v or 'prospect' in v else 'Showing' if 'showing' in v
            else 'Occurrence' if 'occurrence' in v else 'Unknown')


def raw_pool(outline):
    usgs = pd.read_csv(os.path.join(SRC, 'REE_Canada_Carbonatite.csv')).dropna(subset=['Latitude', 'Longitude'])
    usgs = gpd.GeoDataFrame(usgs, geometry=gpd.points_from_xy(usgs.Longitude, usgs.Latitude), crs='EPSG:4326')
    parsa = pd.read_csv(os.path.join(SRC, 'Parsa_2024_REE_occurrences.csv'), encoding='latin-1')
    for c in ['Northing (m, EPSG:3978)', 'Easting (m, EPSG:3978)']:
        parsa[c] = parsa[c].astype(str).str.replace(',', '', regex=False).astype(float)
    parsa = gpd.GeoDataFrame(parsa, geometry=gpd.points_from_xy(parsa['Northing (m, EPSG:3978)'],
                                                                 parsa['Easting (m, EPSG:3978)']),
                             crs=CRS).to_crs('EPSG:4326')
    out = []
    for gdf, source, col in [(usgs, 'REE_Canada', 'Status'), (parsa, 'Parsa_2024', 'Occurrence type')]:
        n = len(gdf)
        gdf = gdf[gdf.within(outline)]
        print(f'  {source}: {len(gdf)} of {n} records inside the national outline')
        out.append(gpd.GeoDataFrame({'Name': gdf.Name.values, 'source': source,
                                     'dep_type': [classify(v) for v in gdf[col].values],
                                     'Latitude': gdf.geometry.y.values, 'Longitude': gdf.geometry.x.values},
                                    geometry=gdf.geometry.values, crs='EPSG:4326'))
    return gpd.GeoDataFrame(pd.concat(out, ignore_index=True), geometry='geometry', crs='EPSG:4326')


def dedup(pool):
    proj = pool.to_crs(CRS)
    xy = np.c_[proj.geometry.x.values, proj.geometry.y.values]
    cid = fcluster(linkage(xy, method='single'), t=DEDUP_KM * 1000.0, criterion='distance')
    keep = np.zeros(len(pool), bool)
    for c in np.unique(cid):
        idx = np.where(cid == c)[0]
        keep[idx[np.argmin(np.linalg.norm(xy[idx] - xy[idx].mean(axis=0), axis=1))]] = True
    return pool[keep].reset_index(drop=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    prov = provinces()
    canada = gpd.GeoDataFrame({'name': ['Canada (onshore provinces, GSC Map 1860A)']},
                              geometry=[prov.geometry.union_all()], crs=CRS)
    canada.to_crs('EPSG:4326').to_file(os.path.join(OUT, 'Canada.geojson'), driver='GeoJSON')
    prov.to_crs('EPSG:4326').to_file(os.path.join(OUT, 'provinces.geojson'), driver='GeoJSON')
    print(f'national outline: {canada.area.iloc[0] / 1e6:,.0f} km2; '
          f'{len(prov)} provinces: {", ".join(prov.province)}')

    outline = canada.to_crs('EPSG:4326').geometry.iloc[0]
    pool = raw_pool(outline)
    study = prov[prov.province.isin(STUDY)].to_crs('EPSG:4326').geometry.union_all()
    regional = []
    for region in STUDY:
        reg = pd.read_csv(os.path.join(REPO, 'data', 'labels', f'{region}_REE_deposits_deduped_{DEDUP_KM}km.csv'))
        regional.append(reg.drop(columns='geometry').assign(province=region))
        print(f'[{region}] regional table: {len(reg)} occurrences '
              f'({int(pool.within(prov.loc[prov.province == region].to_crs("EPSG:4326").geometry.iloc[0]).sum())} '
              f'raw records inside the polygon)')
    regional = pd.concat(regional, ignore_index=True)
    rest = dedup(pool[~pool.within(study)].reset_index(drop=True))
    xy_reg = gpd.GeoSeries.from_xy(regional.Longitude, regional.Latitude, crs='EPSG:4326').to_crs(CRS)
    near = rest.to_crs(CRS).geometry.apply(lambda p: xy_reg.distance(p).min() / 1000.0 <= DEDUP_KM).values
    for _, r in rest[near].iterrows():
        print(f'  dropped {r.Name!r} ({r.source}): within {DEDUP_KM} km of a regional occurrence')
    rest = rest[~near].reset_index(drop=True)
    j = gpd.sjoin(rest.to_crs(CRS), prov, how='left', predicate='within')
    rest['province'] = j.loc[~j.index.duplicated(), 'province'].values
    lab = pd.concat([regional, rest.drop(columns='geometry')], ignore_index=True)
    lab.to_csv(os.path.join(OUT, f'Canada_REE_deposits_deduped_{DEDUP_KM}km.csv'), index=False)
    print(f'national table: {len(lab)} occurrences ({len(regional)} regional + {len(rest)} elsewhere)')
    print(lab.province.value_counts().to_string())


if __name__ == '__main__':
    main()
