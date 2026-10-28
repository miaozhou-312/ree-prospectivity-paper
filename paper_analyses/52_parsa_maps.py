"""
Stage 52 -- ArcGIS Pro layers for the four national models of Table 2 (the
S2-grid design of Parsa et al. (2024) and ours, each with their random forest
and their 1D CNN). One production model per design and classifier is fitted
to all labels of the design and scores every L12 cell of the Stage 43 grid.

    Parsa design: the 264 L13 children of the 66 occurrence cells against 264
        cells drawn from the 5% least cosine-similar to them (the R0 draw),
        not class-weighted, as R0.
    Our design: the 66 occurrence cells against the 34,761-cell training
        background sample (the R5 draw), class-balanced, as R5.
The draws repeat those of Stage 43's ladder() with the same seeds. Training
cells are scored in-sample, as on the paper's other prospectivity maps.

Outputs in GIS_EXPORT_DIR/national_s2/, all in EPSG:3978 (as
src/evaluation/gis_export.py); each model's scores are cached in scores/ and
reused when present:
  {design}_{clf}_score.tif    float32 score, nodata NaN; each 1 km pixel takes
                              the score of the nearest L12 cell centre within
                              2 km (half the largest L12 cell diagonal)
  {design}_{clf}_class.tif    uint8 percentile class over all cells, as the
                              paper's maps (Stage 13): 0 below the top 10%,
                              1 top 10%, 2 top 5%, 3 top 1%; nodata 255
  national_s2_l12.gpkg        layers cells (L12 centre points: token, province,
                              distance to the nearest occurrence, and per model
                              score, percentile and class), occurrences,
                              labels_parsa and labels_ours (training rows)
  national_s2_l12_summary.csv per model: training rows, score range, class
                              thresholds and the share of cells in each class

--step figures draws the four maps in the style of the paper's prospectivity
maps (Fig. 9, Stage 13), without the location inset, into
paper/manuscript/figures_r7/, each alone and the four together (rows: design;
columns: classifier), in three classifications:
  fig_map_national[_{design}_{clf}|_r7].png          percentile classes over
      all cells, as Fig. 9 (each map has 10% of its cells in the top classes)
  fig_map_national_prob[_{design}_{clf}|_r7].png     fixed probability classes
      < 0.5, 0.5-0.7, 0.7-0.9, >= 0.9
  fig_map_national_logprob[_{design}_{clf}|_r7].png  fixed log-spaced classes
      < 0.001, 0.001-0.01, 0.01-0.1, 0.1-0.5, >= 0.5 (our RF scores lie
      almost entirely below 0.1, those of the Parsa design mostly above 0.5)
All four models output a probability in [0, 1]; the fixed classes, common to
the four maps and labelled with their share of the cells, show how much of
Canada each model calls prospective. fig_area_national_r7.png gives the
share of cells at or above every threshold from 0.001 to 1.

--step diagnostics writes national_s2_l12_diagnostics.csv (metric, value): for
each design the share of 50,000 random cells whose nearest training row in
the model's input space (median imputation and min-max scaling fitted on the
training rows) is a positive; for every model the share of cells at or above
0.5 and 0.9; for our RF the cells at or above 0.5 and 0.01 (how many are
occurrence cells, where they lie, how far from an occurrence).

    python -m paper_analyses.52_parsa_maps [--step figures|diagnostics]
"""
import argparse
import importlib
import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import s2sphere as s2
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from paper_analyses.common import config, load_pickle, seed_everything
from paper_analyses.national import LABELS, nat_path
from paper_analyses.protocol import project_xy

_l = importlib.import_module('paper_analyses.43_parsa_ladder')
_m = importlib.import_module('paper_analyses.13_maps')
OUT = os.path.join(config.GIS_EXPORT_DIR, 'national_s2')
CRS = config.PROJECTED_CRS
RES, MAX_DIST = 1000.0, 2000.0            # raster pixel (m); nearest-centre cut-off (m)
MODELS = [('parsa', 'RF'), ('parsa', 'CNN'), ('ours', 'RF'), ('ours', 'CNN')]
TITLES = {('parsa', 'RF'): 'RF MPM\nDesign: Parsa et al. (2024)', ('parsa', 'CNN'): 'CNN MPM\nDesign: Parsa et al. (2024)',
          ('ours', 'RF'): 'RF MPM\nDesign: this study', ('ours', 'CNN'): 'CNN MPM\nDesign: this study'}
FIG = os.path.join(_m.REPO, 'paper', 'manuscript', 'figures_r7')
# fixed probability classes: breaks (float32, as the scores: an RF vote share of 0.9 is >= 0.9), labels, colours
SCHEMES = {'_prob': (np.float32([0.5, 0.7, 0.9]), ['< 0.5', '0.5–0.7', '0.7–0.9', '≥ 0.9'], _m.CLASS_COLORS),
           '_logprob': (np.float32([0.001, 0.01, 0.1, 0.5]), ['< 0.001', '0.001–0.01', '0.01–0.1', '0.1–0.5', '≥ 0.5'],
                        _m.CLASS_COLORS[:2] + ['#a6d96a'] + _m.CLASS_COLORS[2:])}
DESIGN_COLORS = {'parsa': '#d03b3b', 'ours': '#2a78d6'}
BOTTOM = {'': 0.14, '_prob': 0.20, '_logprob': 0.26}   # frame margin below Canada (share of its height): room for the legend


def token(ids):
    return np.array([format(int(i), '016x').rstrip('0') for i in ids])


def child_centres(child_parent):
    """L13 child ids and centres in the order of Stage 43's grid(): the four
    children of each occurrence cell, cell by cell."""
    ids, lat, lon = [], [], []
    for k in range(0, len(child_parent), 4):
        assert (child_parent[k:k + 4] == child_parent[k]).all()
        for c in s2.CellId(int(child_parent[k])).children():
            p = c.to_lat_lng()
            ids.append(c.id())
            lat.append(p.lat().degrees)
            lon.append(p.lng().degrees)
    return np.array(ids, dtype=np.uint64), np.array(lat), np.array(lon)


def designs(g):
    """Training rows of both designs, drawn as in Stage 43's ladder()."""
    X, is_pos = g['X'].astype(float), g['is_pos']
    X_ch = g['X_child'].astype(float)
    cand = np.where(~is_pos)[0]
    far = np.where(~is_pos & (g['d_occ_km'] > config.BUFFER_KM))[0]
    meta = load_pickle(nat_path('national_table.pkl'))
    nat = meta['pts'][meta['pts'].in_national]
    n_ev, n_tr = int((nat.pool == 'eval_neg').sum()), int((nat.pool == 'train_unlabeled').sum())
    perm = np.random.default_rng(config.RANDOM_SEED).permutation(far)
    tr_idx = perm[n_ev:n_ev + n_tr]
    sim = _l.mean_cosine(X_ch, X, cand)
    low = cand[sim <= np.quantile(sim, _l.LOW_SIM_FRAC)]
    neg = np.random.default_rng(config.RANDOM_SEED + 2).choice(low, len(X_ch), replace=False)
    print(f'R0 cosine pool: {len(low):,} cells; median distance of the negatives to the nearest occurrence '
          f'{np.median(g["d_occ_km"][neg]):.0f} km | R5 training background {len(tr_idx):,} cells', flush=True)
    return {'parsa': (np.vstack([X_ch, X[neg]]), np.r_[np.ones(len(X_ch)), np.zeros(len(neg))], False, neg),
            'ours': (np.vstack([X[is_pos], X[tr_idx]]), np.r_[np.ones(int(is_pos.sum())), np.zeros(len(tr_idx))],
                     True, tr_idx)}


def pixel_index(x, y):
    """1 km EPSG:3978 grid over the cells; every pixel's nearest cell centre
    within MAX_DIST (index len(x) where there is none)."""
    x0, y1 = np.floor(x.min() / RES) * RES - RES, np.ceil(y.max() / RES) * RES + RES
    W = int(np.ceil((x.max() + RES - x0) / RES)) + 1
    H = int(np.ceil((y1 - y.min() + RES) / RES)) + 1
    tree = cKDTree(np.c_[x, y])
    idx = np.empty((H, W), dtype=np.int64)
    cx = x0 + (np.arange(W) + 0.5) * RES
    for r0 in range(0, H, 256):
        rows = np.arange(r0, min(r0 + 256, H))
        cy = y1 - (rows + 0.5) * RES
        gx, gy = np.meshgrid(cx, cy)
        idx[rows] = tree.query(np.c_[gx.ravel(), gy.ravel()], distance_upper_bound=MAX_DIST,
                               workers=4)[1].reshape(len(rows), W)
    return idx, from_origin(x0, y1, RES, RES)


def write_tif(path, arr, tf, nodata):
    with rasterio.open(path, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs=CRS, transform=tf, nodata=nodata, compress='lzw') as dst:
        dst.write(arr, 1)


def export():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    g = _l.load_grid()
    n = len(g['ids'])
    X = g['X'].astype(float)
    D = designs(g)
    cells = pd.DataFrame({'cell_token': token(g['ids']), 'lat': g['lat'], 'lon': g['lon'],
                          'province': g['province'].astype(str), 'is_pos': g['is_pos'].astype(np.uint8),
                          'd_occ_km': g['d_occ_km'].round(2)})
    summary = []
    os.makedirs(os.path.join(OUT, 'scores'), exist_ok=True)
    for design, clf in MODELS:
        X_tr, y_tr, balanced, _ = D[design]
        tag = f'{design}_{clf}'
        cache = os.path.join(OUT, 'scores', f'{tag}.npy')
        t = time.time()
        if os.path.exists(cache):
            s = np.load(cache)
        else:
            seed_everything()
            s = _l.CLASSIFIERS[clf](X_tr, y_tr, X, config.RANDOM_SEED, balanced=balanced).astype(np.float64)
            np.save(cache, s)
        cls = _m.classify(s)
        cells[f'{tag}_score'] = s.astype(np.float32)
        cells[f'{tag}_pct'] = (100.0 * (rankdata(s, method='min') - 1) / n).astype(np.float32)
        cells[f'{tag}_class'] = cls.astype(np.uint8)
        q90, q95, q99 = np.percentile(s, [90, 95, 99])
        summary.append({'model': tag, 'design': design, 'classifier': clf, 'n_train_pos': int(y_tr.sum()),
                        'n_train_neg': int((y_tr == 0).sum()), 'class_balanced': balanced,
                        'score_min': s.min(), 'score_median': np.median(s), 'score_max': s.max(),
                        'q90': q90, 'q95': q95, 'q99': q99,
                        **{f'share_class{k}': float((cls == k).mean()) for k in range(4)},
                        'share_score_max': float((s == s.max()).mean())})
        print(f'[{tag}] scored {n:,} cells in {(time.time() - t) / 60:.1f} min; median {np.median(s):.3f}, '
              f'cells at the maximum {summary[-1]["share_score_max"]:.1%}, class shares '
              f'{[round(summary[-1][f"share_class{k}"], 4) for k in range(4)]}', flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(OUT, 'national_s2_l12_summary.csv'), index=False)

    x, y = project_xy(g['lat'], g['lon'], CRS).T
    idx, tf = pixel_index(x, y)
    hit = idx < n
    print(f'raster {idx.shape[1]} x {idx.shape[0]} px of {RES / 1000:.0f} km, {int(hit.sum()):,} px on cells', flush=True)
    for design, clf in MODELS:
        tag = f'{design}_{clf}'
        sc = np.full(idx.shape, np.nan, dtype=np.float32)
        sc[hit] = cells[f'{tag}_score'].values[idx[hit]]
        write_tif(os.path.join(OUT, f'{tag}_score.tif'), sc, tf, np.nan)
        cl = np.full(idx.shape, 255, dtype=np.uint8)
        cl[hit] = cells[f'{tag}_class'].values[idx[hit]]
        write_tif(os.path.join(OUT, f'{tag}_class.tif'), cl, tf, 255)

    gpkg = os.path.join(OUT, 'national_s2_l12.gpkg')
    if os.path.exists(gpkg):
        os.remove(gpkg)
    gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(x, y), crs=CRS).to_file(gpkg, layer='cells', driver='GPKG')
    occ = pd.read_csv(LABELS)
    occ['cell_token'] = token([s2.CellId.from_lat_lng(s2.LatLng.from_degrees(a, b)).parent(_l.LEVEL).id()
                               for a, b in zip(occ.Latitude, occ.Longitude)])
    gpd.GeoDataFrame(occ, geometry=gpd.points_from_xy(*project_xy(occ.Latitude.values, occ.Longitude.values, CRS).T),
                     crs=CRS).to_file(gpkg, layer='occurrences', driver='GPKG')
    c_ids, c_lat, c_lon = child_centres(g['child_parent'])
    neg = D['parsa'][3]
    lab = pd.DataFrame({'label': np.r_[np.ones(len(c_ids)), np.zeros(len(neg))].astype(np.uint8),
                        'kind': ['L13 child of an occurrence cell'] * len(c_ids) + ['cosine-dissimilar negative'] * len(neg),
                        'cell_token': np.r_[token(c_ids), token(g['ids'][neg])],
                        'parent_token': np.r_[token(g['child_parent']), [''] * len(neg)],
                        'lat': np.r_[c_lat, g['lat'][neg]], 'lon': np.r_[c_lon, g['lon'][neg]],
                        'd_occ_km': np.r_[np.zeros(len(c_ids)), g['d_occ_km'][neg].round(2)]})
    gpd.GeoDataFrame(lab, geometry=gpd.points_from_xy(*project_xy(lab.lat.values, lab.lon.values, CRS).T),
                     crs=CRS).to_file(gpkg, layer='labels_parsa', driver='GPKG')
    rows = np.r_[np.where(g['is_pos'])[0], D['ours'][3]]
    lab = cells.loc[rows, ['cell_token', 'lat', 'lon', 'province']].assign(
        label=np.r_[np.ones(int(g['is_pos'].sum())), np.zeros(len(D['ours'][3]))].astype(np.uint8))
    gpd.GeoDataFrame(lab, geometry=gpd.points_from_xy(x[rows], y[rows]), crs=CRS).to_file(
        gpkg, layer='labels_ours', driver='GPKG')
    print(f'wrote {OUT} in {(time.time() - t0) / 60:.1f} min', flush=True)


def share_text(v):
    """Percentage to one decimal that never rounds a non-empty class to 0
    or a non-full one to 100."""
    return '<0.1%' if 0 < v < 0.05 else '>99.9%' if 99.95 <= v < 100 else f'{v:.1f}%'


def frame_of(canada, bottom):
    xmin, ymin, xmax, ymax = canada.total_bounds
    w, h = xmax - xmin, ymax - ymin
    return xmin - 0.06 * w, xmax + 0.06 * w, ymin - bottom * h, ymax + 0.08 * h


def national_panel(ax, tag, title, letter, canada, dep, scheme=None, shares=None, bottom=0.14):
    """One map in the style of Stage 13's map_panel: percentile classes from
    the class raster, or a fixed probability scheme (SCHEMES) applied to the
    score raster, labelled with shares (% of cells per class)."""
    if scheme is None:
        with rasterio.open(os.path.join(OUT, f'{tag}_class.tif')) as src:
            cls, tf = src.read(1), src.transform
        labels, colors = _m.CLASS_LABELS, _m.CLASS_COLORS
    else:
        breaks, names, colors = scheme
        with rasterio.open(os.path.join(OUT, f'{tag}_score.tif')) as src:
            sc, tf = src.read(1), src.transform
        cls = np.where(np.isfinite(sc), np.digitize(sc, breaks), 255)
        labels = [f'{n} ({share_text(v)})' for n, v in zip(names, shares)]
        title += '\nProbability (share of cells)'
    rgba = np.zeros(cls.shape + (4,), dtype=np.uint8)
    for k, c in enumerate(colors):
        rgba[cls == k] = np.round(np.array(to_rgba(c)) * 255).astype(np.uint8)
    H, W = cls.shape
    frame = frame_of(canada, bottom)
    ax.set_xlim(frame[0], frame[1]); ax.set_ylim(frame[2], frame[3]); ax.set_aspect('equal')
    _m.graticule(ax, frame, lons=list(range(-160, 0, 20)), lats=[50, 60, 70, 80], fontsize=10, lat_span=(0, 90),
                 lon_span=(-180, 180), label_top=False)        # meridians meet at the pole, just above the frame
    ax.imshow(rgba, extent=(tf.c, tf.c + W * tf.a, tf.f + H * tf.e, tf.f), origin='upper',
              interpolation='antialiased', zorder=2)
    canada.boundary.plot(ax=ax, color=_m.OUTLINE_COLOR, lw=0.5, zorder=3)
    for t, c in _m.DEP_COLORS.items():
        s = dep[dep.dep_type == t]
        ax.scatter(s.x, s.y, s=45, c=c, edgecolors='black', linewidths=0.9, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(2.0)
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=26, va='top', ha='left')
    handles = [Line2D([], [], marker='o', ls='', mfc=c, mec='black', ms=8, label=f'{t} ({(dep.dep_type == t).sum()})')
               for t, c in _m.DEP_COLORS.items()]
    first = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.08 if letter else 0.0, 0.97),
                      frameon=False, fontsize=11, handletextpad=1.2)
    patches = [Patch(fc=c, ec='none', label=l) for c, l in zip(colors, labels)]
    if scheme is not None:                    # highest probability (red) at the top
        patches = patches[::-1]
    ax.legend(handles=patches, loc='lower left', frameon=False, fontsize=11, title=title, title_fontsize=12,
              alignment='left', handlelength=2.6, handleheight=1.5)
    ax.add_artist(first)
    xa, ya = frame[0] + 0.9 * (frame[1] - frame[0]), frame[2] + 0.17 * (frame[3] - frame[2])
    _m.north_arrow(ax, 0.9, 0.17, 0.07, _m.TO_G.transform(xa, ya))
    _m.scale_bar(ax, 0.62, 0.07, 1000, [0, 1000])


def figures():
    canada = gpd.read_file(os.path.join(os.path.dirname(LABELS), 'Canada.geojson')).to_crs(CRS)
    dep = pd.read_csv(LABELS)
    dep['x'], dep['y'] = _m.TO_P.transform(dep['Longitude'].values, dep['Latitude'].values)
    cols = [f'{d}_{c}_score' for d, c in MODELS]
    cells = gpd.read_file(os.path.join(OUT, 'national_s2_l12.gpkg'), layer='cells', columns=cols, ignore_geometry=True)
    shares = {(k, d, c): 100.0 * np.bincount(np.digitize(cells[f'{d}_{c}_score'].values, br), minlength=len(br) + 1)
              / len(cells) for k, (br, _, _) in SCHEMES.items() for d, c in MODELS}
    pd.DataFrame([{'scheme': k.strip('_'), 'model': f'{d}_{c}', **dict(zip(SCHEMES[k][1], v.round(3)))}
                  for (k, d, c), v in shares.items()]).to_csv(
        os.path.join(OUT, 'national_s2_l12_probability_classes.csv'), index=False)
    area_curve(cells)
    for suffix in ('', *SCHEMES):
        scheme = SCHEMES.get(suffix)
        x0, x1, y0, y1 = frame_of(canada, BOTTOM[suffix])
        ratio = (y1 - y0) / (x1 - x0)
        for design, clf in MODELS:
            fig, ax = plt.subplots(figsize=(8, 8 * ratio))
            national_panel(ax, f'{design}_{clf}', TITLES[design, clf], '', canada, dep, scheme,
                           shares.get((suffix, design, clf)), BOTTOM[suffix])
            name = f'fig_map_national{suffix}_{design}_{clf}.png'
            fig.savefig(os.path.join(FIG, name), dpi=250, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print('saved', name, flush=True)
        fig, axes = plt.subplots(2, 2, figsize=(16, 16 * ratio + 0.6))
        for ax, (design, clf), letter in zip(axes.ravel(), MODELS, 'abcd'):
            national_panel(ax, f'{design}_{clf}', TITLES[design, clf], letter, canada, dep, scheme,
                           shares.get((suffix, design, clf)), BOTTOM[suffix])
        plt.tight_layout(w_pad=1.5, h_pad=2.0)
        fig.savefig(os.path.join(FIG, f'fig_map_national{suffix}_r7.png'), dpi=250, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f'saved fig_map_national{suffix}_r7.png', flush=True)


def area_curve(cells):
    """Share of the cells scoring at or above each probability threshold,
    with the values at 0.5 marked."""
    t = np.logspace(-3, 0, 301).astype(np.float32)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for design, clf in MODELS:
        s = np.sort(cells[f'{design}_{clf}_score'].values)
        at = lambda x: 100.0 * (1.0 - np.searchsorted(s, x, side='left') / len(s))
        name = 'Parsa et al. (2024) design' if design == 'parsa' else 'this study'
        ax.plot(t, at(t), color=DESIGN_COLORS[design], ls='-' if clf == 'RF' else '--', lw=1.8,
                label=f'{clf}, {name}')
        v = at(np.float32(0.5))
        ax.plot(0.5, v, 'o', color=DESIGN_COLORS[design], ms=5, zorder=5)
        ax.annotate(share_text(v), (0.5, v), xytext=(6, 8 if v < 3 else 0), textcoords='offset points', va='center',
                    fontsize=9)
    ax.axvline(0.5, color='#999999', lw=0.8, ls=':', zorder=0)
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 1); ax.set_ylim(0, 100)
    ax.set_xlabel('Probability threshold')
    ax.set_ylabel('Cells at or above the threshold (%)')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='y', color='#e5e5e5', lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc='center left', bbox_to_anchor=(0.03, 0.45))
    fig.savefig(os.path.join(FIG, 'fig_area_national_r7.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved fig_area_national_r7.png', flush=True)


def diagnostics():
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler
    g = _l.load_grid()
    X = g['X'].astype(float)
    D = designs(g)
    rows = []
    samp = np.random.default_rng(0).choice(len(X), 50000, replace=False)
    for design in ('parsa', 'ours'):
        X_tr, y_tr, _, neg = D[design]
        imp = SimpleImputer(strategy='median').fit(X_tr)
        mm = MinMaxScaler().fit(imp.transform(X_tr))
        nn = NearestNeighbors(n_neighbors=1).fit(mm.transform(imp.transform(X_tr)))
        idx = nn.kneighbors(mm.transform(imp.transform(X[samp])), return_distance=False)[:, 0]
        rows.append((f'{design}: share of 50,000 random cells whose nearest training row is a positive',
                     float(y_tr[idx].mean())))
        if design == 'parsa':
            rows.append(('parsa: median distance of the negatives to the nearest occurrence (km)',
                         float(np.median(g['d_occ_km'][neg]))))
    rows.append(('all cells: median distance to the nearest occurrence (km)', float(np.median(g['d_occ_km']))))
    for design, clf in MODELS:
        s = np.load(os.path.join(OUT, 'scores', f'{design}_{clf}.npy')).astype(np.float32)
        for t in (0.5, 0.9):
            rows.append((f'{design}_{clf}: share of cells >= {t}', float((s >= np.float32(t)).mean())))
        rows.append((f'{design}_{clf}: share of cells = 0', float((s == 0).mean())))
        rows.append((f'{design}_{clf}: max score', float(s.max())))
    s = np.load(os.path.join(OUT, 'scores', 'ours_RF.npy')).astype(np.float32)
    hi, pos, d = s >= np.float32(0.5), g['is_pos'], g['d_occ_km']
    rows += [('ours_RF: cells >= 0.5', int(hi.sum())), ('ours_RF: occurrence cells among them', int((hi & pos).sum())),
             ('ours_RF: largest distance of the others to an occurrence (km)', float(d[hi & ~pos].max()))]
    lo = s >= np.float32(0.01)
    prov = g['province'].astype(str)
    rows += [('ours_RF: share of cells >= 0.01', float(lo.mean())),
             ('ours_RF: share of the cells >= 0.01 in Superior, Cordillera or Grenville',
              float(np.isin(prov[lo], ['Superior', 'Cordillera', 'Grenville']).mean())),
             ('ours_RF: median distance of the cells >= 0.01 to an occurrence (km)', float(np.median(d[lo])))]
    df = pd.DataFrame(rows, columns=['metric', 'value'])
    df.to_csv(os.path.join(OUT, 'national_s2_l12_diagnostics.csv'), index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=['export', 'figures', 'diagnostics'], default='export')
    {'export': export, 'figures': figures, 'diagnostics': diagnostics}[ap.parse_args().step]()
