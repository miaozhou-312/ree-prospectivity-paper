"""
Section 12 -- Prediction map.

Builds a regular A∧B-domain-restricted prediction grid per region, scores it
with both production models, saves the scored grid as GeoJSON, and draws two
standalone, true-scale prospectivity maps (RF and DevNet, each its own
figure and PNG) with a P90/P95/P99 upper-tail classification, lat/lon
graticule, north arrow, scale bar and known-deposit + candidate-target
markers.

DevNet's panel label says "semi-supervised" so the two panels' data
paradigms are visible at a glance, and marks each region's top
TOP_N_TARGETS scored unlabeled points as candidate new targets, distinct
from the known-deposit markers both panels already show. Both panels score
the grid using the region's pruned feature set (`trained[name]['feat_cols']`),
so training and the prediction map are never on different feature sets.
"""
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from rasterio.features import geometry_mask as _geomask
from rasterio.transform import Affine
from rasterio.transform import rowcol as _rowcol
from scipy.interpolate import griddata as _griddata

from src.features.glcm import add_glcm_features
from src.features.proximity import add_proximity_line_layers, add_proximity_worms, add_rasters
from src.features.sampleable_domain import SAMPLEABLE, sampleable_lookup
from src.features.terrane import add_terrane_h3
from src.models.devnet import ensemble_score, get_device

_PROSP_COLORS = ["#1a6faf", "#42c4e8", "#f5e21a", "#d73027"]

# Known-deposit confidence: dark = most confirmed. cfg['dep'] carries a
# dep_type column (Deposit/Prospect / Showing / Occurrence); falls back to
# 'Unknown' (grey) if a dep source without that column is ever swapped in.
DEP_TYPE_ORDER = ['Deposit/Prospect', 'Showing', 'Occurrence']
COLOR_DEP_TYPE = {
    'Deposit/Prospect': "#811010",
    'Showing':          "#2ad633",
    'Occurrence':       '#86b6ef',
    'Unknown':          '#898781',
}

# ── True-scale cartography (fixed 1:20,000,000 across every region) ────────
MAP_SCALE_DENOM = 20_000_000   # 1:20,000,000, fixed across all regions/panels
_INCH_M = 0.0254               # metres per inch


def _true_scale_size_in(extent_m, scale_denom=MAP_SCALE_DENOM):
    """Physical panel size (inches) for a ground extent (metres) at a fixed
    map scale -- 1 inch of paper == scale_denom inches == scale_denom *
    _INCH_M metres of ground. Only physically true at whatever DPI the
    figure is actually printed/exported at "100% / actual size"."""
    return extent_m / (scale_denom * _INCH_M)


def _draw_bnd(ax, bnd_proj):
    for geom in bnd_proj.geometry:
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, 'k-', lw=1.2, zorder=3)


def _draw_graticule(ax, projected_crs, lon_step=10, lat_step=10, n_seg=60):
    lonlat_to_proj = Transformer.from_crs('EPSG:4326', projected_crs, always_xy=True)
    proj_to_lonlat = Transformer.from_crs(projected_crs, 'EPSG:4326', always_xy=True)

    bx0, bx1 = ax.get_xlim()
    by0, by1 = ax.get_ylim()

    corner_x = [bx0, bx1, bx0, bx1, (bx0 + bx1) / 2, (bx0 + bx1) / 2, bx0, bx1]
    corner_y = [by0, by0, by1, by1, by0, by1, (by0 + by1) / 2, (by0 + by1) / 2]
    lons, lats = proj_to_lonlat.transform(corner_x, corner_y)
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    lon_lines = np.arange(np.ceil(lon_min / lon_step) * lon_step, lon_max + lon_step, lon_step)
    lat_lines = np.arange(np.ceil(lat_min / lat_step) * lat_step, lat_max + lat_step, lat_step)

    for lon in lon_lines:
        lat_seg = np.linspace(lat_min - lat_step, lat_max + lat_step, n_seg)
        xs, ys = lonlat_to_proj.transform(np.full(n_seg, lon), lat_seg)
        xs, ys = np.asarray(xs), np.asarray(ys)
        ax.plot(xs, ys, color='grey', lw=0.5, ls=':', alpha=0.55, zorder=2)
        j = np.argmin(np.abs(ys - by1))
        if bx0 <= xs[j] <= bx1:
            ax.text(xs[j], by1, f'{lon:.0f}\xb0', ha='center', va='bottom',
                    fontsize=7, color='#333333', clip_on=False)

    for lat in lat_lines:
        lon_seg = np.linspace(lon_min - lon_step, lon_max + lon_step, n_seg)
        xs, ys = lonlat_to_proj.transform(lon_seg, np.full(n_seg, lat))
        xs, ys = np.asarray(xs), np.asarray(ys)
        ax.plot(xs, ys, color='grey', lw=0.5, ls=':', alpha=0.55, zorder=2)
        i = np.argmin(np.abs(xs - bx0))
        if by0 <= ys[i] <= by1:
            ax.text(bx0, ys[i], f'{lat:.0f}\xb0', ha='right', va='center',
                    fontsize=7, color='#333333', clip_on=False)


def _draw_north_arrow(ax, xy_frac=(0.93, 0.90), size_frac=0.07):
    x, y = xy_frac
    ax.annotate('N', xy=(x, y), xytext=(x, y - size_frac),
                xycoords='axes fraction', textcoords='axes fraction',
                ha='center', va='center', fontsize=11, fontweight='bold', zorder=9,
                arrowprops=dict(facecolor='black', edgecolor='black', width=3.5,
                                 headwidth=11, headlength=9, shrink=0.02))


def _draw_scale_bar(ax, length_km=500, x_frac=0.55, y_frac=0.06):
    bx0, bx1 = ax.get_xlim()
    by0, by1 = ax.get_ylim()
    length_m = length_km * 1000.0
    x0 = bx0 + x_frac * (bx1 - bx0)
    y0 = by0 + y_frac * (by1 - by0)
    tick_h = (by1 - by0) * 0.012
    ax.plot([x0, x0 + length_m], [y0, y0], color='black', lw=2.2,
            solid_capstyle='butt', zorder=8)
    for xt in (x0, x0 + length_m / 2, x0 + length_m):
        ax.plot([xt, xt], [y0 - tick_h, y0 + tick_h], color='black', lw=1.4, zorder=8)
    ax.text(x0 + length_m / 2, y0 + tick_h * 2.2, f'{length_km} km',
            ha='center', va='bottom', fontsize=8, zorder=8)


def _draw_study_area_label(ax, name, x_frac=0.03, y_frac=0.03):
    ax.text(x_frac, y_frac, f'Study Area: {name}', transform=ax.transAxes,
            fontsize=9, ha='left', va='bottom', zorder=9,
            bbox=dict(facecolor='white', edgecolor='#999999', alpha=0.85, boxstyle='round,pad=0.3'))


def _sampleable_grid_mask(name, Xi, Yi):
    """Xi/Yi are already in PROJECTED_CRS (== SAMPLEABLE's reference grid
    CRS), so no reprojection is needed here, only a row/col lookup."""
    info = SAMPLEABLE[name]
    ref_tf, ref_shape, mask = info['ref_tf'], info['ref_shape'], info['sampleable']
    rows, cols = _rowcol(ref_tf, Xi.ravel(), Yi.ravel())
    rows = np.asarray(rows).reshape(Xi.shape)
    cols = np.asarray(cols).reshape(Xi.shape)
    H, W = ref_shape
    in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    out = np.zeros(Xi.shape, dtype=bool)
    out[in_bounds] = mask[rows[in_bounds], cols[in_bounds]]
    return out


def _prosp_panel(ax, proba, xs_p, ys_p, bnd_proj, dep_proj, model_name, name, projected_crs,
                  extra_pts_proj=None, extra_label='Potential target', show_title=True):
    """P90/P95/P99 upper-tail classification panel. show_title=False
    suppresses the per-axes title -- used when a figure-level title (see
    _make_prosp_figure) already carries that information."""
    valid = proba[np.isfinite(proba)]

    bx0, by0, bx1, by1 = bnd_proj.total_bounds
    res = 5000

    y_range = np.arange(by1, by0, -res)
    x_range = np.arange(bx0, bx1, res)
    Xi, Yi = np.meshgrid(x_range, y_range)

    Z = _griddata(np.column_stack([xs_p, ys_p]), proba, (Xi, Yi), method='linear')
    tfm = Affine.translation(bx0, by1) * Affine.scale(res, -res)
    outline_msk    = _geomask(bnd_proj.geometry, out_shape=Z.shape, transform=tfm, invert=True)
    sampleable_msk = _sampleable_grid_mask(name, Xi, Yi)
    msk = outline_msk & sampleable_msk
    Z   = np.where(msk, Z, np.nan)

    ext  = [bx0, bx1, by0, by1]
    bks  = [0.0,
            float(np.percentile(valid, 90)),
            float(np.percentile(valid, 95)),
            float(np.percentile(valid, 99)), 1.0]
    cmap = mcolors.ListedColormap(_PROSP_COLORS)
    norm = mcolors.BoundaryNorm(bks, len(_PROSP_COLORS))

    ax.imshow(np.ma.masked_invalid(Z), extent=ext, origin='upper',
              cmap=cmap, norm=norm, aspect='equal', zorder=1)
    ax.set_xlim(bx0, bx1); ax.set_ylim(by0, by1)
    ax.set_aspect('equal')
    _draw_bnd(ax, bnd_proj)
    _draw_graticule(ax, projected_crs)
    dep_type_vals = (dep_proj['dep_type'].fillna('Unknown').values
                     if 'dep_type' in dep_proj.columns
                     else np.array(['Unknown'] * len(dep_proj)))
    dep_types_present = []
    for _t in DEP_TYPE_ORDER + ['Unknown']:
        _sub = dep_proj[dep_type_vals == _t]
        if len(_sub) == 0:
            continue
        ax.scatter(_sub.geometry.x, _sub.geometry.y,
                   c=COLOR_DEP_TYPE.get(_t, '#898781'), s=130, marker='*', alpha=0.85,
                   zorder=5, linewidths=0.5, edgecolors='black')
        dep_types_present.append((_t, len(_sub)))
    if extra_pts_proj is not None and len(extra_pts_proj) > 0:
        ax.scatter(extra_pts_proj.geometry.x, extra_pts_proj.geometry.y,
                   facecolors='none', edgecolors='#eda100', s=200, marker='D',
                   linewidths=2.0, zorder=6)

    _PRIORITY_LABELS = ['Background (lower 90%)',
                        'Priority 3 — top 10% of area',
                        'Priority 2 — top 5% of area',
                        'Priority 1 — top 1% of area']
    patches = [mpatches.Patch(facecolor=_PROSP_COLORS[i], edgecolor='grey', lw=0.5,
               label=_PRIORITY_LABELS[i])
               for i in range(4)]
    for _t, _n in dep_types_present:
        patches.append(plt.Line2D([0], [0], marker='*', color='w',
                                   markerfacecolor=COLOR_DEP_TYPE.get(_t, '#898781'),
                                   markeredgecolor='black', markeredgewidth=0.5,
                                   markersize=11, alpha=0.85, label=f'Known REE: {_t} (n={_n})'))
    if extra_pts_proj is not None and len(extra_pts_proj) > 0:
        patches.append(plt.Line2D([0], [0], marker='D', color='w',
                                   markerfacecolor='none', markeredgecolor='#eda100',
                                   markeredgewidth=2.0, markersize=10,
                                   alpha=0.9, label=extra_label))
    # Legend OUTSIDE the axes (to the right); the figure reserves LEGEND_W_IN
    # of width for exactly this, so it never overlaps the map.
    ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              bbox_transform=ax.transAxes, fontsize=8, title='Legend',
              title_fontsize=9, framealpha=0.9, borderaxespad=0.)
    if show_title:
        ax.set_title(model_name, fontsize=11, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])   # graticule replaces metre-based ticks
    ax.set_xlim(bx0, bx1); ax.set_ylim(by0, by1)

    _draw_north_arrow(ax)
    _draw_scale_bar(ax)
    _draw_study_area_label(ax, name)


def _make_prosp_figure(proba, xs_p, ys_p, bnd_proj, dep_in, name, model_short, model_desc,
                        projected_crs, extra_pts_proj=None, extra_label='Potential target'):
    """Build ONE standalone, true-scale prospectivity figure: a single map
    with its legend to the right and a three-tier title block (region +
    model, CRS/scale, model description). The panel is sized in real inches
    by _true_scale_size_in, so the fixed 1:MAP_SCALE_DENOM scale holds
    regardless of how many figures are produced."""
    bx0, by0, bx1, by1 = bnd_proj.total_bounds
    panel_w_in = _true_scale_size_in(bx1 - bx0)
    panel_h_in = _true_scale_size_in(by1 - by0)

    LEFT_MARGIN_IN, LEGEND_W_IN, RIGHT_PAD_IN = 0.6, 2.6, 0.25
    TOP_MARGIN_IN, BOTTOM_MARGIN_IN = 1.2, 0.35

    fig_w_in = LEFT_MARGIN_IN + panel_w_in + LEGEND_W_IN + RIGHT_PAD_IN
    fig_h_in = TOP_MARGIN_IN + panel_h_in + BOTTOM_MARGIN_IN

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    ax_left   = LEFT_MARGIN_IN / fig_w_in
    ax_bottom = BOTTOM_MARGIN_IN / fig_h_in
    ax_w      = panel_w_in / fig_w_in
    ax_h      = panel_h_in / fig_h_in
    ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])

    # Title centred over the MAP only (the legend occupies the right side).
    title_x = ax_left + ax_w / 2.0
    fig.text(title_x, 1 - 0.30 / fig_h_in, f'{name} — {model_short}',
              ha='center', va='top', fontsize=13, fontweight='bold')
    fig.text(title_x, 1 - 0.62 / fig_h_in,
              f'REE Prospectivity  ·  {projected_crs}  ·  1:{MAP_SCALE_DENOM:,}',
              ha='center', va='top', fontsize=8.5, color='#444444')
    fig.text(title_x, 1 - 0.83 / fig_h_in, model_desc,
              ha='center', va='top', fontsize=7.5, color='#666666', style='italic')

    _prosp_panel(ax, proba, xs_p, ys_p, bnd_proj, dep_in, model_short, name, projected_crs,
                 extra_pts_proj=extra_pts_proj, extra_label=extra_label, show_title=False)
    return fig


def build_grid(cfg, name, grid_deg):
    """Regular prediction grid within the study area (EPSG:4326), then
    restricted to the A∧B sampleable domain."""
    boundary   = gpd.read_file(cfg['outline']).to_crs('EPSG:4326')
    study_area = boundary.unary_union
    xmin, ymin, xmax, ymax = boundary.total_bounds
    xs = np.arange(xmin, xmax, grid_deg)
    ys = np.arange(ymin, ymax, grid_deg)
    xx, yy = np.meshgrid(xs, ys)
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs='EPSG:4326')
    gdf = gdf[gdf.within(study_area)].copy()
    gdf['lat'] = gdf.geometry.y
    gdf['lon'] = gdf.geometry.x

    n_before = len(gdf)
    in_dom = sampleable_lookup(gdf['lat'].values, gdf['lon'].values, name)
    gdf = gdf[in_dom].reset_index(drop=True)
    print(f'  [{name}] prediction grid: {n_before} (within outline) -> {len(gdf)} '
          f'after A∧B sampleable-domain filter ({len(gdf)/n_before*100:.1f}% kept)')
    return gdf, boundary


def predict_grid(cfg, name, rf_model, rf_scaler, rf_imputer, feat_cols,
                  dn_ensemble, dn_scaler, dn_imputer, worm_trees, clipped_rasters, clipped_vectors,
                  config, glcm_rasters, top_targets=None):
    print(f'[{name}] building prediction grid ...')
    device = get_device(config.FORCE_CPU)
    grid, boundary = build_grid(cfg, name, config.GRID_DEG)

    pts = pd.DataFrame({'lat': grid['lat'], 'lon': grid['lon']})
    pts = add_terrane_h3(pts, clipped_vectors.get('h3_grid'))
    pts = add_proximity_line_layers(pts, cfg['vectors'], config.PROJECTED_CRS)
    pts = add_proximity_worms(pts, worm_trees, config.PROJECTED_CRS)
    pts = add_rasters(pts, clipped_rasters)

    print(f'  extracting GLCM texture features for prediction grid ...')
    pts = add_glcm_features(pts, clipped_rasters, glcm_rasters)

    X = pts.reindex(columns=feat_cols).astype(float).values

    X_rf     = rf_scaler.transform(rf_imputer.transform(X))
    rf_proba = rf_model.predict_proba(X_rf)[:, 1]

    X_dn = dn_scaler.transform(dn_imputer.transform(X))
    dn_proba = ensemble_score(dn_ensemble, X_dn, device)  # rank-averaged ensemble, already ~[0, 1]

    if top_targets is not None and len(top_targets) > 0:
        print(f'  [{name}] top-{config.TOP_N_TARGETS} target percentiles within the prediction grid:')
        for _, r in top_targets.iterrows():
            pct = (dn_proba < r['dn_score']).mean() * 100
            print(f"    ({r['lat']:.4f}, {r['lon']:.4f})  grid percentile = {pct:.2f}")

    grid['rf_proba'] = rf_proba
    grid['dn_proba'] = dn_proba
    grid.to_file(f'{name}_prospectivity_combined_dedup.geojson', driver='GeoJSON')
    print(f'  saved -> {name}_prospectivity_combined_dedup.geojson')

    grid_proj = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid['lon'], grid['lat']), crs='EPSG:4326'
    ).to_crs(config.PROJECTED_CRS)
    xs_p = grid_proj.geometry.x.values
    ys_p = grid_proj.geometry.y.values

    bnd_proj = boundary.to_crs(config.PROJECTED_CRS)
    dep      = pd.read_csv(cfg['dep']).dropna(subset=['Latitude', 'Longitude'])
    dep_gdf  = gpd.GeoDataFrame(
        dep,
        geometry=gpd.points_from_xy(dep['Longitude'], dep['Latitude']),
        crs='EPSG:4326',
    ).to_crs(config.PROJECTED_CRS)
    dep_in   = dep_gdf[dep_gdf.within(bnd_proj.unary_union)]

    top_targets_proj = None
    if top_targets is not None and len(top_targets) > 0:
        top_targets_proj = gpd.GeoDataFrame(
            top_targets,
            geometry=gpd.points_from_xy(top_targets['lon'], top_targets['lat']),
            crs='EPSG:4326',
        ).to_crs(config.PROJECTED_CRS)

    # ── Two separate, standalone true-scale figures (RF then DevNet) --
    # each sized in real inches from the region's true ground extent
    # (_true_scale_size_in), not stretched to a fixed canvas, and each saved
    # to its own PNG for the report.
    fig_rf = _make_prosp_figure(
        rf_proba, xs_p, ys_p, bnd_proj, dep_in, name,
        'Random Forest', 'positive vs. shared background (labeled negative)',
        config.PROJECTED_CRS)
    fig_rf.savefig(f'{name}_prospectivity_RF.png', dpi=300, facecolor='white')
    plt.show()

    fig_dn = _make_prosp_figure(
        dn_proba, xs_p, ys_p, bnd_proj, dep_in, name,
        'DevNet', 'semi-supervised: positive vs. shared background (unlabeled)',
        config.PROJECTED_CRS,
        extra_pts_proj=top_targets_proj,
        extra_label=f'Top-{config.TOP_N_TARGETS} unlabeled (potential target)')
    fig_dn.savefig(f'{name}_prospectivity_DevNet.png', dpi=300, facecolor='white')
    plt.show()

    return grid


def generate_prediction_maps(config, trained, worm_trees, clipped_rasters, clipped_vectors, glcm_rasters,
                              run_regions=None):
    """Section "Generate prediction maps for regions in RUN_REGIONS" --
    runs predict_grid for every region and returns GRID_SCORES (the
    per-region RF/DevNet grid-score arrays used by the capture-rate /
    reporting sections)."""
    run_regions = list(config.RUN_REGIONS) if run_regions is None else list(run_regions)
    grid_scores = {}
    for name in run_regions:
        cfg = config.REGIONS[name]
        t = trained[name]
        grid = predict_grid(
            cfg, name,
            t['rf_model'], t['rf_scaler'], t['rf_imp'], t['feat_cols'],
            t['dn_ensemble'], t['dn_scaler'], t['dn_imp'],
            worm_trees, clipped_rasters[name], clipped_vectors[name], config, glcm_rasters,
            top_targets=t.get('top_targets'),
        )
        trained[name]['grid'] = grid
        grid_scores[name] = {'RF': grid['rf_proba'].values,
                              'DevNet(deviation)': grid['dn_proba'].values}
    return grid_scores
