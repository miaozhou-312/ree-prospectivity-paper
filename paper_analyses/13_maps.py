"""
Stage 13 -- prospectivity maps in the cartographic style of the thesis
figures (made in ArcGIS Pro): Canada Atlas Lambert (EPSG:3978), score
percentile classes (background < 90%, top 10%, 5%, 1% of mapped area),
occurrences by development class, lat/lon graticule labelled on the frame,
north arrow, scale bar and a location inset. Colours were sampled from the
thesis maps.

Scores at the 0.1 degree prediction-grid nodes (Stage 12) are interpolated
linearly onto a 2 km raster and clipped to the study-area outline and to
the scored domain; classes are percentiles over mapped pixels.

  fig_map_superior_rev.png  (a) DevNet (final configuration), (b) RF
  fig_ksz_rev.png           Kapuskasing zoom of DevNet, framed to match the
                            thesis geology panel pixel for pixel.

    python -m paper_analyses.13_maps
"""
import os

import matplotlib

matplotlib.use('Agg')
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from shapely.geometry import box  # noqa: E402

from paper_analyses.common import OUT_DIR, config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, 'paper', 'manuscript', 'figures')
CRS = 'EPSG:3978'
CLASS_COLORS = ['#1a6faf', '#42c4e8', '#f5e21a', '#d73027']
CLASS_LABELS = ['Background (<90%)', 'Top 10% area', 'Top 5% area', 'Top 1% area']
DEP_COLORS = {'Deposit/Prospect': '#e4191c', 'Showing': '#4daf4a', 'Occurrence': '#377cb8'}
FAULT_COLOR, OUTLINE_COLOR, GRAT_COLOR = '#717171', '#6e6e6e', '#d9d9d9'
TO_P = Transformer.from_crs('EPSG:4326', CRS, always_xy=True)
TO_G = Transformer.from_crs(CRS, 'EPSG:4326', always_xy=True)
plt.rcParams.update({'font.family': 'DejaVu Sans'})


def classify(scores):
    q90, q95, q99 = np.nanpercentile(scores, [90, 95, 99])
    c = np.zeros(scores.shape)
    c[scores >= q90] = 1
    c[scores >= q95] = 2
    c[scores >= q99] = 3
    c[np.isnan(scores)] = np.nan
    return c


def raster(grid, col, outline_p, res=2000.0, classifier=classify):
    xs, ys = TO_P.transform(grid['lon'].values, grid['lat'].values)
    xmin, ymin, xmax, ymax = outline_p.total_bounds
    gx, gy = np.meshgrid(np.arange(xmin, xmax, res), np.arange(ymin, ymax, res))
    z = griddata(np.c_[xs, ys], grid[col].values, (gx, gy), method='linear')
    d, _ = cKDTree(np.c_[xs, ys]).query(np.c_[gx.ravel(), gy.ravel()])
    z[(d > 9000).reshape(gx.shape)] = np.nan                    # outside the scored domain
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    tf = from_origin(xmin, ymin, res, -res)                       # rows increase northwards
    inside = ~geometry_mask(list(outline_p.geometry), out_shape=gx.shape, transform=tf)
    z[~inside] = np.nan
    return gx, gy, classifier(z), (xmin, xmax, ymin, ymax)


def graticule(ax, extent, lons, lats, label_edges=True, fontsize=11, lat_span=(30, 80), lon_span=(-150, -40),
              label_top=True, color=GRAT_COLOR, lw=0.8):
    x0, x1, y0, y1 = extent
    for lo in lons:
        la = np.linspace(*lat_span, 400)
        x, y = TO_P.transform(np.full_like(la, lo), la)
        ax.plot(x, y, color=color, lw=lw, zorder=1)
    for la in lats:
        lo = np.linspace(*lon_span, 800)
        x, y = TO_P.transform(lo, np.full_like(lo, la))
        ax.plot(x, y, color=color, lw=lw, zorder=1)
    if not label_edges:
        return
    for lo in lons:                           # meridians on bottom / top edges
        la = np.linspace(*lat_span, 4000)
        x, y = TO_P.transform(np.full_like(la, lo), la)
        for edge, va, dy in [(y0, 'top', -0.012), (y1, 'bottom', 0.012)][:2 if label_top else 1]:
            d = np.where((x0 < x) & (x < x1), np.abs(y - edge), np.inf)    # crossings inside the frame only
            i = np.argmin(d)
            if d[i] < 2e4:
                ax.text(x[i], edge + dy * (y1 - y0), f'{lo:.0f}°', ha='center', va=va, fontsize=fontsize,
                        clip_on=False)
    for la in lats:                           # parallels on left / right edges
        lo = np.linspace(*lon_span, 8000)
        x, y = TO_P.transform(lo, np.full_like(lo, la))
        for edge, ha, dx in [(x0, 'right', -0.01), (x1, 'left', 0.01)]:
            d = np.where((y0 < y) & (y < y1), np.abs(x - edge), np.inf)
            i = np.argmin(d)
            if d[i] < 2e4:
                ax.text(edge + dx * (x1 - x0), y[i], f'{la:.0f}°', ha=ha, va='center', fontsize=fontsize,
                        clip_on=False)


def north_arrow(ax, xf, yf, size, lon_lat):
    """ArcGIS-style split arrow pointing to true north at lon_lat."""
    x, y = TO_P.transform(*lon_lat)
    x2, y2 = TO_P.transform(lon_lat[0], lon_lat[1] + 0.5)
    ang = np.arctan2(x2 - x, y2 - y)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    cx, cy = x0 + xf * (x1 - x0), y0 + yf * (y1 - y0)
    L = size * (y1 - y0)
    rot = lambda px, py: (cx + px * np.cos(ang) + py * np.sin(ang), cy - px * np.sin(ang) + py * np.cos(ang))
    tip, left, right, notch = rot(0, L), rot(-0.32 * L, -0.25 * L), rot(0.32 * L, -0.25 * L), rot(0, 0.02 * L)
    ax.add_patch(Polygon([tip, left, notch], closed=True, fc='white', ec='black', lw=1.0, zorder=20))
    ax.add_patch(Polygon([tip, right, notch], closed=True, fc='black', ec='black', lw=1.0, zorder=20))
    nx, ny = rot(0, -0.55 * L)
    ax.text(nx, ny, 'N', ha='center', va='center', fontsize=18, fontweight='bold',
            rotation=-np.degrees(ang), zorder=20)


def scale_bar(ax, xf, yf, length_km, ticks, label='Kilometers', fontsize=11):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    sx, sy = x0 + xf * (x1 - x0), y0 + yf * (y1 - y0)
    h = 0.014 * (y1 - y0)
    L = length_km * 1000.0
    segs = list(zip(ticks[:-1], ticks[1:]))
    for k, (a, b) in enumerate(segs):
        ax.add_patch(Rectangle((sx + a * 1000, sy), (b - a) * 1000, h, fc='black' if k % 2 == 0 else 'white',
                               ec='black', lw=1.0, zorder=20))
    for t in ticks:
        ax.text(sx + t * 1000, sy + h * 1.6, f'{t:g}', ha='center', va='bottom', fontsize=fontsize, zorder=20)
    ax.text(sx + L / 2, sy - h * 0.8, label, ha='center', va='top', fontsize=fontsize, zorder=20)


def occurrences(region):
    dep = pd.read_csv(os.path.join(REPO, 'data', 'labels', f'{region}_REE_deposits_deduped_2.0km.csv'))
    x, y = TO_P.transform(dep['Longitude'].values, dep['Latitude'].values)
    return dep.assign(x=x, y=y)


def province_layers():
    g = gpd.read_file(os.path.join(os.path.dirname(REPO), 'dataset', 'extra', 'gsc_geological_provinces',
                                   'GEOPROV.SHP'))
    g = g[~g['GEOLPROV'].str.contains('Shelf|oceanic', case=False, na=False)].to_crs(CRS)
    return g


def map_panel(ax, grid, col, title, letter, outline_p, dep, canada, region='Superior', show_legend=True,
              classifier=classify, labels=CLASS_LABELS):
    gx, gy, cls, ext = raster(grid, col, outline_p, classifier=classifier)
    pad = 0.06
    w, h = ext[1] - ext[0], ext[3] - ext[2]
    frame = (ext[0] - pad * w, ext[1] + pad * w, ext[2] - 0.14 * h, ext[3] + 0.08 * h)
    ax.set_xlim(frame[0], frame[1]); ax.set_ylim(frame[2], frame[3]); ax.set_aspect('equal')
    graticule(ax, frame, lons=[-100, -90, -80, -70, -60], lats=[50, 60], fontsize=10)
    ax.pcolormesh(gx, gy, cls, cmap=ListedColormap(CLASS_COLORS), vmin=-0.5, vmax=3.5, shading='nearest',
                  zorder=2, rasterized=True)
    outline_p.boundary.plot(ax=ax, color=OUTLINE_COLOR, lw=1.3, zorder=3)
    for t, c in DEP_COLORS.items():
        s = dep[dep.dep_type == t]
        ax.scatter(s.x, s.y, s=70, c=c, edgecolors='black', linewidths=1.0, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():             # all four: 06b's style hides the top and right spines
        sp.set_visible(True)
        sp.set_linewidth(2.0)
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=26, va='top', ha='left')
    if show_legend:
        handles = [Line2D([], [], marker='o', ls='', mfc=c, mec='black', ms=9, label=f'{t} ({(dep.dep_type == t).sum()})')
                   for t, c in DEP_COLORS.items()]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.08, 0.97), frameon=False, fontsize=11,
                  handletextpad=1.2)
    patches = [Patch(fc=c, ec='none', label=l) for c, l in zip(CLASS_COLORS, labels)]
    first = ax.get_legend()
    ax.legend(handles=patches, loc='lower left', frameon=False, fontsize=11, title=title, title_fontsize=12,
              alignment='left', handlelength=2.6, handleheight=1.5)
    if first is not None:
        ax.add_artist(first)
    north_arrow(ax, 0.88, 0.17, 0.07, (-75, 50))
    scale_bar(ax, 0.66, 0.07, 500, [0, 500])
    inset = ax.inset_axes([0.62, 0.79, 0.36, 0.19])
    canada.boundary.plot(ax=inset, color='#8c8c8c', lw=0.25)
    outline_p.boundary.plot(ax=inset, color='#e4191c', lw=0.9)
    inset.set_xticks([]); inset.set_yticks([]); inset.set_aspect('equal')
    for sp in inset.spines.values():
        sp.set_visible(True)
        sp.set_color('#8c8c8c'); sp.set_linewidth(0.8)
    return cls


def superior_maps(panels=(('devnet', 'Superior DevNet MPM', 'a'), ('rf', 'Superior RF MPM', 'b')),
                  outname='fig_map_superior_rev.png', grid_file='superior_grid_scores.csv'):
    grid = pd.read_csv(os.path.join(OUT_DIR, grid_file))
    outline_p = gpd.read_file(os.path.join(REPO, 'data', 'outlines', 'Superior.geojson')).to_crs(CRS)
    dep = occurrences('Superior')
    canada = province_layers()
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 9.2))
    for ax, (col, title, letter) in zip(np.atleast_1d(axes), panels):
        map_panel(ax, grid, col, title, letter, outline_p, dep, canada)
    plt.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG, outname), dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', outname)


STABILITY_LABELS = ['<20% of models', '20–50%', '50–80%', '≥80% (stable)']


def classify_stability(freq):
    c = np.digitize(freq, [0.2, 0.5, 0.8]).astype(float)
    c[np.isnan(freq)] = np.nan
    return c


def stability_maps(outname='fig_map_stability_rev.png', grid_file='superior_grid_uncertainty.csv'):
    """Share of the 15 leave-one-occurrence-out models (Stage 21) that put
    each node in the province's top 5%."""
    grid = pd.read_csv(os.path.join(OUT_DIR, grid_file))
    outline_p = gpd.read_file(os.path.join(REPO, 'data', 'outlines', 'Superior.geojson')).to_crs(CRS)
    dep = occurrences('Superior')
    canada = province_layers()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9.2))
    for ax, (col, title, letter) in zip(axes, [('devnet_top5_freq', 'DevNet: top-5% stability', 'a'),
                                                ('rf_top5_freq', 'RF: top-5% stability', 'b')]):
        map_panel(ax, grid, col, title, letter, outline_p, dep, canada, classifier=classify_stability,
                  labels=STABILITY_LABELS)
    plt.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(FIG, outname), dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', outname)


# Fig. 12: panel (b) is drawn on the pixel canvas of the thesis geology panel
# (a, figures/fig_ksz_geology.png, 2631 x 1600 px), with every map element of
# (a) -- frame, graticule, inset, scale bar, legends, north arrow, labels,
# symbols and font (Tahoma) -- at the position and size measured on (a).
KSZ_W, KSZ_H, KSZ_DPI = 2631, 1600, 250
KSZ_FRAME = (88.5, 2542.0, 65.5, 1536.5)      # frame of (a): centres of its border lines, pixels
KSZ_FONT, KSZ_SMALL = 'Tahoma', 9.5          # legend, scale-bar and occurrence text (pt)


def _fx(x):
    return x / KSZ_W


def _fy(y):
    return 1 - y / KSZ_H


def _pt(px):
    return px * 72 / KSZ_DPI


def ksz_frame():
    """Projected extent of the frame of (a), solved from where its graticule
    crosses lines just inside the frame (pixel positions measured on the
    2631 x 1600 image)."""
    px0, px1, py0, py1 = KSZ_FRAME
    meridians = [(-84, 342.0, 74), (-83, 921.0, 74), (-82, 1503.5, 74), (-81, 2090.0, 74),
                 (-84, 596.0, 1528), (-83, 1199.0, 1528), (-82, 1805.5, 1528), (-81, 2415.5, 1528)]
    parallels = [(49, 98, 587.5), (48, 98, 1477.0), (49, 2532, 100.0), (48, 2532, 1000.0)]

    def to_proj(p, xp, yp):
        X0, X1, Y0, Y1 = p
        return X0 + (xp - px0) / (px1 - px0) * (X1 - X0), Y1 - (yp - py0) / (py1 - py0) * (Y1 - Y0)

    def resid(p):
        r = []
        for lo, xp, yp in meridians:
            x, y = to_proj(p, xp, yp)
            r.append((TO_G.transform(x, y)[0] - lo) * 70e3)
        for la, xp, yp in parallels:
            x, y = to_proj(p, xp, yp)
            r.append((TO_G.transform(x, y)[1] - la) * 111e3)
        return r

    xa, ya = TO_P.transform(-84.6, 47.95)
    xb, yb = TO_P.transform(-80.8, 49.2)
    sol = least_squares(resid, [xa, xb, ya, yb])
    print(f'KSZ frame fit: max residual {np.max(np.abs(sol.fun)) / 1000:.2f} km')
    return sol.x


def _box(fig, x0, y0, x1, y1, lw=1.0):
    fig.add_artist(Rectangle((_fx(x0), _fy(y1)), (x1 - x0) / KSZ_W, (y1 - y0) / KSZ_H, transform=fig.transFigure,
                             fc='white', ec='black', lw=lw, zorder=30))


def _ksz_pixels(ax):
    """Projected (x, y) -> pixel (column, row) on the canvas of (a)."""
    X0, X1 = ax.get_xlim()
    Y0, Y1 = ax.get_ylim()
    sx = (KSZ_FRAME[1] - KSZ_FRAME[0]) / (X1 - X0)
    sy = (KSZ_FRAME[3] - KSZ_FRAME[2]) / (Y1 - Y0)
    return lambda x, y: (KSZ_FRAME[0] + (np.asarray(x) - X0) * sx, KSZ_FRAME[2] + (Y1 - np.asarray(y)) * sy)


def _ksz_graticule_labels(fig, ax, lons, lats, fontsize=12):
    """Graticule labels of (a): a short black tick continues each line 13 px
    beyond the frame line, and the label sits off the end of the tick."""
    to_px = _ksz_pixels(ax)
    fx0, fx1, fy0, fy1 = KSZ_FRAME
    tick = lambda p, q: fig.add_artist(Line2D([_fx(p[0]), _fx(q[0])], [_fy(p[1]), _fy(q[1])], transform=fig.transFigure,
                                              color='black', lw=_pt(4), solid_capstyle='butt', zorder=30))
    text = dict(fontsize=fontsize, zorder=30)
    for lo in lons:
        cx, cy = to_px(*TO_P.transform(np.full(2000, lo), np.linspace(46, 51, 2000)))
        for edge, out in ((fy0, -1), (fy1, 1)):
            i = np.argmin(np.abs(cy - edge))
            if not fx0 < cx[i] < fx1:
                continue
            k = min(i + 1, len(cx) - 1) if cy[min(i + 1, len(cx) - 1)] * out > cy[i] * out else max(i - 1, 0)
            dx, dy = cx[k] - cx[i], cy[k] - cy[i]
            end = (cx[i] + dx / abs(dy) * 13, edge + 13 * out)
            tick((cx[i], edge), end)
            y = end[1] - 12 if out < 0 else end[1] + 41
            fig.text(_fx(end[0]), _fy(y), f'{lo:.0f}°', ha='center', va='baseline', **text)
    for la in lats:
        cx, cy = to_px(*TO_P.transform(np.linspace(-90, -75, 4000), np.full(4000, la)))
        for edge, out in ((fx0, -1), (fx1, 1)):
            i = np.argmin(np.abs(cx - edge))
            if not fy0 < cy[i] < fy1:
                continue
            k = min(i + 1, len(cx) - 1) if cx[min(i + 1, len(cx) - 1)] * out > cx[i] * out else max(i - 1, 0)
            dx, dy = cx[k] - cx[i], cy[k] - cy[i]
            end = (edge + 13 * out, cy[i] + dy / abs(dx) * 13)
            tick((edge, cy[i]), end)
            fig.text(_fx(end[0] + (12 if out > 0 else -11)), _fy(end[1] + (14.5 if out > 0 else 17)), f'{la:.0f}°',
                     ha='left' if out > 0 else 'right', va='baseline', **text)


def _ksz_north_arrow(fig, ax, tip):
    """Outline arrow of (a), pointing to true north at its own position: thick
    right side and base, thin left side; N below it."""
    X0, X1 = ax.get_xlim()
    Y0, Y1 = ax.get_ylim()
    x = X0 + (tip[0] - KSZ_FRAME[0]) / (KSZ_FRAME[1] - KSZ_FRAME[0]) * (X1 - X0)
    y = Y1 - (tip[1] - KSZ_FRAME[2]) / (KSZ_FRAME[3] - KSZ_FRAME[2]) * (Y1 - Y0)
    lon, lat = TO_G.transform(x, y)
    to_px = _ksz_pixels(ax)
    u = np.subtract(to_px(*TO_P.transform(lon, lat + 0.5)), to_px(x, y))
    u /= np.hypot(*u)                                   # north, in pixels (y down)
    v = np.array([-u[1], u[0]])                         # to the right of north
    t = np.array(tip, float)
    right, notch, left = t - 138 * u + 46 * v, t - 126 * u, t - 138 * u - 46 * v
    seg = lambda p, q, lw: fig.add_artist(Line2D([_fx(p[0]), _fx(q[0])], [_fy(p[1]), _fy(q[1])],
                                                 transform=fig.transFigure, color='black', lw=lw,
                                                 solid_capstyle='round', solid_joinstyle='round', zorder=30))
    for p, q in ((t, right), (right, notch), (notch, left)):
        seg(p, q, _pt(9))
    seg(left, t, _pt(1.5))
    n = t - 176 * u + 3 * v
    fig.text(_fx(n[0]), _fy(n[1]), 'N', fontsize=16.5, fontweight='bold', ha='center', va='center',
             rotation=np.degrees(np.arctan2(-u[0], -u[1])), zorder=30)


def ksz_map(col='devnet', outname='fig_ksz_rev.png', grid_file='superior_grid_scores.csv'):
    grid = pd.read_csv(os.path.join(OUT_DIR, grid_file))
    outline_p = gpd.read_file(os.path.join(REPO, 'data', 'outlines', 'Superior.geojson')).to_crs(CRS)
    X0, X1, Y0, Y1 = ksz_frame()
    gx, gy, cls, _ = raster(grid, col, outline_p, res=1000.0)   # classes over the whole province
    dep = occurrences('Superior')
    faults = gpd.read_file(os.path.join(os.path.dirname(REPO), 'dataset', 'vectors', 'Faults_Canada.geojson'))
    faults = faults.to_crs(CRS).clip(box(X0, Y0, X1, Y1))
    with plt.rc_context({'font.family': KSZ_FONT}):
        fig = plt.figure(figsize=(KSZ_W / KSZ_DPI, KSZ_H / KSZ_DPI))
        fx0, fx1, fy0, fy1 = KSZ_FRAME
        ax = fig.add_axes([_fx(fx0), _fy(fy1), (fx1 - fx0) / KSZ_W, (fy1 - fy0) / KSZ_H])
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
        ax.pcolormesh(gx, gy, cls, cmap=ListedColormap(CLASS_COLORS), vmin=-0.5, vmax=3.5, shading='nearest',
                      zorder=1, rasterized=True)
        faults.plot(ax=ax, color=FAULT_COLOR, lw=_pt(7.5), zorder=3)
        graticule(ax, (X0, X1, Y0, Y1), lons=[-84, -83, -82, -81], lats=[48, 49], label_edges=False,
                  color='#828282', lw=_pt(3.5))
        _ksz_graticule_labels(fig, ax, [-84, -83, -82, -81], [48, 49])
        s = dep[(dep.x > X0) & (dep.x < X1) & (dep.y > Y0) & (dep.y < Y1)]
        ax.scatter(s.x, s.y, s=_pt(33) ** 2, c=DEP_COLORS['Occurrence'], edgecolors='black', linewidths=_pt(6),
                   zorder=6)
        lab = dict(fontsize=12, fontweight='bold', ha='center', va='center', zorder=8,
                   bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
        fig.text(_fx(859.5), _fy(721), 'Wawa  Subprovince', **lab)
        fig.text(_fx(1525), _fy(1339), 'Abitibi  Subprovince', **lab)
        fig.text(_fx(1210.5), _fy(991.5), 'KSZ', rotation=55, **dict(lab, fontsize=14))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(_pt(8))
        # inset: outline of the province with the frame of the map
        ins = fig.add_axes([_fx(152.5), _fy(653.5), 625 / KSZ_W, 532 / KSZ_H], zorder=20)
        outline_p.boundary.plot(ax=ins, color='#333333', lw=1.2)
        ox0, oy0, ox1, oy1 = outline_p.total_bounds
        wx = (ox1 - ox0) / ((746 - 173) / 625)
        wy = (oy1 - oy0) / ((627 - 136) / 532)
        ins.set_xlim(ox0 - (173 - 152.5) / 625 * wx, ox0 - (173 - 152.5) / 625 * wx + wx)
        ins.set_ylim(oy0 - (653.5 - 627) / 532 * wy, oy0 - (653.5 - 627) / 532 * wy + wy)
        ins.set_aspect('auto'); ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(_pt(4))
        fig.text(_fx(217), _fy(229), 'b', fontsize=24, ha='left', va='baseline', zorder=31)
        # locator box as drawn in (a): position and size of (a)'s red box, not the extent of the frame
        fig.add_artist(Rectangle((_fx(482.5), _fy(570.5)), 32 / KSZ_W, 21 / KSZ_H, transform=fig.transFigure,
                                 fc='none', ec='#e91f20', lw=_pt(2), zorder=31))
        # scale bar (true scale) in a white box
        kmpx = (fx1 - fx0) / ((X1 - X0) / 1000.0)
        _box(fig, 136, 1321.5, 669.5, 1460)
        bx = lambda km: 159 + km * kmpx
        fig.add_artist(Rectangle((_fx(bx(0)), _fy(1408)), 60 * kmpx / KSZ_W, 41 / KSZ_H, transform=fig.transFigure,
                                 fc='black', ec='none', zorder=31))
        fig.add_artist(Rectangle((_fx(bx(15) + 1), _fy(1400)), (15 * kmpx - 2) / KSZ_W, 25 / KSZ_H,
                                 transform=fig.transFigure, fc='white', ec='none', zorder=32))
        for km in (0, 15, 30, 60):
            fig.text(_fx(bx(km) + 3), _fy(1363), f'{km}', fontsize=KSZ_SMALL, ha='center', va='baseline', zorder=32)
        fig.text(_fx(bx(30) + 3), _fy(1442), 'Kilometers', fontsize=KSZ_SMALL, ha='center', va='baseline', zorder=32)
        # legends: occurrence, then faults and classes (rows of 48.5 px, aligned with the bottom of (a)'s legend)
        _box(fig, 1658, 1391, 1985.5, 1494)
        fig.add_artist(Line2D([_fx(1706)], [_fy(1442)], transform=fig.transFigure, marker='o', ls='',
                              ms=_pt(33), mfc=DEP_COLORS['Occurrence'], mec='black', mew=_pt(6), zorder=31))
        fig.text(_fx(1783), _fy(1455), 'Occurrence', fontsize=KSZ_SMALL, ha='left', va='baseline', zorder=31)
        rows = [1260 + 48.5 * k for k in range(5)]
        _box(fig, 2023, rows[0] - 36, 2493.5, 1494)
        fig.add_artist(Line2D([_fx(2055), _fx(2115)], [_fy(rows[0])] * 2, transform=fig.transFigure,
                              color=FAULT_COLOR, lw=_pt(8), solid_capstyle='butt', zorder=31))
        fig.text(_fx(2160), _fy(rows[0] + 12.5), 'Faults', fontsize=KSZ_SMALL, ha='left', va='baseline', zorder=31)
        for y, c, l in zip(rows[1:], CLASS_COLORS, CLASS_LABELS):
            fig.add_artist(Rectangle((_fx(2055), _fy(y + 14)), 60 / KSZ_W, 28 / KSZ_H, transform=fig.transFigure,
                                     fc=c, ec='none', zorder=31))
            fig.text(_fx(2162), _fy(y + 12.5), l, fontsize=KSZ_SMALL, ha='left', va='baseline', zorder=31)
        _ksz_north_arrow(fig, ax, (2342, 155))
        fig.savefig(os.path.join(FIG, outname), dpi=KSZ_DPI, facecolor='white')
        plt.close(fig)
    print('saved', outname)


if __name__ == '__main__':
    import sys
    if 'stability' in sys.argv:
        stability_maps()
    else:
        superior_maps()
        ksz_map()
