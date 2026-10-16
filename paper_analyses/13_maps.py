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


def raster(grid, col, outline_p, res=2000.0):
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
    return gx, gy, classify(z), (xmin, xmax, ymin, ymax)


def graticule(ax, extent, lons, lats, label_edges=True, fontsize=11):
    x0, x1, y0, y1 = extent
    for lo in lons:
        la = np.linspace(30, 80, 400)
        x, y = TO_P.transform(np.full_like(la, lo), la)
        ax.plot(x, y, color=GRAT_COLOR, lw=0.8, zorder=1)
    for la in lats:
        lo = np.linspace(-150, -40, 800)
        x, y = TO_P.transform(lo, np.full_like(lo, la))
        ax.plot(x, y, color=GRAT_COLOR, lw=0.8, zorder=1)
    if not label_edges:
        return
    for lo in lons:                           # meridians on bottom / top edges
        la = np.linspace(30, 80, 4000)
        x, y = TO_P.transform(np.full_like(la, lo), la)
        for edge, va, dy in [(y0, 'top', -0.012), (y1, 'bottom', 0.012)]:
            i = np.argmin(np.abs(y - edge))
            if abs(y[i] - edge) < 2e4 and x0 < x[i] < x1:
                ax.text(x[i], edge + dy * (y1 - y0), f'{lo:.0f}°', ha='center', va=va, fontsize=fontsize,
                        clip_on=False)
    for la in lats:                           # parallels on left / right edges
        lo = np.linspace(-150, -40, 8000)
        x, y = TO_P.transform(lo, np.full_like(lo, la))
        for edge, ha, dx in [(x0, 'right', -0.01), (x1, 'left', 0.01)]:
            i = np.argmin(np.abs(x - edge))
            if abs(x[i] - edge) < 2e4 and y0 < y[i] < y1:
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


def map_panel(ax, grid, col, title, letter, outline_p, dep, canada, region='Superior', show_legend=True):
    gx, gy, cls, ext = raster(grid, col, outline_p)
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
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=26, va='top', ha='left')
    if show_legend:
        handles = [Line2D([], [], marker='o', ls='', mfc=c, mec='black', ms=9, label=f'{t} ({(dep.dep_type == t).sum()})')
                   for t, c in DEP_COLORS.items()]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.08, 0.97), frameon=False, fontsize=11,
                  handletextpad=1.2)
    patches = [Patch(fc=c, ec='none', label=l) for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
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
        sp.set_color('#8c8c8c'); sp.set_linewidth(0.8)
    return cls


def superior_maps(panels=(('devnet', 'Superior DevNet MPM', 'a'), ('rf', 'Superior RF MPM', 'b')),
                  outname='fig_map_superior_rev.png'):
    grid = pd.read_csv(os.path.join(OUT_DIR, 'superior_grid_scores.csv'))
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


def ksz_frame():
    """Projected frame of the thesis KSZ geology panel, solved from where
    its graticule meets the frame (pixel positions read off the figure,
    2000 x 1216 display; frame x 65-1930, y 48-1168)."""
    px0, px1, py0, py1 = 65, 1930, 48, 1168
    meridians = [(-84, 255, py0), (-81, 1580, py0), (-84, 455, py1), (-81, 1835, py1)]
    parallels = [(49, px0, 447), (49, px1, 70), (48, px1, 752), (48, px0, 1127)]

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


def ksz_map(col='devnet', outname='fig_ksz_rev.png'):
    grid = pd.read_csv(os.path.join(OUT_DIR, 'superior_grid_scores.csv'))
    outline_p = gpd.read_file(os.path.join(REPO, 'data', 'outlines', 'Superior.geojson')).to_crs(CRS)
    X0, X1, Y0, Y1 = ksz_frame()
    gx, gy, cls, _ = raster(grid, col, outline_p, res=1000.0)   # classes over the whole province
    dep = occurrences('Superior')
    faults = gpd.read_file(os.path.join(os.path.dirname(REPO), 'dataset', 'vectors', 'Faults_Canada.geojson'))
    faults = faults.to_crs(CRS).clip(box(X0, Y0, X1, Y1))
    fig = plt.figure(figsize=(2631 / 250, 1600 / 250))
    ax = fig.add_axes([65 / 2000, 1 - 1168 / 1216, (1930 - 65) / 2000, (1168 - 48) / 1216])
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    ax.pcolormesh(gx, gy, cls, cmap=ListedColormap(CLASS_COLORS), vmin=-0.5, vmax=3.5, shading='nearest',
                  zorder=1, rasterized=True)
    faults.plot(ax=ax, color='#4d4d4d', lw=1.6, zorder=3)
    graticule(ax, (X0, X1, Y0, Y1), lons=[-84, -83, -82, -81], lats=[48, 49], fontsize=12)
    s = dep[(dep.x > X0) & (dep.x < X1) & (dep.y > Y0) & (dep.y < Y1)]
    ax.scatter(s.x, s.y, s=260, c='#377cb8', edgecolors='black', linewidths=2.2, zorder=6)
    for txt, xp, yp in [('Wawa  Subprovince', 655, 545), ('Abitibi  Subprovince', 1160, 1015)]:
        x = X0 + (xp - 65) / (1930 - 65) * (X1 - X0); y = Y1 - (yp - 48) / (1168 - 48) * (Y1 - Y0)
        ax.text(x, y, txt, ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85), zorder=8)
    x = X0 + (920 - 65) / (1930 - 65) * (X1 - X0); y = Y1 - (755 - 48) / (1168 - 48) * (Y1 - Y0)
    ax.text(x, y, 'KSZ', rotation=52, ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85), zorder=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(3.0)
    inset = ax.inset_axes([0.027, 0.63, 0.255, 0.335])
    outline_p.boundary.plot(ax=inset, color='#333333', lw=1.2)
    inset.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fc='none', ec='#e4191c', lw=1.2))
    inset.set_xticks([]); inset.set_yticks([]); inset.set_aspect('equal')
    inset.text(0.12, 0.85, 'b', transform=inset.transAxes, fontsize=26)
    patches = [Patch(fc=c, ec='none', label=l) for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
    leg = ax.legend(handles=[Line2D([], [], color='#4d4d4d', lw=2.5, label='Faults')] + patches,
                    loc='lower right', frameon=True, fancybox=False, edgecolor='black', fontsize=12,
                    handlelength=2.4, handleheight=1.3, borderpad=0.8, framealpha=1.0)
    leg.get_frame().set_linewidth(1.5)
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='#377cb8', mec='black', ms=13, label='Occurrence')],
              loc='lower right', bbox_to_anchor=(0.755, 0.0), frameon=True, fancybox=False, edgecolor='black',
              fontsize=12, borderpad=0.9, framealpha=1.0)
    scale_bar(ax, 0.02, 0.06, 60, [0, 15, 30, 60])
    north_arrow(ax, 0.93, 0.84, 0.11, (-81.2, 48.9))
    fig.savefig(os.path.join(FIG, outname), dpi=250, facecolor='white')
    plt.close(fig)
    print('saved', outname)


if __name__ == '__main__':
    superior_maps()
    ksz_map()
