"""
Stage 50: rebuild the derived gravity and magnetic grids from the public
inputs and compare them with the versions in the Getech delivery.

The delivery (GTC_CanadaGandMDataCompilation_2026.pdf) holds the Bouguer
anomaly and TMI regridded from NRCan data, NRCan's BA 1VD and THD, and seven
grids derived from them. This stage recomputes the seven with the methods the
delivery report names and measures how close they are to Getech's (Pearson
and Spearman correlation) over the whole grid, near and away from the data
edge, inside each study province and at the points of the national table.

Gravity (2 km cells, EPSG:3978), from Canada_BA:
  HP40km, HP100km   Butterworth high-pass, order 6 (the report's text; the
                    file list says 4), H = 1 / (1 + (k0/k)^6), k0 = 1/cutoff
  THD_Tilt          tilt of NRCan's THD grid: atan(1VD(THD) / THD(THD))
Magnetic (0.01 degree cells, WGS84), from TMI:
  DRTP              differential reduction to the pole: 8 degree tiles with
                    4 degree overlap, each projected to a 1 km azimuthal
                    equidistant grid; inside a tile the RTP is interpolated
                    between nine (I, D) pairs to each node's own field
                    direction (IGRF-13 at 2020.0, degrees 1-6, declination
                    taken from grid north); tiles blended with cosine tapers
  DRTP_1VD, DRTP_THD, DRTP_THD_Tilt
                    two chains: on the 1 km tile grids (FFT vertical
                    derivative, central differences), then resampled; or on
                    the 0.01 degree DRTP mosaic with metric cell sizes
                    ('_geoderiv'). Applied to Getech's own DRTP neither
                    chain reproduces Getech's derivative grids exactly (tilt
                    of THD r = 0.92-0.95 on three test tiles), so part of the
                    difference lies in the derivative step, not the DRTP
1VD and THD of the Bouguer anomaly are computed too, as checks of the
operators against NRCan's grids.

Our grids go to ../work/derived/ (GeoTIFF), the grid comparison to
out/derived/tab_similarity.csv. `features` then rebuilds the model inputs
that come from these grids (point values, GLCM, 10 km statistics) at the
national-table points as Stage 41 does and compares them with the table
(out/derived/tab_feature_similarity*.csv).

    python -m paper_analyses.50_derivative_reproduction            # everything (~25 min)
    python -m paper_analyses.50_derivative_reproduction gravity    # or magnetic, magnetic_geo, compare,
                                                                   # features, features_geo
"""
import importlib
import os
import sys
import time
from math import factorial

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import fft as sfft
from scipy import ndimage
from scipy.special import lpmv
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(REPO)
GT = os.path.join(ROOT, 'dataset', 'rasters', 'getech')
WORK = os.path.join(ROOT, 'work', 'derived')
OUT = os.path.join(REPO, 'paper_analyses', 'out', 'derived')
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

BUTTER_ORDER = 6
TILE_DEG, OVERLAP_DEG, BLEND_DEG = 8.0, 4.0, 2.0
TILE_CELL_M = 1000.0
TWO_PI = 2 * np.pi


def read(path):
    with rasterio.open(path) as r:
        a = r.read(1).astype('f8')
        a[a == r.nodata] = np.nan
        return a, r.profile


def write(name, a, profile):
    p = dict(profile, driver='GTiff', dtype='float32', nodata=-9999.0, compress='deflate', count=1)
    p.pop('blockxsize', None), p.pop('blockysize', None), p.pop('tiled', None)
    with rasterio.open(os.path.join(WORK, name + '.tif'), 'w', **p) as w:
        w.write(np.where(np.isfinite(a), a, -9999.0).astype('f4'), 1)
    print('wrote', name)


# ---- spectral tools ------------------------------------------------------------------------------

def fill(a, sigma=8, n=6):
    """Nearest-neighbour fill of NaNs, then smoothed so the filled part has no
    steps; the data themselves are left unchanged."""
    m = np.isnan(a)
    if not m.any():
        return a.copy()
    idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
    f = a[tuple(idx)]
    for _ in range(n):
        s = ndimage.gaussian_filter(f, sigma)
        f[m] = s[m]
    return f


def spectral(a, dx, dy, transfer, pad=0.25):
    """Apply a radial transfer function H(|k|) (k in cycles per metre) to a
    gap-free grid, with mirrored padding and a cosine taper."""
    ny, nx = a.shape
    py, px = max(int(ny * pad), 8), max(int(nx * pad), 8)
    Ny, Nx = sfft.next_fast_len(ny + 2 * py), sfft.next_fast_len(nx + 2 * px)
    mu = a.mean()
    b = np.pad(a - mu, ((py, Ny - ny - py), (px, Nx - nx - px)), mode='reflect')
    taper = [np.ones(N) for N in (Ny, Nx)]
    for t, p, N, n in ((taper[0], py, Ny, ny), (taper[1], px, Nx, nx)):
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(p) / p)
        t[:p] = ramp
        q = N - n - p
        t[N - q:] = 0.5 - 0.5 * np.cos(np.pi * np.arange(q)[::-1] / q)
    b *= np.outer(*taper)
    ky = sfft.fftfreq(Ny, dy)[:, None]
    kx = sfft.fftfreq(Nx, dx)[None, :]
    H = transfer(kx, ky)
    return sfft.ifft2(sfft.fft2(b, workers=-1) * H, workers=-1).real[py:py + ny, px:px + nx]


def vdr(kx, ky):
    return TWO_PI * np.hypot(kx, ky)


def highpass(cutoff_m, order=BUTTER_ORDER):
    k0 = 1.0 / cutoff_m
    return lambda kx, ky: 1.0 / (1.0 + (k0 / np.maximum(np.hypot(kx, ky), 1e-15)) ** order)


def thd(a, dx, dy):
    gy, gx = np.gradient(a, dy, dx)
    return np.hypot(gx, gy)


def tilt(a, dx, dy, degrees=False):
    """Tilt angle of a gap-free grid: atan(vertical / total horizontal derivative)."""
    t = np.arctan(spectral(a, dx, dy, vdr) / np.maximum(thd(a, dx, dy), 1e-30))
    return np.degrees(t) if degrees else t


# ---- gravity -------------------------------------------------------------------------------------

def gravity():
    ba, prof = read(os.path.join(GT, 'gravity', 'Canada_BA.flt'))
    nthd, _ = read(os.path.join(GT, 'gravity', 'Canada_BA_THD.flt'))
    dx = dy = prof['transform'].a
    valid = np.isfinite(ba)
    f = fill(ba)
    out = {'Canada_BA_HP40km': spectral(f, dx, dy, highpass(40e3)),
           'Canada_BA_HP100km': spectral(f, dx, dy, highpass(100e3)),
           'Canada_BA_THD_Tilt': tilt(fill(nthd), dx, dy),
           'Canada_BA_1VD': spectral(f, dx, dy, vdr),
           'Canada_BA_THD': thd(f, dx, dy)}
    for k, v in out.items():
        v[~valid] = np.nan
        write(k, v, prof)


# ---- geomagnetic field ---------------------------------------------------------------------------

# IGRF-13 main-field coefficients at epoch 2020.0, degrees 1-6 (nT; Alken et al., 2021)
IGRF2020 = {(1, 0): (-29404.8, 0), (1, 1): (-1450.9, 4652.5),
            (2, 0): (-2499.6, 0), (2, 1): (2982.0, -2991.6), (2, 2): (1677.0, -734.6),
            (3, 0): (1363.2, 0), (3, 1): (-2381.2, -82.1), (3, 2): (1236.2, 241.9), (3, 3): (525.7, -543.4),
            (4, 0): (903.0, 0), (4, 1): (809.5, 281.9), (4, 2): (86.3, -158.4), (4, 3): (-309.4, 199.7),
            (4, 4): (48.0, -349.7),
            (5, 0): (-234.3, 0), (5, 1): (363.2, 47.7), (5, 2): (187.8, 208.3), (5, 3): (-140.7, -121.2),
            (5, 4): (-151.2, 32.3), (5, 5): (13.5, 98.9),
            (6, 0): (66.0, 0), (6, 1): (65.5, -19.1), (6, 2): (72.9, 25.1), (6, 3): (-121.5, 52.8),
            (6, 4): (-36.2, -64.5), (6, 5): (13.5, 8.9), (6, 6): (-64.7, 68.1)}


def _schmidt(n, m, x):
    p = lpmv(m, n, x) * (-1) ** m  # without the Condon-Shortley phase
    return p if m == 0 else p * np.sqrt(2 * factorial(n - m) / factorial(n + m))


def inc_dec(lat, lon):
    """Inclination and declination (degrees) at the Earth's surface, spherical
    approximation."""
    th = np.radians(90 - np.asarray(lat, float))
    ph = np.radians(np.asarray(lon, float))
    e = 1e-6
    X = Y = Z = 0.0
    for (n, m), (g, h) in IGRF2020.items():
        P = _schmidt(n, m, np.cos(th))
        dP = (_schmidt(n, m, np.cos(th + e)) - _schmidt(n, m, np.cos(th - e))) / (2 * e)
        cs = g * np.cos(m * ph) + h * np.sin(m * ph)
        sn = g * np.sin(m * ph) - h * np.cos(m * ph)
        X = X + cs * dP
        Y = Y + m * sn * P / np.sin(th)
        Z = Z - (n + 1) * cs * P
    return np.degrees(np.arctan2(Z, np.hypot(X, Y))), np.degrees(np.arctan2(Y, X))


def rtp_transfer(inc, dec):
    """RTP for induced magnetisation along the field (x east, y north, z down)."""
    I, D = np.radians(inc), np.radians(dec)

    def H(kx, ky):
        k = np.maximum(np.hypot(kx, ky), 1e-15)
        theta = np.sin(I) + 1j * np.cos(I) * (kx * np.sin(D) + ky * np.cos(D)) / k
        out = 1.0 / theta ** 2
        out[(kx == 0) & (ky == 0)] = 1.0
        return out
    return H


# ---- magnetic ------------------------------------------------------------------------------------

def _sample(a, rows, cols):
    return ndimage.map_coordinates(a, [rows, cols], order=1, cval=np.nan, prefilter=False)


def _taper(x, a, b, width):
    """1 inside [a, b), cosine roll-off to 0 over `width` outside."""
    w = np.ones_like(x)
    lo, hi = x < a, x >= b
    w[lo] = 0.5 + 0.5 * np.cos(np.pi * np.clip((a - x[lo]) / width, 0, 1))
    w[hi] = 0.5 + 0.5 * np.cos(np.pi * np.clip((x[hi] - b) / width, 0, 1))
    return w


def geographic_derivatives(drtp, lons, lats, step, band_deg=2.0, blend_deg=0.5, pad_deg=1.0):
    """1VD, THD and tilt of THD on the 0.01 degree grid itself, with metric
    cell sizes: central differences use each row's own east-west spacing, the
    FFT vertical derivative the spacing at the centre of a 2 degree band."""
    ny, nx = drtp.shape
    names = ['DRTP_1VD', 'DRTP_THD', 'DRTP_THD_Tilt']
    acc = {k: np.zeros((ny, nx)) for k in names}
    wsum = np.zeros((ny, nx))
    dy = step * 111195.0
    for tx in np.arange(-141.0, lons[-1], TILE_DEG):
        for ty in np.arange(41.5, lats[0], band_deg):
            ci = np.where((lons >= tx - blend_deg - pad_deg) & (lons < tx + TILE_DEG + blend_deg + pad_deg))[0]
            ri = np.where((lats >= ty - blend_deg - pad_deg) & (lats < ty + band_deg + blend_deg + pad_deg))[0]
            a = drtp[np.ix_(ri, ci)]
            if np.isfinite(a).mean() < 0.001:
                continue
            g = fill(a)
            dxr = dy * np.cos(np.radians(lats[ri]))[:, None]
            dxc = dy * np.cos(np.radians(ty + band_deg / 2))

            def hgrad(f):
                return np.hypot(np.gradient(f, axis=1) / dxr, np.gradient(f, dy, axis=0))
            h = hgrad(g)
            prods = {'DRTP_1VD': spectral(g, dxc, dy, vdr), 'DRTP_THD': h,
                     'DRTP_THD_Tilt': np.degrees(np.arctan(spectral(h, dxc, dy, vdr) / np.maximum(hgrad(h), 1e-30)))}
            w = np.outer(_taper(lats[ri], ty, ty + band_deg, blend_deg), _taper(lons[ci], tx, tx + TILE_DEG, blend_deg))
            for k, v in prods.items():
                acc[k][np.ix_(ri, ci)] += w * v
            wsum[np.ix_(ri, ci)] += w
    return {k: v / np.where(wsum > 0, wsum, np.nan) for k, v in acc.items()}


def magnetic(only=None, write_out=True, deriv='tile', tag='', addback=False):
    """deriv='tile': derivatives on the 1 km projected tile grids;
    'geographic': on the 0.01 degree DRTP mosaic (geographic_derivatives).
    addback: the written DRTP is the TMI plus the RTP correction computed on
    the tiles (keeps the 0.01 degree texture that two resamplings smooth);
    the derivatives still come from the tile DRTP."""
    tmi, prof = read(os.path.join(GT, 'magnetic', 'TMI.flt'))
    T = prof['transform']
    lon0, lat0, step = T.c, T.f, T.a
    ny, nx = tmi.shape
    lons = lon0 + step * (np.arange(nx) + 0.5)
    lats = lat0 - step * (np.arange(ny) + 0.5)
    names = ['DRTP', 'DRTP_1VD', 'DRTP_THD', 'DRTP_THD_Tilt']
    acc = {k: np.zeros((ny, nx)) for k in names + ['DRTP_corr']}
    wsum = np.zeros((ny, nx))
    valid = np.isfinite(tmi)

    tiles = [(x, y) for x in np.arange(-141.0, lons[-1], TILE_DEG) for y in np.arange(41.5, lats[0], TILE_DEG)]
    if only is not None:
        tiles = [t for t in tiles if t in only]
    t_start = time.time()
    for i, (tx, ty) in enumerate(tiles):
        # geographic cells this tile writes to: its core plus the blending margin
        ci = np.where((lons >= tx - BLEND_DEG) & (lons < tx + TILE_DEG + BLEND_DEG))[0]
        ri = np.where((lats >= ty - BLEND_DEG) & (lats < ty + TILE_DEG + BLEND_DEG))[0]
        core_c = (lons[ci] >= tx) & (lons[ci] < tx + TILE_DEG)
        core_r = (lats[ri] >= ty) & (lats[ri] < ty + TILE_DEG)
        if not valid[np.ix_(ri[core_r], ci[core_c])].any():
            continue
        cx, cy = tx + TILE_DEG / 2, min(ty + TILE_DEG / 2, 84.0)
        aeqd = f'+proj=aeqd +lat_0={cy} +lon_0={cx} +datum=WGS84 +units=m'
        fwd = Transformer.from_crs('EPSG:4326', aeqd, always_xy=True)
        inv = Transformer.from_crs(aeqd, 'EPSG:4326', always_xy=True)
        # projected extent of the tile window (core + overlap)
        wl = np.linspace(tx - OVERLAP_DEG, tx + TILE_DEG + OVERLAP_DEG, 60)
        wt = np.clip(np.linspace(ty - OVERLAP_DEG, ty + TILE_DEG + OVERLAP_DEG, 60), -89, 89.9)
        WL, WT = np.meshgrid(wl, wt)
        X, Y = fwd.transform(WL, WT)
        xs = np.arange(X.min(), X.max() + TILE_CELL_M, TILE_CELL_M)
        ys = np.arange(Y.min(), Y.max() + TILE_CELL_M, TILE_CELL_M)  # rows run north
        GX, GY = np.meshgrid(xs, ys)
        glon, glat = inv.transform(GX, GY)
        rows = (lat0 - glat) / step - 0.5
        cols = (glon - lon0) / step - 0.5
        raw = _sample(tmi, rows.ravel(), cols.ravel()).reshape(GX.shape)
        if np.isfinite(raw).mean() < 0.01:
            continue
        g = fill(raw, sigma=10)

        # field direction at each node, declination measured from grid north
        I, D = inc_dec(glat, glon)
        nx_, ny_ = fwd.transform(glon, np.minimum(glat + 0.01, 89.99))
        D = D + np.degrees(np.arctan2(nx_ - GX, ny_ - GY))  # true north lies at this grid azimuth
        Is = np.linspace(np.nanmin(I), np.nanmax(I), 3)
        Ds = np.linspace(np.nanmin(D), np.nanmax(D), 3)
        fields = np.array([[spectral(g, TILE_CELL_M, TILE_CELL_M, rtp_transfer(a, b)) for b in Ds] for a in Is])
        # bilinear interpolation in (I, D) to each node's own direction
        fi = np.clip((I - Is[0]) / max(Is[1] - Is[0], 1e-9), 0, 2 - 1e-9)
        fd = np.clip((D - Ds[0]) / max(Ds[1] - Ds[0], 1e-9), 0, 2 - 1e-9)
        i0, j0 = fi.astype(int), fd.astype(int)
        u, v = fi - i0, fd - j0
        r, c = np.indices(g.shape)
        drtp = ((1 - u) * (1 - v) * fields[i0, j0, r, c] + u * (1 - v) * fields[i0 + 1, j0, r, c]
                + (1 - u) * v * fields[i0, j0 + 1, r, c] + u * v * fields[i0 + 1, j0 + 1, r, c])
        h = thd(drtp, TILE_CELL_M, TILE_CELL_M)
        prods = {'DRTP': drtp, 'DRTP_corr': drtp - g, 'DRTP_1VD': spectral(drtp, TILE_CELL_M, TILE_CELL_M, vdr), 'DRTP_THD': h,
                 'DRTP_THD_Tilt': tilt(h, TILE_CELL_M, TILE_CELL_M, degrees=True)}

        # back to the geographic cells, with cosine weights across the blending margin
        LON, LAT = np.meshgrid(lons[ci], lats[ri])
        px, py = fwd.transform(LON, LAT)
        prow = (py - ys[0]) / TILE_CELL_M
        pcol = (px - xs[0]) / TILE_CELL_M

        w = np.outer(_taper(lats[ri], ty, ty + TILE_DEG, BLEND_DEG), _taper(lons[ci], tx, tx + TILE_DEG, BLEND_DEG))
        for k, a in prods.items():
            acc[k][np.ix_(ri, ci)] += w * _sample(a, prow.ravel(), pcol.ravel()).reshape(LON.shape)
        wsum[np.ix_(ri, ci)] += w
        print(f'tile {i + 1}/{len(tiles)} lon {tx:.0f} lat {ty:.1f} grid {g.shape} '
              f'I {Is[0]:.1f}-{Is[-1]:.1f} D {Ds[0]:.1f}-{Ds[-1]:.1f}  {time.time() - t_start:.0f} s', flush=True)

    res = {k: acc[k] / np.where(wsum > 0, wsum, np.nan) for k in names + ['DRTP_corr']}
    if deriv == 'geographic':
        res.update(geographic_derivatives(np.where(valid, res['DRTP'], np.nan), lons, lats, step))
    if addback:
        res['DRTP'] = tmi + res['DRTP_corr']
    for k in names:
        res[k][~valid] = np.nan
        if write_out:
            write(k + tag, res[k], prof)
    return res


# ---- comparison ----------------------------------------------------------------------------------

def _corr(a, b, m):
    m = m & np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 50:
        return n, np.nan, np.nan
    x, y = a[m], b[m]
    sub = np.random.default_rng(0).choice(n, min(n, 300_000), replace=False)
    return n, pearsonr(x, y)[0], spearmanr(x[sub], y[sub])[0]


def compare(chains=(('tile', ''), ('geographic', '_geoderiv'))):
    import geopandas as gpd
    from rasterio.features import geometry_mask
    prov = gpd.read_file(os.path.join(REPO, 'data', 'national', 'provinces.geojson'))
    prov = prov[prov.province.isin(['Superior', 'Cordillera', 'Grenville'])]
    pts = pd.read_pickle(os.path.join(ROOT, 'work', 'national', 'national_table.pkl'))['pts']
    rows = []
    jobs = [('gravity', n, '', '') for n in ('Canada_BA_HP40km', 'Canada_BA_HP100km', 'Canada_BA_THD_Tilt',
                                             'Canada_BA_1VD', 'Canada_BA_THD')]
    jobs += [('magnetic', 'DRTP', '', '')]
    jobs += [('magnetic', n, chain, tag) for chain, tag in chains for n in ('DRTP_1VD', 'DRTP_THD', 'DRTP_THD_Tilt')]
    for kind, name, chain, tag in jobs:
        path = os.path.join(WORK, name + tag + '.tif')
        if not os.path.exists(path):
            continue
        ours, prof = read(path)
        ref, _ = read(os.path.join(GT, kind, name + '.flt'))
        ref = ref[:ours.shape[0], :ours.shape[1]]  # DRTP_1VD etc. carry one extra column
        T, crs = prof['transform'], prof['crs']
        valid = np.isfinite(ours) & np.isfinite(ref)
        cell_km = (T.a / 1e3,) * 2 if kind == 'gravity' else (111.2 * T.a, 111.2 * T.a * np.cos(np.radians(55)))
        edge_km = ndimage.distance_transform_edt(valid, sampling=cell_km)
        regions = {'all cells': valid, 'cells <= 20 km from data edge': valid & (edge_km <= 20),
                   'cells > 20 km from data edge': valid & (edge_km > 20)}
        pg = prov.to_crs(crs)
        for _, p in pg.iterrows():
            regions[p.province] = valid & ~geometry_mask([p.geometry], ours.shape, T)
        for reg, m in regions.items():
            n, r, s = _corr(ours, ref, m)
            rows.append(dict(grid=name, derivatives=chain, where=reg, n=n, pearson=r, spearman=s))
        # the model inputs: values at the national-table points (nearest cell)
        x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(pts.lon.values, pts.lat.values)
        c = np.floor((np.asarray(x) - T.c) / T.a).astype(int)
        r_ = np.floor((T.f - np.asarray(y)) / -T.e).astype(int)
        ok = (r_ >= 0) & (r_ < ours.shape[0]) & (c >= 0) & (c < ours.shape[1])
        a, b = np.full(len(pts), np.nan), np.full(len(pts), np.nan)
        a[ok], b[ok] = ours[r_[ok], c[ok]], ref[r_[ok], c[ok]]
        for reg, m in [('national-table points', np.ones(len(pts), bool))] + \
                [(f'points, {p}', (pts.province == p).values) for p in ('Superior', 'Cordillera', 'Grenville')]:
            n, r, s = _corr(a, b, m)
            rows.append(dict(grid=name, derivatives=chain, where=reg, n=n, pearson=r, spearman=s))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'tab_similarity.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(df.round(4).to_string(index=False))



SEVEN = ['Canada_BA_HP40km', 'Canada_BA_HP100km', 'Canada_BA_THD_Tilt', 'DRTP', 'DRTP_1VD', 'DRTP_THD',
         'DRTP_THD_Tilt']


def features(tag='', drtp_tag=''):
    """The model inputs that come from the seven grids (point values, GLCM of
    BA_HP40km and DRTP, their 10 km mean and std), recomputed at the
    national-table points from our grids in the way Stage 41 builds them (per
    province), against the values in the table. `tag` picks the magnetic
    derivative chain."""
    from paper_analyses import national as N
    from src.features.glcm import GLCM_PROPS, add_glcm_features
    from src.features.proximity import add_rasters
    from src.features.raster_preprocessing import preprocess_rasters
    s24 = importlib.import_module('paper_analyses.24_extended_features')
    pts = pd.read_pickle(os.path.join(ROOT, 'work', 'national', 'national_table.pkl'))['pts']
    ours = {k: os.path.join(WORK, k + (tag if k.startswith('DRTP_') else drtp_tag if k == 'DRTP' else '') + '.tif')
            for k in SEVEN}
    glcm = ['Canada_BA_HP40km', 'DRTP']
    cols = SEVEN + [f'{k}_glcm_{p}' for k in glcm for p in GLCM_PROPS]
    new = pd.DataFrame(np.nan, index=pts.index, columns=cols)
    for name in pd.unique(pts.province):
        m = (pts.province == name).values
        outline, clip = N.province_extent(name)
        rasters = preprocess_rasters(f'{clip}_public{tag}{drtp_tag}', {'outline': outline, 'raster': ours})
        sub = add_glcm_features(add_rasters(pts.loc[m, ['lat', 'lon']].copy(), rasters), rasters, glcm)
        new.loc[m, cols] = sub[cols].values
        print(f'  {name}: {int(m.sum()):,} points', flush=True)
    for k, path in (('BA_HP40km', ours['Canada_BA_HP40km']), ('DRTP', ours['DRTP'])):
        mu, sd = s24.neighbourhood(path, (pts.lon.values, pts.lat.values), s24.RADIUS_KM)
        new[f'{k}_mean{s24.RADIUS_KM:.0f}km'], new[f'{k}_std{s24.RADIUS_KM:.0f}km'] = mu, sd
    rows = []
    for c in new.columns:
        for reg, m in [('all points', np.ones(len(pts), bool))] + \
                [(p, (pts.province == p).values) for p in ('Superior', 'Cordillera', 'Grenville')]:
            n, r, s = _corr(new[c].values, pts[c].values, m)
            rows.append(dict(feature=c, where=reg, n=n, pearson=r, spearman=s))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f'tab_feature_similarity{tag}{drtp_tag}.csv'), index=False)
    new.to_pickle(os.path.join(WORK, f'features_public{tag}{drtp_tag}.pkl'))
    with pd.option_context('display.width', 200, 'display.max_rows', 300):
        print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if part in ('gravity', 'all'):
        gravity()
    if part in ('magnetic', 'all'):
        magnetic()
    if part in ('magnetic_geo', 'all'):
        magnetic(deriv='geographic', tag='_geoderiv')
    if part in ('magnetic_addback', 'all'):
        magnetic(tag='_addback', addback=True)
    if part in ('compare', 'all'):
        compare()
    if part in ('features', 'all'):
        features()
    if part in ('features_geo', 'all'):
        features('_geoderiv')
    if part in ('features_addback', 'all'):
        features('_geoderiv', '_addback')
