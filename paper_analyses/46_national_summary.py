"""
Stage 46 -- tables and figures of the national part (Stages 40-45).

Reads paper_analyses/out/national/*.csv and the Stage-41 table; writes
    tab_benchmark.csv / .md      N1, Canada and the study provinces in the national out-of-fold scores
    tab_transfer.csv / .md       N2, leave-one-province-out and the transfer matrices
    tab_pooling.csv / .md        N3, regional vs pooled training per province
    tab_shift.csv / .md          N4
    tab_curve.csv / .md          N5
    tab_ladder.csv / .md         Parsa et al. ladder
    fig_national_map.png         occurrences, provinces and background sample
    fig_ladder.png, fig_transfer.png, fig_pooling.png, fig_curve.png

    python -m paper_analyses.46_national_summary
"""
import glob
import importlib
import os

import matplotlib

matplotlib.use('Agg')
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from paper_analyses.common import load_pickle  # noqa: E402
from paper_analyses.national import OUTLINE, PROVINCES, STUDY, nat_path  # noqa: E402

_b = importlib.import_module('paper_analyses.42_national_benchmark')
OUT = _b.OUT
COLORS = {'DevNet': '#d03b3b', 'DevNet-BCE': '#e08a1e', 'RF': '#2a78d6', 'XGBoost': '#1a9bb0', 'BaggingPU': '#3a9e5c',
          'nnPU': '#8a5cc2', 'RF-Parsa': '#52514e', 'DevNet-scaled': '#e87ba4'}
MARKERS = {'DevNet': 'o', 'DevNet-BCE': 'v', 'RF': 's', 'XGBoost': 'D', 'BaggingPU': '^', 'nnPU': 'P',
           'RF-Parsa': 'X', 'DevNet-scaled': 'h'}
LABEL = {'DevNet': 'DevNet', 'DevNet-BCE': 'DevNet-BCE', 'RF': 'RF', 'XGBoost': 'XGBoost', 'BaggingPU': 'Bagging-PU',
         'nnPU': 'nnPU', 'RF-Parsa': 'RF (Parsa et al.)', 'DevNet-scaled': 'DevNet, steps scaled'}
MODELS = ['DevNet', 'DevNet-BCE', 'RF', 'XGBoost', 'BaggingPU', 'nnPU']
PROTOS = ['random', '50 km', '100 km', '200 km']
GROUP_COLORS = {'Superior': '#2a78d6', 'Cordillera': '#eb6834', 'Grenville': '#1baf7a', 'Other': '#6250d6'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9.5, 'axes.titlelocation': 'left'})


def read(pattern):
    files = sorted(glob.glob(os.path.join(OUT, pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def save(df, name, floatfmt='.3f'):
    df.to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
    fmt = lambda v: format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = ['| ' + ' | '.join(map(str, df.columns)) + ' |', '|' + '---|' * len(df.columns)]
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    with open(os.path.join(OUT, f'{name}.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def regional_reference():
    """Regional feature-set-3 results (Stage 25 reencoded): AUC / AP per province."""
    rows = []
    for r in STUDY:
        p = os.path.join(os.path.dirname(OUT), f'reencoded_benchmark_{r}.csv')
        if os.path.exists(p):
            d = pd.read_csv(p)
            d = d[d['keep'].astype(str) == '0'].replace({'MLP-BCE': 'DevNet-BCE'})
            rows.append(d.assign(region=r))
    p = os.path.join(os.path.dirname(OUT), 'xgboost_regional.csv')     # Stage 47
    if os.path.exists(p):
        d = pd.read_csv(p)
        rows.append(d[d.featureset == 'reencoded'])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# -- tables ------------------------------------------------------------------------
def benchmark_table():
    b = read('benchmark_*.csv')
    if b.empty:
        return b
    b = b[b.featureset == 'set3']
    keep = ['scope', 'model', 'protocol', 'auc_med', 'ap_med', 'lift_med', 'cap10_med', 'prevalence', 'n_pos', 'n_bg']
    t = b[keep].copy()
    ref = regional_reference()
    if not ref.empty:
        ref = ref.rename(columns={'region': 'scope'})[['scope', 'model', 'protocol', 'auc_med', 'ap_med']]
        t = t.merge(ref.rename(columns={'auc_med': 'regional_auc', 'ap_med': 'regional_ap'}),
                    on=['scope', 'model', 'protocol'], how='left')
    t['protocol'] = pd.Categorical(t.protocol, PROTOS, ordered=True)
    t = t.sort_values(['scope', 'model', 'protocol'])
    save(t, 'tab_benchmark')
    return t


def transfer_table():
    t = read('transfer_*.csv')
    if t.empty:
        return t
    t = t[t.featureset == 'set3'][['design', 'train', 'test', 'model', 'auc_med', 'auc_lo', 'auc_hi', 'ap_med',
                                   'prevalence', 'n_train_pos']]
    save(t, 'tab_transfer')
    return t


def pooling_table():
    p = read('pooling_*.csv')
    if p.empty:
        return p
    p = p[p.featureset == 'set3']
    base = p[p.variant == 'regional'].set_index(['region', 'model', 'protocol'])
    rows = []
    for _, r in p.iterrows():
        b = base.loc[(r.region, r.model, r.protocol)]
        rows.append({'region': r.region, 'model': r.model, 'protocol': r.protocol, 'variant': r.variant,
                     'auc': r.auc_med, 'ap': r.ap_med, 'cap10': r.cap10_med,
                     'd_auc': r.auc_med - b.auc_med, 'd_ap': r.ap_med - b.ap_med})
    t = pd.DataFrame(rows)
    save(t, 'tab_pooling')
    return t


def curve_table():
    c = read('curve_*.csv')
    if c.empty:
        return c
    c = c[c.featureset == 'set3'][['model', 'k_train_pos', 'auc_med', 'auc_lo', 'auc_hi', 'ap_med', 'lift_med']]
    save(c, 'tab_curve')
    return c


def ladder_table():
    t = read('parsa_ladder_*.csv')
    if t.empty:
        return t
    t = t.drop_duplicates(['rung', 'design', 'model'], keep='last')
    cols = [c for c in ['rung', 'design', 'model', 'auc_balanced_mean', 'auc_balanced_lo', 'auc_balanced_hi',
                        'ap_balanced_mean',
                        'auc_mean', 'auc_lo', 'auc_hi', 'ap_mean', 'lift_mean', 'cap10_mean',
                        'sibling_in_train_mean'] if c in t.columns]
    t = t[cols].sort_values(['rung', 'design', 'model'])
    save(t, 'tab_ladder')
    return t


# -- figures -------------------------------------------------------------------------
def fig_map():
    d = load_pickle(nat_path('national_table.pkl'))
    pts = d['pts']
    nat = pts[pts.in_national]
    prov = gpd.read_file(PROVINCES).to_crs('EPSG:3978')
    canada = gpd.read_file(OUTLINE).to_crs('EPSG:3978')
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    canada.plot(ax=ax, color='#f0efec', edgecolor='none')
    prov.boundary.plot(ax=ax, color='#b5b4ae', lw=0.4)
    for g in STUDY:
        prov[prov.province == g].plot(ax=ax, color=GROUP_COLORS[g], alpha=0.12, edgecolor=GROUP_COLORS[g], lw=0.8)
    bg = gpd.GeoSeries.from_xy(nat[nat.label == 0].lon, nat[nat.label == 0].lat, crs='EPSG:4326').to_crs('EPSG:3978')
    ax.scatter(bg.x, bg.y, s=0.15, color='#898781', lw=0, rasterized=True, label='background sample')
    pos = nat[nat.label == 1]
    grp = np.where(pos.province.isin(STUDY), pos.province, 'Other')
    xy = gpd.GeoSeries.from_xy(pos.lon, pos.lat, crs='EPSG:4326').to_crs('EPSG:3978')
    for g in GROUP_COLORS:
        m = grp == g
        ax.scatter(xy.x[m], xy.y[m], s=26, marker='*', color=GROUP_COLORS[g], edgecolor='white', lw=0.4, zorder=5,
                   label=f'{g if g != "Other" else "rest of Canada"} ({int(m.sum())})')
    ax.set_axis_off()
    ax.legend(frameon=False, loc='lower left', fontsize=7.5, markerscale=1.2)
    fig.savefig(os.path.join(OUT, 'fig_national_map.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def fig_ladder(t, grid=False, path=None):
    """grid: 2 x 2 panels (paper layout) instead of one row."""
    if t.empty:
        return
    if grid:
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), gridspec_kw={'width_ratios': [1.35, 1, 1, 1]})
    ax = axes[0]
    rf = t[t.model == 'RF-Parsa']
    order = [r for r in ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6'] if r in set(rf.rung)]
    for metric, ls, mk in [('auc', '-', 'o'), ('ap', '--', 's')]:
        vals = []
        for r in order:
            row = rf[rf.rung == r].iloc[0]
            c = f'{metric}_balanced' if r in ('R0', 'R1', 'R2', 'R3') else metric
            vals.append(row.get(f'{c}_mean', np.nan))
        r7 = rf[rf.rung == 'R7'][f'{metric}_mean']
        xs = list(range(len(order))) + ([len(order)] if len(r7) else [])
        vs = vals + ([float(r7.mean())] if len(r7) else [])
        cut = order.index('R4') if 'R4' in order else len(xs)      # no line across the change of test set
        for seg in (slice(0, cut), slice(cut, None)):
            ax.plot(xs[seg], vs[seg], ls, marker=mk, color=COLORS['RF-Parsa'], ms=5, lw=1.5,
                    label=metric.upper() if seg.start == 0 else None)
        for i, (xi, v) in enumerate(zip(xs, vs)):
            if np.isfinite(v) and (metric == 'auc' or i >= cut):
                ax.annotate(f'{v:.2f}', (xi, v), textcoords='offset points', xytext=(0, 6 if metric == 'auc' else -12),
                            ha='center', fontsize=7, color=INK)
    cos = {'R2c': 'R2', 'R3c': 'R3', 'R4c': 'R4'}          # cosine-dissimilar negatives without the sibling leak
    for metric, mk in [('auc', 'o'), ('ap', 's')]:
        pts = [(order.index(x), rf[rf.rung == r].iloc[0].get(f'{metric}_balanced_mean' if r != 'R4c' else f'{metric}_mean'))
               for r, x in cos.items() if (rf.rung == r).any() and x in order]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, mfc='white', mec=COLORS['RF-Parsa'], ms=6, mew=1.3,
                    ls='none', label=f'{metric.upper()}, cosine negatives, no leak')
    labels = order + (['R7'] if (rf.rung == 'R7').any() else [])
    ax.set_xticks(range(len(labels)), labels)
    split = labels.index('R4') - 0.5 if 'R4' in labels else None
    if split is not None:
        ax.axvline(split, color=GRID, lw=1)
        ax.text(split - 0.1, 0.02, 'balanced test\n(50% positives)', ha='right', va='bottom', fontsize=7, color=MUTED)
        ax.text(split + 0.1, 0.02, 'realistic test\n(0.3% positives)', ha='left', va='bottom', fontsize=7, color=MUTED)
    ax.set_ylim(0, 1.1)
    ax.set_title('(a) RF of Parsa et al., one fix per rung')
    ax.legend(frameon=False, fontsize=6.5, loc='center left', bbox_to_anchor=(0.0, 0.48))
    models = [m for m in ['RF-Parsa', 'RF', 'XGBoost', 'BaggingPU', 'DevNet'] if m in set(t.model)]
    ax = axes[1]
    rungs = ['R4', 'R5', 'R6']
    for k, m in enumerate(models):
        z = t[t.model == m].drop_duplicates('rung').set_index('rung').reindex(rungs)
        ax.plot(np.arange(3) + (k - (len(models) - 1) / 2) * 0.1, z.ap_mean, MARKERS[m], ms=5.5, color=COLORS[m],
                label=LABEL[m], ls='none')
    prev = 66 / (66 + 20855)
    ax.axhline(prev, color=MUTED, lw=0.8, ls='--', zorder=0)
    ax.text(2.45, prev, 'no skill', ha='right', va='bottom', fontsize=7, color=MUTED)
    ax.set_xticks(np.arange(3), ['R4\nrealistic\ntest', 'R5\nshared\nbackground', 'R6\n100 km\nblocks'])
    ax.set_ylim(0, None)
    ax.set_ylabel('average precision')
    ax.set_title('(b) AP at realistic prevalence')
    ax.legend(frameon=False, fontsize=6.5)
    r7 = t[t.rung == 'R7'].copy()
    held = STUDY + ['Other']
    if not r7.empty:
        r7['held'] = r7.design.str.replace('leave one province out: ', '', regex=False)
        for ax, col, title in [(axes[2], 'auc_mean', '(c) R7 leave one province out: AUC'),
                               (axes[3], 'lift_mean', '(d) R7: AP / prevalence')]:
            for k, m in enumerate(models):
                z = r7[r7.model == m].set_index('held').reindex(held)
                ax.plot(np.arange(4) + (k - (len(models) - 1) / 2) * 0.1, z[col], MARKERS[m], ms=5.5,
                        color=COLORS[m], ls='none')
            ax.axhline(0.5 if col == 'auc_mean' else 1.0, color=MUTED, lw=0.8, ls='--', zorder=0)
            ax.set_xticks(np.arange(4), ['Superior', 'Cordillera', 'Grenville', 'rest'])
            ax.set_title(title)
            if col == 'lift_mean':
                ax.set_yscale('log')
            else:
                ax.set_ylim(0.2, 1.0)
    fig.tight_layout()
    fig.savefig(path or os.path.join(OUT, 'fig_ladder.png'), dpi=220 if path is None else 300, bbox_inches='tight')
    plt.close(fig)


def fig_transfer(t, lopo=True, path=None):
    """lopo=False: only the transfer matrices (paper layout; leave-one-province-out is tabulated)."""
    if t.empty:
        return
    div = LinearSegmentedColormap.from_list('div', ['#d03b3b', '#f0efec', '#2a78d6'])
    seq = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#86b6ef', '#184f95'])
    models = [m for m in ['DevNet', 'RF', 'XGBoost', 'BaggingPU'] if m in set(t.model)]
    ncol = len(models) + (1 if lopo else 0)
    fig, axes = plt.subplots(2, ncol, figsize=(2.75 * ncol + (0.45 if lopo else 0.6), 5.8))
    mat = t[t.design != 'leave-one-province-out']
    for row, metric in enumerate(['auc_med', 'ap_med']):
        for ax, m in zip(axes[row], models):
            z = mat[mat.model == m].pivot(index='train', columns='test', values=metric).reindex(STUDY)[STUDY]
            prev = mat[mat.model == m].groupby('test').prevalence.first().reindex(STUDY).values
            if metric == 'auc_med':
                ax.imshow(z.values, cmap=div, vmin=0.0, vmax=1.0)
            else:                                   # colour by AP / prevalence (log), print AP
                ax.imshow(np.log10(np.clip(z.values / prev[None, :], 0.5, 200)), cmap=seq, vmin=np.log10(0.5),
                          vmax=np.log10(200))
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{z.values[i, j]:.2f}' if metric == 'auc_med' else f'{z.values[i, j]:.3f}',
                            ha='center', va='center', fontsize=8, color=INK, fontweight='bold' if i == j else 'normal')
            ax.set_xticks(range(3), STUDY if row == 1 else [''] * 3, fontsize=7.5)
            ax.set_yticks(range(3), STUDY if ax is axes[row][0] else [''] * 3, fontsize=7.5)
            if row == 1:
                ax.set_xlabel('test province')
            if ax is axes[row][0]:
                ax.set_ylabel('training province')
            ax.set_title(f"{LABEL[m]}: {'AUC' if metric == 'auc_med' else 'AP'}", fontsize=9)
            for sp in ax.spines.values():
                sp.set_visible(False)
    lo = t[t.design == 'leave-one-province-out']
    held = STUDY + ['Other']
    lm = [m for m in MODELS if m in set(lo.model)]
    for row, (col, ref, title) in enumerate([] if not lopo else [('auc_med', 0.5, 'Leave one province out: AUC'),
                                             ('lift', 1.0, 'Leave one province out: AP / prevalence')]):
        ax = axes[row][-1]
        for k, m in enumerate(lm):
            z = lo[lo.model == m].set_index('test').reindex(held)
            v = z.auc_med if col == 'auc_med' else z.ap_med / z.prevalence
            ax.plot(np.arange(4) + (k - (len(lm) - 1) / 2) * 0.09, v, MARKERS[m], ms=5, color=COLORS[m],
                    label=LABEL[m], ls='none')
        ax.axhline(ref, color=MUTED, lw=0.8, ls='--', zorder=0)
        ax.set_xticks(np.arange(4), ['Superior', 'Cordillera', 'Grenville', 'rest'], fontsize=7.5)
        if col == 'lift':
            ax.set_yscale('log')
        else:
            ax.set_ylim(0.2, 1.0)
            ax.legend(frameon=False, fontsize=6.3, loc='lower left', ncol=2)
        ax.set_title(title, fontsize=9)
    if lopo:
        fig.text(0.01, 0.005, 'AP cells: colour = AP / prevalence of the test province (log scale); diagonal = '
                 'random 5x6 CV within the province.', fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path or os.path.join(OUT, 'fig_transfer.png'), dpi=220 if path is None else 300, bbox_inches='tight')
    plt.close(fig)


def fig_pooling(t):
    if t.empty:
        return
    variants = ['+background', '+positives', '+both']
    models = [m for m in ['DevNet', 'DevNet-scaled', 'DevNet-BCE', 'RF', 'XGBoost', 'BaggingPU'] if m in set(t.model)]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2), sharey='row')
    for i, proto in enumerate(['random', '100 km']):
        for j, g in enumerate(STUDY):
            ax = axes[i, j]
            for k, m in enumerate(models):
                z = t[(t.region == g) & (t.model == m) & (t.protocol == proto)].set_index('variant').reindex(variants)
                ax.plot(np.arange(3) + (k - (len(models) - 1) / 2) * 0.1, z.d_ap, MARKERS[m], ms=5, color=COLORS[m],
                        label=LABEL[m], ls='none')
            ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
            ax.set_xticks(np.arange(3), variants)
            ax.set_title(f'{g}, {proto}')
            if j == 0:
                ax.set_ylabel('AP change vs regional training')
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=len(models), frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, 'fig_pooling.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def fig_curve(c):
    if c.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for m in [m for m in ['DevNet', 'DevNet-BCE', 'RF', 'XGBoost', 'BaggingPU'] if m in set(c.model)]:
        z = c[c.model == m].sort_values('k_train_pos')
        axes[0].plot(z.k_train_pos, z.auc_med, '-', marker=MARKERS[m], ms=4, lw=1.5, color=COLORS[m], label=LABEL[m])
        axes[1].plot(z.k_train_pos, z.ap_med, '-', marker=MARKERS[m], ms=4, lw=1.5, color=COLORS[m], label=LABEL[m])
    axes[0].set_ylabel('AUC (national random CV)')
    axes[1].set_ylabel('average precision')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks([3, 5, 10, 20, 40, 52], ['3', '5', '10', '20', '40', 'all'])
        ax.set_xlabel('training positives per fold')
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_curve.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def audit_table():
    """Stage 48 (country-extent table): AP of tables A-D per province."""
    a = read(os.path.join('before_fix', 'feature_audit_*.csv'))
    if a.empty:
        return a
    t = a.pivot_table(index=['region', 'model', 'protocol'], columns='table', values='ap_med').reset_index()
    save(t, 'tab_feature_audit')
    return t


def devnet_tables():
    o = read('devnet_national_offsets.csv')
    if not o.empty:
        save(o, 'tab_devnet_offsets')
    v = read('devnet_national_variants_*.csv')
    if v.empty:
        return o, v
    b = read('benchmark_*.csv')
    b = b[(b.featureset == 'set3') & b.model.isin(set(v.model)) & b.protocol.isin(set(v.protocol))]
    base = b.assign(variant='as in N1')[['model', 'variant', 'protocol', 'scope', 'auc_med', 'ap_med']]
    t = pd.concat([base, v[['model', 'variant', 'protocol', 'scope', 'auc_med', 'ap_med']]], ignore_index=True)
    save(t, 'tab_devnet_variants')
    return o, t


def fig_devnet(t):
    if t is None or t.empty:
        return
    t = t[t.scope == 'Canada']
    variants = ['as in N1', 'local reference', 'province one-hot']
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=False)
    for ax, proto in zip(axes, ['random', '100 km']):
        for k, m in enumerate([m for m in ['DevNet', 'DevNet-BCE', 'RF'] if m in set(t.model)]):
            z = t[(t.model == m) & (t.protocol == proto)].set_index('variant').reindex(variants)
            ax.plot(np.arange(3) + (k - 1) * 0.12, z.ap_med, MARKERS[m], ms=6, color=COLORS[m], label=LABEL[m], ls='none')
        ax.set_xticks(np.arange(3), ['as in N1', 'background of the\nown province', 'province\none-hot'])
        ax.set_ylim(0, None)
        ax.set_title(f'National AP, {proto}')
    axes[0].set_ylabel('average precision')
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_devnet_national.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    audit_table()
    _, dv = devnet_tables()
    fig_devnet(dv)
    benchmark_table()
    tr = transfer_table()
    po = pooling_table()
    cu = curve_table()
    la = ladder_table()
    shift = read('shift.csv')
    if not shift.empty:
        save(shift, 'tab_shift')
    fig_map()
    fig_ladder(la)
    fig_transfer(tr)
    fig_pooling(po)
    fig_curve(cu)
    print('written to', OUT)


if __name__ == '__main__':
    main()
