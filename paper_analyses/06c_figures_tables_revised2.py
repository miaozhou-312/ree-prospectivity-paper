"""
Figures and LaTeX table bodies for the SECOND revision
(paper/manuscript/main_src_revised2.tex), whose main analysis uses feature
set 3 (re-encoded raw H3 geology + extended layers, Stages 24-26).

Set-1 analyses that the second revision reports unchanged (augmentation,
protocol audit, geography-only baseline, province diagnostics, nested design
selection, ablation, margin) are written by the Stage-06b functions into the
new table folder; everything about feature set 3 is generated here.

Outputs: paper/manuscript/generated_revised2/*.tex and
paper/manuscript/figures/*_r2.png. Earlier versions are untouched.

    python -m paper_analyses.06c_figures_tables_revised2
"""
import ast
import glob
import importlib
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from paper_analyses.common import OUT_DIR, load_pickle, load_region  # noqa: E402
from paper_analyses.figstyle import pretty  # noqa: E402

g = importlib.import_module('paper_analyses.06b_figures_tables_revised')
maps = importlib.import_module('paper_analyses.13_maps')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, 'paper', 'manuscript', 'figures')
GEN = os.path.join(REPO, 'paper', 'manuscript', 'generated_revised2')
os.makedirs(GEN, exist_ok=True)
REGIONS = ['Superior', 'Cordillera', 'Grenville']
MODELS = ['DevNet', 'MLP-BCE', 'RF', 'BaggingPU', 'nnPU']
ALL_MODELS = list(MODELS)  # CSV side outputs always cover all five
SUFFIX = '_r2'            # figure-name suffix; 06d (third revision) sets '_r3' and drops MLP-BCE from MODELS
WORKFLOW_MODELS = 'DevNet (deviation loss,\nquantile inputs, 16+240 rows)\ncontrols: MLP-BCE, RF, HGB\nPU: bagging-PU, nnPU'
WORKFLOW_VALIDATION = ('pooled out-of-fold ranks\nrandom CV and 50/100/200 km\nblocks, background purged\npaired bootstrap, Holm\n'
                       'label scarcity, contamination\nnested design selection')
WORKFLOW_INPUTS = ('Three feature sets', 'set 1: thesis (4 lithology groups)\nset 2: no point-level map classes\n  + regional distances, 10 km stats\nset 3: raw H3 lithology, era, contacts\n  + set-2 layers (main analysis)')
OTHERS = lambda: [m for m in MODELS if m != 'DevNet']
LABEL = {'DevNet': 'DevNet', 'MLP-BCE': 'MLP-BCE', 'RF': 'RF', 'BaggingPU': 'Bagging-PU', 'nnPU': 'nnPU',
         'RF-tuned': 'RF (tuned)', 'HGB-tuned': 'HGB (tuned)'}
COLORS = {**g.COLORS, 'MLP-BCE': '#e08a1e'}
PROTOS = [0, 50, 100, 200]
PNAME = {0: 'random', 50: '50 km', 100: '100 km', 200: '200 km'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def o(name):
    return os.path.join(OUT_DIR, name)


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f:
        f.write(text)
    print('wrote', name)


def ci(med, lo, hi, bold=False):
    return (f'\\textbf{{{med:.3f}}}' if bold else f'{med:.3f}') + f' [{lo:.3f}, {hi:.3f}]'


def fp(p):
    return '$<$0.001' if p < 0.0005 else f'{p:.3f}'


def bold_if(v, best):
    return f'\\textbf{{{v:.3f}}}' if np.isclose(v, best) else f'{v:.3f}'


def fs3():
    t = pd.concat([pd.read_csv(f) for f in glob.glob(o('reencoded_benchmark_*.csv'))], ignore_index=True)
    t['keep'] = t['keep'].astype(str)
    return t


def fs3_boot():
    b = pd.concat([pd.read_csv(f) for f in glob.glob(o('reencoded_bootstrap_*.csv'))], ignore_index=True)
    b['lo'] = b.d_ap_ci.map(lambda s: ast.literal_eval(s)[0])
    b['hi'] = b.d_ap_ci.map(lambda s: ast.literal_eval(s)[1])
    return b


def fs3_store(region):
    return load_pickle(o(f'reencoded_benchmark_{region}.pkl'))


# ---------------------------------------------------------------- benchmark
def benchmark():
    t = fs3()
    t0 = t[t['keep'] == '0']
    lines = []
    for r in REGIONS:
        sub = t0[t0.region == r]
        rnd = sub[sub.block_km == 0].set_index('model')
        best_ap, best_auc = rnd.loc[MODELS, 'ap_med'].max(), rnd.loc[MODELS, 'auc_med'].max()
        npos = int(load_region(r)['pruned']['y'].sum())
        for k, m in enumerate(MODELS):
            x = rnd.loc[m]
            lab = f'\\multirow{{{len(MODELS)}}}{{*}}{{{r} ({npos})}}' if k == 0 else ''
            auc = f"\\textbf{{{x.auc_med:.3f}}}" if np.isclose(x.auc_med, best_auc) else f'{x.auc_med:.3f}'
            lines.append(f"{lab} & {LABEL[m]} & {auc} & {ci(x.ap_med, x.ap_lo, x.ap_hi, np.isclose(x.ap_med, best_ap))} \\\\")
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_benchmark.tex', '\n'.join(lines))
    b = fs3_boot()
    lines = []
    for r in REGIONS:
        for m in OTHERS():
            x = b[(b.region == r) & (b.protocol == 'random') & (b.comparison == f'DevNet - {m}')].iloc[0]
            lines.append(f"{r} & DevNet $-$ {LABEL[m]} & {x.d_ap:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}] & {fp(x.p_ap)} & "
                         f"{x.wins} & {x.d_auc:+.3f} & {fp(x.p_auc)} \\\\")
        lines.append('\\addlinespace')
    write('tab_bootstrap.tex', '\n'.join(lines[:-1]))


def spatial():
    t = fs3()
    t0 = t[t['keep'] == '0']
    for metric, name in [('ap_med', 'tab_spatial.tex'), ('auc_med', 'tab_spatial_auc.tex')]:
        lines = []
        for r in REGIONS:
            for k, b in enumerate(PROTOS):
                s = t0[(t0.region == r) & (t0.block_km == b)].set_index('model')
                best = s.loc[MODELS, metric].max()
                lab = f'\\multirow{{4}}{{*}}{{{r}}}' if k == 0 else ''
                lines.append(f"{lab} & {PNAME[b]} & " + ' & '.join(bold_if(s.loc[m, metric], best) for m in MODELS) + ' \\\\')
            lines.append('\\midrule' if r != REGIONS[-1] else '')
        write(name, '\n'.join(lines))
    bt = fs3_boot()
    sup = bt[(bt.region == 'Superior') & (bt.block_km > 0) & ~bt.protocol.str.contains('k=3')].copy()
    fam = sup[sup.comparison != 'DevNet - MLP-BCE'].sort_values('p_ap').reset_index(drop=True)
    mlen = len(fam)   # Holm family: DevNet against the three baselines at three block sizes (MLP-BCE is a control)
    fam['holm'] = np.minimum(np.maximum.accumulate([p * (mlen - i) for i, p in enumerate(fam.p_ap)]), 1)
    sup = sup.merge(fam[['block_km', 'comparison', 'holm']], on=['block_km', 'comparison'], how='left')
    sup = sup.sort_values(['block_km', 'comparison'])
    lines = []
    for b in [50, 100, 200]:
        for m in OTHERS():
            x = sup[(sup.block_km == b) & (sup.comparison == f'DevNet - {m}')].iloc[0]
            holm = '--' if np.isnan(x.holm) else fp(x.holm)
            lines.append(f"{b} km & DevNet $-$ {LABEL[m]} & {x.d_ap:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}] & {fp(x.p_ap)} & "
                         f"{holm} & {x.wins} \\\\")
        lines.append('\\addlinespace')
    write('tab_spatial_bootstrap.tex', '\n'.join(lines[:-1]))
    sup.to_csv(o('reencoded_spatial_holm_Superior.csv'), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    xs = np.arange(4)
    for ax, r in zip(axes, REGIONS):
        for k, m in enumerate(MODELS):
            s = t0[(t0.region == r) & (t0.model == m)].set_index('block_km').reindex(PROTOS)
            ax.errorbar(xs + (k - (len(MODELS) - 1) / 2) * 0.1, s.ap_med, yerr=[s.ap_med - s.ap_lo, s.ap_hi - s.ap_med], fmt='o-',
                        color=COLORS[m], ms=4, lw=1.3, capsize=2, label=LABEL[m])
        ax.set_xticks(xs, [PNAME[p] for p in PROTOS])
        ax.set_title(r, loc='left')
        ax.set_xlabel('cross-validation split')
    axes[0].set_ylabel('average precision (median, 95% interval)')
    axes[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f'fig_spatial_cv{SUFFIX}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def capture():
    y = load_region('Superior')['pruned']['y']
    st = fs3_store('Superior')
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), sharey=True)
    xg = np.linspace(0, 10, 201)
    for ax, (title, proto) in zip(axes, [('(a) random stratified CV', 'random'), ('(b) 100 km spatial blocks', 100.0)]):
        for m in MODELS:
            curves = []
            for oof in st[('Superior', m, proto, None)]['oof']:
                order = np.argsort(-oof)
                cum = np.cumsum(y[order]) / y.sum()
                frac = np.arange(1, len(oof) + 1) / len(oof)
                curves.append(np.interp(np.linspace(0, 0.1, 201), frac, cum))
            curves = np.array(curves)
            ax.plot(xg, np.median(curves, 0), color=COLORS[m], lw=1.6, label=LABEL[m])
            ax.fill_between(xg, curves.min(0), curves.max(0), color=COLORS[m], alpha=0.1, lw=0)
        ax.plot([0, 10], [0, 0.1], 'k--', lw=0.8, label='random')
        ax.set_xlim(0, 10); ax.set_ylim(0, 1)
        ax.set_xlabel('top-ranked share of labelled pool (%)')
        ax.set_title(title, loc='left', fontsize=10)
    axes[0].set_ylabel('share of held-out occurrences captured')
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f'fig_capture_superior{SUFFIX}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------ feature sets
def featuresets():
    t = pd.read_csv(o('featureset_comparison.csv'))
    t = t[t['keep'] == 0]
    sets = ['1 thesis', '2 extended', '3 re-encoded']
    rows_models = MODELS + ['DevNet (no alkalic)', 'RF (no alkalic)']
    for r in REGIONS:
        lines = []
        for m in rows_models:
            cells = []
            for fs in sets:
                for b in PROTOS:
                    v = t[(t.region == r) & (t.model == m) & (t.featureset == fs) & (t.block_km == b)].ap_med
                    cells.append(f'{v.iloc[0]:.3f}' if len(v) else '--')
            name = LABEL.get(m, m).replace(' (no alkalic)', '$^\\dagger$')
            lines.append(f"{name} & " + ' & '.join(cells) + ' \\\\')
        write(f'tab_featuresets_{r}.tex', '\n'.join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, fs in zip(axes, sets):
        for m in [m for m in MODELS if m != 'nnPU']:
            s = t[(t.region == 'Superior') & (t.model == m) & (t.featureset == fs)].set_index('block_km').reindex(PROTOS)
            ax.plot(np.arange(4), s.ap_med, 'o-', color=COLORS[m], lw=1.5, ms=4, label=LABEL[m])
        if fs == '3 re-encoded':
            s = t[(t.region == 'Superior') & (t.model == 'DevNet (no alkalic)') & (t.featureset == fs)].set_index(
                'block_km').reindex(PROTOS)
            ax.plot(np.arange(4), s.ap_med, 'o--', color=COLORS['DevNet'], lw=1.2, ms=3, label='DevNet w/o alkalic class')
        ax.set_xticks(np.arange(4), [PNAME[p] for p in PROTOS])
        ax.set_title({'1 thesis': '(a) Set 1: grouped geology classes', '2 extended': '(b) Set 2: no point-level map classes',
                      '3 re-encoded': '(c) Set 3: raw H3 classes'}[fs], loc='left', fontsize=9)
    axes[0].set_ylabel('Superior average precision (median)')
    axes[2].legend(frameon=False, fontsize=7.5)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f'fig_featuresets{SUFFIX}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def scarcity():
    rows = []
    c = pd.read_csv(o('contamination_devnet_vs_mlp.csv'))
    for proto, pn in [('random', 'random'), ('100', '100 km')]:
        x = c[(c.features == 'all') & (c.protocol.astype(str) == proto) & (c['mode'] == 'drop') & (c.keep.astype(str) == '3')].iloc[0]
        rows.append(('Set 1 (grouped classes)', pn, x.devnet, x.mlp, x.d_ap, x.p, x.devnet_wins))
    for prefix, label in [('extended', 'Set 2 (no point-level map classes)'), ('reencoded', 'Set 3 (raw classes)')]:
        t = pd.concat([pd.read_csv(f) for f in glob.glob(o(f'{prefix}_benchmark_*.csv'))])
        t['keep'] = t['keep'].astype(str)
        bt = pd.concat([pd.read_csv(f) for f in glob.glob(o(f'{prefix}_bootstrap_*.csv'))])
        for proto, b in [('random', 0), ('100 km', 100)]:
            dv = t[(t.region == 'Superior') & (t.model == 'DevNet') & (t.block_km == b) & (t['keep'] == '3')].ap_med.iloc[0]
            ml = t[(t.region == 'Superior') & (t.model == 'MLP-BCE') & (t.block_km == b) & (t['keep'] == '3')].ap_med.iloc[0]
            x = bt[(bt.region == 'Superior') & (bt.protocol == f'{proto}, k=3')].iloc[0]
            rows.append((label, proto, dv, ml, x.d_ap, x.p_ap, x.wins))
    lines = [f"{a} & {b} & {c_:.3f} & {d:.3f} & {e:+.3f} & {fp(f)} & {w} \\\\" for a, b, c_, d, e, f, w in rows]
    write('tab_scarcity.tex', '\n'.join(lines))
    s = pd.read_csv(o('contamination_summary.csv'))
    lines = []
    for feats, label in [('all', 'Set 1'), ('no-terrane', 'Geophysics only')]:
        for m in ['DevNet', 'MLP-BCE', 'RF', 'BaggingPU', 'nnPU']:
            cells = []
            for k in ['8', '5', '3']:
                x = s[(s.features == feats) & (s.protocol.astype(str) == 'random') & (s.model == m) & (s.keep.astype(str) == k)].iloc[0]
                cells.append(f'{x.ap_drop:.3f} / {x.ap_hide:.3f}')
            lines.append(f"{label if m == 'DevNet' else ''} & {LABEL[m]} & " + ' & '.join(cells) + ' \\\\')
        lines.append('\\addlinespace')
    write('tab_contamination.tex', '\n'.join(lines[:-1]))


def controls():
    f = sorted(glob.glob(o('controls_fs3_*.csv')))
    if not f:
        print('skip controls: Stage 28 outputs missing')
        return
    c = pd.concat([pd.read_csv(x) for x in f])
    t = fs3()
    t0 = t[t['keep'] == '0']
    lines = []
    for r in REGIONS:
        for k, b in enumerate(PROTOS):
            v = {m: t0[(t0.region == r) & (t0.model == m) & (t0.block_km == b)].ap_med.iloc[0] for m in [m for m in ['DevNet', 'MLP-BCE', 'RF'] if m in MODELS]}
            for m in ['RF-tuned', 'HGB-tuned']:
                q = c[(c.region == r) & (c.model == m) & (c.block_km == b)]
                v[m] = q.ap_med.iloc[0] if len(q) else np.nan
            best = np.nanmax(list(v.values()))
            lab = f'\\multirow{{4}}{{*}}{{{r}}}' if k == 0 else ''
            lines.append(f"{lab} & {PNAME[b]} & " + ' & '.join('--' if np.isnan(v[m]) else bold_if(v[m], best)
                                                                for m in [m for m in ['DevNet', 'MLP-BCE', 'RF'] if m in MODELS] + ['RF-tuned', 'HGB-tuned']) + ' \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_controls.tex', '\n'.join(lines))


def occurrence():
    y = load_region('Superior')['pruned']['y']
    st = fs3_store('Superior')
    diag = pd.read_csv(o('region_diagnostics_positives.csv'))
    diag = diag[diag.region == 'Superior'].reset_index(drop=True)
    cid = diag.loc[diag.name == 'Montviel', 'cluster_50km'].iloc[0]
    group = np.where(diag.cluster_50km == cid, 'Abitibi cluster', 'dispersed')
    pct = importlib.import_module('paper_analyses.20_occurrence_breakdown').percentile_of_positives
    lines, out = [], []
    for proto, pn in [('random', 'random'), (100.0, '100 km')]:
        for k, gname in enumerate(['Abitibi cluster', 'dispersed']):
            cells = []
            for m in ALL_MODELS:
                p = pct(y, st[('Superior', m, proto, None)]['oof'])[group == gname]
                if m in MODELS:
                    cells.append(f'{np.median(p):.3f} ({int((p >= 0.99).sum())}/{len(p)})')
                out.append({'protocol': pn, 'group': gname, 'model': m, 'median_pct': float(np.median(p)),
                            'top1': int((p >= 0.99).sum()), 'n': len(p)})
            lab = f'\\multirow{{2}}{{*}}{{{pn}}}' if k == 0 else ''
            lines.append(f"{lab} & {gname} ({int((group == gname).sum())}) & " + ' & '.join(cells) + ' \\\\')
        lines.append('\\addlinespace')
    write('tab_occurrence.tex', '\n'.join(lines[:-1]))
    pd.DataFrame(out).to_csv(o('occurrence_breakdown_fs3.csv'), index=False)


# ------------------------------------------------------- interpretation, maps
def shap_outputs():
    arrow = lambda lab: {'higher -> more': '$\\uparrow$', 'higher -> less': '$\\downarrow$',
                         'present -> more': '$\\uparrow$', 'present -> less': '$\\downarrow$'}.get(lab, '$\\circ$')
    for r in REGIONS:
        t = pd.read_csv(o(f'shap_fs3_top10_{r}.csv'))
        write(f'tab_shap_{r}.tex', '\n'.join(
            f'{pretty(x.feature)} & {int(x.dn_rank)} & {arrow(x.dn_direction)} ({x.dn_rho:+.2f}) & {int(x.rf_rank)} & '
            f'{arrow(x.rf_direction)} ({x.rf_rho:+.2f}) \\\\' for _, x in t.iterrows()))
    sr = load_pickle(o('shap_fs3_results.pkl'))['Superior']
    names = [pretty(c) for c in sr['cols']]
    fig = plt.figure(figsize=(14, 6.5))
    for i, (key, title) in enumerate([('shap_dn', '(a) DevNet (ensemble-mean SHAP)'), ('shap_rf', '(b) Random Forest (TreeSHAP)')]):
        ax = fig.add_subplot(1, 2, i + 1)
        plt.sca(ax)
        np.random.seed(0)
        shap.summary_plot(sr[key], sr['X_raw'], feature_names=names, max_display=12, show=False, plot_size=None,
                          color_bar=(i == 1))
        ax.set_title(title, fontsize=12, loc='left')
        ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f'fig_shap_superior{SUFFIX}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def map_figures():
    maps.superior_maps(outname=f'fig_map_superior{SUFFIX}.png', grid_file='superior_grid_scores_fs3.csv')
    maps.ksz_map(outname=f'fig_ksz{SUFFIX}.png', grid_file='superior_grid_scores_fs3.csv')
    maps.stability_maps(outname=f'fig_map_stability{SUFFIX}.png', grid_file='superior_grid_uncertainty_fs3.csv')


def geochem():
    """Stage 31: three provinces, exact-location map percentiles, clr-PCA affinity."""
    v = pd.read_csv(o('geochem3_validation.csv'))
    rn = pd.read_csv(o('geochem3_rocknames.csv'))
    v, rn = v[v.exclusion_km == 10], rn[rn.exclusion_km == 10]
    order = [('3', 'DevNet'), ('3', 'MLP-BCE'), ('3', 'RF'), ('1', 'DevNet'), ('1', 'RF')]
    ci2 = lambda x, a, b, f='+.2f': f'{x:{f}} [{a:{f}}, {b:{f}}]'
    lines = []
    for r in REGIONS:
        for k, (fs, m) in enumerate(order):
            q = lambda ind: v[(v.region == r) & (v.featureset.astype(str) == fs) & (v.model == m) & (v.indicator == ind)].iloc[0]
            le, af = q('LREE'), q('affinity')
            y = rn[(rn.region == r) & (rn.featureset.astype(str) == fs) & (rn.model == m)].iloc[0]
            lab = f'\\multirow{{5}}{{*}}{{{r}}}' if k == 0 else ''
            carb = (f"{y.share_top10:.0%} ({y.base_share_top10:.0%}) & {ci2(y.enrichment, y.enrich_lo, y.enrich_hi, '.2f')}"
                    if y.n_locations >= 3 else '-- & --')
            lines.append(f"{lab} & Set {fs} {LABEL[m]} & {le.share_locations_top10:.0%} & "
                         f"{ci2(le.rho, le.rho_lo, le.rho_hi)} & {ci2(le.enrichment, le.enrich_lo, le.enrich_hi, '.2f')} & "
                         f"{ci2(af.rho, af.rho_lo, af.rho_hi)} & {carb} \\\\".replace('%', '\\%'))
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_geochem.tex', '\n'.join(lines))
    n = rn.groupby('region')[['n_locations', 'n_clusters_10km']].first()
    write('tab_geochem_n.tex', '; '.join(f"{r} {int(n.loc[r, 'n_locations'])} ({int(n.loc[r, 'n_clusters_10km'])} clusters)"
                                         for r in REGIONS))

    # Figure: (a) clr-PCA biplot, (b) Superior enrichment against the sampling baseline
    sc = pd.read_csv(o('geochem3_pca_scores.csv'))
    pca = pd.read_csv(o('geochem3_pca.csv'))
    summ = pd.read_csv(o('geochem3_pca_summary.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1.05, 1]})
    ax = axes[0]
    fam = np.select([sc.carb_named,
                     sc.rock_type.str.contains('syenite|monzonite|alkali', case=False, na=False),
                     sc.rock_type.str.contains('granit|tonalite|trondhjemite|granodiorite|rhyolite|dacite', case=False, na=False),
                     sc.rock_type.str.contains('gabbro|basalt|komatiite|picrite|diorite|peridotite|pyroxenite|andesite',
                                               case=False, na=False)],
                    ['carbonatite / foid-bearing', 'syenite, monzonite', 'granitoid, felsic volcanic',
                     'mafic, ultramafic, intermediate'], 'other')
    cols_f = {'mafic, ultramafic, intermediate': '#9e9e9e', 'granitoid, felsic volcanic': '#f4a6a6',
              'syenite, monzonite': '#9ecae1', 'other': '#d9d9d9', 'carbonatite / foid-bearing': '#d62728'}
    for f_ in ['other', 'mafic, ultramafic, intermediate', 'granitoid, felsic volcanic', 'syenite, monzonite',
               'carbonatite / foid-bearing']:
        s_ = fam == f_
        ax.scatter(sc.PC1[s_], sc.PC2[s_], s=6 if f_ != 'carbonatite / foid-bearing' else 16, c=cols_f[f_],
                   alpha=0.6 if f_ != 'carbonatite / foid-bearing' else 0.9, lw=0, label=f'{f_} ({s_.sum()})')
    scale = 0.8 * np.abs(sc[['PC1', 'PC2']].values).max()
    for _, x in pca.iterrows():
        ax.annotate('', xy=(x.PC1_loading * scale, x.PC2_loading * scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=0.8))
        ax.text(x.PC1_loading * scale * 1.1, x.PC2_loading * scale * 1.1, x.element, fontsize=8, ha='center', va='center')
    ev = summ.explained.values
    ax.set_xlabel(f'PC1 ({ev[0]:.0%} of clr variance)'); ax.set_ylabel(f'PC2 ({ev[1]:.0%})')
    ax.set_title('(a) clr-PCA, ten trace elements, three provinces', loc='left', fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, markerscale=2, loc='lower left')
    ax = axes[1]
    inds = [('carbonatite / foid-bearing rocks', None), ('La+Ce+Nd top 5%', 'LREE'), ('affinity top 5%', 'affinity')]
    ypos = 0
    ticks, labs = [], []
    for fs, m in order:
        for j, (lab_, ind) in enumerate(inds):
            if ind is None:
                x = rn[(rn.region == 'Superior') & (rn.featureset.astype(str) == fs) & (rn.model == m)].iloc[0]
            else:
                x = v[(v.region == 'Superior') & (v.featureset.astype(str) == fs) & (v.model == m) & (v.indicator == ind)].iloc[0]
            c = ['#d62728', '#1f77b4', '#7f7f7f'][j]
            ax.errorbar(x.enrichment, ypos, xerr=[[x.enrichment - x.enrich_lo], [x.enrich_hi - x.enrichment]], fmt='o',
                        color=c, ms=4, capsize=2, label=lab_ if (fs, m) == order[0] else None)
            ypos -= 1
        ticks.append(ypos + 2); labs.append(f'Set {fs} {LABEL[m]}')
        ypos -= 0.8
    ax.axvline(1, color='black', lw=0.8, ls='--')
    ax.set_yticks(ticks, labs); ax.set_xscale('symlog', linthresh=1); ax.set_xlim(-0.05, 8)
    ax.set_xlabel('share in the map\'s top 10% / share of all sampled locations')
    ax.set_title('(b) Superior: enrichment against the sampling baseline', loc='left', fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f'fig_geochem_pca{SUFFIX}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Map: unlabelled carbonatite/foid locations on the set-3 DevNet map, coloured by nothing but their class
    loc = pd.read_csv(o('geochem3_locations.csv'))
    loc = loc[(loc.region == 'Superior') & (loc.occ_km > 10) & loc.carb_named]
    grid = pd.read_csv(o('superior_grid_scores_fs3.csv'))
    import geopandas as gpd
    from matplotlib.lines import Line2D
    outline = gpd.read_file(os.path.join(REPO, 'data', 'outlines', 'Superior.geojson')).to_crs(maps.CRS)
    dep = maps.occurrences('Superior')
    canada = maps.province_layers()
    fig, axes = plt.subplots(1, 2, figsize=(18, 9.2))
    for ax, col, title, letter, key in [(axes[0], 'devnet', 'Superior DevNet MPM (set 3)', 'a', '3|DevNet'),
                                        (axes[1], 'rf', 'Superior RF MPM (set 3)', 'b', '3|RF')]:
        maps.map_panel(ax, grid, col, title, letter, outline, dep, canada, show_legend=False)
        sx, sy = maps.TO_P.transform(loc.lon.values, loc.lat.values)
        hit = (loc[key] >= 0.9).values
        ax.scatter(sx[~hit], sy[~hit], s=60, marker='D', c='white', edgecolors='black', linewidths=1.0, zorder=7)
        ax.scatter(sx[hit], sy[hit], s=60, marker='D', c='#00e5ff', edgecolors='black', linewidths=1.0, zorder=8)
        h = [Line2D([], [], marker='o', ls='', mfc=c, mec='black', ms=8, label=f'Training: {t}')
             for t, c in maps.DEP_COLORS.items()] + [
            Line2D([], [], marker='D', ls='', mfc='#00e5ff', mec='black', ms=7,
                   label='Unlabelled carbonatite / foid rock, in top 10%'),
            Line2D([], [], marker='D', ls='', mfc='white', mec='black', ms=7, label='... below top 10%')]
        leg = ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.08, 0.97), frameon=False, fontsize=10)
        ax.add_artist(leg)
    fig.savefig(os.path.join(FIG, f'fig_geochem{SUFFIX}.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def pa():
    y = load_region('Superior')['pruned']['y']
    st = fs3_store('Superior')
    pa_curve = importlib.import_module('paper_analyses.20_occurrence_breakdown').pa_curve
    th = np.linspace(0, 0.999, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    rows = []
    for ax, (title, proto) in zip(axes, [('(a) random stratified CV', 'random'), ('(b) 100 km spatial blocks', 100.0)]):
        for m in ALL_MODELS:
            p, a = pa_curve(y, st[('Superior', m, proto, None)]['oof'], th)
            k = np.argmin(np.abs(p - (1 - a)))
            rows.append({'protocol': title[4:], 'model': m, 'captured': p[k], 'area': a[k]})
            if m not in MODELS:
                continue
            ax.plot(th * 100, p * 100, color=COLORS[m], lw=1.6, label=LABEL[m])
            ax.plot(th * 100, a * 100, color=COLORS[m], lw=1.0, ls='--')
            ax.plot(th[k] * 100, p[k] * 100, 'o', color=COLORS[m], ms=5)
        ax.set_xlim(80, 100); ax.set_xlabel('score percentile threshold (%)')
        ax.set_title(title, loc='left', fontsize=10)
    axes[0].set_ylabel('occurrences above threshold (solid, %)\narea above threshold (dashed, %)')
    axes[1].legend(frameon=False, fontsize=7.5)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, f'fig_pa_superior{SUFFIX}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(o('pa_intersections_fs3.csv'), index=False)


def workflow():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis('off')

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', fc=fc, ec='#333333', lw=1))
        ax.text(x + w / 2, y + h - 0.22, title, ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(x + w / 2, y + h - 0.55, body, ha='center', va='top', fontsize=8.2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1))
    box(0.1, 4.5, 3.0, 2.0, '1  Evidence layers', 'potential fields, LAB, Moho\nfaults, margins, worms\nH3 geology grid (lithology,\nera, contacts)\n15 Superior occurrences', '#eef3fb')
    box(3.4, 4.5, 3.3, 2.0, '2  ' + WORKFLOW_INPUTS[0], WORKFLOW_INPUTS[1], '#eef3fb')
    box(7.0, 4.5, 2.9, 2.0, '3  Models', WORKFLOW_MODELS, '#fdf0e6')
    box(0.1, 1.9, 3.1, 2.2, '4  Validation', WORKFLOW_VALIDATION, '#eaf6ee')
    box(3.5, 1.9, 3.0, 2.2, '5  Method checks', 'fold-safe synthetic positives\n(jitter, SMOTE, WGAN-GP)\nS2 parent-child protocol audit\ncoordinate-only baseline\nprovince diagnostics', '#eaf6ee')
    box(6.8, 1.9, 3.1, 2.2, '6  Independent checks', 'Clay-Howells Fe-REE-Nb\nresource (not a label)\nKapuskasing Structural\nZone (not an input)', '#eaf6ee')
    box(1.6, 0.1, 6.8, 1.4, '7  Interpretation and maps', 'TreeSHAP / DeepSHAP (ensemble mean), Spearman direction rule\npercentile prospectivity maps; leave-one-occurrence-out stability', '#f5eefa')
    arrow(3.1, 5.5, 3.4, 5.5); arrow(6.7, 5.5, 7.0, 5.5)
    arrow(8.4, 4.5, 1.65, 4.1); arrow(8.4, 4.5, 5.0, 4.1); arrow(8.4, 4.5, 8.35, 4.1)
    arrow(1.65, 1.9, 3.0, 1.5); arrow(8.35, 1.9, 7.0, 1.5)
    fig.savefig(os.path.join(FIG, f'fig_workflow{SUFFIX}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def set1_reused():
    g.GEN = GEN
    for fn in [g.geography, g.augmentation, g.audit, g.ablation, g.region_diagnostics, g.nested, g.table_benchmark]:
        try:
            fn()
        except (FileNotFoundError, IndexError, KeyError) as e:
            print(f'skip {fn.__name__}: {e}')


if __name__ == '__main__':
    for fn in [set1_reused, benchmark, spatial, capture, featuresets, scarcity, controls, occurrence, shap_outputs,
               map_figures, geochem, pa, workflow]:
        try:
            fn()
        except (FileNotFoundError, IndexError, KeyError) as e:
            print(f'skip {fn.__name__}: {type(e).__name__}: {e}')
