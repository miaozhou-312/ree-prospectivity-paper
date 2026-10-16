"""
Figures and LaTeX table bodies for the REVISED manuscript
(paper/manuscript/main_src_revised.tex).

In the revised manuscript "DevNet" is the final configuration (Stage 11:
quantile-normal inputs, background-heavy batches) and "nnPU" uses the same
quantile-normal inputs (Stage 14). RF and bagging-PU are unchanged
(Stages 1-2). The configuration with z-scored inputs and 16 + 16 batches,
and the other training-budget variants, appear only in the ablation table
(Appendix D of the revised manuscript).

Outputs: paper/manuscript/generated_revised/*.tex and
paper/manuscript/figures/*_rev.png. The original manuscript's generator
(06_figures_tables.py) and outputs are untouched.

    python -m paper_analyses.06b_figures_tables_revised
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from paper_analyses.common import OUT_DIR, load_pickle, load_region  # noqa: E402
from paper_analyses.figstyle import pretty  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, 'paper', 'manuscript', 'figures')
GEN = os.path.join(REPO, 'paper', 'manuscript', 'generated_revised')
os.makedirs(GEN, exist_ok=True)
MODELS = ['DevNet', 'RF', 'BaggingPU', 'nnPU']
LABEL = {'DevNet': 'DevNet', 'RF': 'RF', 'BaggingPU': 'Bagging-PU', 'nnPU': 'nnPU', 'RF_fixedseed': 'RF (fixed seed)'}
COLORS = {'DevNet': '#d03b3b', 'RF': '#2a78d6', 'BaggingPU': '#3a9e5c', 'nnPU': '#8a5cc2'}
REGIONS = ['Superior', 'Cordillera', 'Grenville']
PROTOS = [0, 50, 100, 200]
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def o(name):
    return os.path.join(OUT_DIR, name)


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f:
        f.write(text)
    print('wrote', name)


def ci(med, lo, hi, bold=False):
    return (f'\\textbf{{{med:.3f}}}' if bold else f'{med:.3f}') + f' [{lo:.3f}, {hi:.3f}]'


def bold_if(v, best):
    return f'\\textbf{{{v:.3f}}}' if v == best else f'{v:.3f}'


# ── assembled result tables ─────────────────────────────────────────────────
def results():
    """One row per (region, model, feature_set, block_km) with medians and
    intervals, for the four main models."""
    bm = pd.read_csv(o('benchmark_Superior_Cordillera_Grenville.csv'))
    bm = bm[bm.model.isin(['RF', 'BaggingPU', 'RF_fixedseed'])].assign(block_km=0)
    sp = pd.read_csv(o('spatial_cv_Superior_Cordillera_Grenville.csv'))
    sp = sp[sp.model.isin(['RF', 'BaggingPU'])].assign(feature_set='pruned')
    dn = pd.read_csv(o('devnet_final.csv')).assign(model='DevNet', feature_set='pruned')
    fs = pd.read_csv(o('final_suite.csv'))
    fs = fs.replace({'model': {'nnPU-QT': 'nnPU'}})
    cols = ['region', 'model', 'feature_set', 'block_km', 'auc_med', 'auc_lo', 'auc_hi', 'ap_med', 'ap_lo', 'ap_hi']
    return pd.concat([bm[cols], sp[cols], dn[cols], fs[cols]], ignore_index=True)


def n_pos():
    bm = pd.read_csv(o('benchmark_Superior_Cordillera_Grenville.csv'))
    return bm.groupby('region').n_pos.first().to_dict()


def table_benchmark():
    R, npos = results(), n_pos()
    for fs, name in [('pruned', 'tab_benchmark.tex'), ('full', 'tab_benchmark_full.tex')]:
        sub_all = R[(R.feature_set == fs) & (R.block_km == 0)]
        lines = []
        for r in REGIONS:
            sub = sub_all[sub_all.region == r].set_index('model')
            best_ap, best_auc = sub.loc[MODELS, 'ap_med'].max(), sub.loc[MODELS, 'auc_med'].max()
            for k, m in enumerate(MODELS + ['RF_fixedseed']):
                x = sub.loc[m]
                lab = f'\\multirow{{5}}{{*}}{{{r} ({npos[r]})}}' if k == 0 else ''
                lines.append(f'{lab} & {LABEL[m]} & {ci(x.auc_med, x.auc_lo, x.auc_hi, m in MODELS and x.auc_med == best_auc)} & '
                             f'{ci(x.ap_med, x.ap_lo, x.ap_hi, m in MODELS and x.ap_med == best_ap)} \\\\')
            lines.append('\\midrule' if r != REGIONS[-1] else '')
        write(name, '\n'.join(lines))

    dn = pd.read_csv(o('devnet_final.csv'))
    fb = pd.read_csv(o('final_suite_bootstrap.csv'))
    lines = []
    for r in REGIONS:
        x = dn[(dn.region == r) & (dn.block_km == 0)].iloc[0]
        for m in ['RF', 'BaggingPU']:
            lines.append(f"{r} & DevNet $-$ {LABEL[m]} & {x[f'd_ap_vs_{m}']:+.3f} [{x[f'd_ap_vs_{m}_lo']:+.3f}, "
                         f"{x[f'd_ap_vs_{m}_hi']:+.3f}] & {x[f'p_vs_{m}']:.3f} & {x[f'wins_vs_{m}']} & "
                         f"{x[f'd_auc_vs_{m}']:+.3f} & {x[f'p_auc_vs_{m}']:.3f} \\\\")
        y = fb[(fb.region == r) & (fb.block_km == 0)].iloc[0]
        lo, hi = eval(y.d_ap_ci)
        lines.append(f'{r} & DevNet $-$ nnPU & {y.d_ap:+.3f} [{lo:+.3f}, {hi:+.3f}] & {y.p_ap:.3f} & {y.wins} & '
                     f'{y.d_auc:+.3f} & {y.p_auc:.3f} \\\\')
        if r != REGIONS[-1]:
            lines.append('\\addlinespace')
    write('tab_bootstrap.tex', '\n'.join(lines))

    fs = pd.read_csv(o('final_suite.csv'))
    rows = []
    for _, x in fs[(fs.region == 'Superior') & fs.model.str.startswith('nnPU-QT') & (fs.feature_set == 'pruned')
                   & (fs.block_km == 0)].iterrows():
        pi = 0.01 if x.model == 'nnPU-QT' else float(x.model.split('=')[1].rstrip(')'))
        rows.append((pi, f'{pi:g} & {ci(x.auc_med, x.auc_lo, x.auc_hi)} & {ci(x.ap_med, x.ap_lo, x.ap_hi)} \\\\'))
    write('tab_nnpu_prior.tex', '\n'.join(r for _, r in sorted(rows)))

    lines = []
    for r in REGIONS:
        base = dn[(dn.region == r) & (dn.block_km == 0)].iloc[0]
        cells = []
        for a in [2, 3, 5, 8]:
            if a == 5:
                cells.append(f'{base.ap_med:.3f}')
            else:
                x = fs[(fs.region == r) & (fs.model == f'DevNet(a={a})')].iloc[0]
                cells.append(f'{x.ap_med:.3f}')
        lines.append(f'{r} & ' + ' & '.join(cells) + ' \\\\')
    write('tab_margin.tex', '\n'.join(lines))


def spatial():
    R = results()
    R = R[R.feature_set == 'pruned']
    for metric, fname in [('ap_med', 'tab_spatial.tex'), ('auc_med', 'tab_spatial_auc.tex')]:
        lines = []
        for r in REGIONS:
            for k, blk in enumerate(PROTOS):
                s = R[(R.region == r) & (R.block_km == blk)].set_index('model')
                best = s.loc[MODELS, metric].max()
                lab = f'\\multirow{{4}}{{*}}{{{r}}}' if k == 0 else ''
                name = 'random' if blk == 0 else f'{blk} km blocks'
                lines.append(f'{lab} & {name} & ' + ' & '.join(bold_if(s.loc[m, metric], best) for m in MODELS) + ' \\\\')
            lines.append('\\midrule' if r != REGIONS[-1] else '')
        write(fname, '\n'.join(lines))

    dn = pd.read_csv(o('devnet_final.csv'))
    fb = pd.read_csv(o('final_suite_bootstrap.csv'))
    lines = []
    for blk in [50, 100, 200]:
        x = dn[(dn.region == 'Superior') & (dn.block_km == blk)].iloc[0]
        for m in ['RF', 'BaggingPU']:
            lines.append(f"{blk} km & DevNet $-$ {LABEL[m]} & {x[f'd_ap_vs_{m}']:+.3f} [{x[f'd_ap_vs_{m}_lo']:+.3f}, "
                         f"{x[f'd_ap_vs_{m}_hi']:+.3f}] & {x[f'p_vs_{m}']:.3f} & {x[f'wins_vs_{m}']} \\\\")
        y = fb[(fb.region == 'Superior') & (fb.block_km == blk)].iloc[0]
        lo, hi = eval(y.d_ap_ci)
        lines.append(f'{blk} km & DevNet $-$ nnPU & {y.d_ap:+.3f} [{lo:+.3f}, {hi:+.3f}] & {y.p_ap:.3f} & {y.wins} \\\\')
        if blk != 200:
            lines.append('\\addlinespace')
    write('tab_spatial_bootstrap.tex', '\n'.join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    xs = np.arange(4)
    for ax, r in zip(axes, REGIONS):
        for k, m in enumerate(MODELS):
            s = R[(R.region == r) & (R.model == m)].set_index('block_km').loc[PROTOS]
            ax.errorbar(xs + (k - 1.5) * 0.12, s.ap_med, yerr=[s.ap_med - s.ap_lo, s.ap_hi - s.ap_med], fmt='o-',
                        color=COLORS[m], ms=4, lw=1.2, capsize=2, label=LABEL[m])
        ax.set_xticks(xs, ['random', '50 km', '100 km', '200 km'])
        ax.set_title(r, loc='left', fontsize=10)
        ax.set_xlabel('cross-validation split')
    axes[0].set_ylabel('average precision (median, 95% interval)')
    axes[1].legend(frameon=False, fontsize=8, loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_spatial_cv_rev.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def stores_superior():
    bench = load_pickle(o('benchmark_Superior_Cordillera_Grenville.pkl'))
    spat = load_pickle(o('spatial_cv_Superior_Cordillera_Grenville.pkl'))
    dn = load_pickle(o('devnet_final.pkl'))
    fs = load_pickle(o('final_suite.pkl'))
    rnd = {'DevNet': dn[('Superior', 'random')], 'RF': bench[('Superior', 'pruned', 'RF')],
           'BaggingPU': bench[('Superior', 'pruned', 'BaggingPU')], 'nnPU': fs[('Superior', 'nnPU-QT', 'pruned', 'random')]}
    blk = {'DevNet': dn[('Superior', 100.0)], 'RF': spat[('Superior', 100.0, 'RF')],
           'BaggingPU': spat[('Superior', 100.0, 'BaggingPU')], 'nnPU': fs[('Superior', 'nnPU-QT', 'pruned', 100.0)]}
    return rnd, blk


def capture():
    y = load_region('Superior')['pruned']['y']
    rnd, blk = stores_superior()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), sharey=True)
    xg = np.linspace(0, 10, 201)
    for ax, (title, st) in zip(axes, [('(a) random stratified CV', rnd), ('(b) 100 km spatial blocks', blk)]):
        for m in MODELS:
            curves = []
            for oof in st[m]['oof']:
                order = np.argsort(-oof)
                cum = np.cumsum(y[order]) / y.sum()
                frac = np.arange(1, len(oof) + 1) / len(oof)
                curves.append(np.interp(np.linspace(0, 0.1, 201), frac, cum))
            curves = np.array(curves)
            ax.plot(xg, np.median(curves, 0), color=COLORS[m], lw=1.6, label=LABEL[m])
            ax.fill_between(xg, curves.min(0), curves.max(0), color=COLORS[m], alpha=0.12, lw=0)
        ax.plot([0, 10], [0, 0.1], 'k--', lw=0.8, label='random')
        ax.set_xlim(0, 10); ax.set_ylim(0, 1)
        ax.set_xlabel('top-ranked share of labelled pool (%)')
        ax.set_title(title, loc='left', fontsize=10)
    axes[0].set_ylabel('share of held-out occurrences captured')
    axes[1].legend(frameon=False, fontsize=8, loc='lower right', bbox_to_anchor=(1.0, 0.15))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_capture_superior_rev.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def geography():
    diag = pd.concat([pd.read_csv(o(f'devnet_diagnostics_{r}.csv')) for r in REGIONS], ignore_index=True)
    R = results()
    R = R[R.feature_set == 'pruned']
    lines = []
    for r in REGIONS:
        rows = [('RF, coordinates only', diag[(diag.region == r) & (diag.model == 'RF_xy')]),
                ('RF, all features', R[(R.region == r) & (R.model == 'RF')]),
                ('Logistic regression', diag[(diag.region == r) & (diag.model == 'LogReg')]),
                ('DevNet', R[(R.region == r) & (R.model == 'DevNet')])]
        for k, (label, src) in enumerate(rows):
            cells = []
            for proto, blk in [('random', 0), ('100 km', 100)]:
                x = src[src.protocol == proto].iloc[0] if 'protocol' in src and src.protocol.notna().all() else src[src.block_km == blk].iloc[0]
                cells += [f'{x.ap_med:.3f}', f'{x.auc_med:.3f}']
            lab = f'\\multirow{{4}}{{*}}{{{r}}}' if k == 0 else ''
            lines.append(f'{lab} & {label} & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_geography.tex', '\n'.join(lines))


def augmentation():
    names = {'jitter': 'Gaussian jitter', 'smote': 'SMOTE', 'wgan': 'WGAN-GP'}
    srcs = {'DevNet': (pd.read_csv(o('augmentation_final_Superior.csv')), load_pickle(o('augmentation_final_Superior.pkl'))),
            'RF': (pd.read_csv(o('augmentation_Superior.csv')), load_pickle(o('augmentation_Superior.pkl')))}
    lines = []
    for model in ['DevNet', 'RF']:
        ag, st = srcs[model]
        ag = ag[ag.model == model]
        base = ag[ag.method == 'none'].iloc[0]
        lines.append(f'\\multirow{{10}}{{*}}{{{model}}} & none & -- & {base.auc_med:.3f} & '
                     f'{ci(base.ap_med, base.ap_lo, base.ap_hi)} & -- & -- \\\\')
        for meth in ['jitter', 'smote', 'wgan']:
            for mult in [1, 3, 10]:
                x = ag[(ag.method == meth) & (ag.mult == mult)].iloc[0]
                dd = st[(model, meth, mult)]['aps'] - st[(model, 'none', 0)]['aps']
                lines.append(f' & {names[meth]} & {mult}$\\times$ & {x.auc_med:.3f} & {ci(x.ap_med, x.ap_lo, x.ap_hi)} & '
                             f'{np.median(dd):+.3f} [{dd.min():+.3f}, {dd.max():+.3f}] & {int((dd > 0).sum())}/{len(dd)} \\\\')
        lines.append('\\midrule' if model == 'DevNet' else '')
    write('tab_augmentation.tex', '\n'.join(lines))

    dg = pd.read_csv(o('augmentation_diagnostics_Superior.csv'))
    fd = pd.read_csv(o('augmentation_final_diag_Superior.csv')).set_index('source')
    src_names = {'real positives (LOO)': 'Real occurrences (leave-one-out)', 'jitter': 'Gaussian jitter',
                 'smote': 'SMOTE', 'wgan': 'WGAN-GP', 'background (300 random)': 'Background (300 random points)'}
    lines = []
    for _, x in dg.iterrows():
        share = '--' if np.isnan(x.share_nn_is_positive) else f'{x.share_nn_is_positive * 100:.0f}\\%'
        p10 = '--' if np.isnan(x.nn_ratio_p10) else f'{x.nn_ratio_p10:.2f}'
        pct = fd.loc[x.source, 'devnet_pct_med_final'] if x.source in fd.index else x.devnet_pct_med
        lines.append(f'{src_names[x.source]} & {x.nn_ratio_med:.2f} & {p10} & {share} & {pct:.3f} \\\\')
    write('tab_augmentation_diag.tex', '\n'.join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for ax, model in zip(axes, ['DevNet', 'RF']):
        ag = srcs[model][0]
        ag = ag[ag.model == model]
        base = ag[ag.method == 'none'].iloc[0]
        ax.axhspan(base.ap_lo, base.ap_hi, color='#dddddd', zorder=0)
        ax.axhline(base.ap_med, color='k', lw=1, label='no augmentation')
        for k, meth in enumerate(['jitter', 'smote', 'wgan']):
            s = ag[ag.method == meth].sort_values('mult')
            ax.errorbar(np.arange(3) + (k - 1) * 0.15, s.ap_med, yerr=[s.ap_med - s.ap_lo, s.ap_hi - s.ap_med],
                        fmt='o-', ms=4, capsize=2, color=['#0072b2', '#009e73', '#e69f00'][k], label=names[meth])
        ax.set_xticks(np.arange(3), ['1$\\times$', '3$\\times$', '10$\\times$'])
        ax.set_xlabel('synthetic positives per real training positive')
        ax.set_title(f'({"ab"[["DevNet", "RF"].index(model)]}) {model}', loc='left', fontsize=10)
    axes[0].set_ylabel('average precision (median, 95% interval)')
    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_augmentation_performance_rev.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    sd = (pd.read_csv(o('augmentation_final_spatial_Superior.csv')), load_pickle(o('augmentation_final_spatial_Superior.pkl')))
    sr = (pd.read_csv(o('augmentation_spatial_Superior.csv')), load_pickle(o('augmentation_spatial_Superior.pkl')))
    lines = []
    for block in [100.0, 200.0]:
        for model, (df, st) in [('DevNet', sd), ('RF', sr)]:
            sub = df[(df.block_km == block) & (df.model == model)]
            for k, (_, x) in enumerate(sub.iterrows()):
                lab = f'{block:.0f} km & {model}' if k == 0 else ' & '
                if x.method == 'none':
                    lines.append(f'{lab} & none & {ci(x.ap_med, x.ap_lo, x.ap_hi)} & -- & -- \\\\')
                else:
                    dd = st[(block, model, x.method, int(x.mult))]['aps'] - st[(block, model, 'none', 0)]['aps']
                    lines.append(f'{lab} & {names[x.method]} {int(x.mult)}$\\times$ & {ci(x.ap_med, x.ap_lo, x.ap_hi)} & '
                                 f'{np.median(dd):+.3f} [{dd.min():+.3f}, {dd.max():+.3f}] & {int((dd > 0).sum())}/{len(dd)} \\\\')
            lines.append('\\addlinespace')
    write('tab_augmentation_spatial.tex', '\n'.join(lines[:-1]))


def audit():
    a = pd.read_csv(o('protocol_audit.csv'))
    sim = pd.read_csv(o('protocol_audit_sibling_similarity.csv')).set_index('region')
    lines = []
    for r in REGIONS:
        first = True
        for neg in ['cosine', 'random']:
            for split in ['child-level random', 'grouped by parent', 'no augmentation']:
                x = a[(a.region == r) & (a.negatives == neg) & (a.split == split)].iloc[0]
                lab = f'\\multirow{{6}}{{*}}{{{r}}}' if first else ''
                first = False
                sib = '--' if np.isnan(x.test_pos_with_sibling_in_train) else f'{x.test_pos_with_sibling_in_train * 100:.0f}\\%'
                short = {'child-level random': 'child-level', 'grouped by parent': 'parent-grouped',
                         'no augmentation': 'no augm.'}[split]
                lines.append(f'{lab} & {neg} & {short} & {int(x.n_pos_labels)} & '
                             f'{x.auc_mean:.3f} [{x.auc_lo:.3f}, {x.auc_hi:.3f}] & {sib} \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_audit.tex', '\n'.join(lines))
    write('tab_audit_similarity.tex', '\n'.join(
        f'{r} & {sim.loc[r, "sibling_dist_med"]:.2f} & {sim.loc[r, "nearest_other_occurrence_dist_med"]:.2f} & '
        f'{sim.loc[r, "nearest_background_neighbour_dist_med"]:.2f} \\\\' for r in REGIONS))


def shap_tables_and_figure():
    for r in REGIONS:
        t = pd.read_csv(o(f'shap_final_top10_{r}.csv'))
        arrow = lambda lab: {'higher -> more': '$\\uparrow$', 'higher -> less': '$\\downarrow$',
                             'present -> more': '$\\uparrow$', 'present -> less': '$\\downarrow$'}.get(lab, '$\\circ$')
        write(f'tab_shap_{r}.tex', '\n'.join(
            f'{x.label} & {int(x.dn_rank)} & {arrow(x.dn_direction)} ({x.dn_rho:+.2f}) & {int(x.rf_rank)} & '
            f'{arrow(x.rf_direction)} ({x.rf_rho:+.2f}) \\\\' for _, x in t.iterrows()))
    sr = load_pickle(o('shap_final_results.pkl'))['shap']['Superior']
    names = [pretty(c) for c in sr['feat_cols']]
    fig = plt.figure(figsize=(14, 6.5))
    for i, (key, X, title) in enumerate([('shap_dn_ens', sr['X_explain_dn'], '(a) DevNet (ensemble-mean SHAP)'),
                                         ('shap_rf', sr['X_explain_rf'], '(b) Random Forest (TreeSHAP)')]):
        ax = fig.add_subplot(1, 2, i + 1)
        plt.sca(ax)
        np.random.seed(0)  # summary_plot shuffles and jitters dots with np.random
        shap.summary_plot(sr[key], X, feature_names=names, max_display=12, show=False, plot_size=None,
                          color_bar=(i == 1))
        ax.set_title(title, fontsize=12, loc='left')
        ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_shap_superior_rev.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def ablation():
    """Appendix D: DevNet design choices. AP median per region and protocol,
    and the number of repeats in which the final configuration beats each."""
    dn = load_pickle(o('devnet_final.pkl'))
    qt = load_pickle(o('devnet_qt_sensitivity.pkl'))
    bg = load_pickle(o('training_budget_bg_Superior_Cordillera_Grenville.pkl'))
    es = {}
    for r in REGIONS:
        es.update(load_pickle(o(f'training_budget_es_{r}.pkl')))
    bench = load_pickle(o('benchmark_Superior_Cordillera_Grenville.pkl'))
    spat = load_pickle(o('spatial_cv_Superior_Cordillera_Grenville.pkl'))
    diag = {}
    for r in REGIONS:
        diag.update(load_pickle(o(f'devnet_diagnostics_{r}.pkl'))['store'])
    key = lambda blk: 'random' if blk == 0 else float(blk)
    configs = [
        ('quantile & 16\\,+\\,240 & 300 (final)', lambda r, b: dn[(r, key(b))]),
        ('z-score & 16\\,+\\,240 & 300', lambda r, b: bg[(r, key(b))]),
        ('quantile & 16\\,+\\,16 & 300', lambda r, b: qt[(r, key(b))]),
        ('z-score & 16\\,+\\,16 & 300', lambda r, b: bench[(r, 'pruned', 'DevNet')] if b == 0
         else spat[(r, float(b), 'DevNet')]),
        ('z-score & 16\\,+\\,16 & 3,000', lambda r, b: diag.get((r, 'DevNet_long', 'random' if b == 0
                                                                                    else f'{b} km'))),
        ('z-score & 16\\,+\\,16 & nested ES', lambda r, b: es[(r, key(b))]),
    ]
    lines = []
    for r in REGIONS:
        for k, (label, get) in enumerate(configs):
            cells = []
            for b in PROTOS:
                res = get(r, b)
                if res is None:
                    cells.append('--')
                    continue
                if k == 0:
                    cells.append(f'\\textbf{{{res["ap_med"]:.3f}}}')
                else:
                    wins = int((dn[(r, key(b))]['aps'] > res['aps']).sum())
                    cells.append(f'{res["ap_med"]:.3f} ({wins}/6)')
            lab = f'\\multirow{{6}}{{*}}{{{r}}}' if k == 0 else ''
            lines.append(f'{lab} & {label} & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_ablation.tex', '\n'.join(lines))


def workflow():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis('off')

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.08',
                                    fc=fc, ec='#333333', lw=0.8))
        ax.text(x + w / 2, y + h - 0.16, title, ha='center', va='top', fontsize=8.5, weight='bold')
        ax.text(x + w / 2, y + h - 0.45, body, ha='center', va='top', fontsize=7.2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.9, color='#333333'))

    box(0.1, 4.3, 2.6, 1.8, '1  Inputs', '17 potential-field rasters\n(gravity, magnetics, LAB, Moho)\nfault / margin / worm vectors\nH3 terrane era & lithology\n15 Superior occurrences', '#eef3fb')
    box(3.0, 4.3, 3.2, 1.8, '2  Feature table', 'buffer-then-clip; A$\\wedge$B domain\npoint sampling + 9$\\times$9 GLCM textures\n48 features $\\rightarrow$ 35 (label-blind\nSpearman pruning, |$\\rho$| > 0.7)\n3,000 eval + 5,000 train background', '#eef3fb')
    box(6.5, 4.3, 3.4, 1.8, '3  Models (shared background)', 'RF: background = negative\nBagging-PU, nnPU: positive-unlabelled\nDevNet: deviation loss, quantile-normal\ninputs, 16 positive + 240 background\nrows per step; identical folds for all', '#fdf0e6')
    box(0.1, 1.9, 3.1, 2.0, '4  Validation', 'pooled out-of-fold ranks\n(a) random stratified CV, 5$\\times$6\n(b) spatial blocks 50/100/200 km,\n     background purged from test blocks\nAUC, AP (+95% interval), paired\nbootstrap, capture rate, coordinate-only\nbaseline', '#eaf6ee')
    box(3.5, 1.9, 3.0, 2.0, '5  Augmentation (fold-safe)', 'jitter / SMOTE / WGAN-GP\nfit on training-fold positives only\n1$\\times$, 3$\\times$, 10$\\times$ synthetic positives\nheld-out positives always real\n+ memorisation diagnostics', '#eaf6ee')
    box(6.8, 1.9, 3.1, 2.0, '6  Protocol audit', 'S2 level-12 $\\rightarrow$ level-13 children\ncosine-dissimilar vs random negatives\nchild-level vs parent-grouped split\n50 $\\times$ 70/30 balanced splits\n(design of Parsa et al. 2024)', '#eaf6ee')
    box(1.6, 0.1, 6.8, 1.4, '7  Interpretation and maps', 'TreeSHAP / DeepSHAP (ensemble mean), direction = Spearman $\\rho$(feature, SHAP)\nproduction models $\\rightarrow$ percentile prospectivity maps; blind check at Clay-Howells (KSZ)', '#f5eefa')
    arrow(2.7, 5.2, 3.0, 5.2); arrow(6.2, 5.2, 6.5, 5.2)
    arrow(8.2, 4.3, 1.65, 3.9); arrow(8.2, 4.3, 5.0, 3.9); arrow(8.2, 4.3, 8.35, 3.9)
    arrow(1.65, 1.9, 3.0, 1.5); arrow(5.0, 1.9, 5.0, 1.5)
    fig.savefig(os.path.join(FIG, 'fig_workflow_rev.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    for fn in [workflow, table_benchmark, spatial, capture, geography, augmentation, audit,
               shap_tables_and_figure, ablation]:
        try:
            fn()
        except FileNotFoundError as e:
            print(f'skip {fn.__name__}: {e}')
