"""
Stage 6 -- paper figures and LaTeX table bodies, generated from the Stage
1-5 outputs so no number in the manuscript is typed by hand.

Writes paper/manuscript/figures/*.png and paper/manuscript/generated/*.tex;
paper/build.py splices the .tex snippets into the manuscript.

    python -m paper_analyses.06_figures_tables
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paper_analyses.common import OUT_DIR, load_pickle  # noqa: E402
from paper_analyses.figstyle import MODEL_COLORS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, 'paper', 'manuscript', 'figures')
GEN = os.path.join(REPO, 'paper', 'manuscript', 'generated')
os.makedirs(FIG, exist_ok=True)
os.makedirs(GEN, exist_ok=True)
MODELS = ['DevNet', 'RF', 'BaggingPU', 'nnPU']
LABEL = {'DevNet': 'DevNet', 'RF': 'RF', 'BaggingPU': 'Bagging-PU', 'nnPU': 'nnPU',
         'RF_fixedseed': 'RF (fixed seed)'}
REGIONS = ['Superior', 'Cordillera', 'Grenville']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def o(name):
    return os.path.join(OUT_DIR, name)


def fmt_ci(med, lo, hi, bold=False):
    s = f'{med:.3f} [{lo:.3f}, {hi:.3f}]'
    return f'\\textbf{{{med:.3f}}} [{lo:.3f}, {hi:.3f}]' if bold else s


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f:
        f.write(text)
    print('wrote', name)


# ── Table: benchmark under random repeated CV ───────────────────────────────
def table_benchmark():
    allbm = pd.read_csv(o('benchmark_Superior_Cordillera_Grenville.csv'))
    for fs, name in [('pruned', 'tab_benchmark.tex'), ('full', 'tab_benchmark_full.tex')]:
        bm = allbm[(allbm.feature_set == fs) & allbm.model.isin(MODELS + ['RF_fixedseed'])]
        lines = []
        for r in REGIONS:
            sub = bm[bm.region == r].set_index('model')
            best_ap, best_auc = sub.loc[MODELS, 'ap_med'].max(), sub.loc[MODELS, 'auc_med'].max()
            first = True
            for m in MODELS + ['RF_fixedseed']:
                row = sub.loc[m]
                reg = f'\\multirow{{5}}{{*}}{{{r} ({int(row.n_pos)})}}' if first else ''
                first = False
                lines.append(f'{reg} & {LABEL[m]} & '
                             f'{fmt_ci(row.auc_med, row.auc_lo, row.auc_hi, m in MODELS and row.auc_med == best_auc)} & '
                             f'{fmt_ci(row.ap_med, row.ap_lo, row.ap_hi, m in MODELS and row.ap_med == best_ap)} \\\\')
            lines.append('\\midrule' if r != REGIONS[-1] else '')
        write(name, '\n'.join(lines))

    bt = pd.read_csv(o('benchmark_bootstrap_Superior_Cordillera_Grenville.csv'))
    bt = bt[(bt.feature_set == 'pruned') & bt.comparison.isin([f'DevNet - {m}' for m in MODELS[1:]])]
    lines = []
    for r in REGIONS:
        for _, x in bt[bt.region == r].iterrows():
            lo, hi = eval(x.d_ap_ci) if isinstance(x.d_ap_ci, str) else x.d_ap_ci
            alo, ahi = eval(x.d_auc_ci) if isinstance(x.d_auc_ci, str) else x.d_auc_ci
            other = LABEL[x.comparison.split(' - ')[1]]
            lines.append(f'{r} & DevNet $-$ {other} & {x.d_ap:+.3f} [{lo:+.3f}, {hi:+.3f}] & {x.p_ap:.3f} & '
                         f'{x.ap_wins_of_repeats} & {x.d_auc:+.3f} & {x.p_auc:.3f} \\\\')
    write('tab_bootstrap.tex', '\n'.join(lines))

    allb = pd.read_csv(o('benchmark_Superior_Cordillera_Grenville.csv'))
    pri = allb[allb.model.str.startswith('nnPU')]
    pri = pri[pri.region == 'Superior']
    pri = pri[pri.feature_set == 'pruned']
    rows = []
    for _, x in pri.iterrows():
        pi = '0.01' if x.model == 'nnPU' else x.model.split('=')[1].rstrip(')')
        rows.append((float(pi), f'{pi} & {fmt_ci(x.auc_med, x.auc_lo, x.auc_hi)} & {fmt_ci(x.ap_med, x.ap_lo, x.ap_hi)} \\\\'))
    write('tab_nnpu_prior.tex', '\n'.join(r for _, r in sorted(rows)))


# ── Spatial CV table + figure ───────────────────────────────────────────────
def spatial():
    bm = pd.read_csv(o('benchmark_Superior_Cordillera_Grenville.csv'))
    bm = bm[(bm.feature_set == 'pruned') & bm.model.isin(MODELS)].copy()
    bm['block_km'] = 0
    sp = pd.read_csv(o('spatial_cv_Superior_Cordillera_Grenville.csv'))
    allr = pd.concat([bm, sp], ignore_index=True)

    lines = []
    for r in REGIONS:
        sub = allr[allr.region == r]
        for blk in [0, 50, 100, 200]:
            s = sub[sub.block_km == blk].set_index('model')
            best = s.loc[MODELS, 'ap_med'].max()
            name = 'random' if blk == 0 else f'{blk:.0f} km blocks'
            cells = [f'{s.loc[m, "ap_med"]:.3f}' if s.loc[m, 'ap_med'] != best else f'\\textbf{{{s.loc[m, "ap_med"]:.3f}}}'
                     for m in MODELS]
            aucs = [f'{s.loc[m, "auc_med"]:.3f}' for m in MODELS]
            lab = f'\\multirow{{4}}{{*}}{{{r}}}' if blk == 0 else ''
            lines.append(f'{lab} & {name} & ' + ' & '.join(cells) + ' & ' + ' & '.join(aucs) + ' \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_spatial.tex', '\n'.join(lines))

    spb = pd.read_csv(o('spatial_cv_bootstrap_Superior_Cordillera_Grenville.csv'))
    lines = []
    for _, x in spb[spb.region == 'Superior'].iterrows():
        lo, hi = eval(x.d_ap_ci)
        lines.append(f'{x.block_km:.0f} km & DevNet $-$ {LABEL[x.comparison.split(" - ")[1]]} & '
                     f'{x.d_ap:+.3f} [{lo:+.3f}, {hi:+.3f}] & {x.p_ap:.3f} & {x.ap_wins_of_repeats} \\\\')
    write('tab_spatial_bootstrap.tex', '\n'.join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=False)
    xs = np.arange(4)
    for ax, r in zip(axes, REGIONS):
        sub = allr[allr.region == r]
        for k, m in enumerate(MODELS):
            s = sub[sub.model == m].sort_values('block_km')
            off = (k - 1.5) * 0.12
            ax.errorbar(xs + off, s.ap_med, yerr=[s.ap_med - s.ap_lo, s.ap_hi - s.ap_med], fmt='o-',
                        color=MODEL_COLORS[m], ms=4, lw=1.2, capsize=2, label=LABEL[m])
        ax.set_xticks(xs, ['random', '50 km', '100 km', '200 km'])
        ax.set_title(r, loc='left', fontsize=10)
        ax.set_xlabel('cross-validation split')
    axes[0].set_ylabel('average precision (median, 95% interval)')
    axes[1].legend(frameon=False, fontsize=8, loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_spatial_cv.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Capture-rate curves (Superior, random CV OOF, repeat-averaged) ──────────
def capture():
    store = load_pickle(o('benchmark_Superior_Cordillera_Grenville.pkl'))
    sp = load_pickle(o('spatial_cv_Superior_Cordillera_Grenville.pkl'))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), sharey=True)
    for ax, (title, getter) in zip(axes, [
            ('(a) random stratified CV', lambda m: store[('Superior', 'pruned', m)]),
            ('(b) 100 km spatial blocks', lambda m: sp[('Superior', 100.0, m)])]):
        for m in MODELS:
            res = getter(m)
            curves = []
            for oof in res['oof']:
                ycv = _y_superior()
                order = np.argsort(-oof)
                cum = np.cumsum(ycv[order]) / ycv.sum()
                frac = np.arange(1, len(oof) + 1) / len(oof)
                curves.append(np.interp(np.linspace(0, 0.1, 201), frac, cum))
            curves = np.array(curves)
            xg = np.linspace(0, 10, 201)
            ax.plot(xg, np.median(curves, 0), color=MODEL_COLORS[m], lw=1.6, label=LABEL[m])
            ax.fill_between(xg, curves.min(0), curves.max(0), color=MODEL_COLORS[m], alpha=0.12, lw=0)
        ax.plot([0, 10], [0, 0.1], 'k--', lw=0.8, label='random')
        ax.set_xlim(0, 10); ax.set_ylim(0, 1)
        ax.set_xlabel('top-ranked share of labelled pool (%)')
        ax.set_title(title, loc='left', fontsize=10)
    axes[0].set_ylabel('share of held-out occurrences captured')
    axes[1].legend(frameon=False, fontsize=8, loc='lower right', bbox_to_anchor=(1.0, 0.15))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_capture_superior.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


_Y = {}


def _y_superior():
    if 'y' not in _Y:
        from paper_analyses.common import load_region
        _Y['y'] = load_region('Superior')['pruned']['y']
    return _Y['y']


# ── Protocol audit ──────────────────────────────────────────────────────────
def audit():
    a = pd.read_csv(o('protocol_audit.csv'))
    sim = pd.read_csv(o('protocol_audit_sibling_similarity.csv')).set_index('region')
    lines = []
    for r in REGIONS:
        sub = a[a.region == r]
        first = True
        for neg in ['cosine', 'random']:
            for split in ['child-level random', 'grouped by parent', 'no augmentation']:
                x = sub[(sub.negatives == neg) & (sub.split == split)].iloc[0]
                lab = f'\\multirow{{6}}{{*}}{{{r}}}' if first else ''
                first = False
                sib = f'{x.test_pos_with_sibling_in_train * 100:.0f}\\%' if not np.isnan(x.test_pos_with_sibling_in_train) else '--'
                negl = 'cosine' if neg == 'cosine' else 'random'
                short = {'child-level random': 'child-level', 'grouped by parent': 'parent-grouped',
                         'no augmentation': 'no augm.'}[split]
                lines.append(f'{lab} & {negl} & {short} & {int(x.n_pos_labels)} & '
                             f'{x.auc_mean:.3f} [{x.auc_lo:.3f}, {x.auc_hi:.3f}] & {sib} \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_audit.tex', '\n'.join(lines))
    lines = [f'{r} & {sim.loc[r, "sibling_dist_med"]:.2f} & {sim.loc[r, "nearest_other_occurrence_dist_med"]:.2f} & '
             f'{sim.loc[r, "nearest_background_neighbour_dist_med"]:.2f} \\\\' for r in REGIONS]
    write('tab_audit_similarity.tex', '\n'.join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    splits = ['child-level random', 'grouped by parent', 'no augmentation']
    for ax, r in zip(axes, REGIONS):
        sub = a[a.region == r]
        for k, neg in enumerate(['cosine', 'random']):
            s = sub[sub.negatives == neg].set_index('split').loc[splits]
            xs = np.arange(3) + (k - 0.5) * 0.3
            ax.bar(xs, s.auc_mean, width=0.28, color=['#555555', '#bbbbbb'][k],
                   label='cosine-dissimilar negatives' if neg == 'cosine' else 'random negatives')
            ax.errorbar(xs, s.auc_mean, yerr=[s.auc_mean - s.auc_lo, s.auc_hi - s.auc_mean], fmt='none',
                        ecolor='k', lw=0.8, capsize=2)
        ax.set_xticks(np.arange(3), ['child-level\n(as published)', 'grouped by\nparent', 'no\naugmentation'])
        ax.set_ylim(0.2, 1.02)
        ax.set_title(r, loc='left', fontsize=10)
    axes[0].set_ylabel('test AUC (mean, 95% interval; 50 splits)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.06))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_protocol_audit.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Augmentation ────────────────────────────────────────────────────────────
def augmentation():
    ag = pd.read_csv(o('augmentation_Superior.csv'))
    st = load_pickle(o('augmentation_Superior.pkl'))
    lines = []
    names = {'jitter': 'Gaussian jitter', 'smote': 'SMOTE', 'wgan': 'WGAN-GP'}

    def per_repeat(store, key, ref_key):
        """Paired per-repeat AP difference on identical folds: median [min, max], wins."""
        d = store[key]['aps'] - store[ref_key]['aps']
        return f'{np.median(d):+.3f} [{d.min():+.3f}, {d.max():+.3f}]', f'{int((d > 0).sum())}/{len(d)}'

    for model in ['DevNet', 'RF']:
        base = ag[(ag.model == model) & (ag.method == 'none')].iloc[0]
        lines.append(f'\\multirow{{10}}{{*}}{{{model}}} & none & -- & {base.auc_med:.3f} & '
                     f'{fmt_ci(base.ap_med, base.ap_lo, base.ap_hi)} & -- & -- \\\\')
        for meth in ['jitter', 'smote', 'wgan']:
            for mult in [1, 3, 10]:
                x = ag[(ag.model == model) & (ag.method == meth) & (ag.mult == mult)].iloc[0]
                dap, wins = per_repeat(st, (model, meth, mult), (model, 'none', 0))
                lines.append(f' & {names[meth]} & {mult}$\\times$ & {x.auc_med:.3f} & '
                             f'{fmt_ci(x.ap_med, x.ap_lo, x.ap_hi)} & {dap} & {wins} \\\\')
        lines.append('\\midrule' if model == 'DevNet' else '')
    write('tab_augmentation.tex', '\n'.join(lines))

    if os.path.exists(o('augmentation_spatial_Superior.pkl')):
        sp = load_pickle(o('augmentation_spatial_Superior.pkl'))
        spc = pd.read_csv(o('augmentation_spatial_Superior.csv'))
        lines = []
        for block in sorted(spc.block_km.unique()):
            for model in ['DevNet', 'RF']:
                sub = spc[(spc.block_km == block) & (spc.model == model)]
                first = True
                for _, x in sub.iterrows():
                    lab = f'{block:.0f} km & {model}' if first else ' & '
                    first = False
                    if x.method == 'none':
                        lines.append(f'{lab} & none & {x.ap_med:.3f} [{x.ap_lo:.3f}, {x.ap_hi:.3f}] & -- & -- \\\\')
                    else:
                        dap, wins = per_repeat(sp, (block, model, x.method, int(x.mult)), (block, model, 'none', 0))
                        lines.append(f'{lab} & {names[x.method]} {int(x.mult)}$\\times$ & '
                                     f'{x.ap_med:.3f} [{x.ap_lo:.3f}, {x.ap_hi:.3f}] & {dap} & {wins} \\\\')
                lines.append('\\addlinespace')
        write('tab_augmentation_spatial.tex', '\n'.join(lines[:-1]))

    dg = pd.read_csv(o('augmentation_diagnostics_Superior.csv'))
    src_names = {'real positives (LOO)': 'Real occurrences (leave-one-out)', 'jitter': 'Gaussian jitter',
                 'smote': 'SMOTE', 'wgan': 'WGAN-GP', 'background (300 random)': 'Background (300 random points)'}
    lines = []
    for _, x in dg.iterrows():
        share = '--' if np.isnan(x.share_nn_is_positive) else f'{x.share_nn_is_positive * 100:.0f}\\%'
        p10 = '--' if np.isnan(x.nn_ratio_p10) else f'{x.nn_ratio_p10:.2f}'
        lines.append(f'{src_names[x.source]} & {x.nn_ratio_med:.2f} & {p10} & {share} & {x.devnet_pct_med:.3f} \\\\')
    write('tab_augmentation_diag.tex', '\n'.join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for ax, model in zip(axes, ['DevNet', 'RF']):
        base = ag[(ag.model == model) & (ag.method == 'none')].iloc[0]
        ax.axhspan(base.ap_lo, base.ap_hi, color='#dddddd', zorder=0)
        ax.axhline(base.ap_med, color='k', lw=1, label='no augmentation')
        for k, meth in enumerate(['jitter', 'smote', 'wgan']):
            s = ag[(ag.model == model) & (ag.method == meth)].sort_values('mult')
            xs = np.arange(3) + (k - 1) * 0.15
            ax.errorbar(xs, s.ap_med, yerr=[s.ap_med - s.ap_lo, s.ap_hi - s.ap_med], fmt='o-', ms=4, capsize=2,
                        color=['#0072b2', '#009e73', '#e69f00'][k], label=names[meth])
        ax.set_xticks(np.arange(3), ['1$\\times$', '3$\\times$', '10$\\times$'])
        ax.set_xlabel('synthetic positives per real training positive')
        ax.set_title(f'({"ab"[["DevNet", "RF"].index(model)]}) {model}', loc='left', fontsize=10)
    axes[0].set_ylabel('average precision (median, 95% interval)')
    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_augmentation_performance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── SHAP table ──────────────────────────────────────────────────────────────
def shap_table():
    for r in REGIONS:
        t = pd.read_csv(o(f'shap_top10_{r}.csv'))
        lines = []
        for _, x in t.iterrows():
            def d(rho, lab):
                arrow = {'higher -> more': '$\\uparrow$', 'higher -> less': '$\\downarrow$',
                         'present -> more': '$\\uparrow$', 'present -> less': '$\\downarrow$'}.get(lab, '$\\circ$')
                return f'{arrow} ({rho:+.2f})'
            lines.append(f'{x.label} & {int(x.dn_rank)} & {d(x.dn_rho, x.dn_direction)} & '
                         f'{int(x.rf_rank)} & {d(x.rf_rho, x.rf_direction)} \\\\')
        write(f'tab_shap_{r}.tex', '\n'.join(lines))


# ── Geography-only and smooth baselines (Stage 8) ───────────────────────────
def geography():
    diag = pd.concat([pd.read_csv(o(f'devnet_diagnostics_{r}.csv')) for r in REGIONS], ignore_index=True)
    bm = pd.read_csv(o('benchmark_Superior_Cordillera_Grenville.csv'))
    bm = bm[bm.feature_set == 'pruned'].assign(protocol='random')
    sp = pd.read_csv(o('spatial_cv_Superior_Cordillera_Grenville.csv'))
    sp = sp[sp.block_km == 100].assign(protocol='100 km')
    ref = pd.concat([bm, sp], ignore_index=True)
    names = [('RF_xy', 'RF, coordinates only', diag), ('RF', 'RF, all features', ref),
             ('LogReg', 'Logistic regression', diag), ('DevNet', 'DevNet', ref)]
    lines = []
    for r in REGIONS:
        first = True
        for key, label, src in names:
            cells = []
            for proto in ['random', '100 km']:
                x = src[(src.region == r) & (src.model == key) & (src.protocol == proto)].iloc[0]
                cells += [f'{x.ap_med:.3f}', f'{x.auc_med:.3f}']
            lab = f'\\multirow{{4}}{{*}}{{{r}}}' if first else ''
            first = False
            lines.append(f'{lab} & {label} & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_geography.tex', '\n'.join(lines))


# ── Post-diagnostic DevNet sensitivity analyses (Stages 9-10) ────────────────
def sensitivity():
    qt = pd.read_csv(o('devnet_qt_sensitivity.csv')).assign(variant='DevNet-QT')
    parts = [qt]
    for v in ['bg', 'es']:
        parts += [pd.read_csv(f) for f in sorted(__import__('glob').glob(o(f'training_budget_{v}_*.csv')))]
    sv = pd.concat(parts, ignore_index=True)
    variants = ['DevNet-QT', 'DevNet-BG', 'DevNet-ES']
    protos = ['random', '50 km', '100 km', '200 km']
    lines = []
    for r in REGIONS:
        for k, proto in enumerate(protos):
            sub = sv[(sv.region == r) & (sv.protocol == proto)].set_index('variant')
            base = sub.iloc[0]
            vals = {'RF': base.rf_ap_med, 'DevNet': base.devnet_ap_med}
            vals.update({v: sub.loc[v, 'ap_med'] for v in variants})
            best = max(vals.values())
            fmt = lambda v: f'\\textbf{{{v:.3f}}}' if v == best else f'{v:.3f}'
            cells = [fmt(vals['RF']), fmt(vals['DevNet'])]
            cells += [f"{fmt(vals[v])} ({sub.loc[v, 'wins_vs_dn']})" for v in variants]
            lab = f'\\multirow{{4}}{{*}}{{{r}}}' if k == 0 else ''
            name = 'random' if proto == 'random' else f'{proto} blocks'
            lines.append(f'{lab} & {name} & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_sensitivity.tex', '\n'.join(lines))

    lines = []
    sup = sv[sv.region == 'Superior']
    for proto in protos:
        for v in variants:
            x = sup[(sup.protocol == proto) & (sup.variant == v)].iloc[0]
            name = 'random' if proto == 'random' else proto
            short = v.replace('DevNet-', '')
            lines.append(f'{name} & {short} & {x.d_ap_vs_dn:+.3f} [{x.d_ap_vs_dn_lo:+.3f}, {x.d_ap_vs_dn_hi:+.3f}] & '
                         f'{x.p_vs_dn:.3f} & {x.wins_vs_dn} & {x.d_ap_vs_rf:+.3f} [{x.d_ap_vs_rf_lo:+.3f}, '
                         f'{x.d_ap_vs_rf_hi:+.3f}] & {x.p_vs_rf:.3f} & {x.wins_vs_rf} \\\\')
        if proto != protos[-1]:
            lines.append('\\addlinespace')
    write('tab_sensitivity_superior.tex', '\n'.join(lines))


# ── Workflow schematic ──────────────────────────────────────────────────────
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
    box(6.5, 4.3, 3.4, 1.8, '3  Models (shared background)', 'RF: background = negative\nBagging-PU, nnPU: positive-unlabelled\nDevNet: deviation loss,\nbackground = unlabelled reference\nidentical folds & points for all', '#fdf0e6')
    box(0.1, 1.9, 3.1, 2.0, '4  Validation', 'pooled out-of-fold ranks\n(a) random stratified CV, 5$\\times$6\n(b) spatial blocks 50/100/200 km,\n     background purged from test blocks\nAUC, AP (+95% interval),\npaired bootstrap, capture rate', '#eaf6ee')
    box(3.5, 1.9, 3.0, 2.0, '5  Augmentation (fold-safe)', 'jitter / SMOTE / WGAN-GP\nfit on training-fold positives only\n1$\\times$, 3$\\times$, 10$\\times$ synthetic positives\nheld-out positives always real\n+ memorisation diagnostics', '#eaf6ee')
    box(6.8, 1.9, 3.1, 2.0, '6  Protocol audit', 'S2 level-12 $\\rightarrow$ level-13 children\ncosine-dissimilar vs random negatives\nchild-level vs parent-grouped split\n50 $\\times$ 70/30 balanced splits\n(design of Parsa et al. 2024)', '#eaf6ee')
    box(1.6, 0.1, 6.8, 1.4, '7  Interpretation and maps', 'TreeSHAP / DeepSHAP (ensemble mean), direction = Spearman $\\rho$(feature, SHAP)\nproduction models $\\rightarrow$ percentile prospectivity maps; external check against the Kapuskasing Structural Zone', '#f5eefa')
    arrow(2.7, 5.2, 3.0, 5.2); arrow(6.2, 5.2, 6.5, 5.2)
    arrow(8.2, 4.3, 1.65, 3.9); arrow(8.2, 4.3, 5.0, 3.9); arrow(8.2, 4.3, 8.35, 3.9)
    arrow(1.65, 1.9, 3.0, 1.5); arrow(5.0, 1.9, 5.0, 1.5)
    fig.savefig(os.path.join(FIG, 'fig_workflow.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    for fn in [workflow, table_benchmark, spatial, geography, sensitivity, capture, audit, augmentation,
               shap_table]:
        try:
            fn()
        except FileNotFoundError as e:
            print(f'skip {fn.__name__}: {e}')
