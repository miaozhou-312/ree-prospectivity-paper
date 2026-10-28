"""
Figures and LaTeX table bodies for the FIFTH revision
(paper/manuscript/main_src_revised5.tex): the fourth revision with

  * a national (Canada) part in front of the provincial one, built on one
    label table, one feature rule and one evaluation (Stages 40-49): the
    Parsa et al. (2024) ladder, the national benchmark, transfer between
    provinces, the pooling test, the learning curve and the tests of why
    DevNet is weak nationally;
  * XGBoost (fixed parameters of J. Fang's national study, Stage 47) in the
    provincial benchmark, blocked-validation and paired-comparison tables.

Only tables and figures that differ from the fourth revision are written
(generated_revised5/, figures/*_r5.png); `python paper/build.py revised5`
falls back to generated_revised4/, _revised3/ and _revised2/ for the rest.

    python -m paper_analyses.06f_figures_tables_revised5
"""
import ast
import glob
import importlib
import os
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paper_analyses.common import config, load_pickle, load_region  # noqa: E402
from paper_analyses.protocol import paired_bootstrap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, 'paper_analyses', 'out')
NAT = os.path.join(BASE, 'national')
FIG = os.path.join(REPO, 'paper', 'manuscript', 'figures')
GEN = os.path.join(REPO, 'paper', 'manuscript', 'generated_revised5')
os.makedirs(GEN, exist_ok=True)
REGIONS = ['Superior', 'Cordillera', 'Grenville']
MODELS = ['DevNet', 'RF', 'XGBoost', 'BaggingPU', 'nnPU']
LABEL = {'DevNet': 'DevNet', 'DevNet-BCE': 'DevNet-BCE', 'MLP-BCE': 'DevNet-BCE', 'RF': 'RF', 'XGBoost': 'XGBoost',
         'BaggingPU': 'Bagging-PU', 'nnPU': 'nnPU', 'RF-Parsa': 'RF (Parsa et al.)',
         'DevNet-scaled': 'DevNet, steps scaled'}
COLORS = {'DevNet': '#d03b3b', 'RF': '#2a78d6', 'XGBoost': '#1a9bb0', 'BaggingPU': '#3a9e5c', 'nnPU': '#8a5cc2'}
MARKERS = {'DevNet': 'o', 'RF': 's', 'XGBoost': 'D', 'BaggingPU': '^', 'nnPU': 'P'}
PROTOS = [0, 50, 100, 200]
PNAME = {0: 'random', 50: '50 km', 100: '100 km', 200: '200 km'}
NPROTOS = ['random', '50 km', '100 km', '200 km']
GROUPS = ['Superior', 'Cordillera', 'Grenville', 'Other']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f:
        f.write(text)
    print('wrote', name)


def nat(name):
    return pd.read_csv(os.path.join(NAT, name))


def f3(v):
    return '--' if pd.isna(v) else f'{v:.3f}'


def bold_if(v, best, fmt='{:.3f}'):
    s = fmt.format(v)
    return f'\\textbf{{{s}}}' if np.isclose(v, best) else s


def ci3(d, lo, hi):
    return f'{d:+.3f} [{lo:+.3f}, {hi:+.3f}]'


# ----------------------------------------------------------- provincial tables with XGBoost
def regional():
    t = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(BASE, 'reencoded_benchmark_*.csv'))],
                  ignore_index=True)
    t = t[t['keep'].astype(str) == '0']
    x = pd.read_csv(os.path.join(BASE, 'xgboost_regional.csv'))
    return pd.concat([t, x[x.featureset == 'reencoded']], ignore_index=True)


def provincial_tables():
    t = regional()
    lines = []
    for r in REGIONS:
        rnd = t[(t.region == r) & (t.block_km == 0)].set_index('model')
        best_ap, best_auc = rnd.loc[MODELS, 'ap_med'].max(), rnd.loc[MODELS, 'auc_med'].max()
        npos = int(load_region(r)['pruned']['y'].sum())
        for k, m in enumerate(MODELS):
            x = rnd.loc[m]
            lab = f'\\multirow{{{len(MODELS)}}}{{*}}{{{r} ({npos})}}' if k == 0 else ''
            ap = bold_if(x.ap_med, best_ap) + f' [{x.ap_lo:.3f}, {x.ap_hi:.3f}]'
            lines.append(f"{lab} & {LABEL[m]} & {bold_if(x.auc_med, best_auc)} & {ap} \\\\")
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_benchmark.tex', '\n'.join(lines))
    for metric, name in [('ap_med', 'tab_spatial.tex'), ('auc_med', 'tab_spatial_auc.tex')]:
        lines = []
        for r in REGIONS:
            for k, b in enumerate(PROTOS):
                s = t[(t.region == r) & (t.block_km == b)].set_index('model')
                best = s.loc[MODELS, metric].max()
                lab = f'\\multirow{{4}}{{*}}{{{r}}}' if k == 0 else ''
                lines.append(f"{lab} & {PNAME[b]} & " + ' & '.join(bold_if(s.loc[m, metric], best) for m in MODELS)
                             + ' \\\\')
            lines.append('\\midrule' if r != REGIONS[-1] else '')
        write(name, '\n'.join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    xs = np.arange(4)
    for ax, r in zip(axes, REGIONS):
        for k, m in enumerate(MODELS):
            s = t[(t.region == r) & (t.model == m)].set_index('block_km').reindex(PROTOS)
            ax.errorbar(xs + (k - (len(MODELS) - 1) / 2) * 0.1, s.ap_med, yerr=[s.ap_med - s.ap_lo, s.ap_hi - s.ap_med],
                        fmt='-', marker=MARKERS[m], color=COLORS[m], ms=4, lw=1.3, capsize=2, label=LABEL[m])
        ax.set_xticks(xs, [PNAME[p] for p in PROTOS])
        ax.set_title(r, loc='left')
        ax.set_xlabel('cross-validation split')
    axes[0].set_ylabel('average precision (median, 95% interval)')
    axes[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_spatial_cv_r5.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def xgb_boot(region, proto):
    """DevNet - XGBoost on repeat-averaged ranks (the paired comparison of the
    main tables, computed here because XGBoost was added later)."""
    y = load_region(region)['pruned']['y']
    a = load_pickle(os.path.join(BASE, f'reencoded_benchmark_{region}.pkl'))[(region, 'DevNet', proto, None)]
    pn = 'random' if proto == 'random' else f'{proto:.0f} km'
    b = load_pickle(os.path.join(BASE, 'xgboost_regional.pkl'))[('reencoded', region, pn)]
    bt = paired_bootstrap(y, a['oof_mean'], b['oof_mean'], seed=config.RANDOM_SEED)
    return {**bt, 'wins': f"{int((a['aps'] > b['aps']).sum())}/{len(a['aps'])}"}


def comparisons():
    def boot(region):
        b = pd.read_csv(os.path.join(BASE, f'reencoded_bootstrap_{region}.csv'))
        for k in ('d_ap_ci', 'd_auc_ci'):
            b[k] = b[k].map(ast.literal_eval)
        return b
    lines, rows = [], []
    for r in REGIONS:
        b = boot(r)
        b = b[b.protocol == 'random']
        for m in ['RF', 'XGBoost', 'BaggingPU', 'nnPU']:
            x = xgb_boot(r, 'random') if m == 'XGBoost' else b[b.comparison == f'DevNet - {m}'].iloc[0]
            rows.append({'region': r, 'protocol': 'random', 'comparison': f'DevNet - {m}', 'd_ap': x['d_ap'],
                         'wins': x['wins']})
            lines.append(f"{r} & DevNet $-$ {LABEL[m]} & {ci3(x['d_ap'], *x['d_ap_ci'])} & {x['wins']} & "
                         f"{ci3(x['d_auc'], *x['d_auc_ci'])} \\\\")
        lines.append('\\addlinespace')
    write('tab_bootstrap.tex', '\n'.join(lines[:-1]))
    b = boot('Superior')
    lines = []
    for blk in [50, 100, 200]:
        for m in ['RF', 'XGBoost', 'BaggingPU', 'nnPU']:
            if m == 'XGBoost':
                x = xgb_boot('Superior', float(blk))
            else:
                x = b[(b.block_km == blk) & (b.comparison == f'DevNet - {m}') & ~b.protocol.str.contains('k=')].iloc[0]
            rows.append({'region': 'Superior', 'protocol': f'{blk} km', 'comparison': f'DevNet - {m}',
                         'd_ap': x['d_ap'], 'wins': x['wins']})
            lines.append(f"{blk} km & DevNet $-$ {LABEL[m]} & {ci3(x['d_ap'], *x['d_ap_ci'])} & {x['wins']} \\\\")
        lines.append('\\addlinespace')
    write('tab_spatial_bootstrap.tex', '\n'.join(lines[:-1]))
    pd.DataFrame(rows).to_csv(os.path.join(NAT, 'r5_comparisons.csv'), index=False)


# ----------------------------------------------------------------------------- national tables
def ladder():
    lad = nat('tab_ladder.csv')
    lad['held'] = lad.design.str.replace('leave one province out: ', '', regex=False)
    L = lad[lad.model == 'RF-Parsa'].drop_duplicates('rung').set_index('rung')
    steps = [('R0', 'Design of Parsa et al.: four L13 children per occurrence cell; negatives from the 5\\% least '
                    'cosine-similar cells; child-level random 70/30', 'balanced'),
             ('R1', 'Negatives at random ($>$10~km from occurrences)', 'balanced'),
             ('R2', 'Children of one parent cell kept on one side', 'balanced'),
             ('R2c', '\\quad control: R2 with cosine-dissimilar negatives', 'balanced'),
             ('R3', 'No augmentation: one label per occurrence cell', 'balanced'),
             ('R3c', '\\quad control: R3 with cosine-dissimilar negatives', 'balanced'),
             ('R4', 'R3 models scored against realistic background', '0.3\\%'),
             ('R4c', '\\quad control: R3c models against realistic background', '0.3\\%'),
             ('R5', 'Trained on the occurrence cells and a shared background sample', '0.3\\%'),
             ('R6', '100~km spatial blocks, background purged', '0.3\\%')]
    lines = []
    for r, what, test in steps:
        row = L.loc[r]
        bal = test == 'balanced'
        auc = row['auc_balanced_mean'] if bal else row['auc_mean']
        ap = row['ap_balanced_mean'] if bal else row['ap_mean']
        sib = f"{100 * row['sibling_in_train_mean']:.0f}\\%" if pd.notna(row.get('sibling_in_train_mean')) else ''
        lines.append(f"{r} & {what} & {test} & {auc:.3f} & {ap:.3f} & {sib} \\\\")
    r7 = lad[(lad.rung == 'R7') & (lad.model == 'RF-Parsa')].set_index('held')
    prev = nat('tab_transfer.csv').query("design == 'leave-one-province-out'").groupby('test').prevalence.first()
    for k, g in enumerate(GROUPS):
        what = 'Leave one province group out: ' if k == 0 else '\\quad '
        name = 'rest of Canada' if g == 'Other' else g
        lines.append(f"{'R7' if k == 0 else ''} & {what}{name} held out & {100 * prev[g]:.1f}\\% & "
                     f"{r7.loc[g, 'auc_mean']:.3f} & {r7.loc[g, 'ap_mean']:.3f} & \\\\")
    write('tab_nat_ladder.tex', '\n'.join(lines))
    lines = []
    for m in ['RF-Parsa', 'RF', 'XGBoost', 'BaggingPU', 'DevNet']:
        z = lad[lad.model == m].drop_duplicates(['rung', 'held']).set_index(['rung', 'held'])
        r5 = z.xs('R5', level='rung').iloc[0]
        r6 = z.xs('R6', level='rung').iloc[0]
        cells = [f"{r5.ap_mean:.3f} ({r5.auc_mean:.2f})", f"{r6.ap_mean:.3f} ({r6.auc_mean:.2f})"]
        cells += [f"{z.loc[('R7', g), 'ap_mean']:.3f} ({z.loc[('R7', g), 'auc_mean']:.2f})" for g in GROUPS[:3]]
        lines.append(f"{LABEL[m]} & " + ' & '.join(cells) + ' \\\\')
    write('tab_nat_ladder_models.tex', '\n'.join(lines))


def benchmark():
    b = nat('tab_benchmark.csv')
    c = b[b.scope == 'Canada'].set_index(['model', 'protocol'])
    models = ['DevNet', 'DevNet-BCE', 'RF', 'XGBoost', 'BaggingPU', 'nnPU']
    lines = []
    for p in NPROTOS:
        best = max(c.loc[(m, p), 'ap_med'] for m in models)
        cells = [f"{bold_if(c.loc[(m, p), 'ap_med'], best)} ({c.loc[(m, p), 'auc_med']:.2f})" for m in models]
        lines.append(f"{p} & " + ' & '.join(cells) + ' \\\\')
    write('tab_nat_benchmark.tex', '\n'.join(lines))
    t = regional()
    s = b[b.scope == 'Superior'].set_index(['model', 'protocol'])
    lines = []
    for m in ['DevNet', 'RF', 'XGBoost', 'BaggingPU', 'nnPU']:
        cells = []
        for p, blk in [('random', 0), ('100 km', 100)]:
            rr = t[(t.region == 'Superior') & (t.model == m) & (t.block_km == blk)].iloc[0].ap_med
            cells += [f'{rr:.3f}', f"{s.loc[(m, p), 'ap_med']:.3f}"]
        lines.append(f"{LABEL[m]} & " + ' & '.join(cells) + ' \\\\')
    write('tab_nat_superior.tex', '\n'.join(lines))


def transfer():
    tr = nat('tab_transfer.csv')
    lo = tr[tr.design == 'leave-one-province-out']
    lines = []
    for m in ['DevNet', 'DevNet-BCE', 'RF', 'XGBoost', 'BaggingPU', 'nnPU']:
        z = lo[lo.model == m].set_index('test')
        lines.append(f"{LABEL[m]} & " + ' & '.join(f"{z.loc[g, 'ap_med']:.3f} ({z.loc[g, 'auc_med']:.2f})"
                                                     for g in GROUPS) + ' \\\\')
    prev = lo.groupby('test').prevalence.first()
    lines.append('\\midrule')
    lines.append('No skill & ' + ' & '.join(f'{prev[g]:.4f}' for g in GROUPS) + ' \\\\')
    write('tab_nat_lopo.tex', '\n'.join(lines))


def pooling():
    p = nat('tab_pooling.csv')
    lines = []
    for r in REGIONS:
        models = ['DevNet', 'DevNet-BCE', 'RF', 'XGBoost', 'BaggingPU']
        for k, m in enumerate(models):
            z = p[(p.region == r) & (p.model == m) & (p.protocol == '100 km')].set_index('variant')
            base = z.loc['regional', 'ap']
            lab = f'\\multirow{{{len(models)}}}{{*}}{{{r}}}' if k == 0 else ''
            cells = [f'{base:.3f}']
            for v in ['+background', '+positives', '+both']:
                d = z.loc[v, 'ap'] - base
                cells.append(f"{z.loc[v, 'ap']:.3f} ({d:+.3f})")
            lines.append(f"{lab} & {LABEL[m]} & " + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_nat_pooling.tex', '\n'.join(lines))


def devnet():
    v = nat('tab_devnet_variants.csv')
    v = v[v.scope == 'Canada'].set_index(['model', 'variant', 'protocol'])
    o = nat('tab_devnet_offsets.csv').set_index(['model', 'protocol'])
    lines = []
    for m in ['DevNet', 'DevNet-BCE', 'RF']:
        cells = []
        for var in ['as in N1', 'local reference', 'province one-hot']:
            cells.append(' / '.join(f"{v.loc[(m, var, p), 'ap_med']:.3f}" for p in ['random', '100 km']))
        cells.append(f"{o.loc[(m, 'random'), 'ap_within_group']:.3f}")
        cells.append(f"{100 * (o.loc[(m, 'random'), 'bg_top10_Grenville'] + o.loc[(m, 'random'), 'bg_top10_Cordillera']):.0f}\\%")
        lines.append(f"{LABEL[m]} & " + ' & '.join(cells) + ' \\\\')
    write('tab_nat_devnet.tex', '\n'.join(lines))


def audit():
    a = pd.read_csv(os.path.join(NAT, 'tab_feature_audit.csv'))
    lines = []
    for r in REGIONS:
        for k, (m, p) in enumerate([(m, p) for m in ['DevNet', 'RF'] for p in ['random', '100 km']]):
            z = a[(a.region == r) & (a.model == m) & (a.protocol == p)].iloc[0]
            lab = f'\\multirow{{4}}{{*}}{{{r}}}' if k == 0 else ''
            lines.append(f"{lab} & {LABEL[m]} & {p} & " + ' & '.join(f'{z[c]:.3f}' for c in 'ABCD') + ' \\\\')
        lines.append('\\midrule' if r != REGIONS[-1] else '')
    write('tab_nat_audit.tex', '\n'.join(lines))


def curve():
    c = nat('tab_curve.csv')
    models = ['DevNet', 'DevNet-BCE', 'RF', 'XGBoost', 'BaggingPU']
    lines = []
    kmax = c.k_train_pos.max()
    for k in sorted(c.k_train_pos.unique()):
        z = c[c.k_train_pos == k].set_index('model')
        lines.append(f"{k}{' (all)' if k == kmax else ''} & " + ' & '.join(f"{z.loc[m, 'ap_med']:.3f}" for m in models)
                     + ' \\\\')
    write('tab_nat_curve.tex', '\n'.join(lines))


def figures():
    for src in ['fig_national_map', 'fig_pooling', 'fig_curve', 'fig_devnet_national']:
        shutil.copy(os.path.join(NAT, f'{src}.png'), os.path.join(FIG, f'{src}_r5.png'))
        print('copied', f'{src}_r5.png')
    s46 = importlib.import_module('paper_analyses.46_national_summary')    # paper layouts, larger type
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 10.5}):
        s46.fig_ladder(s46.ladder_table(), grid=True, path=os.path.join(FIG, 'fig_ladder_r5.png'))
        s46.fig_transfer(s46.transfer_table(), lopo=False, path=os.path.join(FIG, 'fig_transfer_r5.png'))


def workflow():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.8)
    ax.axis('off')

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', fc=fc, ec='#333333', lw=1))
        ax.text(x + w / 2, y + h - 0.22, title, ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(x + w / 2, y + h - 0.55, body, ha='center', va='top', fontsize=8.2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1))
    box(0.1, 4.6, 3.0, 2.1, '1  Labels and layers', '68 occurrences in Canada\n(one table, 2 km merge)\nnational potential fields,\nLAB, Moho, faults, worms\nH3 geology grid', '#eef3fb')
    box(3.4, 4.6, 3.3, 2.1, '2  Inputs: one rule, two scales', 'rasters, GLCM grey levels and\ngeneralised geology taken per\ngeological province\npruned on each training table\nsensitivity: no map classes', '#eef3fb')
    box(7.0, 4.6, 2.9, 2.1, '3  Models', 'DevNet (deviation loss)\nablation: DevNet-BCE\nRF, XGBoost, tuned RF, HGB\nPU: bagging-PU, nnPU', '#fdf0e6')
    box(0.1, 2.0, 3.1, 2.25, '4  National scale', 'benchmark: random CV,\n50/100/200 km blocks\nleave one province out\nprovince-to-province transfer\npooling test, learning curve\nwhy DevNet is weak', '#eaf6ee')
    box(3.5, 2.0, 3.0, 2.25, '5  Parsa et al. ladder', 'S2 level-12 grid\n(1.62 million cells)\none fix per rung:\nnegatives, siblings,\naugmentation, prevalence,\nblocks, provinces', '#eaf6ee')
    box(6.8, 2.0, 3.1, 2.25, '6  Provinces', 'Superior (15 occurrences)\nCordillera, Grenville\nloss ablation, three tests\nof the unlabelled premise\nsynthetic positives,\ncoordinates-only baseline', '#eaf6ee')
    box(1.6, 0.1, 6.8, 1.5, '7  Interpretation, maps and checks', 'TreeSHAP / DeepSHAP with a rank-based direction rule\npercentile maps, leave-one-occurrence-out stability\nClay-Howells resource and Kapuskasing Structural Zone (not used in training)', '#f5eefa')
    arrow(3.1, 5.6, 3.4, 5.6)
    arrow(6.7, 5.6, 7.0, 5.6)
    arrow(8.4, 4.6, 1.65, 4.25)
    arrow(8.4, 4.6, 5.0, 4.25)
    arrow(8.4, 4.6, 8.35, 4.25)
    arrow(8.35, 2.0, 7.0, 1.6)
    fig.savefig(os.path.join(FIG, 'fig_workflow_r5.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    for fn in [provincial_tables, comparisons, ladder, benchmark, transfer, pooling, devnet, audit, curve, figures,
               workflow]:
        fn()
