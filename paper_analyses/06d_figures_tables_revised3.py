"""
Figures and LaTeX table bodies for the THIRD revision
(paper/manuscript/main_src_revised3.tex).

Same analyses as the second revision (feature set 3 as the main analysis),
restructured: the main comparison is DevNet against established
prospectivity learners (RF, tuned RF and gradient boosting, bagging-PU,
nnPU), and MLP-BCE becomes an ablation of DevNet -- "DevNet-BCE", DevNet
with its deviation loss replaced by binary cross-entropy -- reported in its
own section together with the three tests of the semi-supervised premise
(Stages 32-33).

Only tables and figures that differ from the second revision are written
here (generated_revised3/, figures/*_r3.png). `python paper/build.py
revised3` falls back to generated_revised2/ for every other table, and the
manuscript reuses the *_r2 figures that did not change. Earlier versions
are untouched.

    python -m paper_analyses.06d_figures_tables_revised3
"""
import ast
import importlib
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

g = importlib.import_module('paper_analyses.06c_figures_tables_revised2')
GEN = os.path.join(g.REPO, 'paper', 'manuscript', 'generated_revised3')
os.makedirs(GEN, exist_ok=True)
g.GEN = GEN
g.SUFFIX = '_r3'
g.MODELS = ['DevNet', 'RF', 'BaggingPU', 'nnPU']
g.LABEL['MLP-BCE'] = 'DevNet-BCE'
g.WORKFLOW_MODELS = 'DevNet (deviation loss,\nquantile inputs, 16+240 rows)\nbaselines: RF, tuned RF, HGB\nPU: bagging-PU, nnPU'
g.WORKFLOW_VALIDATION = ('pooled out-of-fold ranks\nrandom CV and 50/100/200 km\nblocks, background purged\n'
                         'paired bootstrap, Holm\nnested design selection\nDevNet loss ablation (4 tests)')
o, write, fp = g.o, g.write, g.fp
PNAME = {0: 'random', 50: '50 km', 100: '100 km', 200: '200 km'}
SETS = [('1', 'Set 1'), ('2', 'Set 2'), ('3', 'Set 3')]


def ci_lo_hi(s):
    lo, hi = ast.literal_eval(s)
    return lo, hi


def loss_ablation():
    """DevNet against DevNet-BCE (identical network, inputs, batches; loss only), Superior, all labels."""
    rows = []
    c1 = pd.read_csv(o('controls_MLPBCE_RFtuned_HGBtuned_Superior.csv'))
    c1 = c1[c1.model == 'MLP-BCE'].drop_duplicates('block_km')
    for _, x in c1.iterrows():
        rows.append(('1', int(x.block_km), x.devnet_ap_med, x.ap_med, x.d_ap_devnet_minus, x.d_ap_lo, x.d_ap_hi,
                     x.p_ap, x.devnet_wins))
    for fs, prefix in [('2', 'extended'), ('3', 'reencoded')]:
        b = pd.read_csv(o(f'{prefix}_benchmark_Superior.csv'))
        b = b[b['keep'].astype(str) == '0']
        bt = pd.read_csv(o(f'{prefix}_bootstrap_Superior.csv'))
        for blk in [0, 50, 100, 200]:
            ap = lambda m: b[(b.model == m) & (b.block_km == blk)].ap_med.iloc[0]
            x = bt[(bt.block_km == blk) & (bt.comparison == 'DevNet - MLP-BCE') & ~bt.protocol.str.contains('k=')].iloc[0]
            lo, hi = ci_lo_hi(x.d_ap_ci)
            rows.append((fs, blk, ap('DevNet'), ap('MLP-BCE'), x.d_ap, lo, hi, x.p_ap, x.wins))
    lines = []
    for fs, label in SETS:
        sub = [r for r in rows if r[0] == fs]
        for k, (_, blk, dn, bce, d, lo, hi, p, w) in enumerate(sub):
            lab = f'\\multirow{{4}}{{*}}{{{label}}}' if k == 0 else ''
            lines.append(f"{lab} & {PNAME[blk]} & {dn:.3f} & {bce:.3f} & {d:+.3f} [{lo:+.3f}, {hi:+.3f}] & {fp(p)} & {w} \\\\")
        lines.append('\\midrule' if fs != '3' else '')
    write('tab_loss_ablation.tex', '\n'.join(lines))
    pd.DataFrame(rows, columns=['featureset', 'block_km', 'devnet', 'devnet_bce', 'd_ap', 'lo', 'hi', 'p', 'wins']).to_csv(
        o('loss_ablation_Superior.csv'), index=False)


def semisup():
    """Stage 33 summaries -> tables and figure (skipped until Stage 33 has run)."""
    path = o('semisup_summary.csv')
    if not os.path.exists(path):
        print('skip semisup: Stage 33 outputs missing')
        return
    s_all = pd.read_csv(path)
    curves_all = pd.read_csv(o('semisup_curves.csv'))
    s, curves = s_all[s_all.region == 'Superior'], curves_all[curves_all.region == 'Superior']
    fsl = {'reencoded': 'Set 3', 'extended': 'Set 2'}
    lines = []
    for exp, label in [('E1', 'E1, mislabelled ground near occurrences ($n = 0 \\rightarrow 50$): AP change of a / of b'),
                       ('E2', 'E2, unlabelled pool ($625 \\rightarrow 5{,}000$): AP gap a $-$ b with 625 / with 5,000 points'),
                       ('E3', 'E3, label scarcity ($k = 1$--3, averaged): AP of a / of b')]:
        lines.append(f'\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}} \\\\')
        for fs in ['reencoded', 'extended']:
            for proto in ['random', '100 km']:
                for pair in ['DevNet vs DevNet-BCE', 'Bagging-PU vs RF']:
                    x = s[(s.experiment == exp) & (s.featureset == fs) & (s.protocol == proto) & (s.pair == pair)]
                    if not len(x):
                        continue
                    x = x.iloc[0]
                    stat = (f'{x.stat_a:+.3f} / {x.stat_b:+.3f}' if exp != 'E3'
                            else f'{x.stat_a:.3f} / {x.stat_b:.3f}')
                    lines.append(f"{fsl[fs]} & {proto} & {pair.replace(' vs ', ' / ')} & {stat} & "
                                 f"{x.effect:+.3f} ({int(x.wins)}/{int(x.n)}) & {fp(x.p_sign)} \\\\")
        lines.append('\\addlinespace')
    write('tab_semisup.tex', '\n'.join(lines[:-1]))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharey='row')
    colors = {'DevNet': g.COLORS['DevNet'], 'MLP-BCE': g.COLORS['MLP-BCE'], 'nnPU': g.COLORS['nnPU'],
              'RF': g.COLORS['RF'], 'BaggingPU': g.COLORS['BaggingPU']}
    xlab = {'E1': 'near-occurrence points per occurrence', 'E2': 'training-background points',
            'E3': 'training positives kept per fold'}
    for i, fs in enumerate(['reencoded', 'extended']):
        for j, exp in enumerate(['E1', 'E2', 'E3']):
            ax = axes[i, j]
            c = curves[(curves.experiment == exp) & (curves.featureset == fs)]
            for m in ['DevNet', 'MLP-BCE', 'nnPU'] + (['RF', 'BaggingPU'] if exp == 'E1' else []):
                for proto, ls in [('random', '-'), ('100 km', '--')]:
                    q = c[(c.model == m) & (c.protocol == proto)].sort_values('x')
                    ax.plot(np.arange(len(q)), q.ap_med, ls, marker='o', ms=3, lw=1.3, color=colors[m],
                            label=f"{g.LABEL[m]}{'' if proto == 'random' else ' (100 km)'}" if (i, j) == (0, 0) else None)
                    ticks = q.x_label.tolist()
            ax.set_xticks(np.arange(len(ticks)), ticks)
            ax.set_xlabel(xlab[exp])
            ax.set_title(f"({'abcdef'[3 * i + j]}) {exp}, {fsl[fs]}", loc='left', fontsize=9.5)
        axes[i, 0].set_ylabel('Superior AP (median of 20 repeats)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False, fontsize=7.5)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(g.FIG, 'fig_semisup_r3.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def semisup_regions():
    """Cordillera and Grenville: the same three tests (appendix)."""
    path = o('semisup_summary.csv')
    if not os.path.exists(path):
        print('skip semisup_regions: Stage 33 outputs missing')
        return
    s = pd.read_csv(path)
    s = s[s.region.isin(['Cordillera', 'Grenville'])]
    if not len(s):
        print('skip semisup_regions: no Cordillera / Grenville runs')
        return
    fsl = {'reencoded': 'Set 3', 'extended': 'Set 2'}
    lines = []
    for exp, label in [('E1', 'E1, mislabelled ground near occurrences: AP change of a / of b'),
                       ('E2', 'E2, unlabelled pool: AP gap with 625 / with 5,000 points'),
                       ('E3', 'E3, label scarcity ($k = 1$--3): AP of a / of b')]:
        lines.append(f'\\multicolumn{{7}}{{l}}{{\\textit{{{label}}}}} \\\\')
        for region in ['Cordillera', 'Grenville']:
            for fs in ['reencoded', 'extended']:
                for proto in ['random', '100 km']:
                    for pair in ['DevNet vs DevNet-BCE', 'Bagging-PU vs RF']:
                        x = s[(s.region == region) & (s.experiment == exp) & (s.featureset == fs) & (s.protocol == proto)
                              & (s.pair == pair)]
                        if not len(x):
                            continue
                        x = x.iloc[0]
                        stat = (f'{x.stat_a:+.3f} / {x.stat_b:+.3f}' if exp != 'E3' else f'{x.stat_a:.3f} / {x.stat_b:.3f}')
                        lines.append(f"{region[:4]}. & {fsl[fs]} & {proto} & {pair.replace(' vs ', ' / ')} & {stat} & "
                                     f"{x.effect:+.3f} ({int(x.wins)}/{int(x.n)}) & {fp(x.p_sign)} \\\\")
        lines.append('\\addlinespace')
    write('tab_semisup_regions.tex', '\n'.join(lines[:-1]))
    curves = pd.read_csv(o('semisup_curves.csv'))
    colors = {m: g.COLORS[m] for m in ['DevNet', 'MLP-BCE', 'RF', 'BaggingPU']}
    xlab = {'E1': 'near-occurrence points per occurrence', 'E2': 'training-background points',
            'E3': 'training positives kept per fold'}
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 11.5))
    rows = [(r, fs) for r in ['Cordillera', 'Grenville'] for fs in ['reencoded', 'extended']]
    for i, (region, fs) in enumerate(rows):
        for j, exp in enumerate(['E1', 'E2', 'E3']):
            ax = axes[i, j]
            c = curves[(curves.region == region) & (curves.experiment == exp) & (curves.featureset == fs)]
            ticks = []
            for m in ['DevNet', 'MLP-BCE', 'RF', 'BaggingPU']:
                q = c[(c.model == m) & (c.protocol == 'random')].sort_values('x')
                if not len(q):
                    continue
                ax.plot(np.arange(len(q)), q.ap_med, '-', marker='o', ms=3, lw=1.3, color=colors[m],
                        label=g.LABEL[m] if (i, j) == (0, 0) else None)
                ticks = q.x_label.tolist()
            ax.set_xticks(np.arange(len(ticks)), ticks)
            if i == len(rows) - 1:
                ax.set_xlabel(xlab[exp])
            ax.set_title(f"({'abcdefghijkl'[3 * i + j]}) {region}, {fsl[fs]}, {exp}", loc='left', fontsize=9)
        axes[i, 0].set_ylabel('AP (median of 20 repeats)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(g.FIG, 'fig_semisup_regions_r3.png'), dpi=250, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    for fn in [g.benchmark, g.spatial, g.capture, g.featuresets, g.controls, g.occurrence, g.pa, g.workflow,
               loss_ablation, semisup, semisup_regions]:
        fn()
