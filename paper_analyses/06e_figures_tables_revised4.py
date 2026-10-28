"""
Figures and LaTeX table bodies for the FOURTH revision
(paper/manuscript/main_src_revised4.tex), a simplified version of the third:

  * only feature set 3 is shown; "without point-level map classes" (set 2 of
    the earlier revisions) appears as a sensitivity of set 3;
  * no p values or Holm adjustments in the tables: paired differences are
    given with their 95% bootstrap interval and the number of repeats won;
  * capture-rate curves for all four protocols;
  * the tests of the semi-supervised premise condensed into one small table
    (three provinces, random CV) and a two-panel figure;
  * every analysis that earlier revisions ran on feature set 1 (design
    ablation, margin, class prior, nested selection, geography-only and
    linear baselines, synthetic positives, protocol audit) is taken from its
    feature-set-3 rerun (REE_FEATURESET=reencoded, outputs in out/fs3/).

Only tables and figures that differ from the earlier revisions are written
(generated_revised4/, figures/*_r4.png); `python paper/build.py revised4`
falls back to generated_revised3/ and generated_revised2/ for the rest.

    python -m paper_analyses.06e_figures_tables_revised4
"""
import ast
import glob
import importlib
import os
import shutil
import tempfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from paper_analyses.common import load_pickle, load_region  # noqa: E402

g = importlib.import_module('paper_analyses.06c_figures_tables_revised2')
g6b = importlib.import_module('paper_analyses.06b_figures_tables_revised')
REPO = g.REPO
BASE = os.path.join(REPO, 'paper_analyses', 'out')
FS3 = os.path.join(BASE, 'fs3')
FIG = g.FIG
GEN = os.path.join(REPO, 'paper', 'manuscript', 'generated_revised4')
os.makedirs(GEN, exist_ok=True)
MODELS = ['DevNet', 'RF', 'BaggingPU', 'nnPU']
LABEL = {**g.LABEL, 'MLP-BCE': 'DevNet-BCE'}
COLORS = g.COLORS
PNAME = {0: 'random', 50: '50 km', 100: '100 km', 200: '200 km'}
REGIONS = ['Superior', 'Cordillera', 'Grenville']


def o(name):
    return os.path.join(BASE, name)


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f:
        f.write(text)
    print('wrote', name)


def ci3(d, lo, hi):
    return f'{d:+.3f} [{lo:+.3f}, {hi:+.3f}]'


def boot(prefix, region):
    b = pd.read_csv(o(f'{prefix}_bootstrap_{region}.csv'))
    for k in ('d_ap_ci', 'd_auc_ci'):
        b[k] = b[k].map(ast.literal_eval)
    return b


# ------------------------------------------------------- paired comparisons
def comparisons():
    lines = []
    for r in REGIONS:
        b = boot('reencoded', r)
        b = b[b.protocol == 'random']
        for m in ['RF', 'BaggingPU', 'nnPU']:
            x = b[b.comparison == f'DevNet - {m}'].iloc[0]
            lines.append(f"{r} & DevNet $-$ {LABEL[m]} & {ci3(x.d_ap, *x.d_ap_ci)} & {x.wins} & "
                         f"{ci3(x.d_auc, *x.d_auc_ci)} \\\\")
        lines.append('\\addlinespace')
    write('tab_bootstrap.tex', '\n'.join(lines[:-1]))
    b = boot('reencoded', 'Superior')
    lines = []
    for blk in [50, 100, 200]:
        for m in ['RF', 'BaggingPU', 'nnPU']:
            x = b[(b.block_km == blk) & (b.comparison == f'DevNet - {m}') & ~b.protocol.str.contains('k=')].iloc[0]
            lines.append(f"{blk} km & DevNet $-$ {LABEL[m]} & {ci3(x.d_ap, *x.d_ap_ci)} & {x.wins} \\\\")
        lines.append('\\addlinespace')
    write('tab_spatial_bootstrap.tex', '\n'.join(lines[:-1]))


def capture():
    y = load_region('Superior')['pruned']['y']
    st = load_pickle(o('reencoded_benchmark_Superior.pkl'))
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4), sharex=True, sharey=True)
    xg = np.linspace(0, 10, 201)
    for ax, (title, proto) in zip(axes.flat, [('(a) random cross-validation', 'random'), ('(b) 50 km blocks', 50.0),
                                              ('(c) 100 km blocks', 100.0), ('(d) 200 km blocks', 200.0)]):
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
        ax.plot([0, 10], [0, 0.1], 'k--', lw=0.8, label='random ranking')
        ax.set_xlim(0, 10); ax.set_ylim(0, 1)
        ax.set_title(title, loc='left', fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel('top-ranked share of the scored pool (%)')
    for ax in axes[:, 0]:
        ax.set_ylabel('held-out occurrences captured')
    axes[1, 1].legend(frameon=False, fontsize=8, loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_capture_superior_r4.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------------------------------------------------- ablation
def loss_ablation():
    lines = []
    for prefix, label in [('reencoded', 'Feature set 3'), ('extended', 'Without map classes')]:
        b = pd.read_csv(o(f'{prefix}_benchmark_Superior.csv'))
        b = b[b['keep'].astype(str) == '0']
        bt = boot(prefix, 'Superior')
        for k, blk in enumerate([0, 50, 100, 200]):
            ap = lambda m: b[(b.model == m) & (b.block_km == blk)].ap_med.iloc[0]
            x = bt[(bt.block_km == blk) & (bt.comparison == 'DevNet - MLP-BCE') & ~bt.protocol.str.contains('k=')].iloc[0]
            lab = f'\\multirow{{4}}{{*}}{{{label}}}' if k == 0 else ''
            lines.append(f"{lab} & {PNAME[blk]} & {ap('DevNet'):.3f} & {ap('MLP-BCE'):.3f} & {ci3(x.d_ap, *x.d_ap_ci)} & "
                         f"{x.wins} \\\\")
        lines.append('\\midrule')
    write('tab_loss_ablation.tex', '\n'.join(lines[:-1]))


def semisup():
    s = pd.read_csv(o('semisup_summary.csv'))
    s = s[s.protocol == 'random']
    rows = [('E1', 'reencoded', 'DevNet vs DevNet-BCE', 'Ground near occurrences labelled barren', 'DevNet vs DevNet-BCE'),
            ('E1', 'reencoded', 'Bagging-PU vs RF', '', 'Bagging-PU vs RF'),
            ('E3', 'reencoded', 'DevNet vs DevNet-BCE', 'One to three training positives', 'DevNet vs DevNet-BCE'),
            ('E3', 'extended', 'DevNet vs DevNet-BCE', '', 'DevNet vs DevNet-BCE, without map classes')]
    lines = []
    for exp, fs, pair, test, label in rows:
        cells = []
        for r in REGIONS:
            x = s[(s.region == r) & (s.experiment == exp) & (s.featureset == fs) & (s.pair == pair)].iloc[0]
            cells.append(f'{x.effect:+.3f} ({int(x.wins)}/{int(x.n)})')
        lines.append(f"{test} & {label} & " + ' & '.join(cells) + ' \\\\')
    write('tab_semisup.tex', '\n'.join(lines))
    c = pd.read_csv(o('semisup_curves.csv'))
    c = c[(c.region == 'Superior') & (c.protocol == 'random')]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, (exp, fs, models, title, xl) in zip(axes, [
            ('E1', 'reencoded', ['DevNet', 'MLP-BCE', 'RF', 'BaggingPU'],
             '(a) Ground near occurrences added to the background', 'background points added per occurrence (2-10 km)'),
            ('E3', 'extended', ['DevNet', 'MLP-BCE', 'RF', 'BaggingPU'],
             '(b) Fewer training positives, without map classes', 'training positives kept per fold')]):
        q = c[(c.experiment == exp) & (c.featureset == fs)]
        for m in models:
            z = q[q.model == m].sort_values('x')
            ax.plot(np.arange(len(z)), z.ap_med, 'o-', ms=4, lw=1.5, color=COLORS[m], label=LABEL[m])
        ax.set_xticks(np.arange(len(z)), z.x_label.tolist())
        ax.set_xlabel(xl)
        ax.set_title(title, loc='left', fontsize=9.5)
    axes[0].set_ylabel('Superior AP (median of 20 repeats)')
    axes[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_semisup_r4.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------- feature-set-3 reruns (out/fs3/)
def _per_region_links():
    """Stages 08 and 10-es were run for all regions in one process; the older
    generator reads one file per region."""
    for stem in ('devnet_diagnostics', 'training_budget_es'):
        for ext in ('csv', 'pkl'):
            src = os.path.join(FS3, f'{stem}_Superior_Cordillera_Grenville.{ext}')
            if not os.path.exists(src):
                continue
            for r in REGIONS:
                dst = os.path.join(FS3, f'{stem}_{r}.{ext}')
                if not os.path.exists(dst):
                    if ext == 'csv':
                        d = pd.read_csv(src)
                        (d[d.region == r] if 'region' in d else d).to_csv(dst, index=False)
                    else:
                        os.symlink(os.path.basename(src), dst)


def fs3_tables():
    if not os.path.exists(os.path.join(FS3, 'ALL_DONE')) and not os.path.exists(os.path.join(FS3, 'nested_selection_Grenville.csv')):
        print('skip fs3_tables: feature-set-3 reruns not finished')
        return
    _per_region_links()
    old_out, old_gen, old_fig = g6b.OUT_DIR, g6b.GEN, g6b.FIG
    tmp = tempfile.mkdtemp()
    g6b.OUT_DIR, g6b.GEN, g6b.FIG = FS3, GEN, tmp
    try:
        for fn in (g6b.geography, g6b.augmentation, g6b.audit, g6b.ablation, g6b.nested):
            fn()
    finally:
        g6b.OUT_DIR, g6b.GEN, g6b.FIG = old_out, old_gen, old_fig
    shutil.move(os.path.join(tmp, 'fig_augmentation_performance_rev.png'),
                os.path.join(FIG, 'fig_augmentation_performance_r4.png'))
    shutil.copy(os.path.join(FS3, 'fig_augmentation_Superior.png'), os.path.join(FIG, 'fig_augmentation_superior_r4.png'))
    # nnPU class prior and deviation margin (Stage 14 on feature set 3)
    fs = pd.read_csv(os.path.join(FS3, 'final_suite.csv'))
    sup = fs[(fs.region == 'Superior') & (fs.block_km == 0) & (fs.feature_set == 'pruned')]
    pri = [(0.01, sup[sup.model == 'nnPU-QT'].iloc[0])] + [
        (float(m.split('=')[1].rstrip(')')), x) for m, x in sup.set_index('model').iterrows() if m.startswith('nnPU-QT(pi=')]
    write('tab_nnpu_prior.tex', '\n'.join(
        f"{p:g} & {x.auc_med:.3f} [{x.auc_lo:.3f}, {x.auc_hi:.3f}] & {x.ap_med:.3f} [{x.ap_lo:.3f}, {x.ap_hi:.3f}] \\\\"
        for p, x in sorted(pri, key=lambda t: t[0])))
    dn = pd.read_csv(os.path.join(FS3, 'devnet_final.csv'))
    lines = []
    for r in REGIONS:
        q = fs[(fs.region == r) & (fs.block_km == 0) & (fs.feature_set == 'pruned')].set_index('model').ap_med
        a5 = dn[(dn.region == r) & (dn.block_km == 0)].ap_med.iloc[0]
        lines.append(f"{r} & {q['DevNet(a=2)']:.3f} & {q['DevNet(a=3)']:.3f} & {a5:.3f} & {q['DevNet(a=8)']:.3f} \\\\")
    write('tab_margin.tex', '\n'.join(lines))
    audit_figure()


def audit_figure():
    a = pd.read_csv(os.path.join(FS3, 'protocol_audit.csv'))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    splits = ['child-level random', 'grouped by parent', 'no augmentation']
    for ax, r in zip(axes, REGIONS):
        for k, (neg, col) in enumerate([('cosine', '#8c2f2f'), ('random', '#1f6b5c')]):
            q = a[(a.region == r) & (a.negatives == neg)].set_index('split').reindex(splits)
            xs = np.arange(3) + (k - 0.5) * 0.18
            ax.errorbar(xs, q.auc_mean, yerr=[np.clip(q.auc_mean - q.auc_lo, 0, None), np.clip(q.auc_hi - q.auc_mean, 0, None)], fmt='o', color=col,
                        capsize=3, label=f'{neg} negatives')
        ax.set_xticks(np.arange(3), ['child-level\n(as published)', 'grouped\nby parent', 'no\naugmentation'], fontsize=8)
        ax.set_title(r, loc='left')
        ax.axhline(0.5, color='grey', lw=0.6, ls=':')
    axes[0].set_ylabel('test AUC (mean, 2.5-97.5% range)')
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_protocol_audit_r4.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def workflow():
    g.SUFFIX = '_r4'
    g.WORKFLOW_INPUTS = ('Inputs', 'raw H3 lithology, era, contacts\ndistances to regional geology\n10 km neighbourhood statistics\nsensitivity: without the\n  point-level map classes')
    g.WORKFLOW_MODELS = 'DevNet (deviation loss,\nquantile inputs, 16+240 rows)\nbaselines: RF, tuned RF, HGB\nPU: bagging-PU, nnPU'
    g.WORKFLOW_VALIDATION = ('pooled out-of-fold ranks\nrandom CV and 50/100/200 km\nblocks, background purged\n'
                             'paired bootstrap\nDevNet loss ablation\n(three tests, 20 repeats)')
    g.workflow()


if __name__ == '__main__':
    for fn in [comparisons, capture, loss_ablation, semisup, fs3_tables, workflow]:
        fn()
