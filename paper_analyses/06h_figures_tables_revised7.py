"""
Figures and LaTeX table bodies for the SEVENTH revision
(paper/manuscript/main_src_revised7.tex): the sixth revision rerun with the
public-data gravity and magnetic grids (Stages 50-51) and GLCM textures of
all eleven gravity and magnetic grids. Every result comes from the rerun in
out/public/ (provincial) and out/public/national/; the table and figure code
of the earlier generators is reused with its paths pointed there.

Writes generated_revised7/ and figures_r7/ (same figure names as before, so
earlier figures are never overwritten).

    source ../work_public/env.sh
    python -m paper_analyses.06h_figures_tables_revised7
"""
import importlib
import os
import re
import shutil
import tempfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from paper_analyses.common import OUT_DIR  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(REPO, 'paper', 'manuscript')
OUT = OUT_DIR
NAT = os.environ.get('REE_NAT_OUT') or os.path.join(OUT, 'national')
GEN = os.path.join(MS, 'generated_revised7')
FIG = os.path.join(MS, 'figures_r7')
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
assert os.path.basename(OUT) == 'public', 'run with ../work_public/env.sh'

g6b = importlib.import_module('paper_analyses.06b_figures_tables_revised')
g6c = importlib.import_module('paper_analyses.06c_figures_tables_revised2')
g6e = importlib.import_module('paper_analyses.06e_figures_tables_revised4')
g6f = importlib.import_module('paper_analyses.06f_figures_tables_revised5')
g6g = importlib.import_module('paper_analyses.06g_figures_tables_revised6')
maps = importlib.import_module('paper_analyses.13_maps')
for mod in (g6b, g6c, g6e, g6f, g6g):
    mod.GEN, mod.FIG = GEN, FIG
g6b.OUT_DIR = g6c.OUT_DIR = OUT
g6e.BASE = g6e.FS3 = g6f.BASE = OUT
g6f.NAT = g6g.NAT = NAT
maps.OUT_DIR, maps.FIG = OUT, FIG
g6c.SUFFIX = '_r2'
g6c.MODELS = ['DevNet', 'RF', 'BaggingPU', 'nnPU']


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f:
        f.write(text)
    print('wrote', name)


def per_region_files():
    """Stage 08 and 10-es ran per region; Stage 15 and the ablation read
    these names directly."""
    for r in ('Superior', 'Cordillera', 'Grenville'):
        for stem in ('devnet_diagnostics', 'training_budget_es'):
            assert os.path.exists(os.path.join(OUT, f'{stem}_{r}.pkl')), f'{stem}_{r} missing'


def results_only(name, drop_last):
    """Drop the paired-difference and repeats-won columns (sixth-revision rule)."""
    rows = open(os.path.join(GEN, name)).read().splitlines()
    out = []
    for r in rows:
        if '&' not in r:
            out.append(r)
            continue
        cells = r.rstrip().removesuffix('\\\\').split(' & ')
        out.append(' & '.join(cells[:-drop_last]).rstrip() + ' \\\\')
    write(name, '\n'.join(out))


def augmentation_pca():
    """Stage 4's PCA view of the augmented positives, redrawn with the
    revised wording (part B's samples and WGAN-GP history, same seeds)."""
    s04 = importlib.import_module('paper_analyses.04_augmentation')
    s04.plot_pca(*s04.pca_inputs('Superior'), os.path.join(FIG, 'fig_augmentation_superior_r4.png'),
                 sample_label='augmented')


def provincial():
    g6f.provincial_tables()                      # tab_benchmark, tab_spatial, tab_spatial_auc, fig_spatial_cv_r5
    g6e.capture()                                # fig_capture_superior_r4
    tmp = tempfile.mkdtemp()
    g6b.FIG = tmp
    g6b.AUG_XLABEL = 'augmented positives per real training positive'
    g6b.augmentation()                           # tab_augmentation(_diag, _spatial)
    g6b.FIG = FIG
    shutil.move(os.path.join(tmp, 'fig_augmentation_performance_rev.png'),
                os.path.join(FIG, 'fig_augmentation_performance_r4.png'))
    augmentation_pca()
    results_only('tab_augmentation.tex', 2)
    results_only('tab_augmentation_spatial.tex', 2)
    per_region_files()
    ablation()
    prior_and_margin()


def ablation():
    """Appendix: DevNet design choices (AP medians; the earlier generator's
    table without the repeats-won counts). The final configuration is Stage 11,
    or Stage 25's DevNet (same factory, folds and seeds) while Stage 11 runs."""
    from paper_analyses.common import load_pickle
    o = lambda n: os.path.join(OUT, n)  # noqa: E731
    final_path = o('devnet_final.pkl')
    if os.path.exists(final_path):
        dn = load_pickle(final_path)
    else:
        dn = {}
        for r in ('Superior', 'Cordillera', 'Grenville'):
            st = load_pickle(o(f'reencoded_benchmark_{r}.pkl'))
            dn.update({(r, p): st[(r, 'DevNet', p, None)] for p in ('random', 50.0, 100.0, 200.0)})
    qt = load_pickle(o('devnet_qt_sensitivity.pkl'))
    bg = load_pickle(o('training_budget_bg_Superior_Cordillera_Grenville.pkl'))
    es = {}
    for r in ('Superior', 'Cordillera', 'Grenville'):
        es.update(load_pickle(o(f'training_budget_es_{r}.pkl')))
    bench = load_pickle(o('benchmark_Superior_Cordillera_Grenville.pkl'))
    spat = load_pickle(o('spatial_cv_Superior_Cordillera_Grenville.pkl'))
    diag = {}
    for r in ('Superior', 'Cordillera', 'Grenville'):
        diag.update(load_pickle(o(f'devnet_diagnostics_{r}.pkl'))['store'])
    key = lambda blk: 'random' if blk == 0 else float(blk)  # noqa: E731
    configs = [
        ('quantile & 16\\,+\\,240 & 300 (final)', lambda r, b: dn[(r, key(b))]),
        ('z-score & 16\\,+\\,240 & 300', lambda r, b: bg[(r, key(b))]),
        ('quantile & 16\\,+\\,16 & 300', lambda r, b: qt[(r, key(b))]),
        ('z-score & 16\\,+\\,16 & 300', lambda r, b: bench[(r, 'pruned', 'DevNet')] if b == 0
         else spat[(r, float(b), 'DevNet')]),
        ('z-score & 16\\,+\\,16 & 3,000', lambda r, b: diag.get((r, 'DevNet_long', 'random' if b == 0 else f'{b} km'))),
        ('z-score & 16\\,+\\,16 & nested ES', lambda r, b: es[(r, key(b))]),
    ]
    lines = []
    for r in ('Superior', 'Cordillera', 'Grenville'):
        for k, (label, get) in enumerate(configs):
            cells = []
            for b in (0, 50, 100, 200):
                res = get(r, b)
                cells.append('--' if res is None else (f'\\textbf{{{res["ap_med"]:.3f}}}' if k == 0
                                                       else f'{res["ap_med"]:.3f}'))
            lab = f'\\multirow{{6}}{{*}}{{{r}}}' if k == 0 else ''
            lines.append(f'{lab} & {label} & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule' if r != 'Grenville' else '')
    write('tab_ablation.tex', '\n'.join(lines))


def final_suite():
    """Stage 14's rows; while it runs, the same rows computed one by one
    (partial14_*.csv: same factories, seeds and folds)."""
    path = os.path.join(OUT, 'final_suite.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    import glob
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'partial14_*.csv'))], ignore_index=True)


def devnet_final_ap(region):
    """Final DevNet, random CV: Stage 11, or Stage 25's DevNet (the same
    factory, folds and seeds) while Stage 11 runs."""
    path = os.path.join(OUT, 'devnet_final.csv')
    if os.path.exists(path):
        dn = pd.read_csv(path)
        return dn[(dn.region == region) & (dn.block_km == 0)].ap_med.iloc[0]
    t = pd.read_csv(os.path.join(OUT, f'reencoded_benchmark_{region}.csv'))
    return t[(t.model == 'DevNet') & (t.block_km == 0) & (t['keep'].astype(str) == '0')].ap_med.iloc[0]


def prior_and_margin():
    fs = final_suite()
    sup = fs[(fs.region == 'Superior') & (fs.block_km == 0) & (fs.feature_set == 'pruned')]
    pri = [(0.01, sup[sup.model == 'nnPU-QT'].iloc[0])] + [
        (float(m.split('=')[1].rstrip(')')), x) for m, x in sup.set_index('model').iterrows() if m.startswith('nnPU-QT(pi=')]
    write('tab_nnpu_prior.tex', '\n'.join(
        f"{p:g} & {x.auc_med:.3f} [{x.auc_lo:.3f}, {x.auc_hi:.3f}] & {x.ap_med:.3f} [{x.ap_lo:.3f}, {x.ap_hi:.3f}] \\\\"
        for p, x in sorted(pri, key=lambda t: t[0])))
    lines = []
    for r in ('Superior', 'Cordillera', 'Grenville'):
        q = fs[(fs.region == r) & (fs.block_km == 0) & (fs.feature_set == 'pruned')].set_index('model').ap_med
        lines.append(f"{r} & {q['DevNet(a=2)']:.3f} & {q['DevNet(a=3)']:.3f} & {devnet_final_ap(r):.3f} & "
                     f"{q['DevNet(a=8)']:.3f} \\\\")
    write('tab_margin.tex', '\n'.join(lines))


def workflow():
    """Fig. 3 without the tuned and geography-only baselines (removed in this revision)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.8)
    ax.axis('off')

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', fc=fc, ec='#333333', lw=1))
        ax.text(x + w / 2, y + h - 0.22, title, ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(x + w / 2, y + h - 0.55, body, ha='center', va='top', fontsize=8.6, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1))
    box(0.1, 4.7, 3.0, 1.95, '1  Labels and layers', '66 occurrences in Canada\n(one table, 2 km merge)\npublic potential fields and\nderivatives, LAB, Moho,\nfaults, worms, H3 geology', '#eef3fb')
    box(3.4, 4.7, 3.3, 1.95, '2  Inputs', 'values computed per\ngeological province\nGLCM on 11 gravity and\nmagnetic grids, pruned on\neach training table', '#eef3fb')
    box(7.0, 4.7, 2.9, 1.95, '3  Models', 'DevNet (deviation loss)\nRF, XGBoost\nPU: bagging-PU, nnPU', '#fdf0e6')
    box(0.1, 2.15, 4.7, 2.15, '4  National scale', 'S2-grid design of Parsa et al. (2024)\nagainst our sampling and evaluation', '#eaf6ee')
    box(5.1, 2.15, 4.8, 2.15, '5  Provinces, modelled separately', 'Superior (15 occurrences)\nCordillera and Grenville as contrasts\nrandom CV and 50/100/200 km blocks\nwith and without map classes\npositive augmentation\ntransfer between provinces\nleave one province out', '#eaf6ee')
    box(1.6, 0.1, 6.8, 1.55, '6  Interpretation, maps and checks', 'TreeSHAP / DeepSHAP with a rank-based direction rule\npercentile prospectivity maps\nClay-Howells resource and Kapuskasing Structural Zone (not used in training)', '#f5eefa')
    arrow(3.1, 5.65, 3.4, 5.65)
    arrow(6.7, 5.65, 7.0, 5.65)
    arrow(8.4, 4.7, 2.45, 4.3)
    arrow(8.4, 4.7, 7.5, 4.3)
    arrow(2.45, 2.15, 4.0, 1.65)
    arrow(7.5, 2.15, 6.0, 1.65)
    fig.savefig(os.path.join(FIG, 'fig_workflow_r7.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved fig_workflow_r7.png')


def interpretation():
    g6c.shap_outputs()                           # tab_shap_<region>, fig_shap_superior_r2
    maps.superior_maps(outname='fig_map_superior_r2.png', grid_file='superior_grid_scores_fs3.csv')
    maps.ksz_map(outname='fig_ksz_r2.png', grid_file='superior_grid_scores_fs3.csv')


def nat_design():
    """The S2-grid design of Parsa et al. (R0) against ours (R5), with their
    random forest and their 1D CNN."""
    lad = pd.read_csv(os.path.join(NAT, 'tab_ladder.csv'))

    def cells(model, rung, auc, ap):
        r = lad[(lad.model == model) & (lad.rung == rung)]
        return '-- & --' if r.empty else f'{r[auc].iloc[0]:.3f} & {r[ap].iloc[0]:.3f}'

    def metrics(rung, auc, ap):
        return f"{cells('RF-Parsa', rung, auc, ap)} & {cells('CNN-Parsa', rung, auc, ap)}"

    lines = [
        "Parsa et al. (2024) & four L13 children per occurrence cell (264) & 264 cells from the 5\\% least "
        "cosine-similar & child labels, random 70/30 & balanced (50\\%) & "
        f"{metrics('R0', 'auc_balanced_mean', 'ap_balanced_mean')} \\\\",
        "This study & one per occurrence cell (66) & 34,761 random cells $>$10~km from occurrences & occurrence "
        f"cells, random 70/30 & realistic (0.3\\%) & {metrics('R5', 'auc_mean', 'ap_mean')} \\\\",
    ]
    write('tab_nat_design.tex', '\n'.join(lines))


def national():
    s46 = importlib.import_module('paper_analyses.46_national_summary')
    s46.ladder_table()                           # tab_ladder.csv
    nat_design()                                 # tab_nat_design
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 10.5}):
        s46.fig_transfer(s46.transfer_table(), lopo=False, path=os.path.join(FIG, 'fig_transfer_r5.png'))


if __name__ == '__main__':
    import sys
    parts = sys.argv[1:] or ['provincial', 'interpretation', 'national', 'workflow']
    for p in parts:
        globals()[p]()
