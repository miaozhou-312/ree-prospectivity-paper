"""
Figures and LaTeX table bodies for the SIXTH revision
(paper/manuscript/main_src_revised6.tex), a shortened fifth revision for
submission (at most ten figures and ten tables in the main text):

  * the national part is an introduction to the provincial study: the S2-grid
    design of Parsa et al. (2024) against ours (one table) and transfer
    between provinces (one figure);
  * tables show results only, not paired differences or repeat counts;
  * the DevNet-BCE ablation, the tests of the semi-supervised premise, nested
    selection, map stability and the province diagnostics are left out.

Writes generated_revised6/ and figures/*_r6.png; `python paper/build.py
revised6` falls back to generated_revised5/, _revised4/, _revised3/ and
_revised2/ for unchanged tables.

    python -m paper_analyses.06g_figures_tables_revised6
"""
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(REPO, 'paper', 'manuscript')
NAT = os.path.join(REPO, 'paper_analyses', 'out', 'national')
FIG = os.path.join(MS, 'figures')
GEN = os.path.join(MS, 'generated_revised6')
os.makedirs(GEN, exist_ok=True)


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f:
        f.write(text)
    print('wrote', name)


def nat_design():
    """The S2-grid design of Parsa et al. (R0) against ours (R5), same forest."""
    lad = pd.read_csv(os.path.join(NAT, 'tab_ladder.csv'))
    L = lad[lad.model == 'RF-Parsa'].drop_duplicates('rung').set_index('rung')
    r0, r5 = L.loc['R0'], L.loc['R5']
    lines = [
        f"Parsa et al. (2024) & four L13 children per occurrence cell (264) & 264 cells from the 5\\% least "
        f"cosine-similar & child labels, random 70/30 & balanced (50\\%) & {r0.auc_balanced_mean:.3f} & "
        f"{r0.ap_balanced_mean:.3f} \\\\",
        f"This study & one per occurrence cell (66) & 34,761 random cells $>$10~km from occurrences & occurrence "
        f"cells, random 70/30 & realistic (0.3\\%) & {r5.auc_mean:.3f} & {r5.ap_mean:.3f} \\\\",
    ]
    write('tab_nat_design.tex', '\n'.join(lines))


def results_only(src, name, drop_last):
    """Copy a fourth-revision table without its last `drop_last` columns
    (paired differences and repeats won)."""
    with open(os.path.join(MS, 'generated_revised4', src)) as f:
        rows = f.read().splitlines()
    out = []
    for r in rows:
        if '&' not in r:
            out.append(r)
            continue
        cells = r.rstrip().removesuffix('\\\\').split(' & ')
        out.append(' & '.join(cells[:-drop_last]).rstrip() + ' \\\\')
    write(name, '\n'.join(out))


def ablation():
    with open(os.path.join(MS, 'generated_revised4', 'tab_ablation.tex')) as f:
        text = f.read()
    write('tab_ablation.tex', re.sub(r' \(\d/6\)', '', text))


def workflow():
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
    box(0.1, 4.7, 3.0, 1.95, '1  Labels and layers', '66 occurrences in Canada\n(one table, 2 km merge)\npotential fields, LAB, Moho,\nfaults, worms, H3 geology', '#eef3fb')
    box(3.4, 4.7, 3.3, 1.95, '2  Inputs', 'values computed per\ngeological province\npruned on each training table\nsensitivity: no map classes', '#eef3fb')
    box(7.0, 4.7, 2.9, 1.95, '3  Models', 'DevNet (deviation loss)\nRF, XGBoost,\ntuned RF, tuned HGB\nPU: bagging-PU, nnPU', '#fdf0e6')
    box(0.1, 2.15, 4.7, 2.15, '4  National scale', 'S2-grid design of Parsa et al. (2024)\nagainst our sampling and evaluation\ntransfer between provinces\nleave one province out', '#eaf6ee')
    box(5.1, 2.15, 4.8, 2.15, '5  Provinces, modelled separately', 'Superior (15 occurrences)\nCordillera and Grenville as contrasts\nrandom CV and 50/100/200 km blocks\nmap classes, coordinates-only baseline\nsynthetic positives', '#eaf6ee')
    box(1.6, 0.1, 6.8, 1.55, '6  Interpretation, maps and checks', 'TreeSHAP / DeepSHAP with a rank-based direction rule\npercentile prospectivity maps\nClay-Howells resource and Kapuskasing Structural Zone (not used in training)', '#f5eefa')
    arrow(3.1, 5.65, 3.4, 5.65)
    arrow(6.7, 5.65, 7.0, 5.65)
    arrow(8.4, 4.7, 2.45, 4.3)
    arrow(8.4, 4.7, 7.5, 4.3)
    arrow(2.45, 2.15, 4.0, 1.65)
    arrow(7.5, 2.15, 6.0, 1.65)
    fig.savefig(os.path.join(FIG, 'fig_workflow_r6.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    nat_design()
    results_only('tab_augmentation.tex', 'tab_augmentation.tex', 2)
    results_only('tab_augmentation_spatial.tex', 'tab_augmentation_spatial.tex', 2)
    ablation()
    workflow()
