"""
Stage 54 -- figures that show what the text states in numbers:

  fig_design_effects_r7.png  (a) test AUC of the S2-grid design and of its
      two elements alone (Stage 43 rungs R0, R3c, R1, R3), RF and CNN;
      (b) AP against realistic background of models trained on cosine-
      dissimilar negatives, random negatives and our shared background
      (R4c, R4, R5).
  fig_map_national_extra_r7.png  panels (e)-(f) of the national map figure:
      (e) share of cells at or above each probability threshold (Stage 52
      scores); (f) the first two principal components of the min-max scaled
      inputs with the occurrence cells, the negatives of the Parsa et al.
      design and our training background.

    source ../work_public/env.sh
    python -m paper_analyses.54_argument_figures
"""
import importlib
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from paper_analyses.common import config  # noqa: E402

_52 = importlib.import_module('paper_analyses.52_parsa_maps')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, 'paper', 'manuscript', 'figures_r7')
NAT = os.environ.get('REE_NAT_OUT') or os.path.join(REPO, 'paper_analyses', 'out', 'public', 'national')
CLF_COLORS = {'RF-Parsa': '#2a78d6', 'CNN-Parsa': '#e08214'}
CLF_NAMES = {'RF-Parsa': 'Random forest', 'CNN-Parsa': 'CNN'}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})


def bars(ax, rows, col, lad, xlabel, fmt):
    y = np.arange(len(rows))[::-1]
    for k, m in enumerate(CLF_COLORS):
        v = [lad.loc[(r, m), col] for r, _ in rows]
        ax.barh(y + (0.2 if k == 0 else -0.2), v, height=0.38, color=CLF_COLORS[m], label=CLF_NAMES[m])
        for yy, vv in zip(y, v):
            ax.text(vv + 0.01 * ax.get_xlim()[1], yy + (0.2 if k == 0 else -0.2), fmt.format(vv), va='center', fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in rows])
    ax.set_xlabel(xlabel)


def design_effects():
    lad = pd.read_csv(os.path.join(NAT, 'tab_ladder.csv')).drop_duplicates(['rung', 'model']).set_index(['rung', 'model'])
    sib = lad.loc[('R1', 'RF-Parsa'), 'sibling_in_train_mean']
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.62))
    a.set_xlim(0, 1.17)
    bars(a, [('R0', 'Parsa et al. (2024) design'), ('R3c', 'Cosine-dissimilar negatives only'),
             ('R1', f'Child cells split at random only\n({100 * sib:.0f}% of test positives have\na sibling in training)'),
             ('R3', 'Neither')], 'auc_balanced_mean', lad, 'Test AUC, balanced test set', '{:.3f}')
    a.set_title('a', loc='left', fontweight='bold')
    b.set_xlim(0, 0.36)
    bars(b, [('R4c', 'Cosine-dissimilar negatives'), ('R4', 'Random negatives'), ('R5', 'Our background (this study)')],
         'ap_mean', lad, 'AP, realistic test set (0.3% prevalence)', '{:.3f}')
    b.set_title('b  trained on', loc='left', fontweight='bold')
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=10.5)
    fig.savefig(os.path.join(FIG, 'fig_design_effects_r7.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved fig_design_effects_r7.png', flush=True)


def national_extra():
    g = _52._l.load_grid()
    D = _52.designs(g)
    fig, (e, f) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw=dict(wspace=0.25))
    t = np.logspace(-3, 0, 301).astype(np.float32)
    for design, clf in _52.MODELS:
        s = np.sort(np.load(os.path.join(_52.OUT, 'scores', f'{design}_{clf}.npy')).astype(np.float32))
        share = 100.0 * (1.0 - np.searchsorted(s, t, side='left') / len(s))
        name = 'design of Parsa et al. (2024)' if design == 'parsa' else 'this study'
        e.plot(t, share, color=_52.DESIGN_COLORS[design], ls='-' if clf == 'RF' else '--', lw=2, label=f'{clf}, {name}')
    e.axvline(0.5, color='#999999', lw=0.8, ls=':')
    e.set_xscale('log'); e.set_xlim(1e-3, 1); e.set_ylim(0, 100)
    e.set_xlabel('Probability threshold'); e.set_ylabel('Cells at or above the threshold (%)')
    e.grid(axis='y', color='#e5e5e5', lw=0.6)
    e.legend(frameon=False, fontsize=10, loc='center left', bbox_to_anchor=(0.02, 0.42))
    e.set_title('e', loc='left', fontweight='bold', fontsize=16)
    # principal components of the scaled inputs (scaler and PCA fitted on a random sample of all cells)
    X = g['X'].astype(float)
    rng = np.random.default_rng(config.RANDOM_SEED)
    samp = rng.choice(len(X), 50000, replace=False)
    imp = SimpleImputer(strategy='median').fit(X[samp])
    mm = MinMaxScaler().fit(imp.transform(X[samp]))
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED).fit(mm.transform(imp.transform(X[samp])))
    proj = lambda idx: pca.transform(mm.transform(imp.transform(X[idx])))
    allc = proj(samp[:20000])
    bg = proj(rng.choice(D['ours'][3], 3000, replace=False))
    neg = proj(D['parsa'][3])
    pos = proj(np.where(g['is_pos'])[0])
    f.scatter(allc[:, 0], allc[:, 1], s=3, color='#d9d9d9', label='All cells (random 20,000)', rasterized=True)
    f.scatter(bg[:, 0], bg[:, 1], s=5, color=_52.DESIGN_COLORS['ours'], alpha=0.5,
              label='Training background, this study (random 3,000)', rasterized=True)
    f.scatter(neg[:, 0], neg[:, 1], s=14, color=_52.DESIGN_COLORS['parsa'], label='Negatives, design of Parsa et al. (2024)')
    f.scatter(pos[:, 0], pos[:, 1], s=28, color='black', marker='^', label='Occurrence cells')
    ev = 100 * pca.explained_variance_ratio_
    f.set_xlabel(f'PC1 ({ev[0]:.0f}% of variance)'); f.set_ylabel(f'PC2 ({ev[1]:.0f}% of variance)')
    f.legend(frameon=False, fontsize=10, markerscale=2, loc='best')
    f.set_title('f', loc='left', fontweight='bold', fontsize=16)
    fig.savefig(os.path.join(FIG, 'fig_map_national_extra_r7.png'), dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved fig_map_national_extra_r7.png; PCA variance', ev.round(1), flush=True)


if __name__ == '__main__':
    design_effects()
    national_extra()
