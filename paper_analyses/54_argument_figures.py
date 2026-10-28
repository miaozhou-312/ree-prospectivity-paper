"""
Stage 54 -- figures that show what the text states in numbers:

  fig_design_effects_r7.png  test AUC of the S2-grid design of Parsa et al.
      (Stage 43 R0), with random instead of cosine-dissimilar negatives (R1),
      and with the four children of an occurrence cell also kept on the same
      side of the split (R2); RF and CNN.
  fig_feature_space_r7.png  the first two principal components of the
      min-max scaled inputs with the occurrence cells, the negatives of the
      Parsa et al. design and our training background.

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
CLF_COLORS = {'RF-Parsa': '#35618f', 'CNN-Parsa': '#a6bddb'}
CLF_NAMES = {'RF-Parsa': 'Random forest', 'CNN-Parsa': 'CNN'}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})


def design_effects():
    lad = pd.read_csv(os.path.join(NAT, 'tab_ladder.csv')).drop_duplicates(['rung', 'model']).set_index(['rung', 'model'])
    rows = [('R0', 'Design of Parsa et al. (2024)'), ('R1', 'Random instead of\ncosine-dissimilar negatives'),
            ('R2', '... and the children of a cell\nkept on one side of the split')]
    with plt.rc_context({'font.size': 9}):
        fig, ax = plt.subplots(figsize=(6.5, 2.6))
        y = np.arange(len(rows))[::-1]
        for k, m in enumerate(CLF_COLORS):
            v = np.array([lad.loc[(r, m), 'auc_balanced_mean'] for r, _ in rows])
            yy = y + (0.18 if k == 0 else -0.18)
            ax.barh(yy, v, height=0.34, color=CLF_COLORS[m], label=CLF_NAMES[m])
            for a_, b_ in zip(yy, v):
                ax.text(b_ + 0.008, a_, f'{b_:.3f}', va='center', fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels([l for _, l in rows])
        ax.set_xlim(0.5, 1.06); ax.set_xlabel('Test AUC (mean of 50 splits)')
        ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
        fig.savefig(os.path.join(FIG, 'fig_design_effects_r7.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print('saved fig_design_effects_r7.png', flush=True)


def feature_space():
    """Scaler and PCA fitted on 50,000 random cells."""
    g = _52._l.load_grid()
    D = _52.designs(g)
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
    with plt.rc_context({'font.size': 9}):
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.scatter(allc[:, 0], allc[:, 1], s=2, color='#d9d9d9', label='All cells (random 20,000)', rasterized=True)
        ax.scatter(bg[:, 0], bg[:, 1], s=3, color=_52.DESIGN_COLORS['ours'], alpha=0.5,
                   label='Training background, this study (random 3,000)', rasterized=True)
        ax.scatter(neg[:, 0], neg[:, 1], s=9, color=_52.DESIGN_COLORS['parsa'], label='Negatives, design of Parsa et al. (2024)')
        ax.scatter(pos[:, 0], pos[:, 1], s=18, color='black', marker='^', label='Occurrence cells')
        ev = 100 * pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({ev[0]:.0f}% of variance)'); ax.set_ylabel(f'PC2 ({ev[1]:.0f}% of variance)')
        ax.legend(frameon=False, fontsize=8, markerscale=2, loc='lower right')
        fig.savefig(os.path.join(FIG, 'fig_feature_space_r7.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print('saved fig_feature_space_r7.png; PCA variance', ev.round(1), flush=True)

if __name__ == '__main__':
    design_effects()
    feature_space()
