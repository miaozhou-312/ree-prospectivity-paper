"""
Section 13d -- SHAP summary (beeswarm) plots.

One beeswarm per model per region: each dot is one explained point, its
x-position is that point's SHAP value for the feature (row), and color is
the (scaled) feature value -- so besides magnitude, these show *direction*
per point. Figures are saved at SHAP_FIG_DPI (600 by default) into
SHAP_FIG_DIR for use in the write-up.
"""
import os

import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap'])
    import shap


TITLE_FONTSIZE = 16
PANEL_LABEL_FONTSIZE = 20

# Panel letters for the combined figure this function's outputs feed into:
# (a, b) Cordillera, (c, d) Superior, (e, f) Grenville -- DevNet left column,
# RF right column. Indexed by position in shap_regions; falls back to
# continuing the alphabet if more than 3 regions are ever passed in.
_LETTER_PAIRS = [('a', 'b'), ('c', 'd'), ('e', 'f')]


def _letters_for(idx):
    if idx < len(_LETTER_PAIRS):
        return _LETTER_PAIRS[idx]
    base = 2 * idx
    return chr(ord('a') + base), chr(ord('a') + base + 1)


def _label_panel(ax, letter):
    """Stamp the panel letter in the top-left corner, just outside the axes."""
    ax.text(-0.02, 1.05, f'{letter}', transform=ax.transAxes,
            fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold',
            va='bottom', ha='right')


def plot_shap_summary(shap_results, shap_regions, shap_fig_dir, shap_fig_dpi=600):
    os.makedirs(shap_fig_dir, exist_ok=True)
    for idx, name in enumerate(shap_regions):
        r = shap_results[name]
        dn_letter, rf_letter = _letters_for(idx)

        plt.figure()
        shap.summary_plot(r['shap_dn'], r['X_explain_dn'], feature_names=r['feat_cols'], show=False)
        plt.title(f'{name}: DevNet ({r["devnet_method"]}) summary', fontsize=TITLE_FONTSIZE, fontweight='bold')
        _label_panel(plt.gca(), dn_letter)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_fig_dir, f'{name}_DevNet_summary.png'),
                    dpi=shap_fig_dpi, bbox_inches='tight')
        plt.show()

        plt.figure()
        shap.summary_plot(r['shap_rf'], r['X_explain_rf'], feature_names=r['feat_cols'], show=False)
        plt.title(f'{name}: RF (TreeSHAP) summary', fontsize=TITLE_FONTSIZE, fontweight='bold')
        _label_panel(plt.gca(), rf_letter)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_fig_dir, f'{name}_RF_summary.png'),
                    dpi=shap_fig_dpi, bbox_inches='tight')
        plt.show()
