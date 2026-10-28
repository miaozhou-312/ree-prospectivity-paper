"""
Replace the bodies of the inline tables of main_src_revised7.tex (between the
header's \\midrule and \\botrule) with the generated bodies in
generated_revised7/.

    python paper/splice_revised7.py
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(REPO, 'paper', 'manuscript')
TEX = os.path.join(MS, 'main_src_revised7.tex')
GEN = os.path.join(MS, 'generated_revised7')
TABLES = {'tab:nat-design': 'tab_nat_design.tex', 'tab:benchmark': 'tab_benchmark.tex', 'tab:spatial': 'tab_spatial.tex',
          'tab:augmentation': 'tab_augmentation.tex', 'tab:aug-diag': 'tab_augmentation_diag.tex',
          'tab:shap': 'tab_shap_Superior.tex', 'tab:spatial-auc': 'tab_spatial_auc.tex', 'tab:prior': 'tab_nnpu_prior.tex',
          'tab:aug-spatial': 'tab_augmentation_spatial.tex', 'tab:shap-cord': 'tab_shap_Cordillera.tex',
          'tab:shap-gren': 'tab_shap_Grenville.tex', 'tab:ablation': 'tab_ablation.tex', 'tab:margin': 'tab_margin.tex'}


def main():
    tex = open(TEX).read()
    for label, name in TABLES.items():
        path = os.path.join(GEN, name)
        if not os.path.exists(path):
            print('missing', name)
            continue
        body = open(path).read().strip('\n')
        start = tex.index(f'\\label{{{label}}}')
        head = tex.index('\\midrule', start) + len('\\midrule')
        end = tex.index('\\botrule', head)
        tex = tex[:head] + '\n' + re.sub(r'\n?\\midrule\s*$', '', body) + '\n' + tex[end:]
        print('spliced', label)
    open(TEX, 'w').write(tex)


if __name__ == '__main__':
    main()
