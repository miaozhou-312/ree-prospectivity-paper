"""
Overleaf files for the seventh revision: main.tex, references.bib, the class
and style files and the regenerated figures (figures_r7/). Figures the
manuscript takes from figures/ or from the author's overleaf_revised6/
folder are already in the Overleaf project and are not repeated.

    python paper/pack_overleaf_revised7.py      # -> paper/overleaf_revised7/ and paper/overleaf_revised7.zip
"""
import os
import re
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(REPO, 'paper', 'manuscript')
DST = os.path.join(REPO, 'paper', 'overleaf_revised7')


def main():
    os.makedirs(os.path.join(DST, 'figures_r7'), exist_ok=True)
    tex = open(os.path.join(MS, 'main_src_revised7.tex')).read()
    with open(os.path.join(DST, 'main.tex'), 'w') as f:
        f.write(tex)
    for name in ('references.bib', 'sn-jnl.cls', 'sn-apacite.bst'):
        shutil.copy(os.path.join(MS, name), DST)
    for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{(figures_r7/[^}]+)\}', tex):
        shutil.copy(os.path.join(MS, path), os.path.join(DST, path))
        print('copied', path)
    shutil.make_archive(DST, 'zip', DST)
    print('wrote', DST + '.zip')


if __name__ == '__main__':
    main()
