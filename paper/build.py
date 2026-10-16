"""
Build the journal manuscript.

1. Copies the analysis figures from paper_analyses/out/ into
   paper/manuscript/figures/.
2. Replaces every `%%GEN:<name>%%` line of manuscript/main_src.tex with
   manuscript/generated/<name>.tex (written by
   paper_analyses/06_figures_tables.py) and writes manuscript/manuscript.tex,
   the single .tex file Springer asks for.
3. Compiles it with Tectonic (runs BibTeX automatically).

    python paper/build.py            # from the repository root
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(HERE, 'manuscript')
OUT = os.path.join(os.path.dirname(HERE), 'paper_analyses', 'out')
COPY = {'fig_shap_superior.png': 'fig_shap_superior.png',
        'fig_augmentation_Superior.png': 'fig_augmentation_superior.png'}


def main():
    for src, dst in COPY.items():
        shutil.copy(os.path.join(OUT, src), os.path.join(MS, 'figures', dst))

    with open(os.path.join(MS, 'main_src.tex')) as f:
        src = f.read()
    missing = []

    def splice(m):
        path = os.path.join(MS, 'generated', f'{m.group(1)}.tex')
        if not os.path.exists(path):
            missing.append(m.group(1))
            return m.group(0)
        with open(path) as g:
            return g.read().rstrip('\n')

    out = re.sub(r'^%%GEN:(\w+)%%$', splice, src, flags=re.M)
    if missing:
        sys.exit(f'missing generated tables: {missing} (run paper_analyses/06_figures_tables.py)')
    left = re.findall(r'\[\[[A-Z_]+\]\]', out)
    if left:
        print(f'WARNING: unfilled placeholders: {left}')
    with open(os.path.join(MS, 'manuscript.tex'), 'w') as f:
        f.write(out)

    subprocess.run(['tectonic', '-X', 'compile', '--keep-logs', 'manuscript.tex'], cwd=MS, check=True)
    print('built', os.path.join(MS, 'manuscript.pdf'))


if __name__ == '__main__':
    main()
