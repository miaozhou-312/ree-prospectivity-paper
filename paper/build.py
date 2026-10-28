"""
Build the journal manuscript.

Two versions are kept side by side:
  python paper/build.py            original:  main_src.tex + generated/          -> manuscript.pdf
  python paper/build.py revised    revised:   main_src_revised.tex + generated_revised/ -> manuscript_revised.pdf
  python paper/build.py revised2   second revision (feature set 3 main):
                                   main_src_revised2.tex + generated_revised2/ -> manuscript_revised2.pdf
  python paper/build.py revised4   fourth revision (simplified, feature set 3 only):
                                   main_src_revised4.tex + generated_revised4/, then _revised3/, _revised2/
  python paper/build.py revised5   fifth revision (national part, XGBoost):
                                   main_src_revised5.tex + generated_revised5/, then _revised4/, _revised3/, _revised2/
  python paper/build.py revised7   seventh revision (public-data grids; tables spliced inline):
                                   main_src_revised7.tex + generated_revised7/ -> manuscript_revised7.pdf
  python paper/build.py revised6   sixth revision (shortened for submission):
                                   main_src_revised6.tex + generated_revised6/, then _revised5/ ... _revised2/
  python paper/build.py revised3   third revision (DevNet-BCE as an ablation):
                                   main_src_revised3.tex + generated_revised3/, falling back to
                                   generated_revised2/ for unchanged tables -> manuscript_revised3.pdf

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
    args = sys.argv[1:]
    if 'revised7' in args:
        src_name, out_name = 'main_src_revised7.tex', 'manuscript_revised7'
        gen_dirs = ['generated_revised7']
    elif 'revised6' in args:
        src_name, out_name = 'main_src_revised6.tex', 'manuscript_revised6'
        gen_dirs = ['generated_revised6', 'generated_revised5', 'generated_revised4', 'generated_revised3',
                    'generated_revised2']
    elif 'revised5' in args:
        src_name, out_name = 'main_src_revised5.tex', 'manuscript_revised5'
        gen_dirs = ['generated_revised5', 'generated_revised4', 'generated_revised3', 'generated_revised2']
    elif 'revised4' in args:
        src_name, out_name = 'main_src_revised4.tex', 'manuscript_revised4'
        gen_dirs = ['generated_revised4', 'generated_revised3', 'generated_revised2']
    elif 'revised3' in args:
        src_name, gen_dirs, out_name = 'main_src_revised3.tex', ['generated_revised3', 'generated_revised2'], 'manuscript_revised3'
    elif 'revised2' in args:
        src_name, gen_dirs, out_name = 'main_src_revised2.tex', ['generated_revised2'], 'manuscript_revised2'
    elif 'revised' in args:
        src_name, gen_dirs, out_name = 'main_src_revised.tex', ['generated_revised'], 'manuscript_revised'
    else:
        src_name, gen_dirs, out_name = 'main_src.tex', ['generated'], 'manuscript'
    for src, dst in COPY.items():
        if os.path.exists(os.path.join(OUT, src)):
            shutil.copy(os.path.join(OUT, src), os.path.join(MS, 'figures', dst))

    with open(os.path.join(MS, src_name)) as f:
        src = f.read()
    missing = []

    def splice(m):
        paths = [os.path.join(MS, d, f'{m.group(1)}.tex') for d in gen_dirs]
        path = next((q for q in paths if os.path.exists(q)), None)
        if path is None:
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
    with open(os.path.join(MS, f'{out_name}.tex'), 'w') as f:
        f.write(out)

    subprocess.run(['tectonic', '-X', 'compile', '--keep-logs', f'{out_name}.tex'], cwd=MS, check=True)
    print('built', os.path.join(MS, f'{out_name}.pdf'))


if __name__ == '__main__':
    main()
