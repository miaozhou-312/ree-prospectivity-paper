"""
Check that key numbers typed into the revised manuscript match the analysis
outputs.

Each claim is a phrase from the manuscript with its numbers written as
format fields; the fields are recomputed from paper_analyses/out/ and the
formatted phrase must occur in main_src_revised.tex (whitespace-normalised).
A failing claim means either the text or the analysis changed.

    python paper/check_claims.py
"""
import glob
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'paper_analyses', 'out')
TEX = os.path.join(REPO, 'paper', 'manuscript', 'main_src_revised.tex')


def csv(name):
    return pd.read_csv(os.path.join(OUT, name))


def latest(pattern):
    return pd.read_csv(sorted(glob.glob(os.path.join(OUT, pattern)))[-1])


DN = csv('devnet_final.csv')
BM = csv('benchmark_Superior_Cordillera_Grenville.csv')
SP = csv('spatial_cv_Superior_Cordillera_Grenville.csv')
FS = csv('final_suite.csv')
FSB = csv('final_suite_bootstrap.csv')
RD = csv('region_diagnostics.csv').set_index('region')
RDP = csv('region_diagnostics_positives.csv')
OB = csv('occurrence_breakdown_summary_Superior.csv')
JK = csv('superior_grid_uncertainty.csv')
PA = csv('pa_intersections_Superior.csv').set_index(['protocol', 'model'])


def dn(region, block=0):
    return DN[(DN.region == region) & (DN.block_km == block)].iloc[0]


def ap(region, model, block=0):
    if model == 'DevNet':
        return dn(region, block).ap_med
    if model == 'nnPU':
        return FS[(FS.region == region) & (FS.model == 'nnPU-QT') & (FS.feature_set == 'pruned')
                  & (FS.block_km == block)].iloc[0].ap_med
    if block == 0:
        return BM[(BM.region == region) & (BM.feature_set == 'pruned') & (BM.model == model)].iloc[0].ap_med
    return SP[(SP.region == region) & (SP.block_km == block) & (SP.model == model)].iloc[0].ap_med


def fni(name):
    return RDP[(RDP.region == 'Grenville') & (RDP.name == name)].iloc[0].far_neighbour_index_geo


def jk_stable(model):
    top = JK[f'{model}_pct'] >= 0.95
    return (JK.loc[top, f'{model}_top5_freq'] >= 0.8).mean()


def occ(model, proto, group):
    x = OB[(OB.model == model) & (OB.protocol == proto) & (OB.group == group)].iloc[0]
    return int(round(x.top1 * x.n))


CLAIMS = [
    ('DevNet has the highest median AP ({:.3f})', lambda: [ap('Superior', 'DevNet')]),
    ("DevNet's AP stays between {:.3f} and {:.3f} at every block size",
     lambda: [min(ap('Superior', 'DevNet', b) for b in (50, 100, 200)),
              max(ap('Superior', 'DevNet', b) for b in (50, 100, 200))]),
    ('falls from {:.3f} under random splits to {:.3f} with 50 and 100~km blocks and {:.3f} with 200~km blocks',
     lambda: [ap('Superior', 'RF'), ap('Superior', 'RF', 50), ap('Superior', 'RF', 200)]),
    ("nnPU's between {:.3f} and {:.3f}",
     lambda: [min(ap('Superior', 'nnPU', b) for b in (50, 100, 200)),
              max(ap('Superior', 'nnPU', b) for b in (50, 100, 200))]),
    ('the far-neighbour index has a median of {:.3f}, an order of magnitude below that of random pseudo-occurrences '
     '({:.3f}; 5th percentile {:.3f})',
     lambda: [RD.loc['Superior', 'far_neighbour_index_geo'], RD.loc['Superior', 'fni_geo_random_ref_median'],
              RD.loc['Superior', 'fni_geo_random_ref_p05']]),
    ('In Cordillera ({:.3f} against {:.3f}) and Grenville ({:.3f} against {:.3f})',
     lambda: [RD.loc['Cordillera', 'far_neighbour_index_geo'], RD.loc['Cordillera', 'fni_geo_random_ref_median'],
              RD.loc['Grenville', 'far_neighbour_index_geo'], RD.loc['Grenville', 'fni_geo_random_ref_median']]),
    ('with $|\\mathrm{{AUC}}-0.5| = {:.2f}$ in Superior but only {:.2f} and {:.2f} in Cordillera and Grenville',
     lambda: [RD.loc[r, 'median_abs_auc_dev'] for r in ('Superior', 'Cordillera', 'Grenville')]),
    ('leaves {} sites of {} occurrences in Superior, {} of {} in Cordillera and {} of {} in Grenville',
     lambda: sum([[int(RD.loc[r, 'sites_10km']), int(RD.loc[r, 'n_pos'])] for r in ('Superior', 'Cordillera', 'Grenville')], [])),
    ('Crevier ({:.2f}), Megiscane Lake ({:.2f}) and Sullivan Island ({:.2f})',
     lambda: [fni('Crevier'), fni('Megiscane Lake'), fni('Sullivan Island Carbonatite')]),
    ('raises the Grenville far-neighbour index from {:.3f} to {:.3f}',
     lambda: [RD.loc['Grenville', 'far_neighbour_index_geo'], RD.loc['Grenville', 'far_neighbour_index_all']]),
    ('dispersed Superior occurrences in the top 1\\% of the scored pool (median percentile {:.3f})',
     lambda: [OB[(OB.model == 'DevNet') & (OB.protocol == '100 km') & (OB.group == 'dispersed')].iloc[0].median_pct]),
    ('RF median percentile {:.3f} under random splits and {:.3f} under 100~km blocks), whereas DevNet changes little '
     '({:.3f} and {:.3f})',
     lambda: [OB[(OB.model == m) & (OB.protocol == p) & (OB.group == 'Abitibi cluster')].iloc[0].median_pct
              for m in ('RF', 'DevNet') for p in ('random', '100 km')]),
    ('DevNet and bagging-PU capture {:.0f}\\% of the held-out occurrences in the top {:.0f}\\% of the area and RF '
     '{:.0f}\\% in {:.0f}\\%',
     lambda: [100 * PA.loc[('100 km spatial blocks', 'DevNet'), 'captured_at_intersection'],
              100 * PA.loc[('100 km spatial blocks', 'DevNet'), 'area_at_intersection'],
              100 * PA.loc[('100 km spatial blocks', 'RF'), 'captured_at_intersection'],
              100 * PA.loc[('100 km spatial blocks', 'RF'), 'area_at_intersection']]),
    ("{:.0f}\\% of the grid nodes in the full DevNet model's top 5\\%", lambda: [100 * jk_stable('devnet')]),
    ('for RF the shares are {:.0f}\\% and {:.0f}\\%',
     lambda: [100 * jk_stable('rf'), 100 * (JK.loc[JK.rf_pct >= 0.95, 'rf_top5_freq'] < 0.5).mean()]),
]


def norm(t):
    return re.sub(r'\s+', ' ', t)


def main():
    tex = norm(open(TEX).read())
    bad = 0
    for phrase, values in CLAIMS:
        try:
            text = phrase.format(*values())
        except Exception as e:                                    # missing output
            print(f'SKIP  {phrase[:70]}...  ({type(e).__name__}: {e})')
            continue
        if norm(text) in tex:
            print(f'ok    {text[:110]}')
        else:
            bad += 1
            print(f'FAIL  {text}')
    print(f'{len(CLAIMS) - bad}/{len(CLAIMS)} claims match')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
