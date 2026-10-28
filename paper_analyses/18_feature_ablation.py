"""
Stage 18 -- feature-set ablations of the final DevNet and RF.

  no-terrane   drop the geological-era and lithology dummies, which can
               encode the target (an alkaline complex mapped at a known
               carbonatite is close to a label); geophysics and structure only.
  no-alkaline-lith  drop only the alkaline-favourable lithology dummy (in
               Superior it holds 6 of the 15 occurrences but ~1% of the
               ground, i.e. the bedrock map marks the known complexes).
  terrane-only only the era and lithology dummies: how much the bedrock map
               alone ranks the occurrences.
  public-only  drop every layer derived from the proprietary Getech grids
               (Bouguer anomaly and its derivatives, TMI, DRTP and
               derivatives, and the GLCM textures computed on them), leaving
               the public lithosphere, deep-source potential-field, proximity
               and terrane layers, so the analysis can be reproduced from
               public data alone.

Columns are dropped from the pruned feature set of each region; folds,
seeds and protocols are those of the main results, and each subset is
compared with the same model on all pruned features.

    nice -n 10 python -m paper_analyses.18_feature_ablation [--regions ...]
"""
import argparse
import importlib

import numpy as np
import pandas as pd

from paper_analyses.common import (REGION_ORDER, config, load_pickle, load_region, out_path, save_pickle,
                                   seed_everything)
from paper_analyses.models import rf_factory
from paper_analyses.protocol import paired_bootstrap, project_xy, run_protocol

_fin = importlib.import_module('paper_analyses.11_devnet_final')
PROTOCOLS = ['random', 50.0, 100.0, 200.0]
GETECH_PREFIXES = ('Canada_BA', 'DRTP', 'TMI')
SUBSETS = {
    'no-terrane': lambda c: not c.startswith(('geo_era_', 'lith_')),
    'public-only': lambda c: not c.startswith(GETECH_PREFIXES),
    'no-alkaline-lith': lambda c: c != 'lith_alkaline_favorable',
    'terrane-only': lambda c: c.startswith(('geo_era_', 'lith_')),
}
DEFAULT_SUBSETS = ['no-terrane', 'public-only']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    ap.add_argument('--subsets', nargs='+', default=DEFAULT_SUBSETS, choices=list(SUBSETS))
    args = ap.parse_args()
    final = load_pickle(out_path('devnet_final.pkl'))
    bench = load_pickle(out_path('benchmark_Superior_Cordillera_Grenville.pkl'))
    spat = load_pickle(out_path('spatial_cv_Superior_Cordillera_Grenville.pkl'))
    rows, store = [], {}
    for region in args.regions:
        rec = load_region(region)
        d = rec['pruned']
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        for subset in args.subsets:
            keep = SUBSETS[subset]
            cols = [c for c in d['cols'] if keep(c)]
            j = [d['cols'].index(c) for c in cols]
            for model, fac in [('DevNet', _fin.final_factory(region)), ('RF', rf_factory(region))]:
                for proto in PROTOCOLS:
                    seed_everything()
                    res = run_protocol(proto, d['X_lab'][:, j], d['y'], xy_lab, d['X_unl'][:, j], xy_bg, fac,
                                       config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
                    store[(region, subset, model, proto)] = {**res, 'cols': cols}
                    if model == 'DevNet':
                        ref = final[(region, proto)]
                    else:
                        ref = bench[(region, 'pruned', 'RF')] if proto == 'random' else spat[(region, proto, 'RF')]
                    bt = paired_bootstrap(d['y'], res['oof_mean'], ref['oof_mean'], seed=config.RANDOM_SEED)
                    dd = res['aps'] - ref['aps']
                    pname = 'random' if proto == 'random' else f'{proto:.0f} km'
                    rows.append({'region': region, 'subset': subset, 'n_features': len(cols), 'model': model,
                                 'protocol': pname, 'block_km': 0 if proto == 'random' else proto,
                                 'auc_med': res['auc_med'], 'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0],
                                 'ap_hi': res['ap_ci'][1], 'all_features_ap_med': ref['ap_med'],
                                 'd_ap_vs_all': bt['d_ap'], 'd_ap_lo': bt['d_ap_ci'][0], 'd_ap_hi': bt['d_ap_ci'][1],
                                 'p_gain': bt['p_ap'], 'wins_vs_all': f'{int((dd > 0).sum())}/{len(dd)}'})
                    print(f"[{region} | {subset} ({len(cols)}) | {pname}] {model} AP {res['ap_med']:.3f} "
                          f"(all features {ref['ap_med']:.3f}, d {bt['d_ap']:+.3f}, wins {rows[-1]['wins_vs_all']})",
                          flush=True)
    tag = '_'.join(args.regions) + ('' if args.subsets == DEFAULT_SUBSETS else '_' + '_'.join(args.subsets))
    pd.DataFrame(rows).to_csv(out_path(f'feature_ablation_{tag}.csv'), index=False)
    save_pickle(store, out_path(f'feature_ablation_{tag}.pkl'))


if __name__ == '__main__':
    main()
