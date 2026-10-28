"""
Stage 23 -- summary of Stage 22 (label scarcity and background contamination).

Per feature set, protocol, model and number of kept positives k:
  * AP with the other positives dropped and with them hidden in the
    background (median over the six repeats);
  * contamination cost = AP(hide) - AP(drop), per repeat on identical folds
    (median and number of repeats in which hiding hurt);
  * DevNet against MLP-BCE in every condition: paired bootstrap on repeat-
    averaged ranks and repeats won.

    python -m paper_analyses.23_contamination_summary
"""
import glob

import numpy as np
import pandas as pd

from paper_analyses.common import config, load_pickle, load_region, out_path
from paper_analyses.protocol import paired_bootstrap


def main():
    y = load_region('Superior')['pruned']['y']
    rows, pairs = [], []
    for f in sorted(glob.glob(out_path('contamination_*.pkl'))):
        fs, proto = f.split('contamination_')[1][:-4].rsplit('_', 1)
        st = load_pickle(f)
        models = sorted({m for m, _, _ in st})
        for m in models:
            full = st[(m, 'all', None)]
            rows.append({'features': fs, 'protocol': proto, 'model': m, 'keep': 'all', 'ap_drop': full['ap_med'],
                         'ap_hide': full['ap_med'], 'cost_med': 0.0, 'cost_neg': ''})
            for k in (8, 5, 3):
                dr, hi = st[(m, 'drop', k)], st[(m, 'hide', k)]
                cost = hi['aps'] - dr['aps']
                rows.append({'features': fs, 'protocol': proto, 'model': m, 'keep': k, 'ap_drop': dr['ap_med'],
                             'ap_hide': hi['ap_med'], 'cost_med': float(np.median(cost)),
                             'cost_neg': f'{int((cost < 0).sum())}/{len(cost)}'})
        for mode, k in [('all', None)] + [(md, k) for k in (8, 5, 3) for md in ('drop', 'hide')]:
            a, b = st[('DevNet', mode, k)], st[('MLP-BCE', mode, k)]
            bt = paired_bootstrap(y, a['oof_mean'], b['oof_mean'], seed=config.RANDOM_SEED)
            pairs.append({'features': fs, 'protocol': proto, 'mode': mode, 'keep': k if k else 'all',
                          'devnet': a['ap_med'], 'mlp': b['ap_med'], 'd_ap': bt['d_ap'], 'p': bt['p_ap'],
                          'devnet_wins': f"{int((a['aps'] > b['aps']).sum())}/{len(a['aps'])}"})
        for k in (8, 5, 3):                       # does DevNet pay less for contamination than MLP-BCE?
            cd = st[('DevNet', 'hide', k)]['aps'] - st[('DevNet', 'drop', k)]['aps']
            cm = st[('MLP-BCE', 'hide', k)]['aps'] - st[('MLP-BCE', 'drop', k)]['aps']
            pairs.append({'features': fs, 'protocol': proto, 'mode': 'cost DevNet - cost MLP', 'keep': k,
                          'devnet': float(np.median(cd)), 'mlp': float(np.median(cm)),
                          'd_ap': float(np.median(cd - cm)), 'p': np.nan,
                          'devnet_wins': f'{int((cd > cm).sum())}/{len(cd)}'})
    rows, pairs = pd.DataFrame(rows), pd.DataFrame(pairs)
    rows.to_csv(out_path('contamination_summary.csv'), index=False)
    pairs.to_csv(out_path('contamination_devnet_vs_mlp.csv'), index=False)
    pd.set_option('display.width', 250)
    for (fs, proto), g in rows.groupby(['features', 'protocol']):
        print(f'\n=== {fs} | {proto}')
        print(g.pivot_table(index='model', columns='keep', values=['ap_drop', 'ap_hide'], aggfunc='first')
              .round(3).to_string())
    print('\nDevNet vs MLP-BCE'); print(pairs.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
