"""
Stage 2 -- spatially blocked cross-validation.

Random stratified CV can place a held-out occurrence a few kilometres from
a training occurrence (deduplication only merges points within 2 km), so
spatial autocorrelation may inflate the thesis metrics. Here whole square
blocks (side 50 / 100 / 200 km) are held out together and the training
background is purged of every point inside a held-out block. Six repeats,
each with a random block-grid offset. Same models, features and scorers as
Stage 1; the drop relative to random CV measures the optimism of the
random-split protocol.

    python -m paper_analyses.02_spatial_cv [--regions Superior] [--blocks 50 100 200]
"""
import argparse
import time

import pandas as pd

from paper_analyses.common import (REGION_ORDER, config, load_region, out_path, save_pickle,
                                   seed_everything)
from paper_analyses.models import MODEL_FACTORIES
from paper_analyses.protocol import paired_bootstrap, project_xy, repeated_spatial_cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    ap.add_argument('--models', nargs='+', default=['RF', 'DevNet', 'BaggingPU', 'nnPU'])
    ap.add_argument('--blocks', nargs='+', type=float, default=[50, 100, 200])
    args = ap.parse_args()

    rows, boot_rows, store = [], [], {}
    for region in args.regions:
        rec = load_region(region)
        d = rec['pruned']
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        for block in args.blocks:
            for m in args.models:
                seed_everything()
                t0 = time.time()
                res = repeated_spatial_cv(d['X_lab'], d['y'], xy_lab, d['X_unl'], xy_bg,
                                          MODEL_FACTORIES[m](region), block_km=block,
                                          n_splits=config.CV_N_SPLITS, n_repeats=config.CV_N_REPEATS,
                                          seed=config.RANDOM_SEED)
                store[(region, block, m)] = res
                fi = pd.DataFrame(res['fold_info'])
                rows.append({'region': region, 'block_km': block, 'model': m,
                             'n_pos': int(d['y'].sum()),
                             'pos_blocks_med': float(fi['n_pos_blocks'].median()),
                             'folds_med': float(fi['n_folds'].median()),
                             'bg_purged_per_fold': float(fi['mean_bg_purged'].mean()),
                             'auc_med': res['auc_med'], 'auc_lo': res['auc_ci'][0], 'auc_hi': res['auc_ci'][1],
                             'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1],
                             'minutes': (time.time() - t0) / 60})
                print(f"[{region} | {block:.0f} km] {m:9s} AUC {res['auc_med']:.3f} "
                      f"[{res['auc_ci'][0]:.3f},{res['auc_ci'][1]:.3f}]  AP {res['ap_med']:.3f} "
                      f"[{res['ap_ci'][0]:.3f},{res['ap_ci'][1]:.3f}]  pos blocks~{rows[-1]['pos_blocks_med']:.0f} "
                      f"purged bg/fold~{rows[-1]['bg_purged_per_fold']:.0f}  ({rows[-1]['minutes']:.1f} min)",
                      flush=True)
            if 'DevNet' in args.models:
                for m in args.models:
                    if m == 'DevNet':
                        continue
                    a, b = store[(region, block, 'DevNet')], store[(region, block, m)]
                    bt = paired_bootstrap(d['y'], a['oof_mean'], b['oof_mean'], seed=config.RANDOM_SEED)
                    boot_rows.append({'region': region, 'block_km': block, 'comparison': f'DevNet - {m}',
                                      'ap_wins_of_repeats': f"{int((a['aps'] > b['aps']).sum())}/{len(a['aps'])}",
                                      **bt})
                    print(f"    DevNet - {m}: dAP {bt['d_ap']:+.3f} [{bt['d_ap_ci'][0]:+.3f},"
                          f"{bt['d_ap_ci'][1]:+.3f}] p={bt['p_ap']:.3f}", flush=True)

    tag = '_'.join(args.regions)
    pd.DataFrame(rows).to_csv(out_path(f'spatial_cv_{tag}.csv'), index=False)
    pd.DataFrame(boot_rows).to_csv(out_path(f'spatial_cv_bootstrap_{tag}.csv'), index=False)
    save_pickle(store, out_path(f'spatial_cv_{tag}.pkl'))


if __name__ == '__main__':
    main()
