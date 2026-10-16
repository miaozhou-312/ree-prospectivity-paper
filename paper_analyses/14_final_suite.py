"""
Stage 14 -- remaining runs for the revised manuscript, in which "DevNet"
denotes the final configuration of Stage 11 (quantile-normal inputs,
background-heavy batches).

  * nnPU with the same quantile-normal inputs (nnPU-QT), so that DevNet vs
    nnPU still differs only in the loss: pruned features under random CV and
    50 / 100 / 200 km blocks, the full feature set under random CV, and the
    Superior class-prior sensitivity;
  * the final DevNet on the full feature set (random CV);
  * the final DevNet's deviation-margin sensitivity (a in {2, 3, 8}, random CV);
  * paired comparisons of the final DevNet with nnPU-QT on identical folds.

    nice -n 10 python -m paper_analyses.14_final_suite
"""
import importlib

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from paper_analyses.common import (REGION_ORDER, config, load_pickle, load_region, model_cfg, out_path,
                                   save_pickle, seed_everything)
from paper_analyses.models import DEVICE, _train_nnpu
from paper_analyses.protocol import paired_bootstrap, project_xy, repeated_random_cv, repeated_spatial_cv

_fin = importlib.import_module('paper_analyses.11_devnet_final')
BLOCKS = [50.0, 100.0, 200.0]


def nnpu_qt_factory(region, prior=0.01):
    cfg = model_cfg(region)

    def make(X_bg_raw):
        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            X_pos_raw = X_tr_raw[y_tr == 1]
            tf = _fin.QuantileNormal(np.vstack([X_pos_raw, X_bg_raw]))
            X_pos, X_u, X_va = tf(X_pos_raw), tf(X_bg_raw), tf(X_va_raw)
            Xv = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
            acc = np.zeros(len(X_va))
            for k in range(config.DN_RESTARTS):
                m = _train_nnpu(X_pos, X_u, prior, seed=config.DN_SEED + seed_counter * 1000 + k,
                                cfg=cfg, epochs=config.DN_EPOCHS, lr=config.DN_LR)
                with torch.no_grad():
                    s = m(Xv).cpu().numpy()
                acc += rankdata(s) / len(s)
            return acc / config.DN_RESTARTS
        return scorer
    return make


def summary(res, **keys):
    return {**keys, 'auc_med': res['auc_med'], 'auc_lo': res['auc_ci'][0], 'auc_hi': res['auc_ci'][1],
            'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1]}


def main():
    final = load_pickle(out_path('devnet_final.pkl'))
    rows, boot, store = [], [], {}
    for region in REGION_ORDER:
        rec = load_region(region)
        d, df = rec['pruned'], rec['full']
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        for proto in ['random'] + BLOCKS:
            seed_everything()
            if proto == 'random':
                res = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], nnpu_qt_factory(region),
                                         config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
            else:
                res = repeated_spatial_cv(d['X_lab'], d['y'], xy_lab, d['X_unl'], xy_bg, nnpu_qt_factory(region),
                                          block_km=proto, n_splits=config.CV_N_SPLITS,
                                          n_repeats=config.CV_N_REPEATS, seed=config.RANDOM_SEED)
            pname = 'random' if proto == 'random' else f'{proto:.0f} km'
            store[(region, 'nnPU-QT', 'pruned', proto)] = res
            rows.append(summary(res, region=region, model='nnPU-QT', feature_set='pruned', protocol=pname,
                                block_km=0 if proto == 'random' else proto))
            dn = final[(region, proto)]
            bt = paired_bootstrap(d['y'], dn['oof_mean'], res['oof_mean'], seed=config.RANDOM_SEED)
            dd = dn['aps'] - res['aps']
            boot.append({'region': region, 'protocol': pname, 'block_km': 0 if proto == 'random' else proto,
                         'comparison': 'DevNet - nnPU-QT', **bt, 'wins': f'{int((dd > 0).sum())}/{len(dd)}'})
            print(f"[{region} | {pname}] nnPU-QT AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f} | "
                  f"DevNet - nnPU-QT dAP {bt['d_ap']:+.3f} p={bt['p_ap']:.3f} wins {boot[-1]['wins']}", flush=True)

        for name, fac in [('DevNet', _fin.final_factory(region)), ('nnPU-QT', nnpu_qt_factory(region))]:
            seed_everything()
            res = repeated_random_cv(df['X_lab'], df['y'], df['X_unl'], fac, config.CV_N_SPLITS,
                                     config.CV_N_REPEATS, config.RANDOM_SEED)
            store[(region, name, 'full', 'random')] = res
            rows.append(summary(res, region=region, model=name, feature_set='full', protocol='random', block_km=0))
            print(f"[{region} | full] {name} AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f}", flush=True)

        for a in [2.0, 3.0, 8.0]:
            seed_everything()
            res = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], _fin.final_factory(region, margin=a),
                                     config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
            store[(region, f'DevNet(a={a:g})', 'pruned', 'random')] = res
            rows.append(summary(res, region=region, model=f'DevNet(a={a:g})', feature_set='pruned',
                                protocol='random', block_km=0))
            print(f"[{region}] DevNet margin {a:g}: AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f}", flush=True)

    d = load_region('Superior')['pruned']
    for prior in [0.005, 0.02, 0.05]:
        seed_everything()
        res = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], nnpu_qt_factory('Superior', prior=prior),
                                 config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
        rows.append(summary(res, region='Superior', model=f'nnPU-QT(pi={prior})', feature_set='pruned',
                            protocol='random', block_km=0))
        print(f"[Superior] nnPU-QT pi={prior}: AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(out_path('final_suite.csv'), index=False)
    pd.DataFrame(boot).to_csv(out_path('final_suite_bootstrap.csv'), index=False)
    save_pickle(store, out_path('final_suite.pkl'))


if __name__ == '__main__':
    main()
