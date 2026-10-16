"""
Stage 8 -- why is DevNet not the best model everywhere? Diagnostics that
separate data, model-class and training explanations.

  RF_xy        RF on projected coordinates only. If it scores well under
               random CV, random-CV skill is largely spatial memorisation.
  LogReg       L2 logistic regression, balanced (a smooth, essentially
               linear discriminative model): does DevNet behave like "a
               smooth model", i.e. is the gap to RF a model-class gap?
  DevNet_long  DevNet with 10x the gradient steps (3,000 instead of 300).
               With 300 steps of 16 background + 16 positive rows, each
               background row is seen ~once while each positive is seen
               ~400 times; the original DevNet trains on ~50x more rows.
  DevNet_qt    DevNet on quantile-normalised inputs instead of z-scores
               (MLPs, unlike trees, are sensitive to skewed inputs).
  DevNet_long_qt  both.

Plus a coarse-layer "fingerprint" count: how many background points share
the exact LAB / Moho value of some occurrence.

Every variant is fixed a priori (10x steps; standard quantile transform);
nothing is tuned on these folds. Run at low priority:

    nice -n 10 python -m paper_analyses.08_devnet_diagnostics [--regions ...]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from paper_analyses.common import (REGION_ORDER, config, dn_train_kwargs, load_region, model_cfg,
                                   out_path, save_pickle, seed_everything)
from paper_analyses.models import DEVICE, rf_factory
from paper_analyses.protocol import project_xy, repeated_random_cv, repeated_spatial_cv
from src.models.devnet import train_devnet

LONG_STEPS = 3000


def devnet_variant_factory(region, steps, transform):
    cfg, kw = model_cfg(region), dn_train_kwargs()
    kw['dn_epochs'] = steps

    def make(X_bg_raw):
        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            X_pos = X_tr_raw[y_tr == 1]
            X_train = np.vstack([X_pos, X_bg_raw])
            y_train = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg_raw))]
            imp = SimpleImputer(strategy='median').fit(X_train)
            tf = (QuantileTransformer(output_distribution='normal', n_quantiles=1000,
                                      random_state=config.RANDOM_SEED) if transform == 'quantile'
                  else StandardScaler()).fit(imp.transform(X_train))
            Xt, Xv = tf.transform(imp.transform(X_train)), tf.transform(imp.transform(X_va_raw))
            Xv_t = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
            acc = np.zeros(len(Xv))
            for k in range(config.DN_RESTARTS):
                m, _ = train_devnet(Xt, y_train, 'deviation', seed=config.DN_SEED + seed_counter * 1000 + k,
                                    model_cfg=cfg, device=DEVICE, **kw)
                m.eval()
                with torch.no_grad():
                    s = m(Xv_t).cpu().numpy()
                acc += rankdata(s) / len(s)
            return acc / config.DN_RESTARTS
        return scorer
    return make


def logreg_factory(region):
    def make(X_bg_raw):
        def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
            X_pos = X_tr_raw[y_tr == 1]
            X = np.vstack([X_pos, X_bg_raw])
            y = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg_raw))]
            pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                             ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000))])
            return pipe.fit(X, y).predict_proba(X_va_raw)[:, 1]
        return scorer
    return make


def fingerprint(rec, cols):
    """Share of background points whose LAB or Moho value equals, exactly,
    that of at least one occurrence (coarse grids as location codes)."""
    X = np.vstack([rec['full']['X_lab'], rec['full']['X_unl']])
    y = np.r_[rec['full']['y'], np.zeros(len(rec['full']['X_unl']))]
    out = {}
    for name in ['GeophysicsLAB_USCanada', 'USCanada_Moho']:
        j = cols.index(name)
        pos_vals = set(np.round(X[y == 1, j], 6))
        bg = np.round(X[y == 0, j], 6)
        out[name] = {'unique_values_background': int(len(np.unique(bg))),
                     'share_bg_matching_an_occurrence_value': float(np.isin(bg, list(pos_vals)).mean()),
                     'occurrences_sharing_value_with_another': int(len(y[y == 1]) - len(pos_vals))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=REGION_ORDER)
    ap.add_argument('--block', type=float, default=100.0)
    args = ap.parse_args()

    rows, store, fps = [], {}, {}
    for region in args.regions:
        rec = load_region(region)
        d = rec['pruned']
        fps[region] = fingerprint(rec, rec['full']['cols'])
        print(f'[{region}] coarse-layer fingerprint: {fps[region]}', flush=True)
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        variants = {
            'RF_xy': (rf_factory(region), xy_lab, xy_bg),
            'LogReg': (logreg_factory(region), d['X_lab'], d['X_unl']),
            'DevNet_qt': (devnet_variant_factory(region, config.DN_EPOCHS, 'quantile'), d['X_lab'], d['X_unl']),
            'DevNet_long': (devnet_variant_factory(region, LONG_STEPS, 'standard'), d['X_lab'], d['X_unl']),
            'DevNet_long_qt': (devnet_variant_factory(region, LONG_STEPS, 'quantile'), d['X_lab'], d['X_unl']),
        }
        for name, (fac, X_lab, X_bg) in variants.items():
            for proto in ['random', f'{args.block:.0f} km']:
                seed_everything()
                t0 = time.time()
                if proto == 'random':
                    res = repeated_random_cv(X_lab, d['y'], X_bg, fac, config.CV_N_SPLITS,
                                             config.CV_N_REPEATS, config.RANDOM_SEED)
                else:
                    res = repeated_spatial_cv(X_lab, d['y'], xy_lab, X_bg, xy_bg, fac, block_km=args.block,
                                              n_splits=config.CV_N_SPLITS, n_repeats=config.CV_N_REPEATS,
                                              seed=config.RANDOM_SEED)
                store[(region, name, proto)] = res
                rows.append({'region': region, 'model': name, 'protocol': proto,
                             'auc_med': res['auc_med'], 'auc_lo': res['auc_ci'][0], 'auc_hi': res['auc_ci'][1],
                             'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1],
                             'minutes': (time.time() - t0) / 60})
                print(f"[{region} | {proto}] {name:15s} AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f} "
                      f"[{res['ap_ci'][0]:.3f},{res['ap_ci'][1]:.3f}] ({rows[-1]['minutes']:.1f} min)", flush=True)
    tag = '_'.join(args.regions)
    pd.DataFrame(rows).to_csv(out_path(f'devnet_diagnostics_{tag}.csv'), index=False)
    save_pickle({'store': store, 'fingerprint': fps}, out_path(f'devnet_diagnostics_{tag}.pkl'))


if __name__ == '__main__':
    main()
