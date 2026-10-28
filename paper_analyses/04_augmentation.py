"""
Stage 4 -- can generative augmentation of the positive class help in the
label-scarce Superior Province?

Part A (performance). Under the Stage-1 protocol (identical 5x6 repeated
folds), each training fold's real positives are augmented with
mult x n_pos synthetic positives from
  * Gaussian jitter (sigma = 0.1 SD),
  * SMOTE-style interpolation (k = 3),
  * a tabular WGAN-GP trained on that fold's positives only,
for mult in {1, 3, 10}, and RF and DevNet are retrained. Held-out positives
are always real, so any gain is a genuine generalisation gain. The
mult = 0 run is the unaugmented baseline and must reproduce Stage 1.

Part B (what the generator learned). Each augmenter is fit to all 15
Superior positives and 150 samples are drawn; reported are the
nearest-neighbour distance ratio (memorisation test), the share of samples
whose nearest labelled neighbour is a positive rather than background, and
the percentile of the samples under the production DevNet relative to the
background. A PCA view is saved as a figure.

    python -m paper_analyses.04_augmentation [--region Superior]
"""
import argparse
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from paper_analyses import augment  # noqa: E402
from paper_analyses.common import (config, dn_train_kwargs, load_region, model_cfg, out_path,  # noqa: E402
                                   save_pickle, seed_everything)
from paper_analyses.models import DEVICE, devnet_factory, rf_factory  # noqa: E402
from paper_analyses.protocol import paired_bootstrap, repeated_random_cv  # noqa: E402
from src.models.devnet import ensemble_score, train_devnet  # noqa: E402

METHODS = ['jitter', 'smote', 'wgan']
MULTS = [1, 3, 10]


def part_a(region):
    d = load_region(region)['pruned']
    bases = {'RF': rf_factory(region), 'DevNet': devnet_factory(region)}
    rows, store = [], {}
    for model, base in bases.items():
        seed_everything()
        ref = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], base,
                                 config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
        store[(model, 'none', 0)] = ref
        rows.append({'model': model, 'method': 'none', 'mult': 0, 'auc_med': ref['auc_med'],
                     'ap_med': ref['ap_med'], 'ap_lo': ref['ap_ci'][0], 'ap_hi': ref['ap_ci'][1],
                     'auc_lo': ref['auc_ci'][0], 'auc_hi': ref['auc_ci'][1]})
        print(f"[{region}] {model} no augmentation: AUC {ref['auc_med']:.3f} AP {ref['ap_med']:.3f}", flush=True)
        for method in METHODS:
            for mult in MULTS:
                seed_everything()
                t0 = time.time()
                fac = augment.augmented_factory(base, method, mult, d['cols'])
                res = repeated_random_cv(d['X_lab'], d['y'], d['X_unl'], fac,
                                         config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
                store[(model, method, mult)] = res
                bt = paired_bootstrap(d['y'], res['oof_mean'], ref['oof_mean'], seed=config.RANDOM_SEED)
                wins = int((res['aps'] > ref['aps']).sum())
                rows.append({'model': model, 'method': method, 'mult': mult,
                             'auc_med': res['auc_med'], 'auc_lo': res['auc_ci'][0], 'auc_hi': res['auc_ci'][1],
                             'ap_med': res['ap_med'], 'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1],
                             'd_ap_vs_none': bt['d_ap'], 'd_ap_lo': bt['d_ap_ci'][0], 'd_ap_hi': bt['d_ap_ci'][1],
                             'p_ap_gain': bt['p_ap'], 'd_auc_vs_none': bt['d_auc'],
                             'ap_wins_vs_none': f'{wins}/{len(res["aps"])}'})
                print(f"[{region}] {model} + {method} x{mult}: AUC {res['auc_med']:.3f} AP {res['ap_med']:.3f} "
                      f"| dAP vs none {bt['d_ap']:+.3f} [{bt['d_ap_ci'][0]:+.3f},{bt['d_ap_ci'][1]:+.3f}] "
                      f"p={bt['p_ap']:.3f} wins {wins}/6 ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pd.DataFrame(rows).to_csv(out_path(f'augmentation_{region}.csv'), index=False)
    save_pickle(store, out_path(f'augmentation_{region}.pkl'))


def part_b(region, n_syn=150):
    rec = load_region(region)
    d = rec['pruned']
    X_pos = d['X_lab'][d['y'] == 1]
    X_bg = d['X_unl']
    space = augment._Space(X_pos, X_bg, d['cols'])
    Zp, Zb = space.to_z(X_pos), space.to_z(X_bg)

    # production DevNet ensemble on the real data (as in run_experiment)
    X_tr = np.vstack([X_pos, X_bg])
    y_tr = np.r_[np.ones(len(X_pos)), np.zeros(len(X_bg))]
    Ztr = space.to_z(X_tr)
    ens = [train_devnet(Ztr, y_tr, 'deviation', seed=config.DN_SEED + k, model_cfg=model_cfg(region),
                        device=DEVICE, **dn_train_kwargs())[0] for k in range(config.DN_RESTARTS)]

    def dn_percentile(Z):
        s_all = ensemble_score(ens, np.vstack([Zb, Z]), DEVICE)
        pct = rankdata(s_all) / len(s_all)
        return pct[len(Zb):]

    _, hist = augment.train_wgan_gp(Zp, seed=2024, return_history=True)
    samples = {m: space.to_z(augment.synthesize(m, X_pos, X_bg, d['cols'], n_syn, seed=2024))
               for m in METHODS}
    nn_lab = NearestNeighbors(n_neighbors=1).fit(np.vstack([Zp, Zb]))
    rows = []
    # Reference row: real positives against each other (leave-one-out NN), so
    # the ratio is 1 at the median by definition.
    d_rr = NearestNeighbors(n_neighbors=2).fit(Zp).kneighbors(Zp)[0][:, 1]
    rows.append({'source': 'real positives (LOO)', 'nn_ratio_med': 1.0,
                 'nn_ratio_p10': float(np.percentile(d_rr / np.median(d_rr), 10)),
                 'share_nn_is_positive': np.nan,
                 'devnet_pct_med': float(np.median(dn_percentile(Zp)))})
    for m, Zs in samples.items():
        ratio = augment.nn_distance_ratio(Zs, Zp)
        idx = nn_lab.kneighbors(Zs, return_distance=False)[:, 0]
        rows.append({'source': m, 'nn_ratio_med': float(np.median(ratio)),
                     'nn_ratio_p10': float(np.percentile(ratio, 10)),
                     'share_nn_is_positive': float((idx < len(Zp)).mean()),
                     'devnet_pct_med': float(np.median(dn_percentile(Zs)))})
    bg_rows = np.random.default_rng(0).choice(len(Zb), 300, replace=False)
    rows.append({'source': 'background (300 random)', 'nn_ratio_med': float(np.median(
        augment.nn_distance_ratio(Zb[bg_rows], Zp))), 'nn_ratio_p10': np.nan,
        'share_nn_is_positive': np.nan, 'devnet_pct_med': 0.5})
    diag = pd.DataFrame(rows)
    print(diag.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    diag.to_csv(out_path(f'augmentation_diagnostics_{region}.csv'), index=False)

    plot_pca(Zb, Zp, samples, hist, out_path(f'fig_augmentation_{region}.png'))


def pca_inputs(region, n_syn=150):
    """The inputs of part B's figure without the DevNet diagnostics: the
    standardised background and positives, the samples and the WGAN-GP
    training history (same seeds as part B)."""
    d = load_region(region)['pruned']
    X_pos, X_bg = d['X_lab'][d['y'] == 1], d['X_unl']
    space = augment._Space(X_pos, X_bg, d['cols'])
    _, hist = augment.train_wgan_gp(space.to_z(X_pos), seed=2024, return_history=True)
    samples = {m: space.to_z(augment.synthesize(m, X_pos, X_bg, d['cols'], n_syn, seed=2024)) for m in METHODS}
    return space.to_z(X_bg), space.to_z(X_pos), samples, hist


def plot_pca(Zb, Zp, samples, hist, path, sample_label='synthetic'):
    """PCA view (fit on background + real positives) + WGAN-GP training curve."""
    pca = PCA(n_components=2).fit(np.vstack([Zb, Zp]))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    titles = {'jitter': 'Gaussian jitter', 'smote': 'SMOTE', 'wgan': 'WGAN-GP'}
    Pb, Pp = pca.transform(Zb), pca.transform(Zp)
    for ax, m in zip(axes[:3], METHODS):
        Ps = pca.transform(samples[m])
        ax.scatter(Pb[:, 0], Pb[:, 1], s=3, c='#bbbbbb', alpha=0.4, label='background', rasterized=True)
        ax.scatter(Ps[:, 0], Ps[:, 1], s=10, c='#e69f00', alpha=0.7, label=sample_label)
        ax.scatter(Pp[:, 0], Pp[:, 1], s=40, c='#d03b3b', edgecolor='k', lw=0.5, label='real positive')
        ax.set_title(f'({"abc"[METHODS.index(m)]}) {titles[m]}', loc='left', fontsize=11)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}%)')
    axes[0].legend(fontsize=8, loc='best')
    h = np.array(hist)
    axes[3].plot(h[:, 0], h[:, 1], color='#333333', lw=1)
    axes[3].set_xlabel('generator step')
    axes[3].set_ylabel('critic estimate of W-distance')
    axes[3].set_title('(d) WGAN-GP training', loc='left', fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


def part_c(region, blocks=(100.0, 200.0), configs=(('none', 0), ('jitter', 10), ('smote', 3), ('wgan', 3),
                                                   ('wgan', 10))):
    """Does any augmentation gain survive spatially blocked validation
    (Stage 2 protocol; synthetic positives generated from each fold's
    training positives, background purged from held-out blocks)?"""
    from paper_analyses.protocol import project_xy, repeated_spatial_cv
    rec = load_region(region)
    d = rec['pruned']
    xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
    xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
    bases = {'RF': rf_factory(region), 'DevNet': devnet_factory(region)}
    rows, store = [], {}
    for block in blocks:
        for model, base in bases.items():
            ref = None
            for method, mult in configs:
                seed_everything()
                fac = base if method == 'none' else augment.augmented_factory(base, method, mult, d['cols'])
                res = repeated_spatial_cv(d['X_lab'], d['y'], xy_lab, d['X_unl'], xy_bg, fac, block_km=block,
                                          n_splits=config.CV_N_SPLITS, n_repeats=config.CV_N_REPEATS,
                                          seed=config.RANDOM_SEED)
                store[(block, model, method, mult)] = res
                row = {'block_km': block, 'model': model, 'method': method, 'mult': mult,
                       'auc_med': res['auc_med'], 'ap_med': res['ap_med'],
                       'ap_lo': res['ap_ci'][0], 'ap_hi': res['ap_ci'][1]}
                if method == 'none':
                    ref = res
                else:
                    bt = paired_bootstrap(d['y'], res['oof_mean'], ref['oof_mean'], seed=config.RANDOM_SEED)
                    row.update({'d_ap_vs_none': bt['d_ap'], 'd_ap_lo': bt['d_ap_ci'][0],
                                'd_ap_hi': bt['d_ap_ci'][1], 'p_ap_gain': bt['p_ap'],
                                'ap_wins_vs_none': f"{int((res['aps'] > ref['aps']).sum())}/{len(res['aps'])}"})
                rows.append(row)
                print(f"[{region} | {block:.0f} km] {model} {method} x{mult}: AUC {res['auc_med']:.3f} "
                      f"AP {res['ap_med']:.3f}" + (f" | dAP vs none {row['d_ap_vs_none']:+.3f} "
                      f"[{row['d_ap_lo']:+.3f},{row['d_ap_hi']:+.3f}] p={row['p_ap_gain']:.3f} "
                      f"wins {row['ap_wins_vs_none']}" if method != 'none' else ''), flush=True)
    pd.DataFrame(rows).to_csv(out_path(f'augmentation_spatial_{region}.csv'), index=False)
    save_pickle(store, out_path(f'augmentation_spatial_{region}.pkl'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default='Superior')
    ap.add_argument('--part', choices=['a', 'b', 'c', 'both'], default='both')
    args = ap.parse_args()
    if args.part in ('b', 'both'):
        part_b(args.region)
    if args.part in ('a', 'both'):
        part_a(args.region)
    if args.part == 'c':
        part_c(args.region)


if __name__ == '__main__':
    main()
