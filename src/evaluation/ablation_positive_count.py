"""
Section 11e -- Cordillera positive-count ablation (small-sample regime,
repeated CV).

Question: as the number of available positives shrinks, does DevNet's
semi-supervised edge over RF grow? Since the shared-background-pool design
(src/evaluation/run_experiment.py), RF and DevNet are both trained on the
identical fixed `train_unlabeled` background pool -- so any AUC/AP gap that
widens as positives shrink can be attributed to the modeling paradigm
itself, not to a difference in which background points each model was
trained on.

Design: Cordillera only, pruned feature set only. Only the positive set is
subsampled -- `eval_neg` and `train_unlabeled` are reused byte-for-byte from
the main experiment's own region_pts['Cordillera']. For each positive-count
level below the full set, `POS_ABLATION_REPEATS` independent random
subsamples are drawn; each is evaluated with repeated CV via
`repeated_pooled_auc_ap`, so every individual repeat value is stored and
reported, none cherry-picked. The floor is CV_N_SPLITS positives (5), since
StratifiedKFold needs at least one positive per fold.

Speed knobs (local to this ablation only, main experiment untouched):
POS_ABLATION_CV_REPEATS / POS_ABLATION_DN_RESTARTS default to roughly 50-60%
of the main experiment's CV_N_REPEATS / DN_RESTARTS.

The self-check re-runs a single-pass pooled_oof_eval at the main
experiment's own settings, purely to confirm the reused eval_neg /
train_unlabeled pools and scorer construction are byte-identical to the
main experiment's -- expected to match exactly regardless of how the sweep
is tuned for speed.
"""
import numpy as np
import pandas as pd

from src.evaluation.cv_protocol import pooled_oof_eval, repeated_pooled_auc_ap
from src.evaluation.run_experiment import _dn_train_kwargs
from src.features.feature_matrix import get_raw_features
from src.models.devnet import devnet_semisup_fold_scorer_factory, get_device
from src.models.random_forest import make_rf_params, rf_shared_bg_fold_scorer_factory

CORD_NAME                = 'Cordillera'
POS_ABLATION_REPEATS     = 5     # random-subsample repeats per (non-full) level
POS_ABLATION_SEED0       = 2001
POS_ABLATION_CV_REPEATS  = 3     # LOCAL override of CV_N_REPEATS (6) -- speed only
POS_ABLATION_DN_RESTARTS = 3     # LOCAL override of DN_RESTARTS (5) -- speed only


def run_positive_count_ablation(config, region_pts, region_feats, trained, results,
                                 pos_ablation_repeats=POS_ABLATION_REPEATS,
                                 seed0=POS_ABLATION_SEED0,
                                 cv_repeats=POS_ABLATION_CV_REPEATS,
                                 dn_restarts=POS_ABLATION_DN_RESTARTS):
    device = get_device(config.FORCE_CPU)
    dn_train_kwargs = _dn_train_kwargs(config)
    rf_params = make_rf_params(config.RANDOM_SEED)

    pts_cord    = region_pts[CORD_NAME]
    feat_cols   = region_feats[CORD_NAME]
    model_cfg   = trained[CORD_NAME]['model_cfg']

    pos_full_tbl  = pts_cord[pts_cord['pool'] == 'positive'].reset_index(drop=True)
    eval_neg_tbl  = pts_cord[pts_cord['pool'] == 'eval_neg'].reset_index(drop=True)
    unlabeled_tbl = pts_cord[pts_cord['pool'] == 'train_unlabeled'].reset_index(drop=True)
    n_pos_full = len(pos_full_tbl)
    print(f'[{CORD_NAME}] full positive count = {n_pos_full}  |  eval_neg = {len(eval_neg_tbl)}  |  '
          f'train_unlabeled = {len(unlabeled_tbl)}  (background pools held FIXED throughout)')

    n_pos_targets = sorted(
        {n for n in [n_pos_full, 20, 15, 10, 7, 5] if config.CV_N_SPLITS <= n <= n_pos_full}, reverse=True)
    print(f'Positive-count levels: {n_pos_targets}  |  subsamples per non-full level: '
          f'{pos_ablation_repeats}  |  CV: {config.CV_N_SPLITS}x{cv_repeats} repeated '
          f'(main experiment uses {config.CV_N_SPLITS}x{config.CV_N_REPEATS})  |  DevNet restarts: '
          f'{dn_restarts} (main experiment uses {config.DN_RESTARTS})')

    X_unlab, _, _ = get_raw_features(unlabeled_tbl, feat_subset=feat_cols)
    rf_scorer_cord = rf_shared_bg_fold_scorer_factory(X_unlab, rf_params)
    dn_scorer_cord = devnet_semisup_fold_scorer_factory(
        X_unlab, model_cfg=model_cfg, device=device, train_kwargs=dn_train_kwargs,
        restarts=dn_restarts, base_seed=config.DN_SEED)

    rows = []
    for n_pos in n_pos_targets:
        is_full   = (n_pos == n_pos_full)
        n_subsamp = 1 if is_full else pos_ablation_repeats
        print(f'\n--- n_pos={n_pos} '
              f'({"full set, single deterministic subsample" if is_full else f"{n_subsamp} random subsamples"}) ---')
        for rep in range(n_subsamp):
            if is_full:
                pos_sub, seed_tag = pos_full_tbl, 'full'
            else:
                seed = seed0 + n_pos * 100 + rep
                idx = np.random.RandomState(seed).choice(n_pos_full, size=n_pos, replace=False)
                pos_sub, seed_tag = pos_full_tbl.iloc[idx].reset_index(drop=True), seed

            pts_labeled = pd.concat([pos_sub, eval_neg_tbl], ignore_index=True)
            X_lab, y, _ = get_raw_features(pts_labeled, feat_subset=feat_cols)

            rf_aucs, rf_aps = repeated_pooled_auc_ap(
                X_lab, y, rf_scorer_cord, n_splits=config.CV_N_SPLITS,
                n_repeats=cv_repeats, seed=config.RANDOM_SEED)
            dn_aucs, dn_aps = repeated_pooled_auc_ap(
                X_lab, y, dn_scorer_cord, n_splits=config.CV_N_SPLITS,
                n_repeats=cv_repeats, seed=config.RANDOM_SEED)

            print(f'  rep={rep} seed={seed_tag}: '
                  f'RF AUC={np.median(rf_aucs):.3f} AP={np.median(rf_aps):.3f}  |  '
                  f'DevNet AUC={np.median(dn_aucs):.3f} AP={np.median(dn_aps):.3f}')

            for r_auc, r_ap in zip(rf_aucs, rf_aps):
                rows.append({'n_pos': n_pos, 'is_full': is_full, 'subsample': rep,
                             'model': 'RF', 'oof_auc': r_auc, 'oof_ap': r_ap})
            for d_auc, d_ap in zip(dn_aucs, dn_aps):
                rows.append({'n_pos': n_pos, 'is_full': is_full, 'subsample': rep,
                             'model': 'DevNet(deviation)', 'oof_auc': d_auc, 'oof_ap': d_ap})

    pos_ablation_df = pd.DataFrame(rows)

    _self_check(config, pos_full_tbl, eval_neg_tbl, feat_cols,
                rf_scorer_cord, X_unlab, model_cfg, device, dn_train_kwargs, results)

    summary = (pos_ablation_df
               .groupby(['n_pos', 'model'])
               .agg(auc_med=('oof_auc', 'median'), auc_lo=('oof_auc', 'min'), auc_hi=('oof_auc', 'max'),
                    ap_med =('oof_ap',  'median'), ap_lo =('oof_ap',  'min'), ap_hi =('oof_ap',  'max'))
               .reset_index()
               .sort_values(['n_pos', 'model'], ascending=[False, True]))
    pd.set_option('display.width', 170)
    print('\n=== Median [min, max] over all subsamples x CV repeats (no cherry-picking) ===')
    print(summary.round(4).to_string(index=False))
    _print_readouts(summary)
    return pos_ablation_df, summary


def _self_check(config, pos_full_tbl, eval_neg_tbl, feat_cols,
                 rf_scorer_cord, X_unlab, model_cfg, device, dn_train_kwargs, results):
    print("\n=== Self-check: full positive set + reused pools reproduce the main experiment's "
          "Cordillera pruned result ===")
    pts_labeled_full = pd.concat([pos_full_tbl, eval_neg_tbl], ignore_index=True)
    X_lab_full, y_full, _ = get_raw_features(pts_labeled_full, feat_subset=feat_cols)
    check_rf_auc, _, _ = pooled_oof_eval(X_lab_full, y_full, rf_scorer_cord,
                                          n_splits=config.CV_N_SPLITS, seed=config.RANDOM_SEED)
    check_dn_auc, _, _ = pooled_oof_eval(
        X_lab_full, y_full,
        devnet_semisup_fold_scorer_factory(X_unlab, model_cfg=model_cfg, device=device,
                                            train_kwargs=dn_train_kwargs,
                                            restarts=config.DN_RESTARTS, base_seed=config.DN_SEED),
        n_splits=config.CV_N_SPLITS, seed=config.RANDOM_SEED)
    ref_rf_auc = results[CORD_NAME]['RF']['pruned']['oof_auc']
    ref_dn_auc = results[CORD_NAME]['DevNet(deviation)']['pruned']['oof_auc']
    print(f'  RF     single-pass AUC: check={check_rf_auc:.4f}  vs  main experiment={ref_rf_auc:.4f}')
    print(f'  DevNet single-pass AUC: check={check_dn_auc:.4f}  vs  main experiment={ref_dn_auc:.4f}')
    if not (np.isclose(check_rf_auc, ref_rf_auc, atol=1e-9) and np.isclose(check_dn_auc, ref_dn_auc, atol=1e-9)):
        raise RuntimeError('Positive-count ablation does not reproduce the main experiment at the '
                            'full-set level -- check that eval_neg/train_unlabeled pools were reused '
                            'unchanged.')
    print('  OK -- matches exactly.')


def _print_readouts(summary):
    print('\n=== Key read-outs ===')
    piv_auc = summary.pivot(index='n_pos', columns='model', values='auc_med').sort_index(ascending=False)
    cross = None
    prev = None
    for n_pos in piv_auc.index:
        rf_a, dn_a = piv_auc.loc[n_pos, 'RF'], piv_auc.loc[n_pos, 'DevNet(deviation)']
        lead = 'DevNet' if dn_a > rf_a else 'RF'
        if prev is not None and lead != prev:
            cross = n_pos
        prev = lead
    print(f'  AUC leader by n_pos (high->low): '
          + ', '.join(f'{n}:{("DevNet" if piv_auc.loc[n,"DevNet(deviation)"]>piv_auc.loc[n,"RF"] else "RF")}'
                       for n in piv_auc.index))
    print(f'  -> AUC crossover at n_pos = {cross}' if cross else '  -> no AUC crossover observed')

    print('  AP spread (max-min) by n_pos, DevNet / RF:')
    for n_pos in piv_auc.index:
        d = summary[(summary.n_pos == n_pos) & (summary.model == 'DevNet(deviation)')]
        r = summary[(summary.n_pos == n_pos) & (summary.model == 'RF')]
        dsp = float(d['ap_hi'].iloc[0] - d['ap_lo'].iloc[0])
        rsp = float(r['ap_hi'].iloc[0] - r['ap_lo'].iloc[0])
        print(f'    n_pos={n_pos:2d}:  DevNet spread={dsp:.3f}   RF spread={rsp:.3f}')


def plot_pos_ablation(summary, cv_repeats=POS_ABLATION_CV_REPEATS,
                       dn_restarts=POS_ABLATION_DN_RESTARTS,
                       pos_ablation_repeats=POS_ABLATION_REPEATS,
                       cv_n_splits=5):
    import matplotlib.pyplot as plt

    colors = {'RF': '#2a78d6', 'DevNet(deviation)': '#d03b3b'}
    models = ['RF', 'DevNet(deviation)']
    levels = sorted(summary['n_pos'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for (metric, mlo, mhi), ax in zip(
            [('auc_med', 'auc_lo', 'auc_hi'), ('ap_med', 'ap_lo', 'ap_hi')], axes):
        for m in models:
            s = summary[summary.model == m].set_index('n_pos').loc[levels]
            ax.plot(levels, s[metric], color=colors[m], lw=2, marker='o', label=m)
            ax.fill_between(levels, s[mlo], s[mhi], color=colors[m], alpha=0.15)
        ax.set_xlabel('Number of Cordillera positives used')
        ax.set_ylabel('OOF ' + ('AUC' if 'auc' in metric else 'AP'))
        ax.grid(alpha=0.25); ax.set_axisbelow(True); ax.legend(fontsize=8)
    fig.suptitle(f'{CORD_NAME}: RF vs DevNet as positives shrink '
                 f'(repeated CV {cv_n_splits}x{cv_repeats}, {dn_restarts} DevNet '
                 f'restarts -- speed-tuned; median line, min-max band over {pos_ablation_repeats} '
                 f'subsamples x repeats; background pools fixed)', fontsize=9)
    plt.tight_layout(); plt.show()
