"""
Section 14 -- Label-confidence stratification: does RF memorize
low-confidence positives?

Question: RF and DevNet(deviation) often look very different in how
confidently they score a positive -- RF tends to score essentially every
known positive high, including the lowest-confidence `Occurrence` records,
while DevNet visibly deprioritizes some of them. Is that a real, measurable
pattern?

Hypothesis: RF discriminatively fits positive vs. eval_neg and can simply
memorize every training positive regardless of how weak its underlying
evidence is (`Occurrence` is the lowest-confidence of the three classes --
Deposit/Prospect > Showing > Occurrence). DevNet never gets a curated
negative class to discriminate against; it only learns what the unlabeled
background typically looks like. If that's right, RF should score all three
occurrence-type groups about equally high, while DevNet's scores should
fall off from Deposit/Prospect to Showing to Occurrence.

**Why this must use out-of-fold (OOF) scores, never the production models'
scores**: the production RF/DevNet were both fit on every positive,
Occurrence included. Scoring positives with those production models would
just show RF "confidently" recognizing every point it was trained on --
training-set recall, not evidence about memorization vs. generalization.
The question is "when a positive is held out and the model has never seen
it, does its confidence tier still predict how it's scored?" -- which is
what OOF scoring answers and production-model scoring cannot.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.evaluation.cv_protocol import repeated_pooled_oof_scores, top_decile_flags
from src.evaluation.run_experiment import _dn_train_kwargs
from src.evaluation.prediction_map import DEP_TYPE_ORDER
from src.features.feature_matrix import get_raw_features
from src.models.devnet import devnet_semisup_fold_scorer_factory, get_device
from src.models.random_forest import make_rf_params, rf_fold_scorer_factory

_OCC_TYPE_COL_CANDIDATES = ['dep_type', 'occurrence_type', 'OccurrenceType', 'type', 'Type', 'DepType']


def _find_occurrence_type_col(df):
    """Auto-detect the occurrence-type column in a positive-sample CSV."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in _OCC_TYPE_COL_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    type_cols = [c for c in df.columns if 'type' in c.lower()]
    if len(type_cols) == 1:
        return type_cols[0]
    raise ValueError(f'Could not auto-detect an occurrence-type column among {list(df.columns)}')


def get_positive_occurrence_types(cfg, pos_tbl):
    """Attach each row of pos_tbl (lat/lon) its occurrence-type label
    (deposit/showing/occurrence confidence tier) by exact-coordinate match
    against the region's raw positive CSV (cfg['dep'])."""
    dep = pd.read_csv(cfg['dep']).dropna(subset=['Latitude', 'Longitude'])
    type_col = _find_occurrence_type_col(dep)
    dep_key = list(zip(np.round(dep['Latitude'].values, 6), np.round(dep['Longitude'].values, 6)))
    dep_lookup = dict(zip(dep_key, dep[type_col].values))

    pos_key = list(zip(np.round(pos_tbl['lat'].values, 6), np.round(pos_tbl['lon'].values, 6)))
    occ_type = np.array([dep_lookup.get(k, 'Unknown') for k in pos_key], dtype=object)
    n_unmatched = int((occ_type == 'Unknown').sum())
    if n_unmatched:
        print(f"    WARNING: {n_unmatched}/{len(pos_tbl)} positives could not be matched to an "
              f"occurrence-type label by lat/lon -- check {cfg['dep']!r} for coordinate drift.")
    return occ_type, type_col


def compute_stratified_oof_scores(config, trained, run_regions=None):
    """Section 14a -- repeated-CV OOF scoring for every region, using each
    region's pruned feature set and production model_cfg, then attach each
    positive's occurrence-type label. RF and DevNet use IDENTICAL folds
    (both driven by the same RepeatedStratifiedKFold(seed=RANDOM_SEED))."""
    run_regions = list(config.RUN_REGIONS) if run_regions is None else list(run_regions)
    device = get_device(config.FORCE_CPU)
    dn_train_kwargs = _dn_train_kwargs(config)
    rf_params = make_rf_params(config.RANDOM_SEED)
    rf_fold_scorer = rf_fold_scorer_factory(rf_params)

    strat_points = {}
    for name in run_regions:
        t   = trained[name]
        cfg = config.REGIONS[name]
        pts_all     = t['pts']
        feat_cols   = t['feat_cols']
        model_cfg   = t['model_cfg']

        pts_labeled   = pts_all[pts_all['pool'].isin(['positive', 'eval_neg'])].reset_index(drop=True)
        pts_unlabeled = pts_all[pts_all['pool'] == 'train_unlabeled'].reset_index(drop=True)

        X_lab,   y_lab, _ = get_raw_features(pts_labeled,   feat_subset=feat_cols)
        X_unlab, _,     _ = get_raw_features(pts_unlabeled, feat_subset=feat_cols)

        print(f'[{name}] repeated-CV OOF scoring ({config.CV_N_SPLITS}-fold x {config.CV_N_REPEATS} '
              f'repeats, IDENTICAL folds for RF and DevNet) ...')
        oof_rf = repeated_pooled_oof_scores(X_lab, y_lab, rf_fold_scorer,
                                             n_splits=config.CV_N_SPLITS, n_repeats=config.CV_N_REPEATS,
                                             seed=config.RANDOM_SEED)
        dn_scorer = devnet_semisup_fold_scorer_factory(
            X_unlab, model_cfg=model_cfg, device=device, train_kwargs=dn_train_kwargs,
            restarts=config.DN_RESTARTS, base_seed=config.DN_SEED)
        oof_dn = repeated_pooled_oof_scores(X_lab, y_lab, dn_scorer,
                                             n_splits=config.CV_N_SPLITS, n_repeats=config.CV_N_REPEATS,
                                             seed=config.RANDOM_SEED)

        top10_rf = top_decile_flags(oof_rf)
        top10_dn = top_decile_flags(oof_dn)

        pos_mask = (y_lab == 1)
        pos_tbl  = pts_labeled[pos_mask].reset_index(drop=True)
        occ_type, type_col_used = get_positive_occurrence_types(cfg, pos_tbl)

        strat_points[name] = pd.DataFrame({
            'region':          name,
            'lat':             pos_tbl['lat'].values,
            'lon':             pos_tbl['lon'].values,
            'occurrence_type': occ_type,
            'rf_oof_score':    oof_rf[pos_mask],
            'rf_top10pct':     top10_rf[pos_mask],
            'dn_oof_score':    oof_dn[pos_mask],
            'dn_top10pct':     top10_dn[pos_mask],
        })
        print(f"  [{name}] occurrence-type column detected: {type_col_used!r}  "
              f"counts: {strat_points[name]['occurrence_type'].value_counts().to_dict()}")

    strat_positives = (pd.concat(strat_points.values(), ignore_index=True)
                        if strat_points else pd.DataFrame())
    print(f'\nTotal positives with OOF scores across {len(strat_points)} region(s): {len(strat_positives)}')
    return strat_points, strat_positives


def stratified_hitrate_table(df, order=DEP_TYPE_ORDER):
    """Section 14b -- for each occurrence-type group, n, each model's mean
    OOF score, and each model's hit rate (fraction landing in the region's
    top-decile OOF score)."""
    rows = []
    for grp in order:
        sub = df[df['occurrence_type'] == grp]
        if len(sub) == 0:
            continue
        rows.append({
            'occurrence_type':         grp,
            'n':                       len(sub),
            'RF_oof_mean':             sub['rf_oof_score'].mean(),
            'RF_hitrate_top10pct':     sub['rf_top10pct'].mean(),
            'DevNet_oof_mean':         sub['dn_oof_score'].mean(),
            'DevNet_hitrate_top10pct': sub['dn_top10pct'].mean(),
        })
    return pd.DataFrame(rows)


def plot_stratified_boxplot(df, title, order=DEP_TYPE_ORDER, random_seed=42):
    """Section 14c -- RF (blue) vs. DevNet (red) OOF score by
    occurrence-type tier. Individual points are jittered on top of each box
    since several groups have single-digit n."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    order = [g for g in order if (df['occurrence_type'] == g).any()]
    if not order:
        print(f'  [{title}] no occurrence-type groups present -- skipping plot.')
        return

    pos_rf = np.arange(len(order)) - 0.18
    pos_dn = np.arange(len(order)) + 0.18
    rf_data = [df.loc[df['occurrence_type'] == g, 'rf_oof_score'].values for g in order]
    dn_data = [df.loc[df['occurrence_type'] == g, 'dn_oof_score'].values for g in order]

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 2.4), 5))
    bp_rf = ax.boxplot(rf_data, positions=pos_rf, widths=0.32, patch_artist=True, showfliers=False)
    bp_dn = ax.boxplot(dn_data, positions=pos_dn, widths=0.32, patch_artist=True, showfliers=False)
    for box in bp_rf['boxes']:
        box.set(facecolor='#2a78d6', alpha=0.5, edgecolor='#2a78d6')
    for box in bp_dn['boxes']:
        box.set(facecolor='#d03b3b', alpha=0.5, edgecolor='#d03b3b')
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp_rf[element]:
            line.set(color='#2a78d6')
        for line in bp_dn[element]:
            line.set(color='#d03b3b')

    jitter_rng = np.random.RandomState(random_seed)
    for i, g in enumerate(order):
        yv_rf = df.loc[df['occurrence_type'] == g, 'rf_oof_score'].values
        ax.scatter(jitter_rng.normal(pos_rf[i], 0.03, size=len(yv_rf)), yv_rf,
                   color='#123a63', s=22, alpha=0.75, zorder=5)
        yv_dn = df.loc[df['occurrence_type'] == g, 'dn_oof_score'].values
        ax.scatter(jitter_rng.normal(pos_dn[i], 0.03, size=len(yv_dn)), yv_dn,
                   color='#6e1414', s=22, alpha=0.75, zorder=5)

    ax.axhline(0.90, color='#898781', ls='--', lw=1)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f'{g}\n(n={int((df["occurrence_type"] == g).sum())})' for g in order])
    ax.set_ylabel('Out-of-fold score\n(pooled percentile rank, repeated-CV mean)')
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title, fontsize=11, fontweight='bold')
    legend_handles = [
        mpatches.Patch(facecolor='#2a78d6', alpha=0.5, edgecolor='#2a78d6', label='RF'),
        mpatches.Patch(facecolor='#d03b3b', alpha=0.5, edgecolor='#d03b3b', label='DevNet'),
        plt.Line2D([0], [0], color='#898781', ls='--', lw=1, label='top-10% threshold'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(axis='y', color='#e1e0d9', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def mannwhitney_deposit_vs_occurrence(df, score_col, label):
    """Section 14d -- one-sided Mann-Whitney U, H1: Deposit/Prospect OOF
    scores > Occurrence OOF scores."""
    dep = df.loc[df['occurrence_type'] == 'Deposit/Prospect', score_col].values
    occ = df.loc[df['occurrence_type'] == 'Occurrence',       score_col].values
    if len(dep) < 3 or len(occ) < 3:
        print(f'    {label}: n_deposit={len(dep)}, n_occurrence={len(occ)} -- '
              f'fewer than 3 in one group, skipping Mann-Whitney U (not testable).')
        return None
    U, p = mannwhitneyu(dep, occ, alternative='greater')
    flag = '[SIGNIFICANT p<0.05]' if p < 0.05 else '[not significant]'
    print(f'    {label}: deposit(n={len(dep)}) vs occurrence(n={len(occ)})  '
          f'U={U:.1f}  p={p:.4f}  {flag}')
    return {'U': float(U), 'p': float(p), 'n_deposit': len(dep), 'n_occurrence': len(occ)}


def run_stratified_tests(strat_points, strat_positives, run_regions):
    print('Mann-Whitney U, H1 = Deposit/Prospect OOF score > Occurrence OOF score (one-sided):')
    strat_tests = {}
    for name in run_regions:
        print(f'\n[{name}]')
        strat_tests[name] = {
            'RF':     mannwhitney_deposit_vs_occurrence(strat_points[name], 'rf_oof_score', 'RF'),
            'DevNet': mannwhitney_deposit_vs_occurrence(strat_points[name], 'dn_oof_score', 'DevNet'),
        }
    print('\n[All regions combined]')
    strat_tests['combined'] = {
        'RF':     mannwhitney_deposit_vs_occurrence(strat_positives, 'rf_oof_score', 'RF'),
        'DevNet': mannwhitney_deposit_vs_occurrence(strat_positives, 'dn_oof_score', 'DevNet'),
    }
    return strat_tests


def _fmt_p(r):
    return f"p={r['p']:.4f}" if r is not None else 'not testable (n<3)'


def print_stratified_conclusion(strat_tests, run_regions):
    """Section 14e -- one-sentence, auto-generated conclusion per region and
    for the combined pool."""
    print('=== Summary ===')
    for name in list(run_regions) + ['combined']:
        rf_r, dn_r = strat_tests[name]['RF'], strat_tests[name]['DevNet']
        rf_sig = (rf_r is not None) and (rf_r['p'] < 0.05)
        dn_sig = (dn_r is not None) and (dn_r['p'] < 0.05)

        if dn_sig and not rf_sig:
            verdict = (f'DevNet significantly down-weights low-confidence occurrence-type positives '
                       f'({_fmt_p(dn_r)}), while RF does not ({_fmt_p(rf_r)}), indicating RF memorizes '
                       f'low-quality labels whereas DevNet is robust to label noise.')
        elif dn_sig and rf_sig:
            verdict = (f'Both models significantly down-weight occurrence-type positives '
                       f'(RF {_fmt_p(rf_r)}, DevNet {_fmt_p(dn_r)}) -- the hypothesized RF/DevNet '
                       f'asymmetry is not supported here.')
        elif not dn_sig and not rf_sig:
            verdict = (f'Neither model shows a significant deposit-vs-occurrence gap '
                       f'(RF {_fmt_p(rf_r)}, DevNet {_fmt_p(dn_r)}) -- likely underpowered (small n) '
                       f'rather than evidence against the hypothesis.')
        else:
            verdict = (f'RF shows a significant gap ({_fmt_p(rf_r)}) but DevNet does not '
                       f'({_fmt_p(dn_r)}) -- the opposite of the hypothesized direction.')
        print(f'[{name}] {verdict}')
