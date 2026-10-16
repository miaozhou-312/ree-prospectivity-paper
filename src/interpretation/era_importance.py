"""
Section 13e-2 -- geo_era occurrence and SHAP importance by region.

geo_era is a full one-hot (see src/features/terrane.py) -- every era gets
its own column, none dropped as a reference, precisely so each era's own
SHAP contribution can be read off directly and compared across regions.
This module reuses shap_results as-is (no new SHAP computation):

1. Occurrence rate per region for all 5 eras, over the full candidate +
   positive pool -- an era with 0 occurrences in a region is a
   zero-variance column there (see src/features/collinearity.py) and
   trivially carries no SHAP importance.
2. SHAP importance per era per region, RF vs. DevNet, using the same
   mean(|SHAP|) importance / sign(mean(SHAP)) direction convention as
   src/interpretation/global_importance.py, plus mean(SHAP | era present) --
   the more directly interpretable "when this specific era IS the terrane,
   which way does it push the score."
"""
import numpy as np
import pandas as pd

from src.features.terrane import ERA_DUMMY_GROUPS


def compute_era_importance(trained, shap_results, shap_regions):
    era_occurrence_rows = []
    era_importance_rows = []

    for name in shap_regions:
        if name not in shap_results:
            print(f'[{name}] no shap_results -- run the SHAP pipeline first, skipping.')
            continue
        r = shap_results[name]
        feat_cols = r['feat_cols']
        pts_all = trained[name]['pts']
        explain_pts = r['explain_pts']

        for era in ERA_DUMMY_GROUPS:
            col = f'geo_era_{era}'
            occ_rate = float(pts_all[col].mean(skipna=True)) if col in pts_all.columns else np.nan
            n_present = int((pts_all[col] == 1).sum()) if col in pts_all.columns else 0
            survived = col in feat_cols
            era_occurrence_rows.append({'region': name, 'era': era, 'occurrence_rate': occ_rate,
                                         'n_points_this_era': n_present, 'n_total_points': len(pts_all),
                                         'survived_pruning': survived})
            if not survived:
                continue

            i = feat_cols.index(col)
            shap_rf_col = r['shap_rf'][:, i]
            shap_dn_col = r['shap_dn'][:, i]
            era_is_1 = (explain_pts[col].values == 1)

            era_importance_rows.append({
                'region': name, 'era': era,
                'RF_mean_abs_shap':     float(np.abs(shap_rf_col).mean()),
                'RF_when_present':      float(shap_rf_col[era_is_1].mean()) if era_is_1.any() else np.nan,
                'DevNet_mean_abs_shap': float(np.abs(shap_dn_col).mean()),
                'DevNet_when_present':  float(shap_dn_col[era_is_1].mean()) if era_is_1.any() else np.nan,
                'n_present_in_explain_set': int(era_is_1.sum()),
            })

    era_occurrence_df = pd.DataFrame(era_occurrence_rows)
    era_importance_df = pd.DataFrame(era_importance_rows)
    return era_occurrence_df, era_importance_df


def print_era_importance_report(era_occurrence_df, era_importance_df, shap_regions):
    pd.set_option('display.width', 170)
    print('=== geo_era occurrence rate per region (candidate + positive pool) ===')
    pivot_occ = era_occurrence_df.pivot(index='era', columns='region', values='occurrence_rate').reindex(ERA_DUMMY_GROUPS)
    print(pivot_occ.round(4).to_string())

    zero_occ = era_occurrence_df[era_occurrence_df['n_points_this_era'] == 0]
    if len(zero_occ):
        print('\nEras with ZERO occurrences (zero-variance column in that region, no SHAP possible):')
        print(zero_occ[['region', 'era']].to_string(index=False))

    pruned_out = era_occurrence_df[~era_occurrence_df['survived_pruning'] & (era_occurrence_df['n_points_this_era'] > 0)]
    if len(pruned_out):
        print('\nEras present in the data but pruned out by prune_collinear (no SHAP value below):')
        print(pruned_out[['region', 'era', 'occurrence_rate']].round(4).to_string(index=False))

    print('\n=== geo_era SHAP importance per region (RF vs DevNet) ===')
    print(era_importance_df.round(4).to_string(index=False))

    print('\n=== Most influential era per region (ranked by RF+DevNet combined mean|SHAP|) ===')
    era_importance_df = era_importance_df.copy()
    era_importance_df['combined_importance'] = era_importance_df['RF_mean_abs_shap'] + era_importance_df['DevNet_mean_abs_shap']
    for name in shap_regions:
        sub = era_importance_df[era_importance_df['region'] == name].sort_values('combined_importance', ascending=False)
        if len(sub) == 0:
            print(f'  [{name}] no geo_era columns survived pruning -- nothing to rank.')
            continue
        for _, row in sub.iterrows():
            rf_dir = 'more prospective' if row['RF_when_present'] > 0 else 'less prospective'
            dn_dir = 'more prospective' if row['DevNet_when_present'] > 0 else 'less prospective'
            print(f"  [{name}] {row['era']:12s}  RF: importance={row['RF_mean_abs_shap']:.4f}, "
                  f"present->{rf_dir} ({row['RF_when_present']:+.4f})  |  "
                  f"DevNet: importance={row['DevNet_mean_abs_shap']:.4f}, present->{dn_dir} ({row['DevNet_when_present']:+.4f})")
    return era_importance_df


def plot_era_importance(era_importance_df, shap_regions):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, name in zip(axes, shap_regions):
        sub = era_importance_df[era_importance_df['region'] == name].set_index('era').reindex(ERA_DUMMY_GROUPS)
        y = np.arange(len(ERA_DUMMY_GROUPS))
        h = 0.38
        ax.barh(y + h / 2, sub['RF_mean_abs_shap'].fillna(0).values, height=h, color='#2a78d6', label='RF')
        ax.barh(y - h / 2, sub['DevNet_mean_abs_shap'].fillna(0).values, height=h, color='#d03b3b', label='DevNet')
        ax.set_yticks(y); ax.set_yticklabels(ERA_DUMMY_GROUPS, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('mean(|SHAP|)')
        ax.set_title(name, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.25); ax.set_axisbelow(True)
    fig.suptitle('geo_era SHAP importance by region', fontsize=12)
    plt.tight_layout()
    plt.show()
