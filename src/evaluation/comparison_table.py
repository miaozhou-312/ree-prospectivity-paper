"""
Section 11 -- Comparison table.

Full-feature vs. pruned-feature ablation, plus a `data_paradigm` column so
RF and DevNet are never mistaken for having been trained/evaluated on the
same labels: RF is discriminative (positive vs. eval_neg); DevNet is
semi-supervised (positive vs. train_unlabeled for training, scored against
eval_neg).

Caveat: eval_neg is a random background subset of the candidate grid, not
confirmed barren ground -- it may contain a small number of undiscovered
deposits. Every AUC/AP is therefore a conservative estimate.
"""
import pandas as pd


def build_comparison_table(results, run_regions, run_losses):
    active_models = ['RF'] + [f'DevNet({lt})' for lt in run_losses]
    feature_set_names = ['full', 'pruned']

    rows = []
    for name in run_regions:
        for model_name in active_models:
            for fs_name in feature_set_names:
                r = results[name][model_name][fs_name]
                rows.append({
                    'region': name,
                    'model': model_name,
                    'data_paradigm': r['data_paradigm'],
                    'feature_set': fs_name,
                    'n_features': r['n_features'],
                    'pooled_oof_auc': r['oof_auc'],
                    'pooled_oof_ap':  r['oof_ap'],
                    'repeatedCV_median_auc': r['rep_median'],
                    'repeatedCV_ci_lo': r['rep_ci'][0],
                    'repeatedCV_ci_hi': r['rep_ci'][1],
                })
    comparison_df = pd.DataFrame(rows)

    pd.set_option('display.width', 180)
    print('Full vs. pruned feature-set ablation (pooled OOF / repeated-CV), by data paradigm:')
    print(comparison_df.round(3).to_string(index=False))
    print('\nCaveat: eval_neg is a random background subset, not confirmed barren ground -- '
          'it may contain a few undiscovered deposits, so the AUC/AP above are a conservative estimate.')
    return comparison_df
