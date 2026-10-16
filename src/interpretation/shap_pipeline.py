"""
Section 13b -- Per-region SHAP computation.

Explain-set construction per region: all positives (rare, most interesting)
plus a random sample of eval_neg up to SHAP_EXPLAIN_N total -- the same
underlying points are used for both models. DevNet's background sample is
drawn separately from the region's train_unlabeled pool (its own training
distribution); RF needs no explicit background.

Same features, same explain-set, model-appropriate preprocessing: both
models are explained on the region's pruned feature set and on the same
underlying points. Each model's own production imputer/scaler transforms
those raw points before SHAP runs, since RF and DevNet were fit on
different pools (RF: positive+eval_neg, DevNet: positive+unlabeled).
"""
import pandas as pd

from src.features.feature_matrix import get_raw_features
from src.interpretation.shap_explainers import explain_devnet, explain_rf
from src.models.devnet import get_device


def compute_shap_results(config, trained, shap_regions, shap_bg_n, shap_explain_n):
    device = get_device(config.FORCE_CPU)
    shap_results = {}

    for name in shap_regions:
        print(f'\n[{name}] computing SHAP ...')
        t = trained[name]
        feat_cols = t['feat_cols']
        pts_all = t['pts']

        pos_rows = pts_all[pts_all['pool'] == 'positive']
        eval_rows = pts_all[pts_all['pool'] == 'eval_neg']
        n_eval_needed = max(0, shap_explain_n - len(pos_rows))
        eval_sample = eval_rows.sample(n=min(n_eval_needed, len(eval_rows)), random_state=config.RANDOM_SEED)
        explain_pts = pd.concat([pos_rows, eval_sample], ignore_index=True)

        unlabeled_rows = pts_all[pts_all['pool'] == 'train_unlabeled']
        bg_pts = unlabeled_rows.sample(n=min(shap_bg_n, len(unlabeled_rows)), random_state=config.RANDOM_SEED)

        X_explain_raw, y_explain, _ = get_raw_features(explain_pts, feat_subset=feat_cols)
        X_bg_raw, _, _ = get_raw_features(bg_pts, feat_subset=feat_cols)
        print(f'  explain set: {len(explain_pts)} points ({int((y_explain==1).sum())} positive, '
              f'{int((y_explain==0).sum())} eval_neg)  |  DevNet background: {len(bg_pts)} unlabeled points')

        # RF: same explain points, RF's own production imputer/scaler
        X_explain_rf = t['rf_scaler'].transform(t['rf_imp'].transform(X_explain_raw))
        shap_rf, ev_rf = explain_rf(t['rf_model'], X_explain_rf)

        # DevNet: same explain points (raw), DevNet's own imputer/scaler --
        # and only the representative ensemble member (dn_ensemble[0], seed=DN_SEED).
        X_explain_dn = t['dn_scaler'].transform(t['dn_imp'].transform(X_explain_raw))
        X_bg_dn      = t['dn_scaler'].transform(t['dn_imp'].transform(X_bg_raw))
        shap_dn, ev_dn, method_dn = explain_devnet(t['dn_ensemble'][0], X_bg_dn, X_explain_dn,
                                                    device, config.DN_SEED)
        print(f'  DevNet explainer used: {method_dn}')

        shap_results[name] = {
            'feat_cols': feat_cols,
            'X_explain_rf': X_explain_rf, 'shap_rf': shap_rf, 'expected_value_rf': ev_rf,
            'X_explain_dn': X_explain_dn, 'shap_dn': shap_dn, 'expected_value_dn': ev_dn,
            'devnet_method': method_dn,
            'X_bg_dn': X_bg_dn,
            'explain_pts': explain_pts,
        }

    print('\nSHAP computed for:', list(shap_results.keys()))
    return shap_results
