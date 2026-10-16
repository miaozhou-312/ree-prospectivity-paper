"""
Superseded code, kept for the record rather than deleted.

Former Section 14f -- "Why does DevNet down-weight some Occurrence points?"
(SHAP, reusing the Section 13 SHAP pipeline). For every Occurrence-tier
positive whose DevNet OOF score fell below its region's DevNet median, this
looked up its SHAP values and reported whether its geophysical/structural
evidence was weak relative to the region -- framed as an explanation for why
DevNet was "down-weighting" that point.

Removed after Section 14a-14e's own hypothesis test (Mann-Whitney U,
Deposit/Prospect vs. Occurrence OOF scores) came back non-significant for
DevNet in this run -- i.e. DevNet does not actually score Occurrence points
systematically lower than Deposit/Prospect points here, so there is no real
effect for this cell to be explaining. The code below would still run
regardless: a median split always puts ~half the Occurrence points "below
median" whether or not the group-level difference is real, so it would keep
generating plausible-looking per-point narratives even under the null result
above -- manufacturing the appearance of an effect that the actual test
didn't support. `src/evaluation/label_confidence_stratification.py` (14a-14e)
is unaffected and still reports that honest null result.

Not reviewed or maintained -- kept only so the notebook's history is
traceable.
"""
import numpy as np
import pandas as pd

_GEOPHYS_KEYWORDS = ['gravity', 'mag_', 'worm', 'fault', 'margin', 'glcm']


def explain_low_score_occurrences(strat_points, shap_results, run_regions):
    low_score_occ_shap = {}

    for name in run_regions:
        if name not in shap_results:
            print(f'[{name}] not in the SHAP explain set -- skipping.')
            continue

        df = strat_points[name]
        dn_median = df['dn_oof_score'].median()
        low_occ = df[(df['occurrence_type'] == 'Occurrence') & (df['dn_oof_score'] < dn_median)]
        if len(low_occ) == 0:
            print(f'[{name}] no occurrence-type positives fall below the DevNet OOF median -- skipping.')
            continue

        r = shap_results[name]
        explain_pts = r['explain_pts'].reset_index(drop=True)
        key_to_row = {
            (round(la, 6), round(lo, 6)): i
            for i, (la, lo) in enumerate(zip(explain_pts['lat'], explain_pts['lon']))
        }

        geophys_idx = [j for j, f in enumerate(r['feat_cols'])
                       if any(kw in f.lower() for kw in _GEOPHYS_KEYWORDS)]
        if not geophys_idx:
            print(f'[{name}] no geophysical/structural features survived collinearity pruning '
                  f'-- cannot assess geophysical-evidence strength, skipping.')
            continue
        region_geophys_abs = np.abs(r['shap_dn'][:, geophys_idx]).mean(axis=1)
        region_geophys_median = float(np.median(region_geophys_abs))

        rows_out = []
        for _, prow in low_occ.iterrows():
            idx = key_to_row.get((round(prow['lat'], 6), round(prow['lon'], 6)))
            if idx is None:
                print(f"    lat={prow['lat']:.3f}, lon={prow['lon']:.3f}: not in this region's SHAP "
                      f"explain set (SHAP_EXPLAIN_N may have excluded it) -- skipping.")
                continue
            sv = r['shap_dn'][idx]
            top_idx = np.argsort(-np.abs(sv))[:5]
            top_feats = '; '.join(f'{r["feat_cols"][j]} ({sv[j]:+.3f})' for j in top_idx)
            this_geophys_abs = float(np.abs(sv[geophys_idx]).mean())
            rows_out.append({
                'lat': prow['lat'], 'lon': prow['lon'],
                'dn_oof_score': prow['dn_oof_score'],
                'top5_shap_features': top_feats,
                'mean_abs_shap_geophys': this_geophys_abs,
                'region_median_abs_shap_geophys': region_geophys_median,
                'weak_geophysical_evidence': this_geophys_abs < region_geophys_median,
            })
        if not rows_out:
            continue

        out_df = pd.DataFrame(rows_out).sort_values('dn_oof_score').reset_index(drop=True)
        low_score_occ_shap[name] = out_df
        n_weak = int(out_df['weak_geophysical_evidence'].sum())
        print(f'\n[{name}] {len(out_df)} DevNet-deprioritized Occurrence points explained:')
        print(out_df.to_string(index=False))
        print(f'    -> {n_weak}/{len(out_df)} have below-region-median geophysical/structural SHAP '
              f'contribution, consistent with DevNet down-weighting them because the geophysical '
              f'case at that point is itself weak, not arbitrarily.')

    print('\nSHAP local explanations available for:', list(low_score_occ_shap.keys()))
    return low_score_occ_shap
