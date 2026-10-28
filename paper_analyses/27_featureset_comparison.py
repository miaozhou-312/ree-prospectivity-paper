"""
Stage 27 -- the three feature sets side by side.

  set 1  thesis features: pruned geophysics + the four grouped lithology and
         era dummies (Stages 1-2, 11, 14, 17, 22);
  set 2  extended, without point-level map classes (Stage 25 --featureset
         extended);
  set 3  re-encoded raw H3 geology + extended layers (+ radiometrics when
         available) (Stage 25 --featureset reencoded).

Same folds, protocols and seeds throughout. Writes a long table of AP
medians (featureset_comparison.csv) and prints region x model x protocol
pivots, plus the k = 3 label-scarcity condition.

    python -m paper_analyses.27_featureset_comparison
"""
import glob

import pandas as pd

from paper_analyses.common import out_path

PNAME = {0: 'random', 50: '50 km', 100: '100 km', 200: '200 km'}


def set1():
    rows = []
    dn = pd.read_csv(out_path('devnet_final.csv'))
    for _, x in dn.iterrows():
        rows.append(('DevNet', x.region, x.block_km, x.ap_med))
    bm = pd.read_csv(out_path('benchmark_Superior_Cordillera_Grenville.csv'))
    sp = pd.read_csv(out_path('spatial_cv_Superior_Cordillera_Grenville.csv'))
    for m in ('RF', 'BaggingPU'):
        for _, x in bm[(bm.feature_set == 'pruned') & (bm.model == m)].iterrows():
            rows.append((m, x.region, 0, x.ap_med))
        for _, x in sp[sp.model == m].iterrows():
            rows.append((m, x.region, x.block_km, x.ap_med))
    fs = pd.read_csv(out_path('final_suite.csv'))
    for _, x in fs[(fs.model == 'nnPU-QT') & (fs.feature_set == 'pruned')].iterrows():
        rows.append(('nnPU', x.region, x.block_km, x.ap_med))
    for f in glob.glob(out_path('controls_MLPBCE_RFtuned_HGBtuned_*.csv')):
        c = pd.read_csv(f)
        for _, x in c[c.model == 'MLP-BCE'].iterrows():
            rows.append(('MLP-BCE', x.region, x.block_km, x.ap_med))
    out = pd.DataFrame(rows, columns=['model', 'region', 'block_km', 'ap_med']).assign(featureset='1 thesis', keep=0)
    k3 = []
    for f in glob.glob(out_path('contamination_all_*.csv')):
        c = pd.read_csv(f)
        for _, x in c[(c['mode'] == 'drop') & (c.keep == 3) & c.model.isin(['DevNet', 'MLP-BCE'])].iterrows():
            k3.append((x.model, 'Superior', 0 if str(x.protocol) == 'random' else 100, x.ap_med))
    k3 = pd.DataFrame(k3, columns=['model', 'region', 'block_km', 'ap_med']).assign(featureset='1 thesis', keep=3)
    return pd.concat([out, k3], ignore_index=True)


def stage25(prefix, label):
    files = glob.glob(out_path(f'{prefix}_benchmark_*.csv'))
    if not files:
        return pd.DataFrame()
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    t['keep'] = t['keep'].astype(str)
    t = t[t['keep'].isin(['0', '3', 'no-alkalic'])]
    t['model'] = t.apply(lambda x: x.model + (' (no alkalic)' if x['keep'] == 'no-alkalic' else ''), axis=1)
    t['keep'] = t['keep'].map({'0': 0, '3': 3, 'no-alkalic': 0})
    return t[['model', 'region', 'block_km', 'ap_med', 'keep']].assign(featureset=label)


def main():
    t = pd.concat([set1(), stage25('extended', '2 extended'), stage25('reencoded', '3 re-encoded')], ignore_index=True)
    t['protocol'] = t.block_km.astype(int).map(PNAME)
    t.to_csv(out_path('featureset_comparison.csv'), index=False)
    pd.set_option('display.width', 220)
    for region in ('Superior', 'Cordillera', 'Grenville'):
        sub = t[(t.region == region) & (t.keep == 0)]
        p = sub.pivot_table(index=['model'], columns=['featureset', 'protocol'], values='ap_med').round(3)
        p = p.reindex(columns=[(f, pr) for f in sorted(sub.featureset.unique()) for pr in PNAME.values()
                               if (f, pr) in p.columns])
        print(f'\n=== {region} (AP, median of six repeats)'); print(p.to_string())
    k = t[t.keep == 3].pivot_table(index=['region', 'model'], columns=['featureset', 'protocol'], values='ap_med').round(3)
    print('\n=== k = 3 kept training positives'); print(k.to_string())


if __name__ == '__main__':
    main()
