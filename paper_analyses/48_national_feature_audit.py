"""
Stage 48 -- why do national features score a province lower than the
regional pipeline? Regional-only training and testing in each study
province (its positives, regional evaluation and training pools, the
folds of the main results), with four feature tables:

    A  regional columns, regional values        (the paper's set 3)
    B  regional columns, national GLCM / generalised-geology values
    C  national columns, national values        (Stages 42-44)
    D  national columns, regional GLCM / generalised-geology values

Only the GLCM texture (grey levels quantised over the study area's raster)
and the generalised-geology distances (map generalised within the study
area's bounding box) take different values at the same point; every other
column is identical. A-B and D-C measure the value effect, A-D and B-C the
column effect.

It reads national_table_country_extent.pkl, the Stage-41 table as first
built (GLCM quantised over Canada, geology generalised over Canada); Stage 41
now computes both by province, which makes B identical to A and D to C.
Results: out/national/before_fix/feature_audit_*.csv.

    python -m paper_analyses.48_national_feature_audit [--regions Superior] [--models DevNet RF]
"""
import argparse
import importlib
import os

import numpy as np
import pandas as pd

from paper_analyses.common import config, save_pickle, seed_everything
from paper_analyses.national import EXT_COLS, STUDY, load_national, regional_rows, summarise
from paper_analyses.protocol import project_xy, run_protocol

_s25 = importlib.import_module('paper_analyses.25_extended_benchmark')
_b = importlib.import_module('paper_analyses.42_national_benchmark')


def is_scale_dependent(c):
    return '_glcm_' in c or c in EXT_COLS and c.startswith('dist_')


def tables(d, region):
    """Feature tables A-D for one province, rows aligned with the regional cache."""
    rec, rcols, XA_lab, XA_unl, ext = _s25.extended_matrices(region, 'reencoded')
    lab_n, unl_n = regional_rows(d['meta'], region)
    out = {}

    def align(reg_df, nat_df):
        key = pd.MultiIndex.from_arrays([nat_df.lat.round(5), nat_df.lon.round(5)])
        pos = pd.Series(np.arange(len(nat_df)), index=key)
        j = pos.reindex(pd.MultiIndex.from_arrays([reg_df.lat.round(5), reg_df.lon.round(5)])).values
        assert not np.isnan(j).any(), 'regional point missing from the national table'
        return nat_df.iloc[j.astype(int)].reset_index(drop=True)
    N_lab, N_unl = align(rec['lab'], lab_n), align(rec['unl'], unl_n)
    reg_lab = pd.concat([rec['lab'].reset_index(drop=True), ext['lab']], axis=1)
    reg_unl = pd.concat([rec['unl'].reset_index(drop=True), ext['unl']], axis=1)
    ncols = d['cols']
    out['A'] = (XA_lab, XA_unl, rcols)
    XB_lab, XB_unl = XA_lab.copy(), XA_unl.copy()
    for k, c in enumerate(rcols):
        if is_scale_dependent(c):
            XB_lab[:, k], XB_unl[:, k] = N_lab[c].values, N_unl[c].values
    out['B'] = (XB_lab, XB_unl, rcols)
    out['C'] = (N_lab[ncols].values.astype(float), N_unl[ncols].values.astype(float), ncols)
    XD_lab, XD_unl = out['C'][0].copy(), out['C'][1].copy()
    for k, c in enumerate(ncols):
        if is_scale_dependent(c):
            XD_lab[:, k], XD_unl[:, k] = reg_lab[c].values, reg_unl[c].values
    out['D'] = (XD_lab, XD_unl, ncols)
    return rec, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=STUDY)
    ap.add_argument('--models', nargs='+', default=['DevNet', 'RF'])
    ap.add_argument('--protocols', nargs='+', default=['random', '100'])
    args = ap.parse_args()
    d = load_national('set3', table='national_table_country_extent.pkl')
    rows = []
    for region in args.regions:
        rec, T = tables(d, region)
        y = rec['pruned']['y']
        xy_lab = project_xy(rec['lab']['lat'], rec['lab']['lon'], config.PROJECTED_CRS) / 1000.0
        xy_bg = project_xy(rec['unl']['lat'], rec['unl']['lon'], config.PROJECTED_CRS) / 1000.0
        for model in args.models:
            for p in args.protocols:
                proto = 'random' if p == 'random' else float(p)
                for name, (Xl, Xu, cols) in T.items():
                    seed_everything()
                    res = run_protocol(proto, Xl, y, xy_lab, Xu, xy_bg, _b.FACTORIES[model](region),
                                       config.CV_N_SPLITS, config.CV_N_REPEATS, config.RANDOM_SEED)
                    rows.append(summarise(y, res['oof'], region=region, model=model, protocol=_b.pname(proto),
                                          table=name, n_features=len(cols)))
                    r = rows[-1]
                    print(f"[{region} | {model} | {_b.pname(proto)} | table {name}] AP {r['ap_med']:.3f} "
                          f"AUC {r['auc_med']:.3f}", flush=True)
    tag = '_'.join(args.regions) + '_' + '_'.join(args.models)
    pd.DataFrame(rows).to_csv(_b.out(os.path.join('before_fix', f'feature_audit_{tag}.csv')), index=False)


if __name__ == '__main__':
    main()
