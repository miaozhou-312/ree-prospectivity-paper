"""
Fits one XGBoost model in a fresh interpreter for paper_analyses.models.xgb_factory
(and returns TreeSHAP contributions instead of scores when the job asks for 'shap').

On macOS the OpenMP runtime bundled with PyTorch and the one xgboost links
cannot live in the same process (it crashes), and every analysis process
imports PyTorch. The factory therefore pickles (X, y, X_va, params) to this
module's stdin and reads the scores back; nothing here imports PyTorch.
"""
import pickle
import sys

from xgboost import XGBClassifier


def main():
    d = pickle.load(sys.stdin.buffer)
    m = XGBClassifier(**d['params']).fit(d['X'], d['y'])
    if d.get('shap'):                  # TreeSHAP contributions (log-odds), bias column dropped
        from xgboost import DMatrix
        out = m.get_booster().predict(DMatrix(d['X_va']), pred_contribs=True)[:, :-1]
    else:
        out = m.predict_proba(d['X_va'])[:, 1]
    sys.stdout.buffer.write(pickle.dumps(out))


if __name__ == '__main__':
    main()
