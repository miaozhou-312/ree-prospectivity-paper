"""
Fits one XGBoost model in a fresh interpreter for paper_analyses.models.xgb_factory.

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
    sys.stdout.buffer.write(pickle.dumps(m.predict_proba(d['X_va'])[:, 1]))


if __name__ == '__main__':
    main()
