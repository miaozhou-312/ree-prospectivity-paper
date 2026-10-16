"""
Section 7 -- Random Forest, fold-safe pipeline.

`rf_fold_scorer` builds a fresh Pipeline(SimpleImputer(median) ->
StandardScaler -> RandomForestClassifier) and fits it only on the training
fold, then predicts probabilities on the validation fold -- imputer/scaler
are never fit on the whole dataset before CV, which would leak.

`rf_shared_bg_fold_scorer_factory` is the scorer actually used by the main
experiment (Section 9): each fold, RF trains on (that fold's positive
subset + the FULL shared `train_unlabeled` background pool, labeled 0) --
mirroring DevNet's semi-supervised fold scorer's training-set construction
exactly (same external background pool, just labeled 0 instead of left
unlabeled), so RF and DevNet are trained on the IDENTICAL set of background
locations and differ only in label semantics / loss.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_rf_params(random_seed):
    return dict(n_estimators=300, max_features='sqrt',
                class_weight='balanced', random_state=random_seed)


def make_rf_pipeline(rf_params):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('rf',      RandomForestClassifier(**rf_params)),
    ])


def rf_fold_scorer_factory(rf_params):
    """Plain discriminative fold scorer: trains on this fold's own
    (positive, eval_neg) rows only. Superseded by
    rf_shared_bg_fold_scorer_factory for the main experiment, kept for
    reference/reuse."""
    def rf_fold_scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
        pipe = make_rf_pipeline(rf_params)
        pipe.fit(X_tr_raw, y_tr)
        return pipe.predict_proba(X_va_raw)[:, 1]
    return rf_fold_scorer


def rf_shared_bg_fold_scorer_factory(X_train_bg_raw, rf_params):
    """RF fold scorer trained on (this fold's positive subset + the FULL
    shared TRAINING background pool, labeled 0) -- mirrors
    devnet_semisup_fold_scorer_factory's training-set construction exactly
    (same external background pool, just labeled 0 instead of left
    unlabeled), so RF and DevNet are trained on the IDENTICAL set of
    background locations and differ only in label semantics / loss. The
    fold's own eval-pool rows in X_tr_raw are intentionally ignored for
    training (same rationale as DevNet's own scorer) -- the shared
    evaluation pool exists only to be scored against, never to teach either
    model what "negative" looks like during CV."""
    def scorer(X_tr_raw, y_tr, X_va_raw, seed_counter=0):
        X_pos_tr = X_tr_raw[y_tr == 1]
        X_train_raw = np.vstack([X_pos_tr, X_train_bg_raw])
        y_train     = np.concatenate([np.ones(len(X_pos_tr)), np.zeros(len(X_train_bg_raw))])
        pipe = make_rf_pipeline(rf_params)
        pipe.fit(X_train_raw, y_train)
        return pipe.predict_proba(X_va_raw)[:, 1]
    return scorer
