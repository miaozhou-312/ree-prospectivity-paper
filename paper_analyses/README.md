# Journal-paper analyses

Scripts that produce every number, table and figure in the journal
manuscript (`paper/manuscript/`), built on the packaged pipeline in
[`../src/`](../src/). They extend the thesis analysis with AP intervals and
paired tests, two positive–unlabelled baselines, spatially blocked
cross-validation, fold-safe positive augmentation (jitter, SMOTE, WGAN-GP),
an audit of the S2 parent–child validation design of Parsa et al. (2024),
the corrected SHAP direction rule (KNOWN_ISSUES #1) and a blind check at
Clay-Howells.

Set up the environment and data as in the [main README](../README.md), plus
`pip install s2sphere==0.2.5` (Stage 5 only). Run from the repository root:

```bash
export PYTHONPATH=. MPLBACKEND=Agg
python -m paper_analyses.00_build_features         # ~4 min, caches region tables in ../work/
python -m paper_analyses.01_benchmark              # random CV, 4 models, AP/AUC intervals, bootstrap
python -m paper_analyses.02_spatial_cv             # 50 / 100 / 200 km blocked CV
python -m paper_analyses.03_shap_directions        # SHAP with Spearman-based directions
python -m paper_analyses.04_augmentation           # jitter / SMOTE / WGAN-GP, random CV + diagnostics
python -m paper_analyses.04_augmentation --part c  # the same under spatial blocks
python -m paper_analyses.05_protocol_audit         # S2 parent–child audit
python -m paper_analyses.07_external_check         # Clay-Howells blind check
python -m paper_analyses.08_devnet_diagnostics     # coordinates-only RF, logistic regression, DevNet variants
python -m paper_analyses.09_devnet_qt_sensitivity  # appendix D: DevNet with quantile-normal inputs
python -m paper_analyses.10_training_budget --variant bg   # appendix D: background-heavy batches
python -m paper_analyses.10_training_budget --variant es   # appendix D: nested spatial early stopping
python -m paper_analyses.06_figures_tables         # figures + LaTeX table bodies
python paper/build.py                              # splice tables, compile manuscript.pdf (Tectonic)
```

Stages 01–05 and 07 only read the Stage-0 cache and can run in parallel;
each takes minutes on a laptop CPU. Results go to `paper_analyses/out/`
(git-ignored).

| Module | Contents |
|---|---|
| `common.py` | Paths, discovery with the notebook's pinned GLCM rasters, region cache. |
| `models.py` | Scorer factories: RF (per-fold seed), RF fixed seed (thesis), DevNet, bagging-PU, nnPU. |
| `protocol.py` | Repeated random CV, spatially blocked CV with background purging, paired bootstrap. |
| `augment.py` | Fold-safe synthetic positives and the memorisation diagnostic. |
| `figstyle.py` | Feature labels and model colours. |

**Reproducibility.** Seeds are fixed and, since the KNOWN_ISSUES #5 fix to
`src/models/devnet.py`, reruns are bit-for-bit identical on one machine.
RF AUC is sensitive to the RF seed at this sample size (most points receive
P = 0 exactly), which is why `RF` varies its seed per fold and
`RF_fixedseed` reproduces the thesis configuration.
