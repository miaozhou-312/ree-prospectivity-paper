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

**Revised manuscript** (`main_src_revised.tex` → `manuscript_revised.pdf`).
Here "DevNet" is the final configuration (quantile-normal inputs, 16 positive
+ 240 background rows per step, 300 steps), nnPU uses the same inputs, and
the z-scored DevNet variants of Stages 01–02 and 08–10 appear only in the
ablation appendix. The original manuscript and its generator are unchanged.

```bash
python -m paper_analyses.11_devnet_final           # final DevNet: random CV + 50/100/200 km blocks, paired tests
python -m paper_analyses.12_final_production       # production fits, SHAP, 0.1 deg grid scores, Clay-Howells
python -m paper_analyses.14_final_suite            # nnPU with quantile inputs, full features, margin, class prior
python -m paper_analyses.15_augmentation_final     # augmentation experiments with the final DevNet
python -m paper_analyses.13_maps                   # ArcGIS-style Superior and Kapuskasing maps
python -m paper_analyses.16_nested_selection       # DevNet design chosen inside each outer fold (selection optimism)
python -m paper_analyses.17_controls               # MLP-BCE (supervised control), RF and HGB tuned by inner spatial CV
python -m paper_analyses.18_feature_ablation       # no terrane dummies; public layers only
python -m paper_analyses.19_region_diagnostics     # label structure, univariate signal, far-neighbour index
python -m paper_analyses.20_occurrence_breakdown   # per-occurrence percentiles, prediction-area curves
python -m paper_analyses.21_map_uncertainty        # leave-one-occurrence-out stability of the Superior maps
python -m paper_analyses.13_maps stability         # stability maps from Stage 21
python -m paper_analyses.06b_figures_tables_revised  # figures + table bodies -> generated_revised/
python paper/build.py revised                      # compile manuscript_revised.pdf
```

**Second revision** (`main_src_revised2.tex` → `manuscript_revised2.pdf`).
The main analysis uses feature set 3: the geophysical and structural
features, distances to a generalised regional geology and 10 km
neighbourhood statistics (Stage 24), and the raw H3 lithology, era and
contact classes (Stage 26). Feature set 1 is the thesis encoding (four
lithology groups + era), set 2 drops every point-level map class. Analyses
the second revision reports unchanged (augmentation, protocol audit,
geography-only baseline, nested selection, ablation, margin, province
diagnostics) use feature set 1 and say so. Earlier versions are unchanged.

```bash
python -m paper_analyses.22_label_contamination --features all --protocol random   # also: no-terrane, 100
python -m paper_analyses.23_contamination_summary  # drop vs hide, DevNet vs MLP-BCE at k = 8 / 5 / 3
python -m paper_analyses.24_extended_features      # regional geology distances, 10 km neighbourhood statistics
python -m paper_analyses.26_geology_reencode       # raw H3 lithology / era one-hot, contact flags
python -m paper_analyses.25_extended_benchmark                          # feature set 2, all models and protocols
python -m paper_analyses.25_extended_benchmark --featureset reencoded   # feature set 3 (+ k = 3, no-alkalic runs)
python -m paper_analyses.27_featureset_comparison  # the three feature sets side by side
python -m paper_analyses.28_controls_fs3           # RF and HGB tuned by inner spatial CV, feature set 3
python -m paper_analyses.29_production_fs3         # production fits, SHAP, grid scores, Clay-Howells, stability
python -m paper_analyses.06c_figures_tables_revised2  # figures + table bodies -> generated_revised2/
python paper/build.py revised2                     # compile manuscript_revised2.pdf
python paper/check_claims_revised2.py              # numbers typed in the text against the outputs
```

**Third revision** (`main_src_revised3.tex` → `manuscript_revised3.pdf`). The
same analyses with MLP-BCE recast as an ablation of DevNet ("DevNet-BCE":
the deviation loss replaced by cross-entropy), plus three tests of the
semi-supervised premise whose predictions were written to
`out/semisup_predictions.txt` before the runs:

```bash
python -m paper_analyses.32_semisupervised_tests --experiment prepare   # predictions + near-occurrence points
python -m paper_analyses.32_semisupervised_tests --experiment E1 --featureset reencoded   # also E2, E3; extended
python -m paper_analyses.33_semisupervised_summary   # sign tests of the predicted effects, curves
python -m paper_analyses.06d_figures_tables_revised3  # changed tables -> generated_revised3/, figures *_r3.png
python paper/build.py revised3                        # falls back to generated_revised2/ for unchanged tables
python paper/check_claims_revised3.py
```

**Fourth revision** (`main_src_revised4.tex` → `manuscript_revised4.pdf`). A
simplified version showing only feature set 3 (with "without point-level map
classes" as a sensitivity), no p values, and a condensed summary of Stages
32-33 for all three provinces. Every analysis that earlier revisions ran on
the thesis encoding was rerun on feature set 3 by setting
`REE_FEATURESET=reencoded`, which makes `load_region()` serve the set-3
matrices and sends outputs to `out/fs3/` (originals untouched):

```bash
../work/run_fs3.sh                                    # Stages 01, 02, 04, 05 (phase A)
../work/run_fs3_bc.sh                                 # Stages 08-11, 14-16, scheduled by dependency
python -m paper_analyses.06e_figures_tables_revised4  # changed tables -> generated_revised4/, figures *_r4.png
python paper/build.py revised4                        # falls back to generated_revised3/ and _revised2/
python paper/check_claims_revised4.py
```

**Fifth revision** (`main_src_revised5.tex` → `manuscript_revised5.pdf`). The
fourth revision with the national part below in front of the provincial one
and XGBoost (Stage 47) in the provincial tables; the provincial protocol
audit moves to an appendix.

```bash
python -m paper_analyses.06f_figures_tables_revised5  # changed tables -> generated_revised5/, figures *_r5.png
python paper/build.py revised5                        # falls back to generated_revised4/, _revised3/, _revised2/
python paper/check_claims_revised5.py                 # 56 claims, incl. every national number in the text
```

**Sixth revision** (`main_src_revised6.tex` → `manuscript_revised6.pdf`). The
fifth revision shortened for submission: at most ten figures and ten tables
in the main text, tables of results only (no paired differences or repeat
counts), the national part as the introduction to the provincial study
(S2-grid design against ours, transfer between provinces), and without the
DevNet-BCE ablation, the tests of the semi-supervised premise, nested
selection, map stability and the province diagnostics.

```bash
python -m paper_analyses.06g_figures_tables_revised6  # changed tables -> generated_revised6/, fig_workflow_r6.png
python paper/build.py revised6                        # falls back to generated_revised5/ ... _revised2/
python paper/check_claims_revised6.py                 # 34 claims
```

**Seventh revision: public data only** (`main_src_revised7.tex` →
`manuscript_revised7.pdf`). The seven derived gravity and magnetic grids
(BA 40/100 km high-pass, BA THD tilt, DRTP and its 1VD, THD and THD tilt) are
computed from the public Bouguer anomaly, THD and TMI by Stage 50, which also
measures how close they are to the versions used before (Pearson r 0.90-0.999
over the grids). GLCM textures are taken on all eleven gravity and magnetic
grids (`REE_GLCM=all`) and pruning keeps the useful ones. Every result is
recomputed in a separate work and output directory (`../work_public/env.sh`:
`REE_PAPER_WORK=../work_public`, `REE_RASTER_DIR=../dataset/rasters_public`,
outputs in `out/public/` and `out/public/national/`), so the earlier results
stay untouched. The regional tables are rebuilt by Stages 00, 24 and 26 (the
points are identical); Stage 51 recomputes the grid-derived columns of the
national table and the S2 grid.

```bash
python -m paper_analyses.50_derivative_reproduction gravity      # also: magnetic_geo, compare, features_geo
source ../work_public/env.sh
python -m paper_analyses.00_build_features
(unset REE_FEATURESET; python -m paper_analyses.24_extended_features; python -m paper_analyses.26_geology_reencode)
python -m paper_analyses.51_public_features national
python -m paper_analyses.51_public_features s2
# then Stages 01, 02, 04 (+c), 08-11, 14, 15, 25 (both feature sets), 28, 47, 42 (transfer), 43 (R0-R5), 45
# Stage 43 also with the 1D CNN of Parsa et al. (2024); R5 ran on an AWS Spot c7a.16xlarge (about 15 min, --workers 21)
python -m paper_analyses.43_parsa_ladder --step ladder --rungs R0 R1 R2 R2c R3c R3 R4 --clf CNN
python -m paper_analyses.43_parsa_ladder --step ladder --rungs R5 --models CNN-Parsa --clf CNN --workers 21
# ArcGIS Pro layers (EPSG:3978) of the four Table 2 models on every S2 L12 cell -> ../work_public/gis_export/national_s2/
python -m paper_analyses.52_parsa_maps
python -m paper_analyses.52_parsa_maps --step figures   # figures_r7/fig_map_national[_prob|_logprob]_*.png, fig_area_national_r7.png
python -m paper_analyses.52_parsa_maps --step diagnostics   # numbers quoted with the national maps (Fig. 4)
# Superior maps of XGBoost, bagging-PU and nnPU in the style of Fig. 10 (reuses Stage 29's grid inputs)
(unset REE_FEATURESET; python -m paper_analyses.53_superior_maps_others)
# design-effect figure (Fig. 4) and the feature-space figure of the national negatives
python -m paper_analyses.54_argument_figures
# SHAP of XGBoost, bagging-PU and nnPU and the five-model SHAP figure
(unset REE_FEATURESET; python -m paper_analyses.55_shap_others)
(unset REE_FEATURESET; python -c "import importlib; importlib.import_module('paper_analyses.21_map_uncertainty').grid_features()"; python -m paper_analyses.29_production_fs3)
python -m paper_analyses.06h_figures_tables_revised7              # generated_revised7/, figures_r7/
python paper/splice_revised7.py                                  # table bodies into main_src_revised7.tex
python paper/build.py revised7
python paper/check_claims_revised7.py
```

**National scale** (Stages 40-49). One label table, one feature pipeline
and one evaluation for Canada and the three provinces, so that differences
between the two scales can be attributed to scale. The national table
reuses the regional pools inside the study provinces (every national point
there is also a regional one) and draws new pools elsewhere at Superior's
density; feature set 3 is built with the regional functions and rules
(`national.py`). Rasters, GLCM quantisation and the generalised geology are
taken per geological province, so a point has the same value at both scales
(Stage 41 checks that the three provinces reproduce the regional values
exactly; Stage 48 shows what computing them over the whole country cost).
Each model uses the columns pruned on its own training table: national
models the national columns, province models (transfer, pooling) the
province's regional columns and network capacity. XGBoost uses fixed
parameters from J. Fang's national study and runs each fit in a
PyTorch-free subprocess (`xgb_worker.py`; the two OpenMP runtimes crash
in one process on macOS). Outputs go to `out/national/` and
`../work/national/`; results from the first, country-extent table are kept
in `out/national/before_fix/`.

```bash
python -m paper_analyses.40_national_labels        # national outline, provinces, label table (data/national/)
python -m paper_analyses.41_national_features      # national table, feature set 3 (~2 min)
python -m paper_analyses.47_xgboost_regional       # XGBoost in the three provinces (main folds)
../work/run_national_all.sh                        # Stages 42 (N1, N2, N5), 44 (N3), 45 (N4), 43 (S2 grid + ladder), 49
python -m paper_analyses.48_national_feature_audit # why the country-extent table scored provinces lower
python -m paper_analyses.46_national_summary       # tables and figures
```

Stage 49 tests, with RF as the control, why DevNet is weak nationally:
province offsets in its scores (P1), a per-province background reference
(P2) and province one-hot inputs (P3); the hypothesis and predictions are
written in the script.

Stages 30 and 31 test the Superior (30) and all three (31) maps against
whole-rock geochemistry (Gard et al., 2019 compilation; La+Ce+Nd, a clr-PCA
affinity index, carbonatite/foid rock names). They are not part of the
second revision. Stage 30 is superseded: it took the nearest 0.1 degree grid
node (median 3.5 km away) on maps that vary at the scale of single H3 cells,
whereas Stage 31 scores every sample location directly; both compare shares
with the sampling baseline rather than with area.

With several stages running at once, set `OMP_NUM_THREADS=2`: the
networks are tiny and PyTorch's default thread pool otherwise competes with
the tree models for cores (results are identical, runs several times
faster). Stages 16 and 17 use the nested inner spatial CV of `nested.py`.

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
