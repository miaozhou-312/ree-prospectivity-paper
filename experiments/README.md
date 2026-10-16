# Experiment log

Every analysis run during the project (May–August 2026), including the
directions that were abandoned and the attempts that made things worse.
The notebooks are archived **exactly as they were run, with their stored
outputs**, so each result below can be checked against the notebook that
produced it without re-running anything. The reproducible pipeline that the
paper reports is [`src/`](../src/) plus
[`notebooks/end_to_end_notebook.ipynb`](../notebooks/end_to_end_notebook.ipynb);
these notebooks are the record of how it got there.

**Status key.** *Adopted* — its result or design is part of the final
pipeline. *Superseded* — replaced by a later iteration. *Negative result* —
tried, did not help, not used. *Abandoned* — line of work dropped.
*Diagnostic* — measured a problem, changed no model.

## Index

| # | Notebook | Last run | Question | Status |
|---|---|---|---|---|
| 0a | [`00_data_preparation/extract_canada_data.ipynb`](00_data_preparation/extract_canada_data.ipynb) | 2026-05-30 | Extract Canadian samples from the global geochemistry compilation | Abandoned (input to 1a/1b) |
| 0b | [`00_data_preparation/select_ppm.ipynb`](00_data_preparation/select_ppm.ipynb) | 2026-06-08 | Select element columns from that subset | Abandoned |
| 1a | [`01_geochemistry/multiregion_integrated_pca_lda.ipynb`](01_geochemistry/multiregion_integrated_pca_lda.ipynb) | 2026-06-22 | Integrated geochemistry + terrane + structure + raster table, PCA/LDA, first RF/DevNet drafts | Abandoned |
| 1b | [`01_geochemistry/geochem_pca_lda.ipynb`](01_geochemistry/geochem_pca_lda.ipynb) | 2026-07-02 | Do geochemical signatures separate REE occurrences from background? | Abandoned |
| 2a | [`02_label_analysis/alkaline_vs_carbonatite_similarity.ipynb`](02_label_analysis/alkaline_vs_carbonatite_similarity.ipynb) | 2026-06-27 | Can alkaline-hosted and carbonatite-hosted occurrences share one positive class? | Adopted (no) |
| 3a | [`03_model_iterations/multiregion_rf_devnet.ipynb`](03_model_iterations/multiregion_rf_devnet.ipynb) | 2026-07-05 | v1: geophysics-only RF vs DevNet | Superseded |
| 3b | [`03_model_iterations/multiregion_rf_devnet_cosine_400negative.ipynb`](03_model_iterations/multiregion_rf_devnet_cosine_400negative.ipynb) | 2026-07-06 | v4: fold-safe protocol, corrected DevNet loss | Superseded |
| 3c | [`03_model_iterations/multiregion_rf_devnet_semisup.ipynb`](03_model_iterations/multiregion_rf_devnet_semisup.ipynb) | 2026-07-07 | v5: DevNet trained semi-supervised on an unlabelled pool | Superseded |
| 3d | [`03_model_iterations/multiregion_rf_devnet_semisup_combined_dedup.ipynb`](03_model_iterations/multiregion_rf_devnet_semisup_combined_dedup.ipynb) | 2026-07-20 | v6: combined + deduplicated positives | Superseded |
| 3e | [`03_model_iterations/multiregion_rf_devnet_semisup_v2_devnet_opt.ipynb`](03_model_iterations/multiregion_rf_devnet_semisup_v2_devnet_opt.ipynb) | 2026-07-21 | v6-opt: can DevNet be tuned to close the gap? | Negative result |
| 4a | [`04_diagnostics/raster_coverage_diagnostic.ipynb`](04_diagnostics/raster_coverage_diagnostic.ipynb) | 2026-07-22 | How much of each region has real data in every layer? | Diagnostic |
| 4b | [`04_diagnostics/positive_coverage_check.ipynb`](04_diagnostics/positive_coverage_check.ipynb) | 2026-08-02 | Does buffer-then-clip give every positive full coverage? Build the A∧B domain | Adopted |
| 2b | [`02_label_analysis/positive_label_dedup.ipynb`](02_label_analysis/positive_label_dedup.ipynb) | 2026-08-03 | Merge two occurrence compilations and remove near-duplicates | Adopted |
| 5a | [`05_ablations/multiregion_rf_devnet_no_leakage_guards.ipynb`](05_ablations/multiregion_rf_devnet_no_leakage_guards.ipynb) | 2026-08-03 | What happens without the deposit buffer and label deduplication? | Adopted (ablation) |
| 3f | [`03_model_iterations/multiregion_rf_devnet_sampleable_domain.ipynb`](03_model_iterations/multiregion_rf_devnet_sampleable_domain.ipynb) | 2026-08-19 | v7: final monolithic research notebook | Adopted → refactored into `src/` |
| 6a | [`06_figures/glcm_texture_demo.ipynb`](06_figures/glcm_texture_demo.ipynb) | 2026-08-23 | Illustrate what GLCM textures capture | Figure |

"Last run" is the notebook's last-saved date; several were edited over
weeks. Notebook headers refer to intermediate versions (v2, v3,
`multiregion_rf_devnet_v3.ipynb`, `cordilleran_positive_dedup.ipynb`) whose
changes were folded into the files above; no separate copies survive.

## How the headline numbers moved

Pooled out-of-fold AUC (AP in brackets), full feature set. **These are not
comparable across rows**: the positives, the evaluation background and the
protocol all changed between iterations, and several changes deliberately
made the task harder and more honest. The table shows why each change was
made, not a leaderboard.

| Iteration | Positives C / S / G | Cordillera RF | Cordillera DevNet | Superior RF | Superior DevNet | Grenville RF | Grenville DevNet |
|---|---|---|---|---|---|---|---|
| v1 (3a)* | 43 / 8 / 6 | 0.982 (0.890) | 0.959 | 0.926 (0.458) | 0.819 | 0.823 (0.625) | 0.688 |
| v4 (3b) | 27 / 8 / 5 | 0.970 (0.633) | 0.948 (0.473) | 0.893 (0.154) | 0.899 (0.158) | 0.641 (0.019) | 0.443 (0.009) |
| v5 (3c) | 43 / 8 / 6 | 0.957 (0.872) | 0.941 (0.771) | 0.895 (0.198) | 0.905 (0.212) | 0.808 (0.350) | 0.739 (0.333) |
| v6 (3d) | 29 / 15 / 12 | 0.917 (0.396) | 0.895 (0.105) | 0.953 (0.136) | 0.945 (0.368) | 0.702 (0.123) | 0.698 (0.033) |
| v6-opt (3e) | 29 / 15 / 12 | — | 0.891 (0.109) | — | 0.886 (0.328) | — | 0.698 (0.058) |
| v7 (3f) | 27 / 15 / 12 | 0.919 (0.351) | 0.897 (0.138) | 0.955 (0.227) | 0.963 (0.404) | 0.820 (0.148) | 0.671 (0.042) |
| No leakage guards (5a) | 50 / 21 / 13 | 0.976 (0.767) | 0.926 (0.299) | 0.981 (0.484) | 0.973 (0.474) | 0.808 (0.281) | 0.769 (0.161) |
| **Packaged pipeline** | **27 / 15 / 12** | **0.897 (0.311)** | **0.874 (0.092)** | **0.926 (0.210)** | **0.963 (0.411)** | **0.750 (0.123)** | **0.678 (0.050)** |

\* v1 reports the mean of per-fold AUCs from a single 5-fold split (AP for
RF only), not pooled out-of-fold scores.

The packaged pipeline is the reference for the paper. Its method is v7's,
but its numbers differ because the refactor into `src/` changed the random
background draws (the Cordillera candidate pool, for one, is 16,473 A∧B
points against 16,316 in v7). With 12–27 positives per region, a different
draw moves AP noticeably — a reminder of how sensitive these metrics are at
this sample size.

## Phase 0 — Data preparation

**0a. `extract_canada_data.ipynb`.** Streams the global whole-rock
compilation (Gard et al., 2019, `complete.csv`) in 100k-row chunks and keeps
`country_id == 32`, giving 78,077 Canadian samples × 192 columns. Input paths
are the original Windows paths.

**0b. `select_ppm.ipynb`.** A one-cell helper to keep a few element columns
and drop rows with a null target column. Its only stored run failed with an
`OSError` (the input path was double-quoted inside the string); kept for
completeness, no later work depends on it.

## Phase 1 — Geochemistry (abandoned)

**1a. `multiregion_integrated_pca_lda.ipynb`.** A first attempt at one
integrated feature table per region: geochemistry assigned to points by
inverse-distance weighting, terrane attributes by point-in-polygon,
structure by distance, rasters by point sampling, followed by PCA/LDA and
drafts of RF (v1–v3) and DevNet (v1–v2, including cosine-similarity negative
selection). **No stored outputs** — it was never run to completion, and it
reads intermediate files (`Cordilleran_Terrane.geojson`,
`Cordilleran_structure.geojson`) that were not kept. Some markdown is in
Chinese. Kept because the geophysics-only notebooks (3a onward) grew out of
it.

**1b. `geochem_pca_lda.ipynb`.** Per region: positives get a geochemical
signature by IDW of up to 8 stations within 25 km (power 2); background is
400 measured stations more than 10 km from any occurrence; KNN imputation,
CLR transform, PCA and LDA.

- Element coverage is poor: La is missing in 56–78% of samples, Ce in
  58–76%, Nb in 55–63%, depending on region.
- LDA cross-validated AUC: Cordilleran 0.888 ± 0.023 (38 positives),
  Superior 0.841 ± 0.105 (7 positives); Grenville skipped (3 positives).
- RF on the 23 raw elements: Cordilleran 0.989, Superior 0.679 ± 0.290.

**Why it was dropped** (supervisor meetings of 19 and 30 June 2026, see the
thesis logbook): sampling is far too sparse to give a value at every grid
cell, so a continuous prospectivity map would rest on interpolation across
large gaps. The separability above is also optimistic, because positives are
smoothed interpolations while background points are raw measurements. The
model moved to continuously sampled geophysical rasters.

## Phase 2 — Positive labels

**2a. `alkaline_vs_carbonatite_similarity.ipynb`.** 61 carbonatite and 39
alkaline occurrences, 23 geophysical/proximity features, binary classifiers
under spatial leave-one-block-out CV (4° × 4° blocks). RF AUC 0.732 ± 0.425,
logistic regression 0.804 ± 0.326; excluding the carbonatite-heavy
Cordilleran, RF AUC rises to 0.917 ± 0.186. **Decision: do not merge** —
the two host types are geophysically distinct, so only carbonatite-hosted
occurrences are positives.

**2b. `positive_label_dedup.ipynb`.** Pools the carbonatite records of
Orris et al. (2018) with the Parsa et al. (2024) table (repairing its
encoding and swapped coordinate headers), then merges any points within
2 km (single-linkage, transitive), keeping the member nearest each cluster
centroid. Cordilleran 51 → 28, Superior 21 → 15, Grenville 13 → 12. Many
merges are the same deposit under two names (e.g. `Aley` / `Aley
Carbonatite`). Each point keeps a `dep_type` confidence class. **Output:**
the label tables in [`data/labels/`](../data/labels/).

## Phase 3 — Model iterations

**3a. v1 — `multiregion_rf_devnet.ipynb`.** First geophysics-only model: 17
rasters, fault/margin/worm proximities, H3 terrane attributes and 16 GLCM
textures (5×5 window); 400 negatives per region, half most cosine-dissimilar
to the positives and half random; plain 5-fold CV. Per-fold AUCs swing from
0.125 to 1.0 in Grenville, because each fold's AUC hinges on one or two
positives. A negative-sampling ablation shows cosine-only negatives push
DevNet to AUC 0.999–1.000: the task becomes trivially easy, not the model
good. Further problems found afterwards: the DevNet loss resampled its
Gaussian reference every batch, and the imputer and scaler were fit on all
data before CV (leakage).

**3b. v4 — `multiregion_rf_devnet_cosine_400negative.ipynb`.** Keeps the
data pipeline, replaces modelling and evaluation: deviation loss with a
fixed reference pool, a pairwise ranking-loss alternative, a smaller
regularised network, RF inside a fold-safe `Pipeline`, pooled out-of-fold
AUC/AP with per-fold rank normalisation, and repeated stratified CV
(5 folds × 6 repeats) reported as a median with a 95% interval; adds
unsupervised collinearity pruning. Grenville collapses (DevNet AUC 0.443,
below chance). DevNet is still being asked to separate curated positives
from curated negatives — a discriminative task it has no advantage on.

**3c. v5 — `multiregion_rf_devnet_semisup.ipynb`.** Uses DevNet as designed:
it trains on the positives plus a large **unlabelled** pool and never sees
the evaluation background; RF keeps training on positives versus labelled
background. The evaluation background becomes a uniform random draw instead
of cosine-selected points, removing the artificial AUC saturation. The
ranking loss is dropped (it asserts every unlabelled point ranks below every
positive). Top-scoring unlabelled points are saved as candidate targets.

**3d. v6 — `multiregion_rf_devnet_semisup_combined_dedup.ipynb`.** Only
change: positives come from 2b (combined, deduplicated). DevNet's AP now
clearly exceeds RF's in Superior (0.368 vs 0.136; in v5 the two were level
at ~0.2); RF leads elsewhere.

**3e. v6-opt — `multiregion_rf_devnet_semisup_v2_devnet_opt.ipynb`.
Negative result.** Tries to raise DevNet without touching data, protocol or
RF: positive jitter, relaxed per-region capacity, a deeper network, each
toggleable, with a Superior-only toggle ablation. None helps: Superior AUC
falls from 0.945 to 0.886, Cordilleran is unchanged within noise. Two further
ideas were tried and deleted from the code after a full run: Deep-SAD-style
autoencoder pretraining (Grenville AUC fell to ~0.41–0.49; the encoder
learned the background, positive and unlabelled embeddings were nearly
identical) and an asymmetric inlier loss weight (a smaller net loss). The
header documents both.

**3f. v7 — `multiregion_rf_devnet_sampleable_domain.ipynb`.** The final
research notebook, which `src/` and the end-to-end notebook were refactored
from. Adds, on top of v6: the A∧B sampleable domain from 4a/4b, rasters
clipped to an 8 km-buffered outline so GLCM windows near the edge read real
data, nodata-aware 9×9 GLCM, candidates kept at least 10 km from any
deposit, and **shared background pools** — RF (as label 0) and DevNet (as
unlabelled) train on the same points, so the learning paradigm is the only
difference between them. Also contains the margin, background-sampling and
positive-count ablations, SHAP, capture-rate curves, the label-confidence
test and GIS export. Its former Section 14f is archived in
[`dev/superseded_occurrence_shap.py`](../dev/superseded_occurrence_shap.py)
with the reason it was removed.

## Phase 4 — Coverage diagnostics

**4a. `raster_coverage_diagnostic.ipynb`.** Finds that v6 kept any point
where at least one raster had data and median-imputed the rest. Measures
where all 17 rasters have data simultaneously and, by eroding that mask by
one GLCM window, where textures are computable: erosion would exclude 5
positives (Goéland, Martison Lake, Megiscane Lake, St. Honoré, Terres
Rompues – Se), and 65 of 600 sampled candidates have GLCM windows that touch
nodata. Changes no model; leads to 4b.

**4b. `positive_coverage_check.ipynb`.** Replaces erosion with
buffer-then-clip (8 km). Every positive then has real data in all layers.
Defines the final domain A∧B (all layers present ∧ GLCM window computable):
73.1% of the Cordilleran outline, 98.6% of Superior, 97.2% of Grenville;
54 of 55 positives fall inside (`Lempriere`, Cordilleran, does not), and
each region has enough candidates for the background pools.

## Phase 5 — Ablation

**5a. `multiregion_rf_devnet_no_leakage_guards.ipynb`.** A copy of the v7
pipeline as of 3 August with the two leakage guards removed: no 10 km
exclusion buffer around deposits (`BUFFER_KM = 0`) and no positive
deduplication (`data/labels/ablation_no_dedup/`, 50 / 21 / 13 positives).
Every metric inflates — Cordilleran RF AP 0.767 vs 0.351 in v7, Superior RF
AP 0.484 vs 0.227 — because near-duplicate positives land in both training
and test folds and background points sit next to deposits. Supports keeping
both guards. Section 14 was stopped by hand (`KeyboardInterrupt`); the main
results in Sections 9–12 are complete.

## Phase 6 — Figures

**6a. `glcm_texture_demo.ipynb`.** Synthetic ring-shaped anomaly versus
noise, scored with the pipeline's GLCM settings (8 grey levels, distance 1,
four directions): contrast 0.484 vs 8.507, homogeneity 0.775 vs 0.324,
correlation 0.905 vs 0.012. Used for the method figure.

## Re-running these notebooks

They are archival: they read the providers' original folder names from a
flat working directory (`../dataset/...`, `../geochem/...`, label CSVs in the
working directory) and use the old region name `Cordilleran`.
[`data/README.md` §5](../data/README.md#5-running-the-archived-experiments)
gives the layout and the renames. They were run in the same Python 3.11
environment as the pipeline as it evolved over the summer;
[`requirements.txt`](../requirements.txt) reflects its final state.
Phase 1 additionally needs the geochemistry files described in
[`data/README.md` §4](../data/README.md#4-how-the-derived-inputs-were-built),
and every notebook from 3a on needs the eleven Getech rasters
([`data/README.md` §2.2](../data/README.md#22-proprietary-layers-getech)).
