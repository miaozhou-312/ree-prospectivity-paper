# Known issues

Problems found while packaging the code for publication (October 2026).
Only #5 is fixed (on branch `paper/nrr-superior-analyses`); for the others the
code, the notebook and its stored outputs still match the thesis run. Each code location carries a one-line
`KNOWN ISSUE #n` comment pointing here.

| # | Issue | Affects results? | Affects the paper text? |
|---|---|---|---|
| 1 | SHAP "direction" is computed as the sign of the mean SHAP value, which is not a feature direction | Section 13e table and 13e-2 era directions are wrong | Yes — re-check every statement about which way a feature pushes prospectivity |
| 2 | Table 1 lists a geological-province (cratonic-margin) proximity layer that no run ever used | No (the runs are internally consistent) | Yes — Table 1 and the feature count |
| 3 | Provenance of `Faults_Canada.geojson` is unverified | No | Possibly — the data source credited in Table 1 |
| 4 | `explain_rf` silently returns wrong values with `shap` < 0.45 | No (pinned `shap==0.46.0` is correct) | No |
| 5 | `src/models/devnet.py` drew DevNet's reference pool `_REF` from torch's unseeded start-up generator — **fixed** on branch `paper/nrr-superior-analyses` | DevNet results via `src/` varied run to run (median AP ±0.003, single repeats up to ±0.03); the notebook is unaffected | No — the journal-paper numbers were produced after the fix |

---

## 5. DevNet reference pool was unseeded in `src/` (fixed)

**Where.** `src/models/devnet.py`, module level: `_REF = torch.randn(50000)`.

**What was wrong.** The deviation loss standardises scores against the mean
and standard deviation of `_REF`. In `src/` this tensor is created when the
module is imported, before any caller can seed torch, and torch's initial
seed differs in every process (`torch.initial_seed()` changes run to run).
The reference mean therefore moved (e.g. 0.0036 vs 0.0073 in two runs), and
identical `src/` runs of the Superior DevNet benchmark gave median AP 0.2732
and 0.2756, with single repeats differing by up to 0.03; single-threaded
runs were equally affected, so thread scheduling was not the cause. The
end-to-end notebook seeds torch in Section 1a before Section 8 builds
`_REF`, so its stored outputs were reproducible.

**Fix (applied).** Draw `_REF` from a dedicated generator:
`torch.randn(50000, generator=torch.Generator().manual_seed(42))`.
Every `paper_analyses/` result was regenerated after the fix.

---

## 1. SHAP direction uses sign(mean SHAP)

**Where**
- `src/interpretation/global_importance.py`, `direction_consistency_table`
- `notebooks/end_to_end_notebook.ipynb`, Section 13e (inline copy of the same function)
- `src/interpretation/era_importance.py` and notebook Section 13e-2 (`RF_when_present` / `DevNet_when_present` and the printed "present -> more/less prospective")

**What is wrong.** A feature is labelled "higher -> more prospective" when
its SHAP values average above zero over the explain set. That average
measures whether the feature pulls the explained points' scores above or
below the explainer's baseline; it says nothing about whether *higher values*
of the feature raise the score. The two baselines also differ:

- RF uses `class_weight='balanced'`, so TreeSHAP's baseline (expected value)
  is about 0.5, while RF predicts close to 0 for the background points that
  make up most of the explain set (all positives plus a random `eval_neg`
  sample). Almost every feature therefore gets a negative mean SHAP, and in
  the stored run RF is reported as "higher -> less prospective" for **all 30**
  top features (10 per region).
- DevNet's baseline is its own unlabelled background, which the explain set
  resembles, so its mean SHAP values are small and mostly positive.

The "agree on direction" counts (Cordillera 0/10, Superior 3/10, Grenville
3/10) are
therefore an artefact of the two baselines, not a disagreement between the
models. The era "when present" means in 13e-2 carry the same baseline
offset.

**Evidence.** Synthetic check with the pipeline's own `make_rf_params` and
`explain_rf` (27 positives, 5,000 background, explain set = positives + 273
background; feature 0 truly raises, feature 1 truly lowers prospectivity):

```
expected_value 0.496 | mean P(pos) on explain set 0.066
feature 0 (truth: more): sign(mean SHAP) -> less | spearman(x, SHAP) = +0.684
feature 1 (truth: less): sign(mean SHAP) -> less | spearman(x, SHAP) = -0.702
noise features sign(mean SHAP): less, less, less, less
```

The SHAP values themselves are correct (exact additivity, positive class);
only the direction rule is wrong. Global importance (`mean |SHAP|`), the
beeswarm plots (Figure 7) and the waterfall plots are unaffected.

**How to fix.** Use the rank correlation between a feature's value and its
SHAP value over the explain set. `SHAP_RESULTS` already stores the inputs
(`X_explain_rf`, `X_explain_dn`), so nothing needs recomputing except
Section 13e/13e-2 themselves.

```python
from scipy.stats import spearmanr

def feature_direction(x, shap_col):
    """Spearman rho between feature value and its SHAP value. > 0: higher
    values raise the score. For 0/1 dummies this matches comparing mean SHAP
    when present vs absent. NaN for a constant column."""
    if np.nanstd(x) == 0:
        return np.nan
    return spearmanr(x, shap_col, nan_policy='omit')[0]
```

- Change `direction_consistency_table(shap_rf, shap_dn, feat_cols, top_k)`
  to also take `X_rf, X_dn`, compute `rho_rf` / `rho_dn` per feature, label
  `|rho| < 0.1` as "no clear direction", and report the rho values.
- Update the call in notebook Section 13e to pass `r['X_explain_rf']` and
  `r['X_explain_dn']`, and fix the inline copy in the same cell.
- In `era_importance.py`, replace the `*_when_present` direction with
  `mean(SHAP | present) - mean(SHAP | absent)` (or `feature_direction`).

**How to verify.** Re-run the synthetic check above: feature 0 must come out
"more", feature 1 "less", noise features near zero. Then re-run Section 13
(it needs Section 9's production models).

**Paper text to re-check after the fix:** report §5.3 ("DevNet ranks terrane
age highest (Mesozoic negative, Proterozoic positive)") and the Discussion
("both models ... treat Mesozoic terranes as unfavourable"). These may have
been read from the beeswarm plots, which are correct, but the stored 13e
table currently contradicts them (it lists DevNet `geo_era_Mesozoic` as
"higher -> more prospective").

---

## 2. Table 1 lists a layer the pipeline never used

**Where.** Thesis report Table 1 (and any table derived from it in the
paper): "Mantle partial-melting events — Geological provinces — Line —
proximity to cratonic margin — Wheeler et al. (1996)".

**What is wrong.** Every run, from the first geophysics-only notebook to the
packaged pipeline, used exactly **two** line layers: `AncientPassiveMargins`
and `Faults_Canada` (stored output: "Discovered 17 raster, 2 vector, 4 point
layer(s)"; 48 features before pruning). No notebook ever builds a
cratonic-margin or province-boundary feature. The thesis-era `data/README.md`
also claimed three vector layers.

Where the province data *was* used (traced October 2026):
- **Study-area outlines.** The three regions are the Cordilleran Orogen,
  Superior and Grenville polygons of GSC Map 1860A (`GEOPROV.SHP`, field
  `GEOLPROV`); `data/outlines/*.geojson` carry those attributes. So Wheeler
  et al. (1996) is a genuine input — it defines the regions — just not a
  distance feature.
- **Early structure feature, dropped.** The June exploratory notebook
  (`experiments/01_geochemistry/multiregion_integrated_pca_lda.ipynb`) had a
  `struct_proximity` feature from `dataset_old/<Region>_structure.geojson`
  (OneDrive, `Desktop/graduate IC/dataset_old/`). Those files are the
  **Getech structural database** (faults, thrusts, folds; `refs_cit` "Getech
  Group plc"), not province boundaries, and the feature did not survive the
  move to the geophysics-only models.

So the Table 1 row could instead be rewritten as "Geological provinces —
polygon — study-area definition — Wheeler et al. (1996)" and moved out of
the feature list.

**How to fix — choose one.**
- **A. Correct the paper** (no re-run): drop the row from Table 1 and make
  sure the feature count and the mineral-system narrative match the two line
  layers actually used.
- **B. Implement the layer** (re-run everything; all numbers change):
  1. Take the province polygons from GSC Map 1860A (`GEOPROV.SHP`, field
     `GEOLPROV`). The prepared data archive carries them in
     `dataset/extra/gsc_geological_provinces/`, outside the folders the
     pipeline scans.
  2. Decide which boundaries count as cratonic margins, convert those
     polygons to **lines** (`gdf.boundary`), and save as e.g.
     `dataset/vectors/CratonicMargins.geojson`. Do not put the polygons in
     `vectors/` directly: distance to a polygon is 0 everywhere inside it.
  3. Discovery picks the file up automatically as
     `CratonicMargins_proximity`; re-run the notebook end to end.

---

## 3. Provenance of the fault layer is unverified

`Faults_Canada.geojson` (7,602 lines) carries only `FID` and `Fault_ID`
attributes, and it was kept in its own folder rather than with the USGS
release files. Table 1 and `data/README.md` credit McCafferty et al. (2023),
following the thesis. Confirm where the file came from (USGS release, GSC,
or Getech) and correct `data/README.md` §2.1 and the paper if needed.

---

## 4. `explain_rf` depends on the `shap` output format

**Where.** `src/interpretation/shap_explainers.py`, `explain_rf`.

`shap` ≥ 0.45 returns `(n_samples, n_features, n_classes)` for a classifier,
and the code keeps class 1. Older versions return a list `[class0, class1]`;
`np.array(...)` makes that `(2, n_samples, n_features)`, still 3-D, so
`sv[:, :, 1]` silently picks **feature 1** instead of the positive class, with
no error. The module also `pip install`s an unpinned `shap` if the import
fails. With the pinned `shap==0.46.0` the results are correct.

**How to fix.** Handle the list form explicitly
(`if isinstance(raw, list): sv = raw[1]`), assert
`sv.shape == X_explain.shape`, and replace the auto-install with a plain
import so the pinned version is always used.
