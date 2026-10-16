# dev/

Superseded code removed from the final research notebook during the
refactor into `src/`, kept for the record rather than deleted. Nothing here
is part of the reproducible pipeline (`src/`) or its reference notebook
(`notebooks/`). The full history of earlier notebooks is in
[`../experiments/`](../experiments/).

## Contents

- **`superseded_capture_rate_plot.py`** -- the original notebook's Section
  12b defined `plot_capture_curves` twice, back-to-back, in the same cell.
  The second definition silently overwrote the first before it was ever
  called, so the first version is dead code that never actually ran. Kept
  here verbatim; the version that was actually used is in
  `src/evaluation/capture_rate.py`.
- **`superseded_occurrence_shap.py`** -- former Section 14f, removed from
  both the notebook and `src/interpretation/`. It explained why DevNet
  "down-weighted" specific Occurrence-tier points via SHAP, but Section
  14a-14e's own hypothesis test found no significant RF/DevNet asymmetry in
  this run -- there was no real effect left for it to explain, and its
  median-split logic would keep producing plausible-looking per-point
  narratives regardless (see the file's own docstring for detail). Kept
  here verbatim; `src/evaluation/label_confidence_stratification.py`
  (14a-14e) is unaffected and still reports the honest null result.

## Design decisions not carried into `src/`

A few pieces of the original notebook are narrow, one-off print/report
snippets rather than reusable logic, and were deliberately left out of the
modular `src/` breakdown (not moved here either, since they aren't
non-functional -- they're just too specific to a single narrative point to
generalize):

- The end of Section 11c (background-sampling sensitivity) printed a
  hardcoded check specific to the Superior region ("is DevNet AP > RF AP in
  all 5 background-sampling draws?"). The same computation is trivially
  derivable from the `bg_sensitivity_df` DataFrame returned by
  `src/evaluation/ablation_background_sampling.run_background_sensitivity`,
  filtered to whichever region is of interest -- it just isn't hardcoded to
  "Superior" as a function in `src/`.

Everything else in the original notebook that defines reusable logic has a
home in `src/` -- see the table in
`notebooks/end_to_end_notebook.ipynb`'s header cell.
