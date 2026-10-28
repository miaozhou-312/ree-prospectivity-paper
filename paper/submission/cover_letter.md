# Cover letter — Natural Resources Research

<!-- TODO(author): date, editor's name, corresponding-author details, final title,
     suggested reviewers. Numbers below come from the revised manuscript. -->

Dear Editor,

We submit the manuscript "[[TITLE]]" for consideration as an original research article in *Natural Resources Research*.

Data-driven prospectivity mapping of carbonatite-hosted rare earth element deposits has to learn from a few tens of known occurrences and no confirmed barren ground. Recent work in this journal has modelled these deposits at national scale with random forests and convolutional networks (Parsa et al., 2024) and has recommended GAN-generated labels (Parsa et al., 2025). Our study asks what such models actually learn when the evaluation mimics the exploration task, ranking ground far from any known occurrence, and every model sees the same data and folds.

The main findings are:

1. **Validation protocol.** Under spatially blocked cross-validation in the Superior Province, smooth neural rankers with rank-normalised inputs keep an average precision of [[DEVNET_SPATIAL]], whereas random forests fall to [[RF_SPATIAL]]; the tree models' advantage under random splits in the Cordilleran Orogen and Grenville Province is matched by a forest that sees only coordinates, i.e. it is location memorisation.
2. **What drives transfer.** A supervised network with the same inputs and training as the semi-supervised Deviation Network performs as well under spatial blocking, and tuned tree ensembles do not close the gap; the gain comes from the input representation and the smooth model class, not from the anomaly-detection loss. Choosing the design inside each training fold costs [[NESTED_COST]] AP, so the result is not an artefact of design selection.
3. **Synthetic positives.** Jitter, SMOTE and a Wasserstein GAN, generated strictly within training folds, gave no consistent gain.
4. **Protocol audit.** Rebuilding a published design in which S2 child cells of one occurrence are split between training and test sets yields test AUC near 1 in every province, inflating AUC by up to 0.37 where real skill is weakest.
5. **Why provinces differ.** Model-free diagnostics show that only in Superior do distant occurrences share a geophysical signature; the Grenville label set has few independent sites and mixes carbonatites of different ages and tectonic settings.

We believe these results will interest readers of *Natural Resources Research* who build or evaluate prospectivity models under label scarcity, and they offer concrete recommendations for validation. Code for every analysis is openly available; eleven processed gravity and magnetic grids are licensed from Getech Group plc, and we show that the main result is [[PUBLIC_ONLY]] with public layers only.

The manuscript has not been published and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests. <!-- TODO(author): confirm -->

Yours sincerely,

[[AUTHOR]]
