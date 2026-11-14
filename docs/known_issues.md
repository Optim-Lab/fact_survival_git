# Known open points

Things where the code, the manuscript, or the verification are not (yet) in full agreement. Listed so that nothing has to be
discovered by the reader. "Manuscript" = the paper draft; "original code" = the code that produced the paper's numbers
(not part of this repository).

##  Degree definition
The manuscript says `object_degree` counts incoming edges. The code that produced the results counts all edges touching the
entity (direction ignored) for FinWiki, FinSurvKG and ICEWSWiki, and incoming edges for YAGO (`configs/*.json`, key
`object_degree`).

##  Model settings per dataset
The experimental-setup text says the epoch is the only tuned quantity and all other settings are shared. The configs follow
that for ICEWSWiki, FinSurvKG and YAGO (hidden 32, dropout 0.1, lr 0.003, weight decay 0.001, batch 8192). But:
- FinWiki: `configs/finwiki.json` uses the Optuna trial-25 values (hidden 16, dropout 0.5, lr 0.00758, weight decay 2.4e-6)
- FinSurvKG: an Optuna trial 28 (hidden 64, dropout 0.4, lr 3.98e-4, weight decay 2.36e-6, batch 2048) exists; the appendix
  lists the shared values. The config uses the shared values.
- ICEWSWiki and YAGO: the config values are the code defaults;

##  How the epoch is chosen
The code pools the development rows per seed, takes the Brier per horizon, averages over horizons and seeds, and picks the
epoch with the lowest value from {1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100} ("rule B"). The manuscript uses 5 seeds (42-46);
some ablation runs used 3 seeds (42-44). For YAGO the chosen epoch is 100, the edge of the grid, so the optimum may lie beyond it.
Whether early stopping should replace or complement rule B is undecided.

## E. YAGO data
The YAGO file used is **not** YAGO11k. It is a YAGO130K-type benchmark file whose authors withdrew it, so it is not
redistributed here. The entity count in the manuscript (156,235) differs from the file (196,799); the manuscript's
dataset citation is still (link the the better~~)

## F. Not in this repository yet
Deep-learning baselines (DeepSurv, LogisticHazard, DeepHit) and graph-encoder variants (RE-GCN, LogCL, EvoKG), bootstrap
confidence intervals, and the reference result tables.
