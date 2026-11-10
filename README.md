# Fact survival forecasting (Neural Landmark Hazard)

Code for predicting how long a temporal-knowledge-graph fact stays valid. At a landmark time `c`, the model outputs the
probability `S(h_k | c)` that a fact is still valid at `c + h_k`. It mixes a neural piecewise-constant hazard with a half-life
curve through a learned gate (symbols in the code comments follow the paper).

> **Status.** `fact_survival/` is a readable re-implementation of the code that produced the paper's numbers. Each part (labels,
> features, half-life, data reading, model, training, metrics, baselines) was checked against that original implementation
> (the original code is not part of this repository). The full paper tables have **not** yet been reproduced end to end with
> the new code (work in progress).

## Layout
| Path | What it is |
|---|---|
| `fact_survival/` | the method: `labels.py`, `features.py`, `half_life.py`, `data.py`, `hazard_model.py`, `training.py`, `metrics.py`, `baselines.py`, `classical.py` |
| `main.py` | the experiment in 4 steps: `develop`, `select`, `test`, `report` |
| `configs/` | one JSON per dataset (horizons, model settings, seeds, candidate epochs) |
| `tests/` | equivalence tests against the original implementation; they are skipped when `legacy/` is not present |

## Install
```bash
pip install -r requirements.txt
```

## Data
Data files are not part of the code. See `data/README.md` for the expected paths and SHA-256 hashes.

## Run
```bash
python main.py develop configs/finwiki.json --data data/FinWiki/train_valid_test.pkl --out results/finwiki --device cuda
python main.py select  configs/finwiki.json --out results/finwiki
python main.py test    configs/finwiki.json --data data/FinWiki/train_valid_test.pkl --out results/finwiki --device cuda
python main.py report  configs/finwiki.json --out results/finwiki
```
Rolling origin: the model is refitted from scratch at every origin on the labels known by then, and predicts the facts active
at that origin. `--part i --parts n` splits the origins over several jobs.

## Tests
```bash
FINWIKI_PKL=data/FinWiki/train_valid_test.pkl python -m pytest tests
```
Tests that need a data file are skipped when its environment variable is not set.

## Known open points
See `docs/known_issues.md` (to be written): object-degree definition per dataset, how the epoch is chosen, model settings per dataset.
