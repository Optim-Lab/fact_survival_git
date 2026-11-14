# Data layout

The data files are not part of this repository (size, and the terms of the original sources). Put the files at the paths below;
the SHA-256 shows which file the paper's numbers were produced with.

| Dataset | Required path | SHA-256 | Original source |
|---|---|---|---|
| FinWiki | `data/FinWiki/train_valid_test.pkl` | `09cbfbc154e1052730036eb92715787a94867645487e5ac8567624416ca12d5e` | TODO (to be filled by the authors) |
| ICEWSWiki | `data/raw/ICEWSWiki.pkl` | `7cabceb2ec0723ea3c5598de9fb0bb00f80354208f13b6f4069fe124d96209da` | TODO |
| YAGO (raw) | `data/YAGO/train_valid_test.pkl` | `fd81ba3fd6867400405c8c136d4aa339b1172433473659b228a99e864df47f31` | YAGO130K-type benchmark file; **withdrawn by its authors, not redistributed** (see `docs/known_issues.md`, E) |
| YAGO landmark table | `data/YAGO/processed/yago_survival_rows.csv` | `cf9cd07b6987301cc104479f1c4636df8bb5c19d979bbf7149b63d580f172e57` | built from the raw YAGO file by the original code; used only as a test reference |

FinSurvKG is configured in `configs/finsurvkg.json`; its file path and hash are still to be added.

The YAGO landmark table is not read by the code; it is only the reference that `tests/test_yago_matches_csv.py` compares `data.py`'s output with (set `YAGO_PKL` and `YAGO_CSV`).
A column that marks the end time of a fact for audits (`*_DO_NOT_USE_AS_FEATURE.csv`) must never be used as a model input.

Check a file: `sha256sum data/FinWiki/train_valid_test.pkl`.
