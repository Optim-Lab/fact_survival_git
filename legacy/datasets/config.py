"""Dataset-specific settings for the shared survival experiment."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMON = {
    "bootstrap_repetitions": 1000,
    "random_seed": 42,
    "bootstrap_seed": 4200,
}

FINWIKI_CONFIG = {
    **COMMON,
    "data_path": PROJECT_ROOT / "data/FinWiki/train_valid_test.pkl",
    "output_dir": PROJECT_ROOT / "results/finwiki_rolling_development",
    "time_unit": "month",
    "horizons": [3, 6, 12],
    "horizon_months": 1,
    "development_start": None,
    "development_end": None,
    "logistic_c": 0.01,
    "calendar_block_months": 3,
    "validation_step_months": 3,
}

YAGO_CONFIG = {
    **COMMON,
    "data_path": PROJECT_ROOT / "data/YAGO/processed/yago_survival_rows.csv",
    "output_dir": PROJECT_ROOT / "results/yago_shared_rolling_development",
    "time_unit": "year",
    "horizons": [1, 2, 3],
    "horizon_months": 12,
    "development_start": "2003-01",
    "development_end": "2008-01",
    "logistic_c": 1.0,
    "calendar_block_months": 12,
    "validation_step_months": 12,
}

FINSURV_CONFIG = {
    **COMMON,
    "data_path": PROJECT_ROOT / "data/FinSurvKG/train_valid_test.pkl",
    "facts_path": PROJECT_ROOT / "data/FinSurvKG/facts.csv",
    "output_dir": PROJECT_ROOT / "results/finsurvkg_rolling_development",
    "time_unit": "month",
    "horizons": [12, 24, 36],
    "horizon_months": 1,
    "development_start": None,
    "development_end": None,
    "logistic_c": 0.01,
    "calendar_block_months": 12,
    "validation_step_months": 12,
}

ICEWSWIKI_CONFIG = {
    **FINWIKI_CONFIG,
    "data_path": PROJECT_ROOT / "data/raw/ICEWSWiki.pkl",
    "output_dir": PROJECT_ROOT / "results/icewswiki_rolling_test",
}

DATASET_CONFIGS = {
    "finwiki": FINWIKI_CONFIG,
    "icewswiki": ICEWSWIKI_CONFIG,
    "yago": YAGO_CONFIG,
    "finsurvkg": FINSURV_CONFIG,
}

# Backward compatibility for older FinWiki experiment scripts.
CONFIG = FINWIKI_CONFIG
