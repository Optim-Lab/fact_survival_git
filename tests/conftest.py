"""The equivalence tests compare the new code with the original implementation in legacy/.
If legacy/ is not in this copy of the repository, those tests are left out (the other tests still run)."""
from pathlib import Path

LEGACY_TESTS = ["test_labels_match_legacy.py", "test_half_life_matches_legacy.py", "test_features.py",
                "test_baselines_match_legacy.py", "test_classical_matches_legacy.py", "test_metrics_match_legacy.py",
                "test_training_matches_legacy.py", "test_data_matches_legacy.py"]

collect_ignore = [] if (Path(__file__).resolve().parents[1] / "legacy").exists() else LEGACY_TESTS
