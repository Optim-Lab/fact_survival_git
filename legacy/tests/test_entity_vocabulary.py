"""Training-only entity identities are shared across subject/object roles."""
import numpy as np

from modules.proposed_experiment import fold_categories


def test_entity_vocabulary_shares_roles_and_excludes_future_entities():
    categories = np.array([[0, 0, 2], [1, 1, 0], [2, 0, 1], [3, 2, 4]])
    train, evaluation, vocabularies = fold_categories(categories, np.array([0, 1]), np.array([2, 3]))
    assert train[0, 0] == train[1, 2]  # entity 0 in both roles
    assert evaluation[0, 0] == train[0, 2]  # entity 2 was seen only as object
    assert evaluation[0, 2] == train[1, 0]  # entity 1 was seen only as subject
    np.testing.assert_array_equal(evaluation[1], [0, 0, 0])
    np.testing.assert_array_equal(vocabularies[0], vocabularies[2])
    changed = categories.copy()
    changed[3] = [10, 10, 10]
    other = fold_categories(changed, np.array([0, 1]), np.array([2, 3]))
    np.testing.assert_array_equal(train, other[0])
    for expected, actual in zip(vocabularies, other[2]):
        np.testing.assert_array_equal(expected, actual)
