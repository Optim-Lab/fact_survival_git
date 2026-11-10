"""ID collisions, trainable entity vectors, and train-only checkpoint lookup."""
import io

import numpy as np
import torch

from modules.neural_hazard import NeuralLandmarkHazard, fit_predict, predict_checkpoint


def _active_head(variant):
    model = NeuralLandmarkHazard(12, [3, 2, 3], [1, 3], variant=variant, hidden=1).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.baseline.fill_(-3.)
        model.encoder[0].weight[0, 12] = 1.
        model.encoder[0].weight[0, 28] = 2.
        model.encoder[3].weight[0, 0] = 1.
        model.hazard_head.weight.fill_(1.)
        model.gate.bias.fill_(3.)
        if variant == "neural_id":
            model.entity_embedding.weight[1, 0] = 1.
            model.entity_embedding.weight[2, 0] = 2.
    return model


def test_same_numeric_features_collide_without_ids_but_id_model_distinguishes():
    x = torch.zeros(2, 12)
    cat = torch.tensor([[1, 1, 2], [2, 1, 1]])
    tau = torch.full((2,), 3.)
    for variant, should_collide in (("neural", True), ("neural_id", False)):
        model = _active_head(variant)
        prediction = model(x, cat, tau)
        assert torch.equal(prediction[0], prediction[1]) == should_collide
        assert (torch.diff(prediction, dim=1) <= 0).all()


def test_shared_entity_vectors_receive_gradients_and_unknown_stays_zero():
    model = _active_head("neural_id")
    cat = torch.tensor([[1, 1, 2], [2, 1, 1], [0, 1, 0]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    (-model(torch.zeros(3, 12), cat, torch.full((3,), 3.)).sum()).backward()
    grad = model.entity_embedding.weight.grad
    assert torch.count_nonzero(grad[1]) > 0
    assert torch.count_nonzero(grad[2]) > 0
    assert torch.count_nonzero(grad[0]) == 0
    optimizer.step()
    assert torch.count_nonzero(model.entity_embedding.weight[0]) == 0


def test_checkpoint_uses_entity_union_and_maps_unseen_ids_to_unknown():
    torch.set_num_threads(1)
    x = np.zeros((4, 12), np.float32)
    cat = np.array([[1, 1, 3], [2, 1, 4], [1, 2, 4], [2, 2, 3]])
    evaluation = cat[:, [2, 1, 0]]
    tau = np.full(4, 3, np.float32)
    labels = np.array([[1., 1.], [0., 0.], [1., 0.], [1., 1.]], np.float32)
    predictions, _, checkpoint = fit_predict(
        x, cat, labels, np.ones_like(labels, bool), tau,
        x, evaluation, tau, [1, 3], variant="neural_id", epochs=[2], batch_size=4,
    )
    assert checkpoint["config"]["cardinalities"] == [5, 3, 5]
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    np.testing.assert_array_equal(
        predictions[2], predict_checkpoint(restored, x, evaluation, tau),
    )
    unseen = np.full_like(cat, 999)
    np.testing.assert_array_equal(
        predict_checkpoint(restored, x, unseen, tau),
        predict_checkpoint(restored, x, np.zeros_like(cat), tau),
    )
    assert np.isfinite(predictions[2]).all()
