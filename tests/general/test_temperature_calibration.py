# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from swift.callbacks.temperature_calibration import TemperatureCalibrationCallback


@pytest.mark.parametrize('loss_name', ['brier', 'nll'])
def test_calibration_metrics_use_held_out_records(loss_name):
    callback = TemperatureCalibrationCallback(None, None)
    callback.loss_name = loss_name
    callback.min_items = 10
    items = []
    for record in range(20):
        for kind in range(3):
            items.append({
                # Non-contiguous IDs with the same parity: splitting is by record order.
                'record_id': 100 + 6 * record,
                'kind': kind,
                'logits': torch.tensor([2., 0., 100.]),
                'target': torch.tensor([.8, .2, 0.] if record % 2 == 0 else [.6, .4, 0.]),
                'mask': torch.tensor([True, True, False]),
            })

    metrics = callback._cross_validated_metrics(items, 1., {})

    # Both objectives have an interior optimum matching each half's soft targets.
    # Held-out scoring must use the opposite half's probabilities. Scoring the
    # fitting half instead would give a near-zero Brier score rather than 0.04.
    targets = torch.tensor([[.8, .2], [.6, .4]])
    probabilities = targets.flip(0)
    expected_brier = (probabilities - targets).square().mean().item()
    expected_nll = -(probabilities.log() * targets).mean().item()
    expected_ece = (1 - probabilities[:, 0]).mean().item()

    assert metrics['accuracy'] == 1.
    # Allow the numerical solvers' convergence tolerance around the analytic optimum.
    assert metrics['brier'] == pytest.approx(expected_brier, abs=5e-4)
    assert metrics['nll'] == pytest.approx(expected_nll, abs=5e-4)
    assert metrics['ece'] == pytest.approx(expected_ece, abs=5e-4)
