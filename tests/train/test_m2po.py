# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

from swift.rl_core.m2po import compute_m2po_log_ratio, compute_m2po_mask, compute_m2po_token_loss


def test_m2po_log_ratio_prefers_rollout_policy_and_falls_back_to_old_policy():
    current = torch.tensor([[-1.0, -2.0]])
    old = torch.tensor([[-1.1, -2.1]])
    rollout = torch.tensor([[-1.5, -2.5]])

    torch.testing.assert_close(compute_m2po_log_ratio(current, old, rollout), current - rollout)
    torch.testing.assert_close(compute_m2po_log_ratio(current, old), current - old)


def test_m2po_masks_largest_trust_region_outlier():
    log_ratio = torch.tensor([[0.1, 0.2, 0.3]])
    completion_mask = torch.ones_like(log_ratio, dtype=torch.bool)
    advantages = torch.ones_like(log_ratio)

    mask, metrics = compute_m2po_mask(log_ratio, completion_mask, advantages, m2_threshold=0.04)

    assert torch.equal(mask, torch.tensor([[True, True, False]]))
    assert metrics['m2_before'].item() == pytest.approx((0.01 + 0.04 + 0.09) / 3)
    assert metrics['m2_after'].item() == pytest.approx(0.025)
    assert metrics['masked_fraction'].item() == pytest.approx(1 / 3)
    assert metrics['trust_region_fraction'].item() == pytest.approx(1.0)


def test_m2po_only_constrains_active_ppo_quadrants_and_valid_tokens():
    log_ratio = torch.tensor([[1.0, -1.0, 1.0, -1.0, 2.0]])
    advantages = torch.tensor([[-1.0, 1.0, 1.0, -1.0, 1.0]])
    completion_mask = torch.tensor([[True, True, True, True, False]])

    mask, metrics = compute_m2po_mask(log_ratio, completion_mask, advantages, m2_threshold=0.1)

    assert torch.equal(mask, torch.tensor([[True, True, False, False, False]]))
    assert metrics['trust_region_fraction'].item() == pytest.approx(0.5)


def test_m2po_vectorized_selection_matches_algorithm_one():
    threshold = 0.04
    for seed in range(10):
        generator = torch.Generator().manual_seed(seed)
        log_ratio = 0.8 * torch.randn(4, 7, generator=generator)
        advantages = torch.randn(4, 7, generator=generator)
        completion_mask = torch.rand(4, 7, generator=generator) > 0.2

        actual, _ = compute_m2po_mask(log_ratio, completion_mask, advantages, threshold)

        expected = completion_mask.clone()
        trust_region = completion_mask & (((advantages > 0) & (log_ratio > 0)) | ((advantages < 0) & (log_ratio < 0)))
        active = torch.nonzero(trust_region.reshape(-1), as_tuple=False).squeeze(-1)
        second_moment = log_ratio.float().square().reshape(-1)
        while active.numel() and second_moment[active].mean() > threshold:
            largest = torch.argmax(second_moment[active])
            expected.reshape(-1)[active[largest]] = False
            active = torch.cat((active[:largest], active[largest + 1:]))

        assert torch.equal(actual, expected)


def test_m2po_loss_keeps_original_denominator_and_masks_gradients():
    log_ratio = torch.tensor([[0.0, 1.0]], requires_grad=True)
    completion_mask = torch.ones_like(log_ratio, dtype=torch.bool)
    advantages = torch.ones_like(log_ratio)

    per_token_loss, mask, metrics = compute_m2po_token_loss(log_ratio, completion_mask, advantages, m2_threshold=0.1)
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    loss.backward()

    assert torch.equal(mask, torch.tensor([[True, False]]))
    assert loss.item() == pytest.approx(-0.5)
    assert torch.allclose(log_ratio.grad, torch.tensor([[-0.5, 0.0]]))
    assert metrics['masked_fraction'].item() == pytest.approx(0.5)


def test_m2po_empty_mask_and_extreme_non_trust_ratio_are_finite():
    empty_log_ratio = torch.tensor([[1.0, -1.0]], requires_grad=True)
    empty_mask = torch.zeros_like(empty_log_ratio, dtype=torch.bool)
    empty_loss, final_mask, metrics = compute_m2po_token_loss(empty_log_ratio, empty_mask,
                                                              torch.ones_like(empty_log_ratio))

    assert not final_mask.any()
    assert empty_loss.sum().item() == pytest.approx(0.0)
    assert metrics['masked_fraction'].item() == pytest.approx(0.0)

    extreme_log_ratio = torch.tensor([[100.0]], requires_grad=True)
    extreme_loss, _, _ = compute_m2po_token_loss(extreme_log_ratio,
                                                 torch.ones_like(extreme_log_ratio,
                                                                 dtype=torch.bool), -torch.ones_like(extreme_log_ratio))
    extreme_loss.sum().backward()

    assert torch.isfinite(extreme_loss).all()
    assert torch.isfinite(extreme_log_ratio.grad).all()
    assert extreme_log_ratio.grad.item() > 0


def test_m2po_rejects_invalid_inputs():
    values = torch.zeros(1, 2)
    mask = torch.ones_like(values, dtype=torch.bool)
    advantages = torch.ones_like(values)

    with pytest.raises(ValueError, match='non-negative'):
        compute_m2po_mask(values, mask, advantages, m2_threshold=-0.01)
    with pytest.raises(ValueError, match='finite'):
        compute_m2po_mask(values, mask, advantages, m2_threshold=float('nan'))
    with pytest.raises(ValueError, match='identical shapes'):
        compute_m2po_mask(values, mask[:, :1], advantages)
    with pytest.raises(ValueError, match='non-finite'):
        compute_m2po_mask(torch.tensor([[float('nan'), 0.0]]), mask, advantages)


def _distributed_m2po_worker(rank, world_size, init_method, result_queue):
    dist.init_process_group('gloo', rank=rank, world_size=world_size, init_method=init_method)
    try:
        local_ratios = [torch.tensor([[0.05, 0.4]]), torch.tensor([[0.2, 0.21]])][rank]
        local_mask, metrics = compute_m2po_mask(
            local_ratios,
            torch.ones_like(local_ratios, dtype=torch.bool),
            torch.ones_like(local_ratios),
            m2_threshold=0.04,
        )
        result_queue.put((rank, local_mask.tolist(), {key: value.item() for key, value in metrics.items()}))
    finally:
        dist.destroy_process_group()


def test_m2po_uses_one_threshold_across_distributed_ranks():
    world_size = 2
    spawn_context = mp.get_context('spawn')
    result_queue = spawn_context.SimpleQueue()
    with tempfile.TemporaryDirectory() as tmp_dir:
        init_method = (Path(tmp_dir) / 'm2po_dist_init').as_uri()
        mp.spawn(
            _distributed_m2po_worker,
            args=(world_size, init_method, result_queue),
            nprocs=world_size,
            join=True,
        )

    results = sorted([result_queue.get() for _ in range(world_size)])
    assert results[0][1] == [[True, False]]
    assert results[1][1] == [[True, True]]
    for _, _, metrics in results:
        assert metrics['m2_after'] == pytest.approx((0.05**2 + 0.2**2 + 0.21**2) / 3)
        assert metrics['masked_fraction'] == pytest.approx(0.25)
