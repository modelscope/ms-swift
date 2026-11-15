# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resume the next real AdamW updates and all three CPU RNG streams."""
import json
import numpy as np
import pytest
import random
import torch

from swift.fsdpturbo.adapter_checkpoint import (capture_rng, read_adapter_metadata, restore_rng, write_adapter_metadata,
                                                write_tensor_state)


def test_optimizer_and_rng_resume_continue_the_same_updates(tmp_path):
    torch.manual_seed(42)
    random.seed(17)
    np.random.seed(19)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)

    def update(module, optim, sched):
        inputs = torch.randn(2, 4) * random.random() + float(np.random.random())
        optim.zero_grad()
        loss = module(inputs).square().mean()
        loss.backward()
        optim.step()
        sched.step()
        return loss.detach()

    for _ in range(4):
        update(model, optimizer, scheduler)
    write_tensor_state(tmp_path / 'model.pt', model.state_dict())
    write_tensor_state(tmp_path / 'optimizer.pt', optimizer.state_dict())
    write_tensor_state(tmp_path / 'scheduler.pt', scheduler.state_dict())
    write_tensor_state(tmp_path / 'rng.pt', capture_rng('cpu'))
    expected = [update(model, optimizer, scheduler) for _ in range(2)]
    resumed = torch.nn.Linear(4, 3)
    resumed.load_state_dict(torch.load(tmp_path / 'model.pt', weights_only=True))
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, 2)
    resumed_optimizer.load_state_dict(torch.load(tmp_path / 'optimizer.pt', weights_only=True))
    resumed_scheduler.load_state_dict(torch.load(tmp_path / 'scheduler.pt', weights_only=True))
    restore_rng(torch.load(tmp_path / 'rng.pt', weights_only=True), 'cpu')
    actual = [update(resumed, resumed_optimizer, resumed_scheduler) for _ in range(2)]
    torch.testing.assert_close(torch.stack(expected), torch.stack(actual), rtol=0, atol=0)
    for left, right in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_resume_rejects_incomplete_or_changed_training(tmp_path):
    topology = dict(world_size=32, fsdp_size=32, cp_size=8, ep_size=32)
    with pytest.raises(ValueError, match='completed optimizer'):
        read_adapter_metadata(tmp_path, topology=topology)
    write_adapter_metadata(tmp_path, dict(topology=topology, dataset_fingerprint='dataset-a'))
    with pytest.raises(ValueError, match='topology'):
        read_adapter_metadata(tmp_path, topology=dict(topology, cp_size=4))
    with pytest.raises(ValueError, match='dataset'):
        read_adapter_metadata(tmp_path, topology=topology, dataset_fingerprint='dataset-b')
    assert read_adapter_metadata(tmp_path, topology=topology, dataset_fingerprint='dataset-a')['format_version'] == 1


def test_sampler_and_accumulation_changes_are_rejected_for_early_checkpoints(tmp_path):
    topology = dict(world_size=32)
    signature = dict(
        max_length=32768,
        gradient_accumulation_steps=1,
        data_seed=42,
        per_device_train_batch_size=1,
        train_dataloader_shuffle=True,
        dataloader_drop_last=False)
    write_adapter_metadata(tmp_path, dict(topology=topology))
    (tmp_path / 'args.json').write_text(json.dumps(signature))
    read_adapter_metadata(tmp_path, topology=topology, run_signature=signature)
    for key, changed in (('max_length', 65536), ('data_seed', 44), ('gradient_accumulation_steps', 2)):
        with pytest.raises(ValueError, match='sampler settings'):
            read_adapter_metadata(tmp_path, topology=topology, run_signature=dict(signature, **{key: changed}))
