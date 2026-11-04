# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real two-process Ulysses checks on CPU/Gloo or two Ascend NPUs/HCCL."""
import copy
import datetime
import os
import pytest
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from types import SimpleNamespace

_olmo3 = pytest.importorskip('transformers.models.olmo3.modeling_olmo3', reason='OLMo 3 requires transformers>=4.57')


def _config():
    config = _olmo3.Olmo3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        layer_types=['sliding_attention'] * 3 + ['full_attention'],
        sliding_window=4,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        rope_theta=500000,
        rope_scaling={
            'rope_type': 'yarn',
            'factor': 8.0,
            'original_max_position_embeddings': 8,
            'attention_factor': 1.2079441541679836,
            'beta_fast': 32.0,
            'beta_slow': 1.0,
        })
    config._attn_implementation = 'sdpa'
    return config


def _batch(length, padding, device):
    input_ids = torch.arange(3, 3 + 2 * length, device=device).reshape(2, length)
    attention_mask = torch.ones_like(input_ids)
    if padding != 'none':
        count = length // 3
        selection = slice(-count, None) if padding == 'right' else slice(0, count)
        attention_mask[1, selection] = 0
        input_ids.masked_fill_(attention_mask == 0, 0)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    labels = input_ids.clone().masked_fill(attention_mask == 0, -100)
    labels[torch.arange(2, device=device), attention_mask.argmax(-1)] = -100
    # Shift once globally before splitting; passing these labels to HF would shift twice.
    shifted_labels = F.pad(labels[:, 1:], (0, 1), value=-100)
    return dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids), shifted_labels


def _run_ulysses(rank, store, device_type, dtype, length, padding):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2')
    torch.set_num_threads(1)
    from swift.model import MODEL_MAPPING
    from swift.sequence_parallel import sequence_parallel as sp

    device = torch.device(f'npu:{rank}' if device_type == 'npu' else 'cpu')
    if device_type == 'npu':
        torch.npu.set_device(rank)
        torch.npu.matmul.allow_hf32 = False
    torch.manual_seed(7)
    config = _config()
    reference = _olmo3.Olmo3ForCausalLM(config).to(device=device, dtype=dtype).eval()
    inputs, labels = _batch(length, padding, device)
    count = (labels != -100).sum()

    # SP preparation patches global HF functions. Cache the unpatched reference first.
    reference_logits = reference(**inputs).logits
    reference_loss = F.cross_entropy(
        reference_logits.float().reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100,
        reduction='sum') / count
    reference_loss.backward()
    reference_gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in reference.named_parameters()
    }
    reference_logits = reference_logits.detach().float().cpu()
    reference_loss = reference_loss.detach().float().cpu()
    model = _olmo3.Olmo3ForCausalLM(copy.deepcopy(config)).to(device=device, dtype=dtype).eval()
    model.load_state_dict(reference.state_dict())
    model.model_meta = MODEL_MAPPING['olmo3']
    model.model_info = SimpleNamespace(is_moe_model=False)

    if dtype == torch.bfloat16:
        logits_tolerance, gradient_tolerance, loss_tolerance = (2e-2, 2e-3), (3e-2, 3e-3), (5e-3, 5e-3)
    else:
        logits_tolerance, gradient_tolerance, loss_tolerance = (2e-4, 2e-6), (3e-4, 3e-6), (3e-5, 3e-6)
    try:
        dist.init_process_group(
            'hccl' if device_type == 'npu' else 'gloo',
            init_method=store,
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=120))
        sp.prepare(2, model=model, tokenizer=SimpleNamespace(pad_token_id=0), padding_free=False)
        assert sp.sp_world_size == 2 and sp.rp_world_size == 1
        local_inputs = {name: value.clone() for name, value in inputs.items()}
        sp.prepare_inputs(local_inputs)
        local_logits = model(**local_inputs).logits
        assert local_logits.shape == (2, (length + 1) // 2, config.vocab_size)
        full_logits = sp.gather(local_logits.detach().contiguous(), dim=1)[:, :length].float().cpu()
        valid = inputs['attention_mask'].bool().cpu()
        torch.testing.assert_close(
            full_logits[valid], reference_logits[valid], rtol=logits_tolerance[0], atol=logits_tolerance[1])

        labels = F.pad(labels, (0, local_logits.shape[1] * 2 - length), value=-100)
        local_labels = labels.chunk(2, dim=1)[sp.sp_rank].contiguous()
        local_loss = F.cross_entropy(
            local_logits.float().reshape(-1, config.vocab_size),
            local_labels.reshape(-1),
            ignore_index=-100,
            reduction='sum') / count
        full_loss = local_loss.detach().clone()
        dist.all_reduce(full_loss, group=sp.sp_group)
        torch.testing.assert_close(
            full_loss.float().cpu(), reference_loss, rtol=loss_tolerance[0], atol=loss_tolerance[1])
        local_loss.backward()
        # Each rank differentiated local_ce_sum / global_count. SUM recovers full gradients.
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            gradient = parameter.grad.detach().clone()
            dist.all_reduce(gradient, group=sp.sp_group)
            torch.testing.assert_close(
                gradient.float().cpu(),
                reference_gradients[name],
                rtol=gradient_tolerance[0],
                atol=gradient_tolerance[1],
                msg=name)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize('length', [8, 9])
@pytest.mark.parametrize('padding', ['none', 'left', 'right'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_olmo3_ulysses_matches_unsharded_model(tmp_path, monkeypatch, length, padding, dtype):
    from swift.utils import get_device

    if not dist.is_available():
        pytest.skip('torch.distributed is unavailable')
    device_type = torch.device(get_device()).type
    if device_type == 'npu':
        if torch.npu.device_count() < 2:
            pytest.skip('Two Ascend NPUs are required for HCCL Ulysses checks')
        monkeypatch.setenv('HCCL_CONNECT_TIMEOUT', '180')
    elif device_type == 'cpu':
        if not dist.is_gloo_available():
            pytest.skip('CPU Ulysses checks require Gloo')
        if dtype != torch.float32:
            pytest.skip('BF16 Ulysses is covered on Ascend NPUs')
    else:
        pytest.skip('This regression targets CPU/Gloo and Ascend NPU/HCCL')

    # Fresh subprocesses isolate every case from the global HF patches installed by SP.
    context = mp.spawn(
        _run_ulysses, args=((tmp_path / 'store').as_uri(), device_type, dtype, length, padding), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 240
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, 'Timed out waiting for two-process Ulysses regression'
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
