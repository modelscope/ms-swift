# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loss/grad parity between a single-process baseline and a sequence-parallel run.

Gated by ``SWIFT_RUN_CUDA_E2E=1`` because Ulysses attention requires NCCL.
"""

import math
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from transformers import LlamaConfig, LlamaForCausalLM
from types import SimpleNamespace

from swift.sequence_parallel import sequence_parallel
from swift.trainers.utils import per_token_loss_func, per_token_loss_func_sp

SEQ_LEN = 32
NUM_HEADS = 4


def _build_llama(gradient_checkpointing: bool) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        max_position_embeddings=256,
        attn_implementation='sdpa',
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).cuda().float()


def _build_qwen3_5(gradient_checkpointing: bool):
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        full_attention_interval=4,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=256,
        attn_implementation='sdpa',
    )
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config).cuda().float()
    # Route the linear attention through ms-swift's sequence-parallel implementation.
    from swift.model.models.qwen import _patch_qwen3_5_linear_attention_sequence_parallel
    _patch_qwen3_5_linear_attention_sequence_parallel()
    return model


_BUILDERS = {'llama': _build_llama, 'qwen3_5': _build_qwen3_5}


def _build_model(model_family: str, gradient_checkpointing: bool):
    model = _BUILDERS[model_family](gradient_checkpointing)
    # Attributes normally attached by ms-swift's model loader.
    model.model_info = SimpleNamespace(is_moe_model=False)
    model.model_meta = SimpleNamespace(model_arch=SimpleNamespace(language_model=None), is_multimodal=False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def _run(rank, sp_size, model_family, gradient_checkpointing, output_path, rendezvous):
    # get_dist_setting()/get_device() read these; mp.spawn does not populate them like torchrun does.
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(sp_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl', init_method=rendezvous, rank=rank, world_size=sp_size, timeout=timedelta(seconds=120))
    try:
        model = _build_model(model_family, gradient_checkpointing)
        input_ids = (torch.arange(1, SEQ_LEN + 1, device='cuda') % 128).unsqueeze(0)
        labels = input_ids.clone()
        labels[:, :4] = -100
        position_ids = torch.arange(SEQ_LEN, device='cuda').unsqueeze(0)
        inputs = {
            'input_ids': input_ids.clone(),
            'labels': labels.clone(),
            'position_ids': position_ids.clone(),
            'use_cache': False,
        }
        num_items = int((labels != -100).sum())

        if sp_size == 1:
            outputs = model(**inputs)
            loss = per_token_loss_func(outputs, inputs['labels']).sum() / num_items
        else:
            sequence_parallel.prepare(
                sp_size, model=model, tokenizer=SimpleNamespace(pad_token_id=0), padding_free=True)
            sequence_parallel.prepare_inputs(inputs)
            local_labels = inputs.pop('labels')
            outputs = model(**inputs)
            # Mirrors Seq2SeqTrainer.compute_loss: num_items is gathered over all ranks (SP replicas
            # included), then the loss is scaled by the world size.
            loss = per_token_loss_func_sp(outputs, local_labels).sum() / (num_items * sp_size) * sp_size

        loss.backward()

        # DDP averages gradients across ranks; reproduce that here so the SP pieces are summed.
        world_size = dist.get_world_size()
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().clone()
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grads[name] = (grad / world_size).cpu()
        if rank == 0:
            torch.save({'loss': loss.item(), 'grads': grads}, output_path)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(world_size, model_family, gradient_checkpointing, output_path, rendezvous):
    mp.spawn(
        _run,
        args=(world_size, model_family, gradient_checkpointing, str(output_path), rendezvous),
        nprocs=world_size,
        join=True)


def _check_parity(tmp_path, sp_size, model_family, gradient_checkpointing):
    ref_path = tmp_path / 'reference.pt'
    sp_path = tmp_path / 'sp.pt'
    _spawn(1, model_family, False, ref_path, (tmp_path / 'rv_ref').as_uri())
    _spawn(sp_size, model_family, gradient_checkpointing, sp_path, (tmp_path / 'rv_sp').as_uri())

    reference = torch.load(ref_path, weights_only=False)
    actual = torch.load(sp_path, weights_only=False)
    assert math.isclose(
        reference['loss'], actual['loss'], rel_tol=1e-5, abs_tol=1e-6), (reference['loss'], actual['loss'])
    assert reference['grads'].keys() == actual['grads'].keys()
    # qwen3_5's linear attention runs fla/triton kernels with different head batch shapes on the SP
    # and non-SP paths (measured fp32 absdiff ~2e-6), so its grad tolerance only needs to catch
    # structural errors, which show up as O(1) deviations.
    rtol, atol = (1e-3, 1e-5) if model_family == 'qwen3_5' else (1e-5, 1e-6)
    for name, expected in reference['grads'].items():
        torch.testing.assert_close(actual['grads'][name], expected, rtol=rtol, atol=atol, msg=name)


def _skip_reason():
    if os.environ.get('SWIFT_RUN_CUDA_E2E') != '1':
        return 'Set SWIFT_RUN_CUDA_E2E=1 to run the distributed CUDA E2E test'
    if not torch.cuda.is_available():
        return 'CUDA is not available'
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or '')
@pytest.mark.parametrize('model_family', ['llama', 'qwen3_5'])
@pytest.mark.parametrize('sp_size', [2, 4])
@pytest.mark.parametrize('gradient_checkpointing', [False, True])
def test_sp_loss_grad_matches_single_process(tmp_path, model_family, sp_size, gradient_checkpointing):
    if torch.cuda.device_count() < sp_size:
        pytest.skip(f'Requires {sp_size} GPUs, found {torch.cuda.device_count()}')
    _check_parity(tmp_path, sp_size, model_family, gradient_checkpointing)


if __name__ == '__main__':
    if os.environ.get('SWIFT_RUN_CUDA_E2E') != '1':
        raise SystemExit('Set SWIFT_RUN_CUDA_E2E=1 to run the distributed CUDA E2E test')
    import tempfile
    from pathlib import Path

    for model_family in ('llama', 'qwen3_5'):
        for sp_size in (2, 4):
            for gradient_checkpointing in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    _check_parity(Path(directory), sp_size, model_family, gradient_checkpointing)
                    print(f'{model_family} sp_size={sp_size} gradient_checkpointing={gradient_checkpointing}: OK')
