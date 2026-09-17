# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from itertools import product
from torch.distributed import init_device_mesh
from transformers.modeling_outputs import CausalLMOutputWithPast
from types import SimpleNamespace

from swift.loss.causal_lm import CustomCrossEntropyLoss
from swift.sequence_parallel import sequence_parallel
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer


class LocalLogitsModel(torch.nn.Module):

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)
        self.model_info = SimpleNamespace(is_moe_model=False)

    def forward(self, **kwargs):
        return CausalLMOutputWithPast(logits=self.logits)


def _check_loss(rank, rendezvous, ring_size, sequence_size, data_size):
    parallel_size = ring_size * sequence_size
    world_size = data_size * parallel_size
    dist.init_process_group(
        'gloo', init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    try:
        sp = sequence_parallel
        sp.world_size = parallel_size
        sp.rp_world_size = ring_size
        sp.sp_world_size = sequence_size
        sp.device_mesh = init_device_mesh(
            'cpu', (data_size, ring_size, sequence_size), mesh_dim_names=('data', 'ring', 'sequence'))
        for lengths in ([8], [5], [3, 5]):
            positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
            labels = (torch.arange(sum(lengths)).unsqueeze(0) % 6) + 1
            labels[positions < 2] = -100
            if rank // parallel_size > 0:
                labels[:, -1] = -100
            padded_positions = sp.pad(positions, padding_value=-1, position_ids=positions)
            logits = torch.randn(
                1, sum(lengths), 8, generator=torch.Generator().manual_seed(42 + rank // parallel_size))
            local_logits = sp.split(sp.pad(logits, 0, positions), 1, padded_positions)
            for scale_mode in ('none', 'weighted', 'zero'):
                for denominator, average_tokens, training in product((None, 20), (False, True), (False, True)):
                    results = []
                    for custom in (False, True):
                        inputs = {'input_ids': labels.clamp_min(0), 'labels': labels.clone(), 'position_ids': positions}
                        if scale_mode != 'none':
                            weights = torch.arange(sum(lengths)).unsqueeze(0).float() / 3
                            inputs['loss_scale'] = weights if scale_mode == 'weighted' else torch.zeros_like(weights)
                        sp.prepare_inputs(inputs)
                        model = LocalLogitsModel(local_logits.clone())
                        model.train(training)
                        template = SimpleNamespace(
                            sequence_parallel_size=parallel_size,
                            padding_free=True,
                            compute_sft_loss=lambda model, inputs, **kwargs: model(**inputs))
                        trainer = SimpleNamespace(
                            template=template,
                            model=model,
                            label_smoother=None,
                            model_accepts_loss_kwargs=True,
                            accelerator=SimpleNamespace(unwrap_model=lambda model: model, num_processes=world_size),
                            _compute_acc=lambda *args, **kwargs: None,
                            args=SimpleNamespace(
                                use_liger_kernel=False,
                                past_index=-1,
                                enable_dft_loss=False,
                                enable_channel_loss=False,
                                average_tokens_across_devices=average_tokens,
                                tuner_backend='peft',
                                acc_strategy='token'))
                        if custom:
                            inputs['compute_loss_func'] = CustomCrossEntropyLoss(None, trainer)
                        loss = Seq2SeqTrainer.compute_loss(trainer, model, inputs, num_items_in_batch=denominator)
                        loss.backward()
                        results.append((loss.detach(), model.logits.grad))
                    for actual, expected in zip(results[1], results[0]):
                        torch.testing.assert_close(actual, expected)
                    reference_logits = logits.clone().requires_grad_()
                    token_loss = torch.nn.functional.cross_entropy(
                        reference_logits.reshape(-1, 8), labels.roll(-1, dims=1).reshape(-1), reduction='none')
                    if scale_mode != 'none':
                        weights = torch.arange(sum(lengths)).float() / 3
                        if scale_mode == 'zero':
                            weights.zero_()
                        token_loss = token_loss * weights.roll(-1)
                    count = denominator
                    if count is None:
                        count = (labels != -100).sum()
                        dist.all_reduce(count)
                        count = count / parallel_size
                    reference_loss = token_loss.sum() / count
                    if average_tokens:
                        reference_loss = reference_loss * world_size
                        if not training:
                            reference_loss = reference_loss / parallel_size
                    reference_loss.backward()
                    torch.testing.assert_close(results[1][0], reference_loss)
                    expected_grad = sp.split(sp.pad(reference_logits.grad, 0, positions), 1, padded_positions)
                    torch.testing.assert_close(results[1][1], expected_grad * parallel_size)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo is not available')
@pytest.mark.parametrize(('ring_size', 'sequence_size', 'data_size'), [(1, 2, 1), (2, 1, 1), (2, 2, 1), (1, 2, 2)])
def test_explicit_cross_entropy_matches_default_sequence_parallel_loss(tmp_path, ring_size, sequence_size, data_size):
    mp.spawn(
        _check_loss,
        args=((tmp_path / 'rendezvous').as_uri(), ring_size, sequence_size, data_size),
        nprocs=ring_size * sequence_size * data_size)
