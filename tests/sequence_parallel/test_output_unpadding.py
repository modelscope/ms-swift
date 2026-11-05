# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from torch.distributed import init_device_mesh
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from types import SimpleNamespace

from swift.model.patcher import gather_sequence_parallel_outputs, revert_padding_free, transformers_seq_cls_forward
from swift.sequence_parallel import sequence_parallel


def _check_outputs(rank, rendezvous, ring_size, sequence_size):
    world_size = ring_size * sequence_size
    dist.init_process_group(
        'gloo', init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(42)
        sp = sequence_parallel
        sp.world_size = world_size
        sp.rp_world_size = ring_size
        sp.sp_world_size = sequence_size
        sp.device_mesh = init_device_mesh(
            'cpu', (1, ring_size, sequence_size), mesh_dim_names=('data', 'ring', 'sequence'))
        for lengths in ([3, 5], [8, 8], [5], [1, 3, 5]):
            positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
            sp.extra_kwargs['text_position_ids'] = positions
            padded_positions = sp.pad(positions, padding_value=-1, position_ids=positions)
            for key in ('last_hidden_state', 'logits'):
                for padding_side in ('left', 'right'):
                    reference = torch.randn(1, sum(lengths), 3, generator=torch.Generator().manual_seed(42))
                    reference.requires_grad_()
                    padded = sp.pad(reference.detach(), padding_value=0, position_ids=positions)
                    local = sp.split(padded, dim=1, position_ids=padded_positions).detach().requires_grad_()
                    attentions = (torch.ones(1), )
                    output_type = BaseModelOutputWithPast if key == 'last_hidden_state' else CausalLMOutputWithPast
                    model_output = output_type(**{key: local}, attentions=attentions)
                    gathered = gather_sequence_parallel_outputs(model_output)
                    assert gathered is model_output
                    assert gathered.attentions is attentions
                    actual = revert_padding_free(gathered, {'position_ids': positions}, padding_side)[key]
                    expected = []
                    for chunk in reference[0].split(lengths):
                        pad_length = max(lengths) - chunk.shape[0]
                        padding = (0, 0, pad_length, 0) if padding_side == 'left' else (0, 0, 0, pad_length)
                        expected.append(torch.nn.functional.pad(chunk, padding))
                    expected = torch.stack(expected)
                    actual_loss = actual.square().sum()
                    expected_loss = expected.square().sum()
                    if padding_side == 'left':
                        classifier = SimpleNamespace(
                            config=SimpleNamespace(use_return_dict=True, problem_type='single_label_classification'),
                            num_labels=2,
                            score=torch.nn.Linear(3, 2, bias=False))
                        kwargs = dict(
                            input_ids=torch.ones(len(lengths), max(lengths), dtype=torch.long),
                            labels=torch.arange(len(lengths)) % 2,
                            padding_side='left')
                        output = transformers_seq_cls_forward(
                            classifier,
                            origin_forward=lambda **_: BaseModelOutputWithPast(last_hidden_state=actual),
                            **kwargs)
                        baseline = transformers_seq_cls_forward(
                            classifier,
                            origin_forward=lambda **_: BaseModelOutputWithPast(last_hidden_state=expected),
                            **kwargs)
                        torch.testing.assert_close(output.logits, baseline.logits)
                        torch.testing.assert_close(output.loss, baseline.loss)
                        actual_loss = actual_loss + output.loss
                        expected_loss = expected_loss + baseline.loss
                    torch.testing.assert_close(actual, expected)
                    actual_loss.backward()
                    expected_loss.backward()
                    expected_grad = sp.split(
                        sp.pad(reference.grad, padding_value=0, position_ids=positions), 1, padded_positions)
                    torch.testing.assert_close(local.grad, expected_grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo is not available')
@pytest.mark.parametrize(('ring_size', 'sequence_size'), [(2, 1), (2, 2), (1, 2), (1, 1)])
def test_gathered_outputs_unpack_like_original_sequences(tmp_path, ring_size, sequence_size):
    mp.spawn(
        _check_outputs,
        args=((tmp_path / 'rendezvous').as_uri(), ring_size, sequence_size),
        nprocs=ring_size * sequence_size)
