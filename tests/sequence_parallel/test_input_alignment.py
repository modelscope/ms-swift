# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from swift.model.models.qwen import _patch_deepstack_process
from swift.sequence_parallel.sequence_parallel import SequenceParallel


@pytest.fixture(params=[(rp, sp, r, s) for rp, sp in [(1, 2), (2, 1), (2, 2)] for r in range(rp) for s in range(sp)])
def parallel(request):
    rp, sp, ring_rank, sequence_rank = request.param
    parallel = SequenceParallel()
    parallel.rp_world_size = rp
    parallel.sp_world_size = sp
    parallel.world_size = rp * sp
    parallel.tokenizer = SimpleNamespace(pad_token_id=0)
    with patch.object(
            SequenceParallel, 'rp_rank', new_callable=PropertyMock, return_value=ring_rank), patch.object(
                SequenceParallel, 'sp_rank', new_callable=PropertyMock, return_value=sequence_rank):
        yield parallel


def reference_indices(lengths, parallel):
    # Index the original tokens; -1 denotes newly inserted padding.
    indices = torch.arange(sum(lengths))
    if parallel.rp_world_size == 1:
        indices = F.pad(indices, (0, -len(indices) % parallel.world_size), value=-1)
    else:
        parts = []
        for sequence in indices.split(lengths):
            sequence = F.pad(sequence, (0, -len(sequence) % (2 * parallel.world_size)), value=-1)
            chunk_size = len(sequence) // (2 * parallel.rp_world_size)
            for rank in (parallel.rp_rank, 2 * parallel.rp_world_size - 1 - parallel.rp_rank):
                parts.append(sequence[rank * chunk_size:(rank + 1) * chunk_size])
        indices = torch.cat(parts)
    return indices.chunk(parallel.sp_world_size)[parallel.sp_rank]


@pytest.mark.parametrize('lengths', ([3, 5], [4, 8], [1, 1, 2], [7]))
def test_extra_fields_follow_input_tokens(parallel, lengths):
    input_ids = torch.arange(10, 10 + sum(lengths)).unsqueeze(0)
    positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
    result = parallel.pad_and_split_inputs(
        input_ids,
        None,
        None,
        positions,
        None,
        None,
        extra_split_values=[(input_ids.clone(), 0, -1), (input_ids.clone(), 0, 1)])

    indices = reference_indices(lengths, parallel)
    expected = input_ids[:, indices.clamp_min(0)].masked_fill(indices[None] < 0, 0)
    torch.testing.assert_close(result[0], expected)
    for extra in result[-1]:
        torch.testing.assert_close(extra, expected)
    torch.testing.assert_close(parallel.real_position_ids, positions)


@pytest.mark.parametrize('lengths', ([3, 5], [4, 8], [1, 1, 2], [7]))
def test_visual_mask_preserves_embedding_alignment(parallel, lengths):
    count = sum(lengths)
    input_ids = torch.arange(10, 10 + count).unsqueeze(0)
    positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
    parallel.extra_kwargs = {'input_ids': input_ids, 'text_position_ids': positions}
    visual_mask = (torch.arange(count) % 3 != 0).unsqueeze(0)
    embeddings = torch.arange(int(visual_mask.sum()) * 2, dtype=torch.float32).reshape(-1, 2).requires_grad_()

    mask, selected = parallel.pad_and_split_mm_tokens(visual_mask, embeddings)

    indices = reference_indices(lengths, parallel)
    expected_mask = visual_mask[:, indices.clamp_min(0)] & (indices[None] >= 0)
    token_to_embedding = visual_mask[0].long().cumsum(0) - 1
    embedding_indices = token_to_embedding[indices[expected_mask[0]]]
    torch.testing.assert_close(mask, expected_mask)
    torch.testing.assert_close(selected, embeddings[embedding_indices])
    selected.sum().backward()
    expected_grad = torch.zeros_like(embeddings)
    expected_grad.index_add_(0, embedding_indices, torch.ones_like(selected))
    torch.testing.assert_close(embeddings.grad, expected_grad)


def test_qwen_deepstack_after_forward_hook(parallel):
    import swift.sequence_parallel as sp_module

    class DeepstackModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(20, 2)

        def forward(self, inputs_embeds, visual_pos_masks, visual_embeds, **kwargs):
            return self._deepstack_process(inputs_embeds.clone(), visual_pos_masks, visual_embeds)

    lengths = [3, 5]
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    parallel.prepare_inputs({'input_ids': torch.arange(10, 18).unsqueeze(0), 'position_ids': positions})
    model = DeepstackModel()
    _patch_deepstack_process(model)
    parallel._prepare_forward_hook(model)
    mask = torch.tensor([[False, True, False, True, False, False, True, False]])
    embeddings = torch.tensor([[10., 11.], [20., 21.], [30., 31.]])
    with patch.object(sp_module, 'sequence_parallel', parallel):
        result = model(
            inputs_embeds=torch.zeros(1, 8, 2), position_ids=positions, visual_pos_masks=mask, visual_embeds=embeddings)
    full = torch.zeros(1, 8, 2)
    full[mask] = embeddings
    indices = reference_indices(lengths, parallel)
    expected = full[:, indices.clamp_min(0)].masked_fill(indices[None, :, None] < 0, 0)
    torch.testing.assert_close(result, expected)
