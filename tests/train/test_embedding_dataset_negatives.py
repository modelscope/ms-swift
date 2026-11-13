# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import pytest
import torch
from torch.nn import functional as F
from unittest.mock import patch

from swift.dataloader.dataset_sampler import DatasetBatchSamplerShard
from swift.dataset import DatasetMeta
from swift.loss.embedding import InfonceLoss, _parse_multi_negative_sentences
from swift.template.base import Template
from swift.template.template_inputs import TemplateInputs


@pytest.mark.parametrize('world_size', [1, 2, 3])
@pytest.mark.parametrize('drop_last', [False, True])
def test_global_batches_and_tail_handling(world_size, drop_last):
    groups = ['retrieval'] * 5 + ['classification'] * 11 + ['tiny']
    batches = []
    with patch('torch.distributed.is_initialized', return_value=True), \
            patch('torch.distributed.get_world_size', return_value=world_size):
        for rank in range(world_size):
            with patch('torch.distributed.get_rank', return_value=rank):
                sampler = DatasetBatchSamplerShard(groups, 2, shuffle=True, drop_last=drop_last, data_seed=8109)
                observed = list(sampler)
                assert len(observed) == len(sampler)
                assert observed == list(sampler)
                batches.append(observed)
    assert len({len(rank_batches) for rank_batches in batches}) == 1
    seen = []
    for step in range(len(batches[0])):
        global_batch = [index for rank_batches in batches for index in rank_batches[step]]
        assert len(global_batch) == world_size * 2
        assert len({groups[index] for index in global_batch}) == 1
        seen.extend(global_batch)
    if drop_last:
        assert len(seen) == len(set(seen))
    else:
        assert set(seen) == set(range(len(groups)))


def test_epoch_shuffle_and_resume_batches():
    from accelerate.data_loader import SkipBatchSampler
    sampler = DatasetBatchSamplerShard(['a'] * 20 + ['b'] * 20, 2, shuffle=True, drop_last=False, data_seed=8109)
    first = list(sampler)
    assert list(SkipBatchSampler(sampler, skip_batches=3)) == first[3:]
    sampler.set_epoch(1)
    second = list(sampler)
    assert second != first
    sampler.set_epoch(1)
    assert list(sampler) == second


@pytest.mark.parametrize('use_batch,hard_negatives', [(True, 0), (True, 1), (False, 1), (False, 2)])
def test_dataset_loss_overrides_values_and_gradients(monkeypatch, use_batch, hard_negatives):
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('INFONCE_USE_BATCH', str(not use_batch))
    monkeypatch.setenv('INFONCE_HARD_NEGATIVES', '4')
    monkeypatch.setenv('INFONCE_TEMPERATURE', '.5')
    torch.manual_seed(8109)
    embeddings = F.normalize(torch.randn(6, 5, dtype=torch.float64), dim=-1).requires_grad_()
    labels = torch.tensor([1., 0., 1., 0.])
    config = {'use_in_batch_negatives': use_batch, 'hard_negatives': hard_negatives}
    environment = dict(os.environ)
    actual = InfonceLoss(None, None)({'last_hidden_state': embeddings}, labels, infonce_config=config)
    reference = embeddings.detach().clone().requires_grad_()
    indices = [1] + [2] * hard_negatives + [4] + [5] * hard_negatives
    docs = reference[indices]
    queries = reference[[0, 3]]
    if use_batch:
        logits = queries @ docs.T / .5
        targets = torch.tensor([0, hard_negatives + 1])
    else:
        logits = (queries[:, None] * docs.reshape(2, hard_negatives + 1, 5)).sum(-1) / .5
        targets = torch.zeros(2, dtype=torch.long)
    expected = F.cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(actual, embeddings)[0], torch.autograd.grad(expected, reference)[0])
    assert dict(os.environ) == environment


def test_zero_hard_negatives_on_positive_only_samples():
    embeddings = torch.randn(4, 3)
    parsed = _parse_multi_negative_sentences(embeddings, torch.tensor([1., 1.]), hard_negatives=0)
    assert [tuple(group.shape) for group in parsed] == [(2, 3), (2, 3)]
    with pytest.raises(ValueError, match='without explicit negatives'):
        _parse_multi_negative_sentences(embeddings, torch.tensor([1., 1.]), hard_negatives=1)


def test_invalid_configurations_and_mixed_collation():
    for config in [{
            'hard_negatives': -1
    }, {
            'use_in_batch_negatives': 'false'
    }, {
            'unknown': 1
    }, {
            'use_in_batch_negatives': False,
            'hard_negatives': 0
    }]:
        with pytest.raises(ValueError):
            DatasetMeta(infonce_config=config)
    template = object.__new__(Template)
    batch = [{
        '_dataset_group': 'a',
        '_infonce_config': json.dumps({})
    }, {
        '_dataset_group': 'b',
        '_infonce_config': json.dumps({})
    }]
    with pytest.raises(ValueError, match='exactly one dataset'):
        template._embedding_data_collator(batch)


def test_nullable_negative_columns_in_concatenated_datasets():
    conversation = [{'role': 'user', 'content': 'text'}]
    inputs = TemplateInputs.from_dict({
        'messages': conversation,
        'positive_messages': [conversation],
        'negative_messages': None,
    })
    assert len(inputs.positive) == 1
    assert inputs.negative == []
    template = object.__new__(Template)
    observed = []

    def collator(batch, **kwargs):
        observed.extend(batch)
        return {}

    template._data_collator = collator
    result = template._embedding_data_collator([{
        'anchor_input_ids': [1, 2],
        'positive_input_ids': [3, 4],
        'negative_input_ids': None,
        'labels': [1.],
        '_dataset_group': 'retrieval',
        '_infonce_config': '{"hard_negatives": 0}',
    }])
    assert len(observed) == 2
    assert result['_dataset_group'] == 'retrieval'
    torch.testing.assert_close(result['labels'], torch.tensor([1.]))
