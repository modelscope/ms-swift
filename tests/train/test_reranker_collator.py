import torch

from swift.template.base import Template


def _build_template(loss_type):
    template = Template.__new__(Template)
    template.is_training = True
    template.loss_type = loss_type
    template._data_collator = lambda batch, padding_to=None: {'encoded_batch': batch}
    return template


def test_pointwise_reranker_collator_supports_negative_only():
    template = _build_template('pointwise_reranker')
    batch = [{
        'input_ids': [[101], [102]],
        'attention_mask': [[1], [1]],
        'labels': [0, 0],
    }]

    res = Template._reranker_data_collator(template, batch)

    assert res['num_samples'] == 2
    assert torch.equal(res['labels'], torch.tensor([0, 0], dtype=torch.long))
    assert torch.equal(res['group_sizes'], torch.tensor([2], dtype=torch.long))
    assert len(res['encoded_batch']) == 2


def test_pointwise_reranker_collator_supports_positive_only():
    template = _build_template('pointwise_reranker')
    batch = [{
        'input_ids': [[201], [202]],
        'attention_mask': [[1], [1]],
        'labels': [1, 1],
    }]

    res = Template._reranker_data_collator(template, batch)

    assert res['num_samples'] == 2
    assert torch.equal(res['labels'], torch.tensor([1, 1], dtype=torch.long))
    assert torch.equal(res['group_sizes'], torch.tensor([1, 1], dtype=torch.long))
    assert len(res['encoded_batch']) == 2


def test_listwise_reranker_collator_still_skips_negative_only():
    template = _build_template('listwise_reranker')
    batch = [{
        'input_ids': [[301], [302]],
        'attention_mask': [[1], [1]],
        'labels': [0, 0],
    }]

    res = Template._reranker_data_collator(template, batch)

    assert res['num_samples'] == 0
    assert 'labels' not in res
    assert 'group_sizes' not in res
    assert res['encoded_batch'] == []
