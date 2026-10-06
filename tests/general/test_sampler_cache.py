import json
import numpy as np
import pytest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from swift.pipelines.sampling.utils import get_messages_md5
from swift.pipelines.sampling.vanilla_sampler import VanillaSampler
from swift.ray_utils import RayHelper


@pytest.mark.parametrize('reward_model', [None, 'orm', 'prm', 'both'])
@pytest.mark.parametrize('cached', [True, False])
@pytest.mark.parametrize('batch_size', [1, 2])
def test_sampling_preserves_choices(tmp_path, monkeypatch, reward_model, cached, batch_size):
    monkeypatch.setattr(RayHelper, 'ray_inited', lambda: False)
    row = {'messages': [{'role': 'user', 'content': 'What is 1 + 1?'}, {'role': 'assistant', 'content': '2'}]}
    choices = ['two', 'three']
    cache_path = tmp_path / 'cache.jsonl'
    cached_rows = []
    for choice in choices:
        cached_row = deepcopy(row)
        cached_row['id'] = get_messages_md5(row)
        cached_row['messages'][-1]['content'] = choice
        cached_rows.append(json.dumps(cached_row) + '\n')
    cache_path.write_text(''.join(cached_rows), encoding='utf-8')

    # Exercise the real cache reader and sampling methods without loading a model.
    sampler = VanillaSampler.__new__(VanillaSampler)
    sampler.args = SimpleNamespace(
        cache_files=[str(cache_path)] if cached else [],
        num_return_sequences=2,
        system=None,
        max_new_tokens=8,
        temperature=1.,
        top_k=50,
        top_p=1.,
        orm_model='orm' if reward_model in ('orm', 'both') else None,
        prm_model='prm' if reward_model in ('prm', 'both') else None,
        n_best_to_keep=1,
        easy_query_threshold=None)
    sampler.caches = sampler.read_cache()
    cache_before = deepcopy(sampler.caches)
    responses = [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=c))]) for c in choices]
    sampler.infer_engine = None if cached else SimpleNamespace(infer=Mock(return_value=responses * batch_size))
    scores = np.array([0.8, 0.1, 1.0])
    mask = np.array([True, False, True])
    sampler.get_orm_score = Mock(return_value=(scores, mask))
    sampler.get_prm_score = Mock(return_value=(scores, mask))
    data = {'messages': [row['messages']] * batch_size}
    data_before = deepcopy(data)

    first = sampler.do_sample(data)
    second = sampler.do_sample(data)

    assert second == first
    assert sampler.caches == cache_before
    assert data == data_before
    assert cache_path.read_text(encoding='utf-8') == ''.join(cached_rows)
    output = [json.loads(line) for line in first]
    if reward_model is None:
        assert [item['messages'][-1]['content'] for item in output] == choices * batch_size
    else:
        assert len(output) == batch_size
        for item in output:
            assert item['messages'][-1]['content'] == '2'
            assert item['rejected_response'] == 'three'
    for item in output:
        assert item['id'] == get_messages_md5(row)
        assert 'choices' not in item
    for model, enabled in [(sampler.get_orm_score, sampler.args.orm_model),
                           (sampler.get_prm_score, sampler.args.prm_model)]:
        assert model.call_count == (2 * batch_size if enabled else 0)
        for call in model.call_args_list:
            requests, ground_truth = call.args
            assert [r['messages'][-1]['content'] for r in requests] == choices + ['2']
            assert ground_truth == '2'
    if not cached:
        assert sampler.infer_engine.infer.call_count == 2
