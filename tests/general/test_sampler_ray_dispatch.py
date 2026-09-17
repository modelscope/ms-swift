import pytest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from swift.pipelines.sampling.distill_sampler import DistillSampler
from swift.pipelines.sampling.utils import get_messages_md5
from swift.pipelines.sampling.vanilla_sampler import VanillaSampler
from swift.ray_utils import RayHelper


@pytest.mark.parametrize('sampler_cls', [VanillaSampler, DistillSampler])
@pytest.mark.parametrize('num_rows,num_workers', [(0, 2), (1, 3), (5, 2), (6, 3)])
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('with_media', [False, True])
def test_ray_sampling_matches_local(monkeypatch, sampler_cls, num_rows, num_workers, cached, with_media):
    rows = []
    for i in range(num_rows):
        row = {
            'messages': [{
                'role': 'user',
                'content': f'question {i}'
            }, {
                'role': 'assistant',
                'content': f'reference {i}'
            }]
        }
        if with_media:
            row.update(
                images=[f'image-{i}.png'],
                videos=[f'video-{i}.mp4'],
                audios=[f'audio-{i}.wav'],
                metadata={'index': i},
                dataset='test-data')
        rows.append(row)
    columns = ['messages', 'images', 'videos', 'audios', 'metadata', 'dataset'] if with_media else ['messages']
    data = {key: [row[key] for row in rows] for key in columns}
    original = deepcopy(data)

    def infer(requests, request_config):
        answers = [f"answer to {row['messages'][-1]['content']}" for row in requests]
        if sampler_cls is DistillSampler:
            return answers
        return [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=a))]) for a in answers]

    def make_sampler():
        sampler = sampler_cls.__new__(sampler_cls)
        sampler.args = SimpleNamespace(
            num_return_sequences=2, system=None, max_new_tokens=8, temperature=1., top_k=50, top_p=1.)
        sampler.caches = {
            get_messages_md5(row): {
                'choices': ['cached one', 'cached two']
            }
            for row in rows
        } if cached else {}
        sampler.infer_engine = None if cached else SimpleNamespace(infer=Mock(side_effect=infer))
        return sampler

    monkeypatch.setattr(RayHelper, 'ray_inited', lambda: False)
    expected = make_sampler().generate(data)
    samplers = [make_sampler() for _ in range(num_workers)]

    def run_worker(sampler, batch):
        # Stand in for Ray transport while retaining the worker-side decorator.
        with monkeypatch.context() as worker_context:
            worker_context.setattr(RayHelper, 'is_worker', lambda: True)
            sampler.group = ['sampler']
            return sampler.generate(batch)

    workers = []
    for sampler in samplers:
        remote = Mock(side_effect=lambda batch, sampler=sampler: run_worker(sampler, batch))
        workers.append(SimpleNamespace(generate=SimpleNamespace(remote=remote)))
    monkeypatch.setattr(RayHelper, 'worker_instance', {'sampler': workers})
    monkeypatch.setattr(RayHelper, 'ray_inited', lambda: True)
    monkeypatch.setattr(RayHelper, 'is_worker', lambda: False)
    monkeypatch.setattr(RayHelper, 'is_called_from_init', lambda: False)
    monkeypatch.setattr(RayHelper, 'execute_all_sync', RayHelper.execute_all_async)

    actual = make_sampler().generate(data)

    assert actual == expected
    assert data == original
    for key in columns:
        assert [value for worker in workers for value in worker.generate.remote.call_args.args[0][key]] == data[key]
    for worker in workers:
        worker.generate.remote.assert_called_once()
    if cached:
        for sampler in samplers:
            assert sampler.caches == make_sampler().caches
    else:
        requests = [
            row for sampler in samplers for call in sampler.infer_engine.infer.call_args_list for row in call.args[0]
        ]
        assert requests == [{**row, 'messages': row['messages'][:-1]} for row in rows for _ in range(2)]
