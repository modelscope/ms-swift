"""Unit tests for multi-teacher model server support.

Tests cover:
- parse_teacher_model_server: single URL / JSON parsing + non-empty & non-overlapping tag validation
- route_samples_to_teachers: single-teacher (all samples) and multi-teacher tag routing + fail-fast
- fetch_teacher_parsed_by_routing: teacher-count-agnostic fetch + scatter back to sample order
- expand_advantage_to_per_token: scalar teacher KL coefficient (all teachers share --teacher_kl_coef)
- streaming multi-dataset load: auto-injected ``dataset`` column + routing parity with non-streaming
"""
import json
import os
import tempfile
import torch
import unittest
from typing import List, Optional

from swift.dataset import load_dataset
from swift.rl_core.advantage import expand_advantage_to_per_token
from swift.rl_core.data import OnPolicySample
from swift.rlhf_trainers.gkd_helpers import (TeacherServerConfig, fetch_teacher_parsed_by_routing, get_sample_tag,
                                             parse_teacher_model_server, route_samples_to_teachers)


def _make_sample(tag: Optional[str] = None, source: Optional[str] = None) -> OnPolicySample:
    extra = {}
    if tag is not None:
        extra['dataset'] = tag
    if source is not None:
        extra['source'] = source
    return OnPolicySample(messages=[], extra=extra)


class TestParseTeacherModelServer(unittest.TestCase):

    def test_parse_none(self):
        self.assertIsNone(parse_teacher_model_server(None))

    def test_parse_single_url(self):
        result = parse_teacher_model_server('http://localhost:8000')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].url, 'http://localhost:8000')
        self.assertEqual(result[0].tags, [])

    def test_parse_multi_json(self):
        config = json.dumps([
            {
                'url': 'http://localhost:8000',
                'tags': ['math']
            },
            {
                'url': 'http://localhost:8001',
                'tags': ['code']
            },
        ])
        result = parse_teacher_model_server(config)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].tags, ['math'])
        self.assertEqual(result[1].tags, ['code'])

    def test_parse_empty_json_list(self):
        with self.assertRaises(ValueError):
            parse_teacher_model_server('[]')

    def test_parse_missing_url(self):
        with self.assertRaises(ValueError):
            parse_teacher_model_server(json.dumps([{'tags': ['math']}]))

    def test_parse_invalid_tags(self):
        config = json.dumps([{'url': 'http://localhost:8000', 'tags': 'math'}])
        with self.assertRaises(ValueError):
            parse_teacher_model_server(config)

    def test_parse_multi_empty_tags_rejected(self):
        """With multiple teachers, empty tags are rejected (each sample needs exactly one teacher)."""
        config = json.dumps([
            {
                'url': 'http://localhost:8000',
                'tags': ['math']
            },
            {
                'url': 'http://localhost:8001',
                'tags': []
            },
        ])
        with self.assertRaises(ValueError):
            parse_teacher_model_server(config)

    def test_parse_overlapping_tags_rejected(self):
        """A tag may not appear in more than one teacher."""
        config = json.dumps([
            {
                'url': 'http://localhost:8000',
                'tags': ['math', 'shared']
            },
            {
                'url': 'http://localhost:8001',
                'tags': ['shared', 'code']
            },
        ])
        with self.assertRaises(ValueError):
            parse_teacher_model_server(config)


class TestRouteSamplesToTeachers(unittest.TestCase):

    def test_route_single_teacher_all_samples(self):
        """Single teacher (empty tags) handles all samples, tags ignored."""
        samples = [_make_sample('math'), _make_sample(), _make_sample('code')]
        configs = [TeacherServerConfig(url='http://t0', tags=[])]
        routing = route_samples_to_teachers(samples, configs)
        self.assertEqual(routing[0], [0, 1, 2])

    def test_route_one_to_one(self):
        samples = [_make_sample('math'), _make_sample('code'), _make_sample('math')]
        configs = [
            TeacherServerConfig(url='http://t0', tags=['math']),
            TeacherServerConfig(url='http://t1', tags=['code']),
        ]
        routing = route_samples_to_teachers(samples, configs)
        self.assertEqual(routing[0], [0, 2])
        self.assertEqual(routing[1], [1])

    def test_route_unmatched_fails_fast(self):
        samples = [_make_sample('unknown')]
        configs = [
            TeacherServerConfig(url='http://t0', tags=['math']),
            TeacherServerConfig(url='http://t1', tags=['code']),
        ]
        with self.assertRaises(ValueError):
            route_samples_to_teachers(samples, configs)

    def test_route_tag_fallback_to_source(self):
        self.assertEqual(get_sample_tag(_make_sample(source='gsm8k')), 'gsm8k')

    def test_route_no_tag_returns_none(self):
        self.assertIsNone(get_sample_tag(_make_sample()))


class TestFetchTeacherParsedByRouting(unittest.TestCase):

    def test_multi_teacher_scatter_back(self):
        samples = [_make_sample('math'), _make_sample('code'), _make_sample('math')]
        configs = [
            TeacherServerConfig(url='http://t0', tags=['math']),
            TeacherServerConfig(url='http://t1', tags=['code']),
        ]
        requests = ['r0', 'r1', 'r2']
        seen = {}

        def fetch_fn(subset_reqs, client):
            seen[client.url] = list(subset_reqs)
            return [f'{client.url}:{r}' for r in subset_reqs]

        parsed = fetch_teacher_parsed_by_routing(samples, requests, configs, configs, fetch_fn=fetch_fn)
        self.assertEqual(parsed, ['http://t0:r0', 'http://t1:r1', 'http://t0:r2'])
        self.assertEqual(seen['http://t0'], ['r0', 'r2'])
        self.assertEqual(seen['http://t1'], ['r1'])

    def test_single_teacher_one_fetch(self):
        """Single teacher: one fetch over all requests in original order."""
        samples = [_make_sample(), _make_sample(), _make_sample()]
        configs = [TeacherServerConfig(url='http://t0', tags=[])]
        requests = ['r0', 'r1', 'r2']
        calls = []

        def fetch_fn(subset_reqs, client):
            calls.append(list(subset_reqs))
            return [f'p:{r}' for r in subset_reqs]

        parsed = fetch_teacher_parsed_by_routing(samples, requests, configs, configs, fetch_fn=fetch_fn)
        self.assertEqual(parsed, ['p:r0', 'p:r1', 'p:r2'])
        self.assertEqual(calls, [['r0', 'r1', 'r2']])  # exactly one fetch


class TestExpandAdvantageScalarCoef(unittest.TestCase):

    def test_scalar_coef(self):
        B, T = 2, 4
        result = expand_advantage_to_per_token(
            torch.tensor([1.0, 1.0]),
            torch.ones(B, T),
            teacher_per_token_logps=torch.tensor([[2.0] * T, [3.0] * T]),
            policy_per_token_logps=torch.tensor([[1.0] * T, [1.0] * T]),
            teacher_kl_coef=0.5,
        )
        # base(1) + 0.5 * (2-1) = 1.5; base(1) + 0.5 * (3-1) = 2.0
        torch.testing.assert_close(result[0], torch.ones(T) * 1.5)
        torch.testing.assert_close(result[1], torch.ones(T) * 2.0)

    def test_no_teacher(self):
        B, T = 2, 4
        result = expand_advantage_to_per_token(torch.tensor([1.0, 2.0]), torch.ones(B, T))
        torch.testing.assert_close(result[0], torch.ones(T) * 1.0)
        torch.testing.assert_close(result[1], torch.ones(T) * 2.0)

    def test_zero_coef_no_injection(self):
        B, T = 1, 4
        result = expand_advantage_to_per_token(
            torch.tensor([1.0]),
            torch.ones(B, T),
            teacher_per_token_logps=torch.tensor([[2.0] * T]),
            policy_per_token_logps=torch.tensor([[1.0] * T]),
            teacher_kl_coef=0.0,
        )
        torch.testing.assert_close(result[0], torch.ones(T) * 1.0)


class TestStreamingDatasetMultiTeacherRouting(unittest.TestCase):
    """End-to-end: streaming load_dataset injects ``dataset`` tags that multi-teacher routing consumes."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.paths: List[str] = []
        for name, n_rows in (('math.jsonl', 3), ('code.jsonl', 2)):
            path = os.path.join(cls.tmpdir, name)
            with open(path, 'w', encoding='utf-8') as f:
                for i in range(n_rows):
                    row = {'messages': [{'role': 'user', 'content': f'{name}-{i}'}]}
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            cls.paths.append(path)

    def _teacher_configs(self) -> List[TeacherServerConfig]:
        return [
            TeacherServerConfig(url='http://t0', tags=[self.paths[0]]),
            TeacherServerConfig(url='http://t1', tags=[self.paths[1]]),
        ]

    def _load_samples(self, *, streaming: bool, interleave_prob: Optional[List[float]] = None) -> List[OnPolicySample]:
        kwargs = dict(datasets=self.paths, streaming=streaming, split_dataset_ratio=0.)
        if interleave_prob is not None:
            kwargs['interleave_prob'] = interleave_prob
            kwargs['stopping_strategy'] = 'all_exhausted'
        train, _ = load_dataset(**kwargs)
        return [OnPolicySample.from_row(row) for row in train]

    def test_streaming_injected_tags_route_correctly(self):
        samples = self._load_samples(streaming=True)
        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertIn(sample.extra.get('dataset'), self.paths)

        routing = route_samples_to_teachers(samples, self._teacher_configs())
        math_indices = [i for i, s in enumerate(samples) if s.extra['dataset'] == self.paths[0]]
        code_indices = [i for i, s in enumerate(samples) if s.extra['dataset'] == self.paths[1]]
        self.assertEqual(routing[0], math_indices)
        self.assertEqual(routing[1], code_indices)
        self.assertEqual(len(math_indices), 3)
        self.assertEqual(len(code_indices), 2)

    def test_streaming_fetch_by_routing(self):
        samples = self._load_samples(streaming=True)
        configs = self._teacher_configs()
        requests = [f'r{i}' for i in range(len(samples))]
        seen = {}

        def fetch_fn(subset_reqs, client):
            seen[client.url] = list(subset_reqs)
            return [f'{client.url}:{r}' for r in subset_reqs]

        parsed = fetch_teacher_parsed_by_routing(samples, requests, configs, configs, fetch_fn=fetch_fn)
        for i, sample in enumerate(samples):
            tag = sample.extra['dataset']
            expected_url = 'http://t0' if tag == self.paths[0] else 'http://t1'
            self.assertEqual(parsed[i], f'{expected_url}:r{i}')
        self.assertEqual(len(seen['http://t0']), 3)
        self.assertEqual(len(seen['http://t1']), 2)

    def test_streaming_routing_matches_non_streaming(self):
        stream_samples = self._load_samples(streaming=True)
        static_samples = self._load_samples(streaming=False)
        configs = self._teacher_configs()
        self.assertEqual(
            route_samples_to_teachers(stream_samples, configs),
            route_samples_to_teachers(static_samples, configs),
        )
        stream_tags = [s.extra['dataset'] for s in stream_samples]
        static_tags = [s.extra['dataset'] for s in static_samples]
        self.assertEqual(stream_tags, static_tags)

    def test_streaming_interleave_preserves_per_dataset_tags(self):
        samples = self._load_samples(streaming=True, interleave_prob=[0.5, 0.5])
        self.assertGreater(len(samples), 0)
        for sample in samples:
            self.assertIn(sample.extra.get('dataset'), self.paths)

        routing = route_samples_to_teachers(samples, self._teacher_configs())
        self.assertEqual(sorted(routing[0] + routing[1]), list(range(len(samples))))
        math_count = sum(1 for s in samples if s.extra['dataset'] == self.paths[0])
        code_count = sum(1 for s in samples if s.extra['dataset'] == self.paths[1])
        self.assertEqual(len(routing[0]), math_count)
        self.assertEqual(len(routing[1]), code_count)


if __name__ == '__main__':
    unittest.main()
