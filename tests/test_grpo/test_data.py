"""Tests for swift.rl_core.data — GRPO data structures."""
import pickle
import pytest
import torch

from swift.rl_core.data import GKDBatch, GKDSample, GRPOBatch, GRPOSample, OnPolicySample
from swift.rlhf_trainers.gkd_loss import DataSource

B, T = 4, 8
DEVICE = 'cpu'


def _make_rl_batch(**overrides):
    defaults = {
        'completion_mask': torch.ones(B, T, dtype=torch.bool, device=DEVICE),
        'truncated_mask': torch.zeros(B, dtype=torch.bool, device=DEVICE),
        'seq_lengths': torch.full((B, ), T, dtype=torch.long, device=DEVICE),
    }
    defaults.update(overrides)
    return GRPOBatch(**defaults)


class TestGRPOBatch:

    def test_required_fields(self):
        rl = _make_rl_batch()
        assert rl.completion_mask.shape == (B, T)
        assert rl.truncated_mask.shape == (B, )
        assert rl.old_per_token_logps is None
        assert rl.advantages is None

    def test_optional_fields(self):
        rl = _make_rl_batch(
            old_per_token_logps=torch.randn(B, T),
            ref_per_token_logps=torch.randn(B, T),
            advantages=torch.randn(B),
            num_items_in_batch=torch.tensor(24.0),
            logits_to_keep=6,
        )
        assert rl.old_per_token_logps.shape == (B, T)
        assert rl.advantages.shape == (B, )
        assert rl.logits_to_keep == 6

    def test_mutable_after_creation(self):
        rl = _make_rl_batch()
        assert rl.advantages is None
        rl.advantages = torch.randn(B)
        assert rl.advantages.shape == (B, )
        rl.num_items_in_batch = torch.tensor(32.0)
        assert rl.num_items_in_batch.item() == 32.0


def _make_sample(cls=OnPolicySample, **overrides):
    defaults = {
        'messages': [{
            'role': 'user',
            'content': 'hi'
        }, {
            'role': 'assistant',
            'content': 'yo'
        }],
        'prompt_id': 'prompt_0',
        'request_id': 'req_0',
    }
    defaults.update(overrides)
    return cls(**defaults)


class TestOnPolicySample:

    def test_required_fields(self):
        s = _make_sample()
        assert s.prompt_id == 'prompt_0'
        assert s.request_id == 'req_0'
        assert s.extra == {}
        assert s.response_token_ids == []
        assert s.encoded is None

    def test_is_truncated_property(self):
        assert _make_sample(finish_reason='length').is_truncated is True
        assert _make_sample(finish_reason='stop').is_truncated is False
        assert _make_sample().is_truncated is False

    def test_multi_turn_nested_shape(self):
        s = _make_sample(
            response_token_ids=[[1, 2], [3, 4, 5]],
            response_loss_mask=[[1, 1], [1, 1, 1]],
            rollout_logprobs=[[-0.1, -0.2], [-0.3, -0.4, -0.5]],
        )
        assert len(s.response_token_ids) == 2
        assert s.response_token_ids[1] == [3, 4, 5]

    def test_to_reward_row_flattens_extra(self):
        s = _make_sample(extra={'solution': '42', 'target': 7})
        row = s.to_reward_row()
        # dataset columns flattened to top level
        assert row['solution'] == '42'
        assert row['target'] == 7
        # core fields present
        assert row['messages'] == s.messages
        assert row['request_id'] == 'req_0'
        assert row['is_truncated'] is False
        # encoded excluded (heavy, model-internal)
        assert 'encoded' not in row

    def test_to_reward_row_excludes_rollout_only_fields(self):
        # routed_experts / response_token_ids / rollout_logprobs are model/rollout-internal,
        # not useful for reward functions; they must not appear in the reward row.
        s = _make_sample(routed_experts=torch.zeros(2, 2), response_token_ids=[[1, 2]])
        row = s.to_reward_row()
        assert 'routed_experts' not in row
        assert 'response_token_ids' not in row
        assert 'rollout_logprobs' not in row

    def test_pickle_roundtrip(self):
        # Ray ships List[OnPolicySample] via cloudpickle; must survive pickle.
        s = _make_sample(
            extra={'solution': '42'},
            response_token_ids=[[1, 2, 3]],
            encoded={
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([-100, 2, 3]),
                'length': 3
            },
        )
        restored = pickle.loads(pickle.dumps(s))
        assert restored.prompt_id == s.prompt_id
        assert restored.extra == {'solution': '42'}
        assert torch.equal(restored.encoded['input_ids'], s.encoded['input_ids'])


class TestGRPOSample:

    def test_inherits_and_adds_fields(self):
        s = _make_sample(cls=GRPOSample)
        assert isinstance(s, OnPolicySample)
        assert s.rewards is None
        assert s.advantages is None

    def test_advantage_assignment(self):
        s = _make_sample(cls=GRPOSample)
        s.advantages = 1.5
        s.rewards = [1.0, 0.0]
        assert s.advantages == 1.5
        assert s.rewards == [1.0, 0.0]

    def test_to_reward_row_works_on_subclass(self):
        s = _make_sample(cls=GRPOSample, extra={'solution': 'x'})
        assert s.to_reward_row()['solution'] == 'x'


class TestOnPolicySampleHelpers:

    def test_from_row_maps_known_and_extra(self):
        row = {
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }],
            'prompt_id': 'p1',
            'request_id': 'r1',
            'response_token_ids': [[1, 2]],
            'finish_reason': 'stop',
            'add_eos': True,
            'is_truncated': False,  # derived -> dropped
            'solution': '42',  # dataset passthrough -> extra
            'images': ['img0'],  # multimodal -> explicit field
        }
        s = OnPolicySample.from_row(row)
        assert s.prompt_id == 'p1'
        assert s.response_token_ids == [[1, 2]]
        assert s.add_eos is True
        assert s.extra == {'solution': '42'}
        # multimodal standard column -> explicit field, not extra
        assert s.images == ['img0']
        assert 'images' not in s.extra
        # derived key not stored
        assert 'is_truncated' not in s.extra

    def test_from_row_defaults_ids(self):
        s = OnPolicySample.from_row({'messages': [{'role': 'user', 'content': 'x'}]})
        assert s.prompt_id == ''
        assert s.request_id == ''

    def test_to_template_dict_passes_chat_template_kwargs_not_other_extra(self):
        # Only chat_template_kwargs is forwarded to encode; other dataset columns
        # (solution/...) are reward-only and excluded from the template dict.
        s = _make_sample(
            extra={
                'solution': '42',
                'chat_template_kwargs': {
                    'enable_thinking': False
                }
            }, images=['img0'], add_eos=True)
        d = s.to_template_dict()
        assert d['messages'] == s.messages
        assert d['images'] == ['img0']
        assert d['add_eos'] is True
        assert d['chat_template_kwargs'] == {'enable_thinking': False}
        assert 'solution' not in d

    def test_from_row_to_template_dict_roundtrip(self):
        row = {'messages': [{'role': 'user', 'content': 'q'}], 'solution': 's', 'add_eos': False}
        s = OnPolicySample.from_row(row)
        d = s.to_template_dict()
        assert d['messages'] == row['messages']
        # dataset-only column stays in extra, not forwarded to encode
        assert 'solution' not in d
        assert s.extra['solution'] == 's'

    def test_apply_rollout_output(self):
        from types import SimpleNamespace as NS
        s = _make_sample()
        # multi-turn-style output: messages provided (full history), explicit token ids,
        # multimodal injected via rollout_infos.
        choice = NS(
            message=NS(content='a'),
            token_ids=[5, 6],
            logprobs=None,
            finish_reason='length',
        )
        rollout_output = NS(
            response=NS(choices=[choice], id='r1'),
            messages=[{
                'role': 'assistant',
                'content': 'a'
            }],
            response_token_ids=[[9, 8]],
            response_loss_mask=[],
            rollout_logprobs=[],
            rollout_infos={'images': ['i']},
        )
        s.apply_rollout_output(rollout_output=rollout_output)
        assert s.response_token_ids == [[9, 8]]
        assert s.is_truncated is True
        assert s.images == ['i']

    def test_to_infer_request_basic(self):
        s = _make_sample(request_id='req_42', images=['/path/a.png'])
        req = s.to_infer_request()
        assert req.messages == s.messages
        assert req.uuid == 'req_42'
        assert req.images == ['/path/a.png']
        # dataset passthrough not forwarded unless include_extra=True
        assert req.data_dict == {}

    def test_to_infer_request_include_extra(self):
        s = _make_sample(extra={'solution': '42'})
        req = s.to_infer_request(include_extra=True)
        assert req.data_dict == {'solution': '42'}

    def test_to_infer_request_image_bytes_to_base64(self):
        s = _make_sample(images=[{'bytes': b'abc'}])
        req = s.to_infer_request()
        import base64
        assert req.images == [base64.b64encode(b'abc').decode('utf-8')]


class TestGKDSample:

    def test_inherits_and_adds_fields(self):
        s = _make_sample(cls=GKDSample)
        assert isinstance(s, OnPolicySample)
        assert s.teacher_prompt is None
        assert s.teacher_messages is None

    def test_teacher_fields_assignment(self):
        s = _make_sample(cls=GKDSample, teacher_prompt='tp')
        assert s.teacher_prompt == 'tp'

    def test_to_reward_row_works_on_subclass(self):
        s = _make_sample(cls=GKDSample, extra={'solution': 'z'})
        assert s.to_reward_row()['solution'] == 'z'

    def test_pickle_roundtrip(self):
        s = _make_sample(cls=GKDSample, teacher_prompt='tp', extra={'solution': '42'})
        restored = pickle.loads(pickle.dumps(s))
        assert restored.teacher_prompt == 'tp'
        assert restored.extra == {'solution': '42'}


class TestGKDBatch:

    def test_required_and_optional_fields(self):
        gb = GKDBatch(data_source=DataSource.STUDENT)
        assert gb.data_source == DataSource.STUDENT
        assert gb.teacher_topk_logprobs is None
        gb.teacher_topk_logprobs = torch.randn(B, T, 5)
        gb.teacher_topk_indices = torch.randint(0, 100, (B, T, 5))
        assert gb.teacher_topk_logprobs.shape == (B, T, 5)
