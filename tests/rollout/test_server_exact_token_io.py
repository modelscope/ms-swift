import asyncio
import sys
from contextlib import nullcontext
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest
import torch

from swift.infer_engine.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    RequestConfig,
    RolloutInferRequest,
    UsageInfo,
)
from swift.rl_core.data import GRPOBatch, GRPOSample
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.rollout.multi_turn import MultiTurnScheduler, RolloutScheduler
from swift.template import Template as SwiftTemplate


class PrefixTokenizer:

    def encode(self, text, add_special_tokens=False):
        assert text == '<prefix>'
        return [9, 8]

    def decode(self, token_ids, skip_special_tokens=False):
        return f'decoded:{",".join(map(str, token_ids))}'


class SeparatorTokenizer:
    mapping = {
        '<end>\n': [9, 10],
        '<pair>\n': [7, 8, 10],
    }

    def __call__(self, text, **kwargs):
        return {'input_ids': self.mapping[text]}


class PrefixTemplate:

    def _get_response_prefix(self, inputs):
        return inputs.chat_template_kwargs.get('response_prefix', '')


class ExactTokenTemplate:
    padding_free = False
    enable_thinking = None

    def encode(self, data, **kwargs):
        content = data['messages'][-1]['content']
        if isinstance(content, dict):
            response_ids = content['token_ids']
            response_mask = content['loss_scale']
        else:
            response_ids = [13]
            response_mask = [1]
        return {
            'input_ids': [100, *response_ids],
            'labels': [-100, *[token_id if mask else -100
                               for token_id, mask in zip(response_ids, response_mask)]],
        }

    def data_collator(self, encoded_data, padding_to=None):
        self.encoded_data = encoded_data
        return {
            key: torch.tensor([item[key] for item in encoded_data])
            for key in ('input_ids', 'labels')
        }


class AsyncTwoTurnEngine:

    def __init__(self, responses):
        self.responses = iter(responses)
        self.inference_messages = []

    async def infer_async(self, infer_request, request_config, **kwargs):
        self.inference_messages.append(deepcopy(infer_request.messages))
        return next(self.responses)


class ServerBoundaryScheduler(MultiTurnScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_messages = []

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        self.hook_messages.append(deepcopy(infer_request.messages))
        return {}

    def check_finished(self, infer_request, response_choice, current_turn):
        return current_turn >= 2

    def step(self, infer_request, response_choice, current_turn):
        infer_request.messages.append({'role': 'user', 'content': 'observation'})
        return {'infer_request': infer_request}


def make_response(token_ids, text, logprobs, finish_reason=None):
    choice = ChatCompletionResponseChoice(
        0,
        ChatMessage('assistant', text),
        finish_reason,
        logprobs={'content': [{'logprob': value} for value in logprobs]},
        token_ids=token_ids)
    return ChatCompletionResponse('fake-model', [choice], UsageInfo(0, len(token_ids), len(token_ids)))


def test_server_scheduler_preserves_exact_token_history():
    request = RolloutInferRequest(
        messages=[{'role': 'user', 'content': 'question'}],
        chat_template_kwargs={'response_prefix': '<prefix>'})
    engine = AsyncTwoTurnEngine([
        make_response([11, 12], 'first action', [-0.2, -0.4]),
        make_response([13], 'second action', [-0.7], finish_reason='stop'),
    ])
    scheduler = ServerBoundaryScheduler(
        infer_engine=engine, tokenizer=PrefixTokenizer(), template=PrefixTemplate())

    result = asyncio.run(scheduler.run(request, RequestConfig(n=1)))

    assert scheduler.hook_messages[0][-1] == {'role': 'assistant', 'content': 'first action'}
    assert engine.inference_messages[1][1] == {'role': 'assistant', 'content': [9, 8, 11, 12]}
    assert scheduler.hook_messages[1][1] == {
        'role': 'assistant',
        'content': 'decoded:9,8,11,12',
    }
    assert scheduler.hook_messages[1][-1] == {'role': 'assistant', 'content': 'second action'}
    assert result.response_token_ids == [[9, 8, 11, 12], [9, 8, 13]]
    assert result.response_loss_mask == [[0, 0, 1, 1], [0, 0, 1]]
    assert result.rollout_logprobs == [[-0.2, -0.4], [-0.7]]


def test_multimodal_chunk_rebuild_preserves_exact_response_tokens():
    sample = GRPOSample(
        messages=[
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'decoded response'},
        ],
        images=[object()],
        response_token_ids=[[9, 8, 11, 12]],
        response_loss_mask=[[0, 0, 1, 1]])
    batch = GRPOBatch(
        completion_mask=torch.ones((1, 5)),
        truncated_mask=torch.zeros(1),
        seq_lengths=torch.tensor([5]))
    template = ExactTokenTemplate()
    trainer = SimpleNamespace(
        is_multimodal=True,
        template=template,
        accelerator=SimpleNamespace(device=torch.device('cpu')),
        _template_context=lambda template: nullcontext())

    model_inputs, _ = GRPOTrainer.get_chunked_inputs(
        trainer, {'input_ids': torch.zeros((1, 5), dtype=torch.long)}, batch, 0, 1, origin_data=[sample])

    assert model_inputs['input_ids'][0].tolist() == [100, 9, 8, 11, 12]
    assert template.encoded_data[0]['labels'] == [-100, -100, -100, 11, 12]


def test_token_backed_response_deduplicates_template_separator_overlap():
    template = object.__new__(SwiftTemplate)
    template.processor = SeparatorTokenizer()

    assert template._remove_response_separator_overlap(
        {'token_ids': [1, 9], 'loss_scale': [1, 1]}, ['<end>\n']) == [[10]]
    assert template._remove_response_separator_overlap(
        {'token_ids': [1, 7, 8], 'loss_scale': [1, 1, 1]}, ['<pair>\n']) == [[10]]
    assert template._remove_response_separator_overlap([1, 9, 10], ['<end>\n']) == []
    no_overlap = ['<end>\n']
    assert template._remove_response_separator_overlap({'token_ids': [1, 2]}, no_overlap) is no_overlap
    assert template._remove_response_separator_overlap('text response', no_overlap) is no_overlap


class RayDriverProbeScheduler(RolloutScheduler):
    pass


@pytest.mark.parametrize('trainer_name', ['GRPOTrainer', 'GKDTrainer'])
def test_ray_driver_scheduler_preserves_deterministic_response_prefix(monkeypatch, trainer_name):
    try:
        import ray  # noqa: F401
    except ModuleNotFoundError:
        ray_module = ModuleType('ray')
        ray_module.__path__ = []
        runtime_env_module = ModuleType('ray.runtime_env')
        runtime_env_module.RuntimeEnv = type('RuntimeEnv', (), {})
        util_module = ModuleType('ray.util')
        util_module.__path__ = []
        scheduling_module = ModuleType('ray.util.scheduling_strategies')
        scheduling_module.PlacementGroupSchedulingStrategy = type('PlacementGroupSchedulingStrategy', (), {})
        ray_module.util = util_module
        monkeypatch.setitem(sys.modules, 'ray', ray_module)
        monkeypatch.setitem(sys.modules, 'ray.runtime_env', runtime_env_module)
        monkeypatch.setitem(sys.modules, 'ray.util', util_module)
        monkeypatch.setitem(sys.modules, 'ray.util.scheduling_strategies', scheduling_module)

    import swift.ray.megatron.gkd_trainer as gkd_module
    import swift.ray.megatron.grpo_trainer as grpo_module

    scheduler_name = 'ray_driver_exact_token_probe'
    monkeypatch.setitem(grpo_module.multi_turns, scheduler_name, RayDriverProbeScheduler)
    monkeypatch.setitem(gkd_module.multi_turns, scheduler_name, RayDriverProbeScheduler)

    trainer_module = grpo_module if trainer_name == 'GRPOTrainer' else gkd_module
    trainer_cls = getattr(trainer_module, trainer_name)
    tokenizer = PrefixTokenizer()
    template = PrefixTemplate()
    template.tokenizer = tokenizer

    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(multi_turn_scheduler=scheduler_name, max_turns=2, gym_env=None)
    trainer.template = template
    trainer._prepare_multi_turn()
    scheduler = trainer._multi_turn_scheduler

    assert scheduler.infer_engine is None
    assert scheduler.tokenizer is tokenizer
    assert scheduler._template is template

    request = SimpleNamespace(chat_template_kwargs={'response_prefix': '<prefix>'}, messages=[])
    response_choice = SimpleNamespace(token_ids=[11, 12])
    token_ids, loss_mask = scheduler.get_response_token_data(request, response_choice)

    assert token_ids == [9, 8, 11, 12]
    assert loss_mask == [0, 0, 1, 1]
