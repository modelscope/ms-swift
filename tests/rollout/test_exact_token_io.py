from copy import deepcopy

import pytest

from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, RequestConfig,
                                         RolloutInferRequest, RolloutOutput, UsageInfo)
from swift.rollout.agent_loop import run_multi_turn
from swift.rollout.multi_turn import MultiTurnScheduler


class PrefixTokenizer:
    """Encode the deterministic response prefix used by scheduler tests."""

    def encode(self, text, add_special_tokens=False):
        """Map the configured prefix to stable token IDs."""
        return {
            '<prefix>': [9, 8],
            '<next-prefix>': [7],
        }[text]

    def decode(self, token_ids, skip_special_tokens=False):
        """Expose exact historical IDs as text to scheduler hooks."""
        return f'decoded:{",".join(map(str, token_ids))}'


class ToolResultTokenizer(PrefixTokenizer):
    """Encode the deterministic tool observation for agentic scheduler tests."""

    def encode(self, text, add_special_tokens=False):
        """Keep the tool observation separate from model sampled tokens."""
        if text == 'Result: 3':
            return [21, 22]
        return super().encode(text, add_special_tokens=add_special_tokens)


class PrefixTemplate:
    """Resolve response prefix from per-request template arguments."""

    def _get_response_prefix(self, inputs):
        """Return the request-specific response prefix."""
        return inputs.chat_template_kwargs.get('response_prefix', '')


class Scheduler(MultiTurnScheduler):
    """Provide the abstract scheduler step for token normalization tests."""

    def step(self, infer_request, response_choice, current_turn):
        """Return an unchanged request; these tests exercise token normalization only."""
        return {'infer_request': infer_request}


def make_choice(token_ids):
    """Build an inference response choice containing exact sampled IDs."""
    return ChatCompletionResponseChoice(
        0, ChatMessage('assistant', 'sampled text'), 'stop', token_ids=token_ids)


def make_request():
    """Build a request with an explicit deterministic assistant prefix."""
    return RolloutInferRequest(
        messages=[{'role': 'user', 'content': 'question'}],
        chat_template_kwargs={'response_prefix': '<prefix>'})


class NonBijectiveTokenizer:
    """Model a tokenizer whose decode/encode round trip loses token identity."""

    def encode(self, text, add_special_tokens=False):
        """Return the canonical encoding for text shared by multiple tokenizations."""
        assert text == 'same text'
        return [13]

    def decode(self, token_ids, skip_special_tokens=False):
        """Map distinct tokenizations to the same visible text."""
        assert token_ids in ([11, 12], [13])
        return 'same text'


def test_text_round_trip_cannot_preserve_sampled_token_ids():
    """Show why decoded assistant text cannot be the training token source."""
    tokenizer = NonBijectiveTokenizer()
    sampled_ids = [11, 12]

    decoded_text = tokenizer.decode(sampled_ids, skip_special_tokens=False)
    reconstructed_ids = tokenizer.encode(decoded_text, add_special_tokens=False)

    assert decoded_text == tokenizer.decode(reconstructed_ids, skip_special_tokens=False)
    assert reconstructed_ids != sampled_ids


def test_exact_response_token_contract():
    """Define alignment among deterministic, sampled, mask, and logprob data."""
    deterministic_prefix_ids = [9, 8]
    sampled_ids = [11, 12]
    sampled_logprobs = [-0.2, -0.4]

    response_token_ids = deterministic_prefix_ids + sampled_ids
    response_loss_mask = [0] * len(deterministic_prefix_ids) + [1] * len(sampled_ids)

    assert len(response_token_ids) == len(response_loss_mask)
    assert sum(response_loss_mask) == len(sampled_ids)
    assert len(sampled_logprobs) == sum(response_loss_mask)
    assert response_token_ids[-len(sampled_ids):] == sampled_ids


def test_scheduler_adds_masked_prefix_to_sampled_ids():
    """Normalize engine sampled IDs into the full assistant token sequence."""
    scheduler = Scheduler(tokenizer=PrefixTokenizer(), template=PrefixTemplate())

    ids, mask = scheduler.get_response_token_data(make_request(), make_choice([11, 12]))

    assert ids == [9, 8, 11, 12]
    assert mask == [0, 0, 1, 1]


def test_sampled_prefix_values_are_not_mistaken_for_deterministic_prefix():
    """Preserve sampled actions even when their values equal the template prefix."""
    scheduler = Scheduler(tokenizer=PrefixTokenizer(), template=PrefixTemplate())

    ids, mask = scheduler.get_response_token_data(make_request(), make_choice([9, 8, 11]))

    assert ids == [9, 8, 9, 8, 11]
    assert mask == [0, 0, 1, 1, 1]


def test_explicit_masked_prefix_is_not_duplicated():
    """Accept a caller-provided prefix only when its zero mask proves its role."""
    scheduler = Scheduler(tokenizer=PrefixTokenizer(), template=PrefixTemplate())

    ids, mask = scheduler.get_response_token_data(
        make_request(),
        make_choice([11]),
        response_token_ids=[9, 8, 7],
        response_loss_mask=[0, 0, 0])

    assert ids == [9, 8, 7]
    assert mask == [0, 0, 0]


def test_continuation_does_not_repeat_response_prefix():
    """Treat continuation IDs as part of the current assistant message."""
    scheduler = Scheduler(tokenizer=PrefixTokenizer(), template=PrefixTemplate())

    ids, mask = scheduler.get_response_token_data(
        make_request(), make_choice([12]), is_continuation=True)

    assert ids == [12]
    assert mask == [1]


@pytest.mark.parametrize('loss_mask', ([1], [2, 1]))
def test_invalid_response_loss_mask_is_rejected(loss_mask):
    """Reject masks with a wrong length or values outside the binary contract."""
    scheduler = Scheduler(tokenizer=PrefixTokenizer(), template=PrefixTemplate())

    with pytest.raises(AssertionError):
        scheduler.get_response_token_data(
            make_request(),
            make_choice([11, 12]),
            response_token_ids=[11, 12],
            response_loss_mask=loss_mask)


def make_output(token_ids, text, logprobs, finish_reason=None):
    """Build one fake engine output for the real multi-turn driver."""
    choice = ChatCompletionResponseChoice(
        0,
        ChatMessage('assistant', text),
        finish_reason,
        logprobs={'content': [{'logprob': value} for value in logprobs]},
        token_ids=token_ids)
    response = ChatCompletionResponse(
        'fake-model', [choice], UsageInfo(0, len(token_ids), len(token_ids)))
    return RolloutOutput(response=response)


class TwoTurnScheduler(Scheduler):
    """Stop after the second response while adding a user observation."""

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        """Expose the first response to the next turn as an observation."""
        return {'done': current_turn >= 2}

    def step(self, infer_request, response_choice, current_turn):
        """Append an observation before the next model inference."""
        infer_request.messages.append({'role': 'user', 'content': 'observation'})
        return {'infer_request': infer_request}


class MutatingPrefixScheduler(TwoTurnScheduler):
    """Change the next turn's prefix in place while completing the current turn."""

    def step(self, infer_request, response_choice, current_turn):
        infer_request.chat_template_kwargs['response_prefix'] = '<next-prefix>'
        return super().step(infer_request, response_choice, current_turn)


class TokenHistoryBoundaryScheduler(TwoTurnScheduler):
    """Record the text-oriented scheduler view at each turn boundary."""

    def __init__(self, *args, **kwargs):
        """Initialize scheduler hook snapshots."""
        super().__init__(*args, **kwargs)
        self.hook_messages = []

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        """Capture scheduler-visible messages before deciding whether to stop."""
        self.hook_messages.append(deepcopy(infer_request.messages))
        return await super().on_turn_end(infer_request, response_choice, current_turn)


def test_colocate_driver_accumulates_exact_ids_masks_and_logprobs():
    """Run the real colocate driver with fake engine outputs across two turns."""
    request = make_request()
    first_output = make_output([11, 12], 'first', [-0.2, -0.4], finish_reason=None)
    second_output = make_output([13], 'second', [-0.7], finish_reason='stop')
    outputs_by_turn = iter([[second_output]])

    def rollout_fn(requests, request_config):
        """Return the next fake engine output and support distributed empty batches."""
        if not requests:
            return []
        assert requests[0].messages[-1] == {'role': 'user', 'content': 'observation'}
        return next(outputs_by_turn)

    result = run_multi_turn(
        [request],
        [first_output],
        TwoTurnScheduler(tokenizer=PrefixTokenizer(), template=PrefixTemplate()),
        rollout_fn,
        RequestConfig(n=1),
        max_turns=2)

    assert result[0].response_token_ids == [[9, 8, 11, 12], [9, 8, 13]]
    assert result[0].response_loss_mask == [[0, 0, 1, 1], [0, 0, 1]]
    assert result[0].rollout_logprobs == [[-0.2, -0.4], [-0.7]]


def test_colocate_completed_turn_uses_prefix_before_scheduler_mutation():
    """Keep each completed turn bound to the prefix active during its inference."""
    request = make_request()
    first_output = make_output([11, 12], 'first', [-0.2, -0.4], finish_reason=None)
    second_output = make_output([13], 'second', [-0.7], finish_reason='stop')

    def rollout_fn(requests, request_config):
        if not requests:
            return []
        return [second_output]

    result = run_multi_turn(
        [request],
        [first_output],
        MutatingPrefixScheduler(tokenizer=PrefixTokenizer(), template=PrefixTemplate()),
        rollout_fn,
        RequestConfig(n=1),
        max_turns=2)

    assert result[0].response_token_ids == [[9, 8, 11, 12], [7, 13]]
    assert result[0].response_loss_mask == [[0, 0, 1, 1], [0, 1]]


def test_colocate_driver_preserves_token_history_across_scheduler_boundary():
    """Keep exact IDs for inference while presenting decoded text to scheduler hooks."""
    request = make_request()
    scheduler = TokenHistoryBoundaryScheduler(tokenizer=PrefixTokenizer(), template=PrefixTemplate())
    first_output = make_output([11, 12], 'first action', [-0.2, -0.4], finish_reason=None)
    second_output = make_output([13], 'second action', [-0.7], finish_reason='stop')
    inference_messages = []

    def rollout_fn(requests, request_config):
        """Capture the exact next-turn inference history."""
        if not requests:
            return []
        inference_messages.append(deepcopy(requests[0].messages))
        return [second_output]

    result = run_multi_turn(
        [request],
        [first_output],
        scheduler,
        rollout_fn,
        RequestConfig(n=1),
        max_turns=2)

    assert scheduler.hook_messages[0][-1] == {'role': 'assistant', 'content': 'first action'}
    assert inference_messages[0][1] == {'role': 'assistant', 'content': [9, 8, 11, 12]}
    assert scheduler.hook_messages[1][1] == {
        'role': 'assistant',
        'content': 'decoded:9,8,11,12',
    }
    assert scheduler.hook_messages[1][-1] == {'role': 'assistant', 'content': 'second action'}
    assert result[0].response_token_ids == [[9, 8, 11, 12], [9, 8, 13]]
    assert result[0].response_loss_mask == [[0, 0, 1, 1], [0, 0, 1]]


class ContinuationScheduler(TwoTurnScheduler):
    """Continue generating the same assistant message on the second turn."""

    def step(self, infer_request, response_choice, current_turn):
        """Keep the assistant message last so the next response is a continuation."""
        return {'infer_request': infer_request}


class ToolCallScheduler(Scheduler):
    """Model the exact-token contract of an agentic tool-call scheduler."""

    def check_finished(self, infer_request, response_choice, current_turn):
        """Continue after a tool call and finish on the next model response."""
        return current_turn >= 2

    def step(self, infer_request, response_choice, current_turn):
        """Append a masked tool observation after sampled model tokens."""
        infer_request.messages[-1]['content'] += 'Result: 3'
        token_ids = list(response_choice.token_ids)
        tool_result_ids = self.tokenizer.encode('Result: 3', add_special_tokens=False)
        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids + tool_result_ids,
            'response_loss_mask': [1] * len(token_ids) + [0] * len(tool_result_ids),
        }


def test_tool_call_scheduler_preserves_sampled_tokens_and_masks_tool_result():
    """Keep tool observations out of the loss while preserving exact history."""
    request = make_request()
    first_output = make_output(
        [11, 12], 'Action: calculator\nAction Input: 1 + 2\n', [-0.2, -0.4], finish_reason=None)
    second_output = make_output([31, 32], 'The answer is 3', [-0.7, -0.8], finish_reason='stop')
    inference_messages = []

    def rollout_fn(requests, request_config):
        """Capture the token-backed assistant history before the final turn."""
        if not requests:
            return []
        inference_messages.append(deepcopy(requests[0].messages))
        return [second_output]

    result = run_multi_turn(
        [request],
        [first_output],
        ToolCallScheduler(tokenizer=ToolResultTokenizer(), template=PrefixTemplate()),
        rollout_fn,
        RequestConfig(n=1),
        max_turns=2)

    assert inference_messages[0][1] == {
        'role': 'assistant',
        'content': [9, 8, 11, 12, 21, 22],
    }
    assert result[0].response_token_ids == [[9, 8, 11, 12, 21, 22, 31, 32]]
    assert result[0].response_loss_mask == [[0, 0, 1, 1, 0, 0, 1, 1]]
    assert result[0].rollout_logprobs == [[-0.2, -0.4, -0.7, -0.8]]


def test_colocate_driver_merges_continuation_without_repeating_prefix():
    """Merge exact continuation data into the current assistant token turn."""
    request = make_request()
    first_output = make_output([11], 'first', [-0.2], finish_reason=None)
    second_output = make_output([12], ' continued', [-0.4], finish_reason='stop')
    outputs_by_turn = iter([[second_output]])

    def rollout_fn(requests, request_config):
        """Return the continuation output and support distributed empty batches."""
        if not requests:
            return []
        assert requests[0].messages[-1] == {'role': 'assistant', 'content': [9, 8, 11]}
        return next(outputs_by_turn)

    result = run_multi_turn(
        [request],
        [first_output],
        ContinuationScheduler(tokenizer=PrefixTokenizer(), template=PrefixTemplate()),
        rollout_fn,
        RequestConfig(n=1),
        max_turns=2)

    assert result[0].response_token_ids == [[9, 8, 11, 12]]
    assert result[0].response_loss_mask == [[0, 0, 1, 1]]
    assert result[0].rollout_logprobs == [[-0.2, -0.4]]
