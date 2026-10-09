"""GLM-5.3 template alignment against the model's own chat_template.jinja.

GLM-5.3 reuses GLM-5.2's `GlmMoeDsa` weights layout, so `model_type` and every config field except
`transformers_version` / `quantization_config` are identical. Its chat template is not: ms-swift's
`swift` backend renders these prompts by hand and never consults the jinja, so each of the
following would silently corrupt training data rather than raise.

* `<|system|>Reasoning Effort: {X}` lost GLM-5.2's `enable_thinking` gate and is now unconditional.
* `reasoning_effort` also accepts `'low'`, not just `'high'` / `'max'`.
* `clear_thinking` defaults to false, so historical `<think>` blocks survive (GLM-5.2 stripped them).
* the generation prompt is always `<|assistant|><think>` -- there is no non-thinking mode.

Every case encodes the same messages through both backends and compares token ids. The GLM-5.2
case at the end is the regression guard for `GLM5_2Template`, which GLM-5.3 subclasses.
"""
import pytest

from swift.model import MODEL_MAPPING, get_processor
from swift.template import TemplateInputs, get_template
from swift.template.templates.glm import GLM5_2Template, GLM5_3Template

GLM5_3_ID = 'ZhipuAI/GLM-5.3'
GLM5_2_ID = 'ZhipuAI/GLM-5.2'


@pytest.fixture(scope='module')
def glm5_3_processor():
    return get_processor(GLM5_3_ID)


@pytest.fixture(scope='module')
def glm5_2_processor():
    return get_processor(GLM5_2_ID)


def _encode(processor, messages, *, backend='swift', template_type, inputs=None, **kwargs):
    template = get_template(processor, template_type=template_type, template_backend=backend, **kwargs)
    payload = {'messages': [dict(m) for m in messages]}
    payload.update(inputs or {})
    encoded = template.encode(TemplateInputs(payload))
    return encoded['input_ids'], template


def _assert_backends_agree(processor, messages, template_type, inputs=None, **kwargs):
    swift_ids, template = _encode(processor, messages, template_type=template_type, inputs=inputs, **kwargs)
    jinja_ids, _ = _encode(processor, messages, backend='jinja', template_type=template_type, inputs=inputs, **kwargs)
    assert swift_ids == jinja_ids, (f'swift backend rendered {template.safe_decode(swift_ids)!r} '
                                    f'but the model chat template rendered {template.safe_decode(jinja_ids)!r}')
    return template.safe_decode(swift_ids)


def test_glm5_3_model_ids_resolve_to_the_glm5_3_template():
    """`ZhipuAI/GLM-5.3` is FP8 and `GLM-5.3-BF16` is not, but both are glm_moe_dsa + glm5_3."""
    model_meta = MODEL_MAPPING['glm_moe_dsa']
    for model_name in ['glm-5.3', 'glm-5.3-bf16']:
        group = model_meta.get_matched_model_group(model_name)
        assert group is not None, f'{model_name} did not resolve to any glm_moe_dsa group'
        assert group.template == 'glm5_3'
    # GLM-5.2 must keep its own template, and GLM-5.3-Flash is a different architecture that
    # glm_moe_dsa must not claim.
    assert model_meta.get_matched_model_group('glm-5.2').template == 'glm5_2'
    assert model_meta.get_matched_model_group('glm-5.3-flash') is None


def test_glm5_3_processor_resolves_the_template_without_being_told(glm5_3_processor):
    assert glm5_3_processor.model_info.model_type == 'glm_moe_dsa'
    template = get_template(glm5_3_processor)
    assert isinstance(template, GLM5_3Template)
    assert template.template_meta.preserve_thinking is True


def test_single_turn_inference(glm5_3_processor):
    text = _assert_backends_agree(glm5_3_processor, [{'role': 'user', 'content': 'hi'}], 'glm5_3')
    # The unconditional system prefix, and always-on thinking at the generation prompt.
    assert text == '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>'


def test_reasoning_effort_is_not_gated_on_thinking(glm5_3_processor):
    """GLM-5.2 dropped the prefix when thinking was off; GLM-5.3 renders it either way."""
    for enable_thinking in [True, False]:
        text = _assert_backends_agree(
            glm5_3_processor, [{
                'role': 'user',
                'content': 'hi'
            }], 'glm5_3', enable_thinking=enable_thinking)
        assert '<|system|>Reasoning Effort: Max' in text
        assert text.endswith('<|assistant|><think>'), text


@pytest.mark.parametrize('effort,expected', [('low', 'Low'), ('high', 'High'), ('max', 'Max')])
def test_reasoning_effort_levels(glm5_3_processor, effort, expected):
    """`low` is new in GLM-5.3; GLM-5.2's template only accepted `high` / `max`."""
    text = _assert_backends_agree(
        glm5_3_processor, [{
            'role': 'user',
            'content': 'hi'
        }],
        'glm5_3',
        inputs={'chat_template_kwargs': {
            'reasoning_effort': effort
        }})
    assert f'Reasoning Effort: {expected}' in text


def test_history_thinking_is_preserved(glm5_3_processor):
    """`clear_thinking` defaults to false, so earlier <think> blocks are kept verbatim."""
    messages = [
        {
            'role': 'user',
            'content': 'q1'
        },
        {
            'role': 'assistant',
            'content': '<think>r1</think>a1'
        },
        {
            'role': 'user',
            'content': 'q2'
        },
    ]
    text = _assert_backends_agree(glm5_3_processor, messages, 'glm5_3')
    assert '<think>r1</think>' in text
    assert text.endswith('<|assistant|><think>')


def test_assistant_turn_without_thinking_gets_an_empty_think_block(glm5_3_processor):
    """`non_thinking_prefix` exists for this case even though GLM-5.3 has no non-thinking mode."""
    messages = [
        {
            'role': 'user',
            'content': 'q1'
        },
        {
            'role': 'assistant',
            'content': 'a1'
        },
        {
            'role': 'user',
            'content': 'q2'
        },
    ]
    text = _assert_backends_agree(glm5_3_processor, messages, 'glm5_3')
    assert '<|assistant|><think></think>a1' in text


def test_explicit_system_message_keeps_the_effort_prefix_first(glm5_3_processor):
    messages = [{'role': 'system', 'content': 'You are terse.'}, {'role': 'user', 'content': 'hi'}]
    text = _assert_backends_agree(glm5_3_processor, messages, 'glm5_3')
    assert text.startswith('[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>You are terse.')


def test_training_mode(glm5_3_processor):
    messages = [{'role': 'user', 'content': 'q1'}, {'role': 'assistant', 'content': '<think>r1</think>a1'}]
    template = get_template(glm5_3_processor, template_type='glm5_3')
    template.set_mode('train')
    encoded = template.encode(TemplateInputs({'messages': [dict(m) for m in messages]}))
    # The assistant answer is supervised; the prompt and the reasoning are not.
    assert any(label != -100 for label in encoded['labels'])
    text = template.safe_decode(encoded['input_ids'])
    assert 'Reasoning Effort: Max' in text
    assert '<think>r1</think>' in text


def test_glm5_3_differs_from_glm5_2_only_where_the_chat_template_does():
    """Pin the two behaviour flags so a later edit cannot quietly revert GLM-5.3 to 5.2 semantics."""
    assert GLM5_2Template.reasoning_effort_needs_thinking is True
    assert GLM5_3Template.reasoning_effort_needs_thinking is False
    assert issubclass(GLM5_3Template, GLM5_2Template)


def test_glm5_2_still_matches_its_own_chat_template(glm5_2_processor):
    """Regression guard: GLM5_3Template subclasses GLM5_2Template, which gained a gate flag."""
    text = _assert_backends_agree(glm5_2_processor, [{'role': 'user', 'content': 'hi'}], 'glm5_2')
    # GLM-5.2 gates the prefix on enable_thinking, and ms-swift defaults that to off.
    assert 'Reasoning Effort' not in text
    assert text.endswith('<|assistant|><think></think>')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
