# Copyright (c) ModelScope Contributors. All rights reserved.
"""H-UT3 ④ — a system-only prefix renders under truncation span recovery
(ms-swift 5560fc49d, runs/H01/fixes/system_only_prefix_render.diff).

Defect: any system-prompt row whose render exceeds max_length crashed under
truncation_strategy left/right: ``project_sliced_messages`` re-encodes ``messages[:1]`` (the
system block alone) to recover span offsets; ``StdTemplateInputs`` lifts the leading system
out of ``messages``, so the prefix render reaches ``_swift_encode`` with ZERO turns --
``_replace_start_image_tags`` indexed ``messages[-1]`` (IndexError) and the encoder asserted
``len(inputs.messages) > 0`` (AssertionError).

Oracle (independent):
* the untruncated render of the SAME row (max_length=None template) -- left truncation must
  keep exactly its tail ``max_length`` tokens (the slice contract, independent of the span
  recovery under test);
* the official jinja ``tokenizer.apply_chat_template`` on the system-only message list --
  the zero-turn swift-backend render must agree token-for-token (the system block alone);
* labels stay pre-shifted (labels[i] == input_ids[i+1] on supervised positions, -100 else).

Reverse (single point, in-process): wrap ``Template._swift_encode`` to re-impose the OLD
``assert len(inputs.messages) > 0`` before delegating. The span-recovery prefix render then
raises AssertionError exactly as pre-fix -> red.

Placement: tests/v5/component/processor/test_system_only_truncation.py.
CPU-only (real tokenizer from the shared cache via TinyModel.tokenizer_dir; no model weights,
no accelerator). Unmarked L1. Estimated runtime: ~5-15 s.
"""
import pytest


def _templates():
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor
    from tests.v5.tiny import TinyModel

    tok_dir = TinyModel.tokenizer_dir()
    _, proc = get_model_processor(tok_dir, model_type='qwen2', load_model=False)
    # Default template_backend is 'swift' -- the backend whose zero-turn render was fixed.
    truncated = build_template(
        TemplateConfig(template='qwen2_5', max_length=48, truncation_strategy='left'), proc)
    full = build_template(TemplateConfig(template='qwen2_5', max_length=None), proc)
    return truncated, full, proc


def _row():
    system = ('You are a meticulous assistant. ' * 12).strip()
    return {
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': 'What is 2+2?'},
            {'role': 'assistant', 'content': 'It is 4.'},
        ]
    }, system


def test_system_only_prefix_survives_left_truncation():
    from swift.dev.template import TRAIN

    truncated, full, proc = _templates()
    row, system = _row()
    full_ids = list(full.encode(row, intent=TRAIN)['input_ids'])
    assert len(full_ids) > 48, 'fixture must exceed max_length so truncation actually slices'

    enc = truncated.encode(row, intent=TRAIN)  # pre-fix: IndexError / AssertionError here
    ids = list(enc['input_ids'])
    assert len(ids) <= 48
    assert ids == full_ids[-48:], 'left truncation must keep the tail of the untruncated render'
    labels = list(enc['labels'])
    supervised = [i for i, v in enumerate(labels) if v != -100]
    assert supervised, 'truncation must keep some supervised (assistant) positions'
    for i in supervised[:-1]:
        assert labels[i] == ids[i + 1], f'label shift broken at {i}'

    # The projected trajectory keeps a system turn whose content is a prefix of the original.
    messages = enc.get('messages')
    if messages:
        systems = [m for m in messages if m.get('role') == 'system']
        # Span recovery decodes a token slice: it may start on a leading-space token and end on
        # the template newline -- strip both edges before the prefix comparison (content contract).
        assert systems and system.startswith(systems[0]['content'].strip()), \
            f'span recovery lost the system prefix: {systems}'


def test_zero_turn_system_render_matches_official_jinja():
    from swift.dev.template import INFER

    truncated, full, proc = _templates()
    _, system = _row()
    # A system-only conversation: StdTemplateInputs lifts it, messages == [] at the encoder.
    enc = full.encode({'messages': [{'role': 'system', 'content': system}],
                       'add_generation_prompt': False}, intent=INFER)
    tok = getattr(proc, 'tokenizer', None) or proc
    ref = tok.apply_chat_template([{'role': 'system', 'content': system}],
                                  add_generation_prompt=False, tokenize=True, return_dict=False)
    assert list(enc['input_ids']) == list(ref), 'zero-turn system render diverges from official jinja'


def test_reverse_old_zero_turn_assert_crashes(monkeypatch):
    """Single-point reverse: re-impose the pre-fix ``len(messages) > 0`` guard at the encoder
    boundary. The truncation span-recovery prefix must then crash (RED without the fix)."""
    from swift.dev.template import TRAIN
    from swift.dev.template.base import Template

    truncated, full, proc = _templates()
    row, _ = _row()
    original = Template._swift_encode

    def _old_guard(self, inputs):
        assert len(inputs.messages) > 0, f'inputs.messages: {inputs.messages}'
        return original(self, inputs)

    monkeypatch.setattr(Template, '_swift_encode', _old_guard)
    with pytest.raises(AssertionError, match='inputs.messages'):
        truncated.encode(row, intent=TRAIN)