# Copyright (c) ModelScope Contributors. All rights reserved.
"""H-UT3 ⑩ — DeepSeek-V4 thinking render parity (ms-swift 5560fc49d,
runs/H05/fixes/dsv4_thinking_render.diff; H05 Phase-1 probe is the registered oracle).

Defect: the V4 registration reused the V3.1 jinja contract (opener '<｜Assistant｜></think>',
history_thinking_prefix=''), but the official V4 renderer (model-shipped
``encoding/encoding_dsv4.py::encode_messages``) opens the think block on the assistant turn
itself and closes history turns with '``'. Both thinking ON and OFF arms diverged
on EVERY case pre-fix.

Oracle (independent): the OFFICIAL encoding module loaded DIRECTLY from the cached
deepseek-ai/DeepSeek-V4-Flash snapshot via importlib (NOT twinkle's wrapper) + the raw HF
tokenizer; token-for-token parity on TRAIN/INFER renders, plus the thinking-segment label
contract (final response reasoning trained, query/system masked; chat mode carries no think
block). CPU-only: only tokenizer/config/encoding files are read, never weights.

Reverse (single point, in-process): monkeypatch the built template instance's
``template_meta.response_opener`` back to the V3.1-style ``['<｜Assistant｜></think>']`` --
the parity oracle must then detect a token divergence (RED without the fix).

Placement: tests/v5/component/processor/test_deepseek_v4_thinking_render.py.
Unmarked L1 (CPU, real tokenizer from the shared MODELSCOPE_CACHE). Estimated runtime: ~10-30 s.
"""
import importlib.util

import pytest

DSV4_ID = 'deepseek-ai/DeepSeek-V4-Flash'


def _snapshot():
    from modelscope import snapshot_download

    return snapshot_download(DSV4_ID, allow_patterns=['*.json', '*.txt', 'encoding/*.py'])


def _official_encoding(model_dir):
    path = f'{model_dir}/encoding/encoding_dsv4.py'
    spec = importlib.util.spec_from_file_location('hut3_official_dsv4_encoding', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dev(enable_thinking):
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor

    model_dir = _snapshot()
    _, proc = get_model_processor(model_dir, load_model=False, model_type='deepseek_v4')
    cfg = TemplateConfig(template='deepseek_v4', max_length=8192, enable_thinking=enable_thinking)
    dev = build_template(cfg, proc)
    tok = getattr(proc, 'tokenizer', None) or proc
    return dev, tok, model_dir


def _cases():
    S = {'role': 'system', 'content': 'You are helpful.'}
    U1 = {'role': 'user', 'content': 'What is 2+2?'}
    A1 = {'role': 'assistant', 'content': '4', 'reasoning_content': 'Two plus two equals four.'}
    U2 = {'role': 'user', 'content': 'And 3+3?'}
    A2 = {'role': 'assistant', 'content': '6', 'reasoning_content': 'Three plus three equals six.'}
    return {
        'train_basic': ([S, U1, A1], 'TRAIN', False),
        'train_multi': ([S, U1, A1, U2, A2], 'TRAIN', False),
        'infer_multi': ([S, U1, A1, U2], 'INFER', True),
        'train_empty_rc': ([S, U1, {'role': 'assistant', 'content': '4', 'reasoning_content': ''}],
                           'TRAIN', False),
    }


def _parity(dev, tok, enc_mod, messages, intent, enable_thinking):
    from swift.dev.template import INFER, TRAIN

    thinking_mode = 'thinking' if enable_thinking else 'chat'
    ref_text = enc_mod.encode_messages(
        [dict(m) for m in messages], thinking_mode=thinking_mode,
        drop_thinking=(intent == 'INFER'), reasoning_effort='high' if enable_thinking else None)
    ref_ids = tok.encode(ref_text, add_special_tokens=False)
    enc = dev.encode({'messages': [dict(m) for m in messages]},
                     intent=INFER if intent == 'INFER' else TRAIN)
    return list(enc['input_ids']), ref_ids, enc


@pytest.mark.parametrize('enable_thinking', [True, False])
def test_dsv4_render_matches_official_encoding(enable_thinking):
    dev, tok, model_dir = _dev(enable_thinking)
    enc_mod = _official_encoding(model_dir)
    for name, (messages, intent, _) in _cases().items():
        got, ref, _ = _parity(dev, tok, enc_mod, messages, intent, enable_thinking)
        assert got == ref, f'{name} (thinking={enable_thinking}): render diverges from official encoding'

    # Label contract on train_basic.
    messages = _cases()['train_basic'][0]
    _, _, enc = _parity(dev, tok, enc_mod, messages, 'TRAIN', enable_thinking)
    ids, labels = list(enc['input_ids']), list(enc['labels'])
    trained_text = tok.decode([l for l in labels if l != -100])
    if enable_thinking:
        assert messages[-1]['reasoning_content'] in trained_text and '4' in trained_text
        assert 'What is 2+2?' not in trained_text and 'You are helpful.' not in trained_text
    else:
        assert '4' in trained_text and 'What is 2+2?' not in trained_text


def test_reverse_v31_opener_diverges_from_official(monkeypatch):
    """Single-point reverse: the pre-fix V3.1-style opener must be DETECTED by the parity
    oracle on the thinking arm (RED without the fix)."""
    dev, tok, model_dir = _dev(True)
    enc_mod = _official_encoding(model_dir)
    monkeypatch.setattr(dev.template_meta, 'response_opener', ['<｜Assistant｜></think>'])
    messages, intent, _ = _cases()['train_basic']
    got, ref, _ = _parity(dev, tok, enc_mod, messages, intent, True)
    assert got != ref, 'reverse control failed: the V3.1 opener must diverge from the official V4 render'