# Copyright (c) ModelScope Contributors. All rights reserved.
"""Laya (`decision` task_type) alignment against the reference ``rl_common.py``.

The convaiinnovations/laya repo ships its own reference implementation (``rl_common.py`` at the
snapshot root), so both sides of the alignment run from the SAME snapshot:

  A  encoding  -- ``LayaPreprocessor`` renders option texts byte-identical to the reference
                  ``render_options``, and ``LayaTemplate._encode`` input_ids / ``laya_markers``
                  are byte-identical to the reference ``build_sequence`` on the REAL tokenizer
                  (covers noul/choice/score, long-state truncation, and the option-budget shrink
                  path). Tokenizer only; cheap.
  B  full fwd  -- the swift-loaded model (``LayaModelLoader``: stub-config bypass, split weight
                  load, patched post-final-norm forward) produces fp32 logits EQUAL to the
                  reference ``DecisionModel`` built from the SAME factory ``model.safetensors``.
                  Swift's eval-time temperature calibration is neutralized for the comparison
                  (the reference ``forward`` applies no temperature; predict-time code does).

Env variables (all optional; missing checkpoint -> alignment tests skip):

    LAYA_MODEL_DIR  Path to the convaiinnovations/laya SNAPSHOT ROOT (must contain rl_common.py,
                    model.safetensors, encoder/, tokenizer/). A checkpoint subfolder
                    (typed-decisions / multilingual) is NOT enough for this test.

Usage::

    LAYA_MODEL_DIR=... pytest tests/test_align/test_decision_laya_align.py
"""
import importlib.util
import json
import os
import pytest
import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from swift.dataset.preprocessor.decision import LayaPreprocessor  # noqa: E402
from swift.model import get_model_processor  # noqa: E402
from swift.model.decision_head import QUESTION_TYPES  # noqa: E402
from swift.template.template_inputs import StdTemplateInputs  # noqa: E402
from swift.template.templates.decision import LayaTemplate  # noqa: E402

MAX_LEN = 512  # root checkpoint rl_agent_config.json: max_len=512
HEAD_MAX_LEN = 192  # root checkpoint rl_agent_config.json: head_max_len=192

# One question per record (Laya scores one question per forward). Covers every kind plus the
# two truncation paths of build_sequence: state truncation and the option-budget shrink.
ALIGN_RECORDS = [
    # The README quickstart example (choice with criteria map).
    ('Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.', {
        't': 'choice',
        'ins': 'Which department should handle this?',
        'crit': {
            'billing': 'invoices, payments, refunds',
            'technical': 'bugs, outages, system errors',
            'other': 'everything else',
        },
        'y': 0
    }),
    # noul: canonical false/true criteria texts.
    ('The refund arrived.', {
        't': 'noul',
        'ins': 'Customer is satisfied?',
        'y': 1
    }),
    # score: level descriptions + a soft distribution target.
    ('Urgent complaint.', {
        't': 'score',
        'ins': 'How urgent?',
        'crit': ['low', 'high'],
        'soft': [0.25, 0.75]
    }),
    # Long state: exercises the state right-truncation budget (identical on both sides).
    ('word ' * 400, {
        't': 'noul',
        'ins': 'Does the encoder truncate the state identically?'
    }),
    # 40 long options: exercises the option-budget even-shrink path (all markers must survive).
    ('many options', {
        't': 'choice',
        'ins': 'pick one',
        'crit': {
            f'opt{i}': 'detail ' + 'x ' * 40
            for i in range(40)
        }
    }),
]


def _resolve_snapshot():
    env = os.environ.get('LAYA_MODEL_DIR')
    if env and os.path.isfile(os.path.join(env, 'rl_common.py')):
        return env
    try:
        from modelscope import snapshot_download
        snap = snapshot_download('convaiinnovations/laya')
        return snap if os.path.isfile(os.path.join(snap, 'rl_common.py')) else None
    except Exception:
        return None


def _load_reference():
    """(rl_common module, tokenizer) from the snapshot; skips when unavailable."""
    snap = _resolve_snapshot()
    if snap is None:
        pytest.skip('Laya snapshot unavailable (set LAYA_MODEL_DIR to enable alignment tests)')
    spec = importlib.util.spec_from_file_location('laya_rl_common', os.path.join(snap, 'rl_common.py'))
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(snap, 'tokenizer'))
    return snap, rl, tok


def _make_template(tok, max_length=MAX_LEN):
    tmpl = object.__new__(LayaTemplate)
    tmpl.processor = tok
    tmpl.max_length = max_length
    return tmpl


def _swift_encode_row(tmpl, state, q):
    """The full swift path for ONE laya-native record: preprocessor fan-out -> template encode."""
    rows = LayaPreprocessor().preprocess({'state': state, 'qs': [q]})
    assert len(rows) == 1, f'expected one row per single-question record, got {len(rows)}'
    encoded = tmpl._encode(StdTemplateInputs.from_dict(rows[0]))
    return rows[0], encoded


def test_laya_encoding_aligns_with_rl_common():
    """Stage A: option rendering + sequence building are byte-identical to the reference."""
    _, rl, tok = _load_reference()
    tmpl = _make_template(tok)
    for state, q in ALIGN_RECORDS:
        row, encoded = _swift_encode_row(tmpl, state, q)
        # Option texts: LayaPreprocessor vs rl_common.render_options.
        assert row['options'][0] == rl.render_options(q), f'option texts diverge for {q["t"]}: {q["ins"]!r}'
        # Sequence + markers: LayaTemplate._encode vs rl_common.build_sequence.
        ids_ref, markers_ref = rl.build_sequence(tok, state, q, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
        assert encoded['input_ids'] == ids_ref, f'input_ids diverge for {q["t"]}: {q["ins"]!r}'
        assert encoded['decision_meta']['laya_markers'][0] == markers_ref, \
            f'markers diverge for {q["t"]}: {q["ins"]!r}'
        assert len(markers_ref) == len(row['options'][0]), 'a marker was lost to truncation'


def test_laya_full_forward_aligns_with_reference():
    """Stage B: swift-loaded model logits == reference DecisionModel logits (fp32, eval)."""
    snap, rl, tok = _load_reference()
    cfg = json.load(open(os.path.join(snap, 'rl_agent_config.json')))
    assert cfg['max_len'] == MAX_LEN and cfg['head_max_len'] == HEAD_MAX_LEN, \
        'root checkpoint constants changed; update MAX_LEN / HEAD_MAX_LEN'

    # Reference side: DecisionModel (encoder + head) from the factory state dict.
    from safetensors.torch import load_file
    reference = rl.build_model(cfg, encoder_dir=os.path.join(snap, 'encoder'))
    missing, unexpected = reference.load_state_dict(load_file(os.path.join(snap, 'model.safetensors')), strict=False)
    assert not missing and not unexpected, f'reference load failed: missing={missing}, unexpected={unexpected}'
    reference.eval()

    # Swift side: the full LayaModelLoader path (stub config bypass, split load, patched forward).
    model, _ = get_model_processor(snap, model_type='laya', task_type='decision', torch_dtype=torch.float32)
    # Neutralize eval-time calibration: the reference forward applies no temperature.
    model.scoring_head.temperature = torch.ones(3)
    model.scoring_head.temperature_by_options = {}
    model.eval()

    # One batch, one question per row, from the reference encoder (Stage A proved byte-equality
    # with swift's own encoding, so this drives BOTH models through the same tensors).
    seqs = [
        rl.build_sequence(tok, state, q, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN) for state, q in ALIGN_RECORDS[:3]
    ]
    batch, kmax = len(seqs), max(len(m) for _, m in seqs)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    max_len = max(len(ids) for ids, _ in seqs)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    marker_pos = torch.zeros((batch, kmax), dtype=torch.long)
    marker_mask = torch.zeros((batch, kmax), dtype=torch.bool)
    for i, (ids, markers) in enumerate(seqs):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1
        marker_pos[i, :len(markers)] = torch.tensor(markers)
        marker_mask[i, :len(markers)] = True

    kind_of_t = {'noul': 'noul', 'choice': 'choice', 'score': 'score'}
    meta = {
        'n_options': [len(markers) for _, markers in seqs],
        'kinds': [QUESTION_TYPES[kind_of_t[q['t']]] for _, q in ALIGN_RECORDS[:3]],
        'record_index': list(range(batch)),
        'counts': [1] * batch,
        'laya_markers': [markers for _, markers in seqs],
        'seq_lens': [len(ids) for ids, _ in seqs],
        'left_padding': False,
        'max_opt': kmax,
    }
    with torch.no_grad():
        swift_out = model(input_ids=input_ids, attention_mask=attention_mask, decision_meta=meta)
        ref_logits, _ = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_pos=marker_pos,
            marker_mask=marker_mask,
            qtype=torch.tensor([rl.QTYPES[q['t']] for _, q in ALIGN_RECORDS[:3]]))
    torch.testing.assert_close(swift_out.logits, ref_logits)
    # Every question's masked-softmax over its own options also matches (the serving contract).
    for i in range(batch):
        k = meta['n_options'][i]
        torch.testing.assert_close(swift_out.logits[i, :k].softmax(-1), ref_logits[i, :k].softmax(-1))
