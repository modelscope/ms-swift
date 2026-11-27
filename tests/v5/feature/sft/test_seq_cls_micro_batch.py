# Copyright (c) ModelScope Contributors. All rights reserved.
"""H-UT3 ⑧ — task head (seq_cls) micro-batching: every micro-batch row is unpacked, trained,
and accounted for in the loss (ms-swift 5560fc49d "recipe(supervised): task head 微批解包";
E06 train/seq_cls example, runs/H-EX/VERDICT_B.md; H09 P1b).

Existing coverage (inventory): test_train_task_head_group.py::test_seq_cls_example_smokes
(bs2 smoke, finite loss + ckpt) and test_task_types.py::test_run_seq_cls_single_label_end_to_end
(bs2, assert_trained) cover the happy path but carry NO independent loss oracle and NO reverse
control; test_recipe_micro_batch.py covers the loop plumbing on stub models. This case adds the
missing oracle + single-point reverse for the micro-batch UNPACKING seam.

Setup: seq_cls num_labels=2, per_device_train_batch_size=1, gradient_accumulation_steps=2 ->
each update submits TWO rows as TWO one-row micro-batches (no padding inside a micro-batch),
4 rows / 2 updates, fp32 for determinism, save_steps=1, SWIFT_TRAIN_TENSOR_DUMP on.

Oracles (independent):
* the tensor dump records exactly the 2 consumed rows per update (labels + lengths), and an
  observation-only spy on ``Template.batch_encode`` (the encode seam right BEFORE micro-batching,
  twinkle transformers.py _normalize_local_batch) captures the actual encoded rows: dump and
  encode agree 1:1 on label and length, and the two updates consume all four dataset rows exactly
  once (input_ids identity against an independent seq_cls INFER render -- the batch sampler picks
  the pairing and the planner may reorder inside an update, so the contract is row identity);
* update-2's logged loss equals an independent plain-HF ``Qwen2ForSequenceClassification``
  recomputation in a FRESH subprocess (immune to process-global kernel patches left by earlier
  tests, e.g. Liger rotary): load checkpoint-1 (the PRE-update-2 weights), forward each captured
  row of update-2 alone (no padding), CE against its label, mean over the two rows (fp32 band 1e-4);
* grad_norm finite >0 <1e3; the classification head ``score.weight`` Δθ nonzero between
  checkpoint-1 and checkpoint-final.

Reverse (single point, in-process): monkeypatch ``TransformersModel._build_micro_batch_plan``
to return only the FIRST micro-batch's indices. The second row is then never forwarded/backwarded:
the logged loss collapses to the surviving row's CE and diverges from the two-row independent oracle -> red.

Placement: tests/v5/feature/sft/test_seq_cls_micro_batch.py (uses ``text_model_path``).
Markers: slow + accel(1). Estimated runtime: ~60-120 s.
"""
import json
import os

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _rows(tmp_path):
    data = [
        {'messages': [{'role': 'user', 'content': 'This product is wonderful, I love it.'}], 'label': 1},
        {'messages': [{'role': 'user', 'content': 'Terrible quality, it broke immediately.'}], 'label': 0},
        {'messages': [{'role': 'user', 'content': 'Absolutely fantastic experience overall.'}], 'label': 1},
        {'messages': [{'role': 'user', 'content': 'Worst purchase I have ever made.'}], 'label': 0},
    ]
    path = tmp_path / 'seq_cls_micro.jsonl'
    with open(path, 'w') as f:
        for row in data:
            f.write(json.dumps(row) + '\n')
    return str(path), data


def _train(model_path, data_path, out_dir, dump_dir):
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, LoggingConfig,
                                  ModelConfig, TemplateConfig, TrainConfig)
    from swift.dev.recipe.sft.run_sft import run_seq_cls

    os.environ['SWIFT_TRAIN_TENSOR_DUMP'] = str(dump_dir)
    try:
        history = run_seq_cls(
            ModelConfig(model=model_path, task_type='seq_cls', torch_dtype='float32',
                        num_labels=2, problem_type='single_label_classification'),
            TemplateConfig(template='qwen2_5', max_length=128),
            DatasetConfig(dataset=[data_path], dataset_shuffle=False),
            TrainConfig(learning_rate=1e-6, per_device_train_batch_size=1,
                        gradient_accumulation_steps=2, max_steps=2),
            DistributedConfig(),
            CheckpointConfig(save_strategy='steps', save_steps=1),
            logging_config=LoggingConfig(logging_steps=1),
            output_dir=str(out_dir),
        )
    finally:
        os.environ.pop('SWIFT_TRAIN_TENSOR_DUMP', None)
    return history


def _spy_batch_encode(monkeypatch):
    """Observation-only wrap of the pre-micro-batch encode seam: record the trainer's actual
    encoded rows per update call (no behavior change)."""
    from swift.dev.template.base import Template

    captured = []
    original = Template.batch_encode

    def spy(self, trajectories, *args, **kwargs):
        out = original(self, trajectories, *args, **kwargs)
        captured.append([{'input_ids': list(r['input_ids']), 'labels': r.get('labels')} for r in out])
        return out

    monkeypatch.setattr(Template, 'batch_encode', spy)
    return captured


def _dump_records(dump_dir):
    path = os.path.join(dump_dir, 'index.rank0.jsonl')
    with open(path) as f:
        return [json.loads(line) for line in f]


def _dataset_row_ids(model_path, data):
    """Independent seq_cls render of each dataset row (INFER intent: the trainer pools the last
    position, so the render carries the assistant opener). Returns ids tuples in dataset order."""
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.dev.template import INFER
    from swift.model import get_model_processor

    _, proc = get_model_processor(model_path, model_type='qwen2', load_model=False)
    template = build_template(TemplateConfig(template='qwen2_5', max_length=128), proc, task_type='seq_cls')
    return [tuple(template.encode({'messages': d['messages']}, intent=INFER)['input_ids']) for d in data]


def _hf_rows_ce(ckpt_dir, rows):
    """Independent plain-HF seq_cls CE for each row (no padding), weights from a real checkpoint,
    recomputed in a FRESH subprocess (CPU fp32 eager).

    Process isolation is load-bearing, not cosmetic: an earlier test in the same pytest process
    may have installed process-global kernel patches -- twinkle ``kernelize`` installs dotted
    FUNCTION targets via ``setattr`` on the owning transformers module (kernel/core.py), e.g. the
    Liger rotary replaces ``modeling_qwen2.apply_rotary_pos_emb`` for the WHOLE process (class
    targets are instance-scoped, module functions are not). A same-process "plain HF" oracle would
    silently inherit those patches and crash on CPU (Triton kernels reject CPU tensors:
    ``Pointer argument cannot be accessed from Triton``). A fresh interpreter guarantees the oracle
    is the unpatched HF forward, independent of any in-process global state.
    """
    import subprocess
    import sys

    script = r'''
import json, os, sys
import torch
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForSequenceClassification
ckpt_dir, rows = json.loads(sys.stdin.read())
cfg = Qwen2Config.from_pretrained(ckpt_dir, num_labels=2)
model = Qwen2ForSequenceClassification(cfg)
state = {}
for name in sorted(os.listdir(ckpt_dir)):
    if name.endswith('.safetensors'):
        state.update(load_file(os.path.join(ckpt_dir, name)))
model.load_state_dict(state, strict=False)
model.eval().float()
ces = []
for input_ids, label in rows:
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids=ids, labels=torch.tensor([label], dtype=torch.long))
    ces.append(float(out.loss))
print(json.dumps(ces))
'''
    payload = json.dumps([str(ckpt_dir), [[r['input_ids'], r['labels']] for r in rows]])
    proc = subprocess.run([sys.executable, '-c', script], input=payload, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError(f'independent HF oracle subprocess failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}')
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _head_tensor(ckpt_dir):
    from safetensors.torch import load_file

    for name in sorted(os.listdir(ckpt_dir)):
        if name.endswith('.safetensors'):
            state = load_file(os.path.join(ckpt_dir, name))
            for key, value in state.items():
                if key.endswith('score.weight'):
                    return value
    raise AssertionError(f'no score.weight head in {ckpt_dir}')


def _updates(dump_dir):
    by_step = {}
    for rec in _dump_records(dump_dir):
        by_step.setdefault(rec['global_step'], []).append(rec)
    assert sorted(by_step) == [0, 1] or sorted(by_step) == [1, 2], f'unexpected dump steps: {sorted(by_step)}'
    return [sorted(by_step[s], key=lambda r: r['sample']) for s in sorted(by_step)]


def test_seq_cls_micro_batch_unpacking_matches_independent_oracle(tmp_path, text_model_path, monkeypatch):
    import torch

    data_path, data = _rows(tmp_path)
    out_dir = tmp_path / 'out'
    dump_dir = tmp_path / 'dump'
    captured = _spy_batch_encode(monkeypatch)
    history = _train(text_model_path, data_path, out_dir, dump_dir)

    assert len(history) == 2
    for rec in history:
        assert rec['loss'] == rec['loss'] and abs(rec['loss']) != float('inf')
        assert 0 < rec['grad_norm'] < 1e3, f'grad_norm out of range: {rec}'
    assert history[-1]['loss'] < 5.0, f'converged loss out of sane range: {history}'

    # Per-row unpacking: the encode seam saw BOTH rows of each update, and the dump recorded
    # exactly those 2 rows (label + length agree 1:1; the planner may reorder inside an update).
    updates = _updates(dump_dir)
    assert len(captured) == 2, f'expected one encode call per update, got {len(captured)}'
    ref_ids = _dataset_row_ids(text_model_path, data)
    for k, recs in enumerate(updates):
        assert len(recs) == 2, f'update {k}: micro-batch rows not unpacked 1:1 ({len(recs)} records)'
        assert len(captured[k]) == 2, f'update {k}: encode seam did not receive both rows'
        for rec, row in zip(recs, captured[k]):
            assert rec['scalars']['label'] == row['labels'], \
                f'update {k}: dump label {rec["scalars"]["label"]} != encoded label {row["labels"]}'
            tensors = torch.load(os.path.join(dump_dir, rec['file']), weights_only=True)
            assert tensors['tensors']['lengths'].tolist() == [len(row['input_ids'])], \
                f'update {k}: dump length != encoded length'
        # Each consumed row IS a dataset row (token-for-token identity against the independent
        # render); the batch sampler decides the pairing, so coverage is asserted across updates.
        for row in captured[k]:
            assert tuple(row['input_ids']) in set(ref_ids), \
                f'update {k}: consumed row is not a dataset row'
    seen = [tuple(r['input_ids']) for call in captured for r in call]
    assert len(seen) == 4 and set(seen) == set(ref_ids), \
        'the two updates must consume each of the 4 dataset rows exactly once'

    # Independent loss oracle for update 2: checkpoint-1 holds the PRE-update-2 weights.
    ckpt1 = os.path.join(str(out_dir), 'checkpoint-1')
    final = os.path.join(str(out_dir), 'checkpoint-final')
    assert os.path.isdir(ckpt1) and os.path.isdir(final)
    ces = _hf_rows_ce(ckpt1, captured[1])
    oracle_loss = sum(ces) / len(ces)
    logged = history[1]['loss']
    assert abs(logged - oracle_loss) <= 1e-4, \
        f'update-2 loss {logged} != independent two-row HF oracle {oracle_loss} (micro-batch merge broken)'

    # The head really trained.
    head1, head_final = _head_tensor(ckpt1), _head_tensor(final)
    assert not torch.equal(head1, head_final), 'seq_cls head Δθ is zero across updates'


def test_reverse_dropped_micro_batch_diverges_from_oracle(tmp_path, text_model_path, monkeypatch):
    """Single-point reverse: plan only the FIRST micro-batch. The second row is never trained;
    the logged loss must then diverge from the two-row independent oracle (RED without the seam)."""
    from twinkle.model.transformers.transformers import TransformersModel

    data_path, data = _rows(tmp_path)
    out_dir = tmp_path / 'out_rev'
    dump_dir = tmp_path / 'dump_rev'
    captured = _spy_batch_encode(monkeypatch)

    original = TransformersModel._build_micro_batch_plan

    def _first_only(self, inputs, config, optimizer_config):
        plan = original(self, inputs, config, optimizer_config)
        return plan[:1]

    monkeypatch.setattr(TransformersModel, '_build_micro_batch_plan', _first_only)
    history = _train(text_model_path, data_path, out_dir, dump_dir)

    ckpt1 = os.path.join(str(out_dir), 'checkpoint-1')
    ces = _hf_rows_ce(ckpt1, captured[1])
    oracle_loss = sum(ces) / len(ces)
    logged = history[1]['loss']
    assert abs(logged - oracle_loss) > 1e-4, \
        'reverse control failed: dropping a micro-batch must diverge from the two-row oracle'
    # Exactly ONE row's CE is what the broken run reports (the loss normalizes over the EXECUTED
    # micro-batches, so dropping one leaves the survivor's full CE); the planner may keep either
    # micro-batch, so accept either single-row CE but never the two-row mean.
    assert any(abs(logged - ce) <= 1e-4 for ce in ces), \
        f'unexpected reverse loss shape: {logged} vs {ces}'