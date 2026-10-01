# Copyright (c) ModelScope Contributors. All rights reserved.
"""评测方式 (维度 6) x sampler (维度 7) x 启动方式 (维度 8) -- the in-training eval paths.

``run_sft`` scores a run two ways, chosen by ``TrainConfig.predict_with_generate`` and dispatched in
``SFTLoop.evaluate`` (``train_loop.py``):

  - **validation-loss** (``predict_with_generate=False``, the default): ``_evaluate_loss`` runs
    ``forward_only`` over the ``eval_dataloader`` that ``split_dataset_ratio`` / ``val_dataset`` carved
    out, and records ``eval_loss``. No sampler, no engine, no network -- pure scoring on the weights
    being trained.
  - **generative** (``predict_with_generate=True``): ``_evaluate_generate`` runs an EvalScope benchmark
    (``eval_dataset``) through a resident sampler the assembly built in ``_build_eval_sampler``, and
    folds the report into ``eval_history``. Two sampler backends:
      * ``transformers`` -- a facade over the LIVE training module (``TransformersSampler(model=..)``):
        no second weight copy, no sync, generation dispatched across the model's own data ranks.
      * ``vllm`` / ``sglang`` -- a co-resident engine in the trainer's 'model' DeviceGroup, weight-synced
        through a ``CheckpointEngineManager`` and bracketed by a colocate sleep/wake schedule
        (``eval_enter`` / ``eval_exit``). Ray-mode only (``validate._check_eval_generation`` refuses it
        otherwise, since a local/torchrun launch has no way to place the engine group).

Coverage map (this file adds to it, never re-runs it):

  - val-loss on the Megatron backend (``forward_only`` + ``calculate_loss``-NotImplementedError
    fallback): ``test_e2e.py::test_run_sft_megatron_evaluate_returns_metrics``.
  - val-loss on the transformers backend, single card and under torchrun (per-rank ``evaluate``):
    ``test_val_loss_eval_single_card`` / ``test_val_loss_eval_under_torchrun`` below.
  - generative + transformers-live sampler, single card and under Ray (driver dispatches generate
    across the DP group): ``test_generative_eval_transformers_sampler_single_card`` /
    ``::test_generative_eval_transformers_sampler_ray`` below.
  - generative + vllm co-resident engine under Ray (weight-sync + colocate schedule):
    ``test_generative_eval_vllm_sampler_ray`` below.

The generative-under-torchrun combination is deliberately NOT gated here: with the transformers-live
sampler it is correct but rank0-only (peers idle at the eval barrier -- ``_evaluate_generate`` warns
about the 1/N throughput), and ``validate._check_eval_generation`` REFUSES it outright under a
param-sharding strategy (FSDP / DeepSpeed ZeRO-3) because rank0's all-gather would deadlock against
peers parked at the barrier. The efficient multi-GPU generative path is Ray, which is what the Ray
tests below drive; ``sglang`` mirrors ``vllm`` but is not installed in this environment.

The generative tests score a HERMETIC ``general_qa`` benchmark pointed at a local jsonl (no benchmark
download), with Chinese rows so EvalScope's Rouge scorer takes its jieba path. EvalScope's BLEU scorer
still reaches for NLTK ``punkt_tab`` whenever the real model emits a non-Chinese token, so the tests
provision it through EvalScope's own mirror-backed ``check_nltk_data`` (the same call its Rouge path
makes) rather than assume it pre-exists.
"""
import json
import os
from contextlib import contextmanager

import pytest
import torch

from swift.dev.tests._runners import Runners

MODEL_TEXT = 'Qwen/Qwen2.5-0.5B-Instruct'

#: Chinese question/answer rows for the hermetic ``general_qa`` benchmark, doubled as the SFT train set
#: so one tiny file drives both the training steps and the eval scoring. Chinese keeps EvalScope's Rouge
#: scorer on its jieba path (offline-safe); see the module docstring for the BLEU/punkt_tab caveat.
_QA_PAIRS = [
    ('二加二等于几？', '四'),
    ('法国的首都是哪里？', '巴黎'),
    ('天空是什么颜色？', '蓝色'),
    ('水的化学式是什么？', '水的化学式是 H2O。'),
    ('一年有多少个月？', '一年有十二个月。'),
    ('太阳从哪个方向升起？', '太阳从东方升起。'),
    ('冰是什么温度融化的？', '冰在零摄氏度融化。'),
    ('猫会飞吗？', '猫不会飞。'),
]


def _write_train_jsonl(path):
    """Write the SFT train set (``{messages: [user, assistant]}`` per row)."""
    with open(path, 'w', encoding='utf-8') as f:
        for question, answer in _QA_PAIRS:
            f.write(
                json.dumps(
                    {
                        'messages': [{
                            'role': 'user',
                            'content': question
                        }, {
                            'role': 'assistant',
                            'content': answer
                        }]
                    },
                    ensure_ascii=False) + '\n')


def _write_qa_jsonl(path):
    """Write the hermetic ``general_qa`` benchmark (``{question, answer}`` per row)."""
    with open(path, 'w', encoding='utf-8') as f:
        for question, answer in _QA_PAIRS:
            f.write(json.dumps({'question': question, 'answer': answer}, ensure_ascii=False) + '\n')


def _ensure_nltk_punkt():
    """Provision NLTK ``punkt_tab`` through EvalScope's mirror-backed downloader (idempotent).

    EvalScope's ``general_qa`` BLEU scorer calls ``nltk.word_tokenize`` on any non-Chinese span the real
    model generates, which raises ``LookupError`` without ``punkt_tab``; the adapter swallows it and
    returns a ``None`` score that EvalScope then rejects. ``check_nltk_data`` is the same helper
    EvalScope's own Rouge path uses, and falls back to a ModelScope OSS mirror when nltk.org is
    unreachable -- so this is environment provisioning, not a test shortcut.
    """
    from evalscope.utils.resource_utils import check_nltk_data
    check_nltk_data('punkt_tab')


@contextmanager
def _captured_loop():
    """Capture the ``SFTLoop`` the assembly builds, so a test can read its ``eval_history`` afterwards.

    ``run_sft`` returns only the training loss history; the eval metrics live on the loop it built and
    discarded. Wrapping ``build_loop`` observes the REAL loop (the same technique ``hf_dp.py`` uses to
    read the model's mesh off a discarded assembly) rather than stubbing ``evaluate``.
    """
    import swift.dev.recipe.assembly as assembly_mod
    captured = {}
    original = assembly_mod.TrainAssembly.build_loop

    def capturing(self, *args, **kwargs):
        loop = original(self, *args, **kwargs)
        captured['loop'] = loop
        return loop

    assembly_mod.TrainAssembly.build_loop = capturing
    try:
        yield captured
    finally:
        assembly_mod.TrainAssembly.build_loop = original


def _evalscope_reports(out_dir):
    """Every EvalScope benchmark report json written under ``out_dir/eval`` (one per scored dataset)."""
    eval_root = os.path.join(out_dir, 'eval')
    found = []
    for root, _, files in os.walk(eval_root):
        for name in files:
            if name.endswith('.json') and os.sep + 'reports' + os.sep in os.path.join(root, name) + os.sep:
                found.append(os.path.join(root, name))
    return found


def _assert_trained(history):
    """The task-agnostic training post-condition shared by every eval test here."""
    assert history, 'run_sft produced no optimizer steps'
    losses = [r['loss'] for r in history]
    assert all(loss == loss and abs(loss) != float('inf') for loss in losses), f'non-finite loss: {losses}'
    return losses


# --- dimension 6 = False: the validation-loss path ------------------------------------------------


@pytest.mark.slow
@pytest.mark.accel(1)
def test_val_loss_eval_single_card(tmp_path, text_model_path):
    """``predict_with_generate=False`` scores the split-off validation set and records a finite eval_loss.

    Single card, transformers backend. ``split_dataset_ratio=0.25`` carves 2 of the 8 rows into a val
    set; ``eval_steps=1`` makes ``evaluate()`` run every optimizer step plus a final pass. The assertion
    reads ``eval_loss`` off the captured loop's ``eval_history`` -- a real number produced by
    ``_evaluate_loss``'s ``forward_only`` + ``calculate_metric`` over the eval dataloader, not merely a
    green run.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig, TunerConfig)
    from swift.dev.recipe import run_sft

    data_path = str(tmp_path / 'train.jsonl')
    out_dir = str(tmp_path / 'out')
    _write_train_jsonl(data_path)

    with _captured_loop() as captured:
        history = run_sft(
            ModelConfig(model=text_model_path, torch_dtype='bfloat16'),
            TemplateConfig(template='qwen2_5', max_length=256),
            DatasetConfig(dataset=[data_path], dataset_shuffle=False, split_dataset_ratio=0.25),
            TrainConfig(
                learning_rate=1e-4,
                lr_scheduler='constant',
                warmup_ratio=0.0,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,
                eval_strategy='steps',
                eval_steps=1,
                max_steps=2,
                max_grad_norm=1.0),
            DistributedConfig(),
            CheckpointConfig(),
            tuner_config=TunerConfig(tuner='lora'),
            output_dir=out_dir)

    losses = _assert_trained(history)
    eval_history = captured['loop'].eval_history
    assert eval_history, 'no eval ran: split_dataset_ratio/eval_steps did not wire the val-loss path'
    eval_losses = [e['eval_loss'] for e in eval_history]
    assert all(v == v and abs(v) != float('inf') for v in eval_losses), f'non-finite eval_loss: {eval_losses}'
    assert os.path.isdir(os.path.join(out_dir, 'checkpoint-final'))
    print(f'\nval-loss single: train={losses} eval={eval_losses}')


@pytest.mark.slow
@pytest.mark.accel(2)
def test_val_loss_eval_under_torchrun(tmp_path):
    """The val-loss path completes ``evaluate()`` on EVERY rank of a real 2-GPU torchrun launch.

    Under torchrun each rank drives its own loop over its shard of the eval dataloader (the
    ``DeviceMeshSampler`` split), so this is the launch mode the in-process single-card test and the
    Ray-driver test (loop runs once on the driver) structurally cannot reach. Both ranks must report a
    finite ``eval_loss``.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs')

    import sys

    data_path = str(tmp_path / 'train.jsonl')
    out_dir = str(tmp_path / 'out')
    result_path = str(tmp_path / 'result.json')
    _write_train_jsonl(data_path)

    cmd = [
        sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2', *Runners.RENDEZVOUS,
        Runners.path('hf_eval'), '--data', data_path, '--out', result_path, '--out_dir', out_dir
    ]
    proc = Runners.launch(cmd, timeout=1200)
    for rank in (0, 1):
        per_rank = f'{result_path}.rank{rank}.json'
        assert os.path.exists(per_rank), (
            f'rank {rank} produced no result. stdout tail:\n{proc.stdout[-1500:]}\nstderr tail:\n{proc.stderr[-2500:]}')
        with open(per_rank) as f:
            result = json.load(f)
        assert result['train_losses'], f'rank {rank}: no training steps'
        assert result['eval_losses'], f'rank {rank}: evaluate() never ran'
        assert all(v is not None and v == v and abs(v) != float('inf') for v in result['eval_losses']), \
            f'rank {rank}: non-finite eval_loss: {result["eval_losses"]}'
        print(f'\nval-loss torchrun rank{rank}: train={result["train_losses"]} eval={result["eval_losses"]}')


# --- dimension 6 = True: the generative (EvalScope) path ------------------------------------------


def _run_generative(tmp_path, text_model_path, *, backend, distributed_config, per_device_batch):
    """One ``predict_with_generate=True`` run over the hermetic general_qa; returns ``(history, loop, out)``."""
    from swift.dev.config import (CheckpointConfig, DatasetConfig, ModelConfig, TemplateConfig, TrainConfig,
                                  TunerConfig)
    from swift.dev.recipe import run_sft

    _ensure_nltk_punkt()
    data_path = str(tmp_path / 'train.jsonl')
    qa_path = str(tmp_path / 'qa.jsonl')
    out_dir = str(tmp_path / 'out')
    _write_train_jsonl(data_path)
    _write_qa_jsonl(qa_path)

    with _captured_loop() as captured:
        history = run_sft(
            ModelConfig(model=text_model_path, torch_dtype='bfloat16'),
            TemplateConfig(template='qwen2_5', max_length=256),
            DatasetConfig(dataset=[data_path], dataset_shuffle=False),
            TrainConfig(
                learning_rate=1e-4,
                lr_scheduler='constant',
                warmup_ratio=0.0,
                per_device_train_batch_size=per_device_batch,
                gradient_accumulation_steps=1,
                max_steps=1,
                max_grad_norm=1.0,
                predict_with_generate=True,
                eval_strategy='steps',
                eval_steps=1,
                eval_dataset=['general_qa'],
                eval_dataset_args={'general_qa': {
                    'local_path': qa_path
                }},
                eval_limit=3,
                eval_generation_config={'max_tokens': 16,
                                        'temperature': 0.0},
                eval_sampler_backend=backend),
            distributed_config,
            CheckpointConfig(),
            tuner_config=TunerConfig(tuner='lora'),
            output_dir=out_dir)
    return history, captured['loop'], out_dir


def _assert_generative_scored(history, loop, out_dir, label):
    """The generative post-condition: EvalScope really ran and wrote a benchmark report."""
    _assert_trained(history)
    eval_history = loop.eval_history
    assert eval_history, f'{label}: no generative eval ran'
    # _evaluate_generate stores the raw EvalScope summary under 'eval_report' on rank 0 / the driver.
    assert any('eval_report' in e for e in eval_history), f'{label}: eval_history carries no EvalScope report'
    reports = _evalscope_reports(out_dir)
    assert reports, f'{label}: EvalScope wrote no benchmark report under {os.path.join(out_dir, "eval")}'
    with open(reports[0]) as f:
        report = json.load(f)
    assert report, f'{label}: empty EvalScope report at {reports[0]}'
    return reports


@pytest.mark.slow
@pytest.mark.accel(1)
def test_generative_eval_transformers_sampler_single_card(tmp_path, text_model_path):
    """``predict_with_generate=True`` with the transformers-live sampler scores a benchmark on one card.

    The sampler is a facade over the live LoRA module (``TransformersSampler(model=..)``): generation runs
    in-place on the weights being trained, no second copy. This is the end-to-end gate for the whole
    generative wiring -- ``_generative_eval_kwargs`` -> ``_build_eval_sampler`` -> ``_evaluate_generate``
    -> ``run_evalscope`` -> EvalScope Native -- which builds a TaskConfig from TrainConfig. A green run
    with a real report on disk means that chain is intact.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from swift.dev.config import DistributedConfig
    history, loop, out_dir = _run_generative(
        tmp_path, text_model_path, backend='transformers', distributed_config=DistributedConfig(),
        per_device_batch=1)
    reports = _assert_generative_scored(history, loop, out_dir, 'generative/transformers/single')
    print(f'\ngenerative transformers single: reports={[os.path.basename(r) for r in reports]}')


@pytest.mark.slow
@pytest.mark.accel(2)
def test_generative_eval_transformers_sampler_ray(tmp_path, text_model_path):
    """The transformers-live generative path under Ray (mode='ray', nproc=2).

    This is the efficient multi-GPU generative layout the torchrun path cannot offer: the driver runs the
    loop once and ``TransformersModel.generate``'s ``dispatch='slice_dp'`` scatters the benchmark prompts
    across the two DP workers, so generation uses both cards rather than idling one at a barrier.
    per_device_train_batch_size >= the DP size (2) for the same slice_dp reason the training Ray test notes.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs')

    from swift.dev.config import DistributedConfig
    history, loop, out_dir = _run_generative(
        tmp_path,
        text_model_path,
        backend='transformers',
        distributed_config=DistributedConfig(mode='ray', nproc_per_node=2),
        per_device_batch=2)
    reports = _assert_generative_scored(history, loop, out_dir, 'generative/transformers/ray')
    print(f'\ngenerative transformers ray: reports={[os.path.basename(r) for r in reports]}')


@pytest.mark.slow
@pytest.mark.accel(1)
def test_generative_eval_vllm_sampler_ray(tmp_path, text_model_path):
    """``eval_sampler_backend='vllm'``: a co-resident engine, weight-synced, under Ray colocate.

    The distinct assembly this gates (``_build_eval_sampler``'s vllm branch): ``build_sampler`` places a
    vLLM engine in the trainer's 'model' DeviceGroup with ``enable_sleep_mode`` on, a
    ``CheckpointEngineManager`` syncs the trained LoRA-merged weights into it, and ``eval_enter`` /
    ``eval_exit`` run the colocate hand-over (wake engine -> sync weights -> offload trainer -> generate
    -> sleep engine -> reload trainer) around the eval. Ray-mode only; a single worker (nproc=1) keeps the
    colocate hand-over on one card. sglang would take the same branch but is not installed here.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    pytest.importorskip('vllm', reason='vllm not installed')

    from swift.dev.config import DistributedConfig
    history, loop, out_dir = _run_generative(
        tmp_path,
        text_model_path,
        backend='vllm',
        distributed_config=DistributedConfig(mode='ray', nproc_per_node=1),
        per_device_batch=1)
    reports = _assert_generative_scored(history, loop, out_dir, 'generative/vllm/ray')
    print(f'\ngenerative vllm ray: reports={[os.path.basename(r) for r in reports]}')
