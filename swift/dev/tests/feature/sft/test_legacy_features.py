# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end SFT training across the *legacy* knobs dev carried over (dimension 10).

The other files in this suite vary one structural axis (task_type / framework / distribution /
optimizer). This one holds a plain causal_lm run on the shared 0.5B fixed and exercises the grab-bag of
data-, checkpoint-, and logging-side features legacy swift exposed, each driven through the real recipe
and asserted by an OBSERVABLE effect rather than a config echo -- a knob that parses but is never
consumed must not be able to pass here.

Consumed features (real end-to-end, each pinned to its consumption point):

  - **packing** -- ``DatasetConfig.packing`` auto-enables ``padding_free`` (config/process._derive_packing)
    and routes the eagerly-encoded split through ``_pack`` -> ``PackingDataset`` (builders/dataset.py:222),
    which concatenates samples into ``packing_length`` sequences. Proven by a step-count differential: the
    same rows yield FEWER optimizer steps packed than unpacked.
  - **padding_free** -- ``TemplateConfig.padding_free`` reaches ``set_processor`` (assembly.py:392) so the
    InputProcessor emits position-id/attention-mask-free packed batches; driven standalone (no packing).
  - **gradient_checkpointing** -- ``TrainConfig.gradient_checkpointing=False`` is honoured by
    ``_disable_gradient_checkpointing`` (builders/model.py:605), undoing twinkle's unconditional enable;
    both ends of the flag train (the False end is the regression guard -- it was silently ignored before).
  - **max_length + truncation_strategy** -- ``TemplateConfig.truncation_strategy`` maps to the template
    (builders/template.py:33: ``None``/``'delete'`` -> ``'raise'``). ``'left'``/``'right'`` truncate an
    over-budget row and keep training; ``'delete'`` refuses it (``MaxLengthError``) instead of silently
    dropping or truncating -- both driven on rows that genuinely exceed ``max_length``.
  - **save_total_limit** -- forwarded to ``rotate_checkpoints`` (assembly.py:432 -> twinkle base.py:34),
    which prunes all but the newest N completed checkpoints. Proven by a checkpoint-count differential.
  - **save_only_model** -- ``no_save_optim or save_only_model`` (assembly.py:427) flips ``save_optimizer``
    off, so ``_save_training_state`` is skipped and the checkpoint carries weights but NO optimizer.pt /
    trainer_state.json. Proven by a file-set differential against the default.
  - **report_to** -- ``LoggingConfig.report_to=['tensorboard']`` builds a ``SummaryWriter`` under
    ``output_dir/runs`` (recipe/tracking.py:116) and writes scalars each ``logging_steps``. Proven by the
    event file existing after a run that logs.
  - **dataset mixing + dataset_num_proc** -- ``DatasetConfig.dataset=[a, b]`` concatenates the two files
    (builders/dataset.load_dataset) and ``dataset_num_proc>1`` drives the multiprocessing encode. Proven by
    the merged step count exceeding a single file's.
  - **resume_from_checkpoint** -- the transformers save/resume closed loop: a periodic checkpoint carries
    optimizer.pt + trainer_state.json, and resuming it restores global_step + optimizer momentum so the
    continued trajectory matches an uninterrupted run. (The Megatron backend's bit-exact save/resume is
    covered in test_e2e.py::test_run_sft_megatron_local_save_resume; this is the distinct HF path.)

Cross-referenced rather than duplicated: ``split_dataset_ratio`` (carving a val split + the eval cadence it
feeds) is exercised end to end in test_eval_sampler.py::test_val_loss_eval_single_card.

Documented GAP (skipped with a precise reason, NOT stubbed with a vacuous pass -- see the test):
``neftune_noise_alpha`` parses and validates but has no consumer in the dev training path, so it cannot be
asserted honestly. Wiring (or explicitly refusing) it is product work tracked separately. (``seed`` /
``full_determinism`` and ``freeze_parameters`` / ``trainable_parameters`` were once listed here; both are
now wired -- the former covered in test_e2e.py, the latter in tests/component/optimizer/test_freeze_parameters.py.)

No mocks: every run loads real weights and drives the real recipe. All tests are ``@pytest.mark.slow``
(+ ``@pytest.mark.accel(1)``); run with
``CUDA_VISIBLE_DEVICES=<card> pytest swift/dev/tests/feature/sft/test_legacy_features.py -m slow``.
"""
import glob
import os

import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: causal_lm cross-entropy over the ~151k Qwen vocab settles near ln(vocab) ~= 11.9, so 20.0 sits above
#: a correctly mean-reduced run yet far below a sum-reduced (scales with sequence length) or diverged one.
#: Shared with test_frameworks/test_optim_tuner -- the ceiling is a property of the task, not the knob.
_CAUSAL_LM_MAX_LOSS = 20.0

#: A ``max_length`` below the encoded length of ``_long_sft_rows`` so truncation actually fires, yet large
#: enough that a 64-token window over a short-prompt/long-response row still keeps real assistant tokens
#: (hence labels) under either 'left' or 'right'.
_TRUNC_MAX_LENGTH = 64


def _tiny_sft_rows():
    """Four short plain causal_lm rows -- the legacy knob is the variable under test, not the data."""
    return [{
        'messages': [{
            'role': 'user',
            'content': q
        }, {
            'role': 'assistant',
            'content': a
        }]
    } for q, a in (('What is 2+2?', '2 + 2 equals 4.'), ('Say hello.', 'Hello! How can I help you?'),
                    ('Capital of France?', 'The capital of France is Paris.'), ('Name a color.', 'Blue is a color.'))]


def _long_sft_rows():
    """Rows whose encoded length comfortably exceeds ``_TRUNC_MAX_LENGTH``, to drive truncation.

    A short user turn plus a long assistant response (~110 tokens): under the default strategy
    ('delete' -> the template's 'raise') these refuse to encode once ``max_length`` drops below their
    length, while 'left'/'right' truncate them to the budget and keep training.
    """
    long_answer = ('The capital of France is Paris, which sits on the Seine river in the north of the '
                   'country and has been a major European city since the Middle Ages. Paris is known for '
                   'its museums, its architecture, its cuisine and its role in art, fashion and science. '
                   'The city hosts the Louvre, the Eiffel Tower and Notre-Dame cathedral, and it remains '
                   'one of the most visited destinations in the world by international travellers each year.')
    return [{
        'messages': [{
            'role': 'user',
            'content': 'Tell me about Paris.'
        }, {
            'role': 'assistant',
            'content': long_answer
        }]
    } for _ in range(4)]


def _flash_attn_available():
    """True when a FlashAttention backend is importable (padding_free / packed batches require one).

    Mirrors test_e2e's gate: twinkle's processor refuses a padding_free/packed transformers batch unless
    ``hf_config._attn_implementation`` is flash (processor/base.py:486), since SDPA/eager cannot isolate
    logical sequences once they are concatenated into one physical row.
    """
    try:
        from transformers.modeling_flash_attention_utils import is_flash_attn_available
        return is_flash_attn_available()
    except Exception:
        return False


def _run_causal_lm(model_path,
                   out_dir,
                   *,
                   train_config,
                   data_path=None,
                   dataset_config=None,
                   template_config=None,
                   checkpoint_config=None,
                   logging_config=None,
                   attn_impl=None):
    """Drive one real causal_lm run_sft on the shared 0.5B, overriding only the config under test.

    ``data_path`` is shorthand for the common single-file dataset; pass ``dataset_config`` instead when the
    dataset side is the variable (packing / mixing / num_proc). ``attn_impl`` selects the attention backend
    (padding_free / packing need ``'flash_attention_2'``). Unspecified configs fall back to the same defaults
    the sibling test files use, so a test states only what it varies.
    """
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig)
    from swift.dev.recipe import run_sft
    return run_sft(
        ModelConfig(model=model_path, task_type='causal_lm', torch_dtype='bfloat16', attn_impl=attn_impl),
        template_config or TemplateConfig(template='qwen2_5', max_length=256),
        dataset_config or DatasetConfig(dataset=[data_path], dataset_shuffle=False),
        train_config,
        DistributedConfig(),
        checkpoint_config or CheckpointConfig(),
        None,
        logging_config,
        output_dir=out_dir,
    )


def test_packing_packs_sequences_and_trains(tmp_path, text_model_path, assert_trained, write_jsonl):
    """packing concatenates samples into fewer sequences: the same rows run FEWER steps packed.

    ``_derive_packing`` turns on ``padding_free`` and derives ``packing_length`` from ``max_length``; the
    eagerly-encoded split then goes through ``_pack`` -> ``PackingDataset``, which bin-packs rows up to
    ``packing_length``. Four short rows (~25 tokens each) fit one 512-token bin, so a 1-epoch bs=1 run does
    1 packed step against 4 unpacked -- a differential that only holds if packing really merged them. A
    finite, normalized loss on the packed run means the packed batch (concatenated sequences, block-diagonal
    attention via padding_free) trained correctly rather than cross-contaminating samples.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    if not _flash_attn_available():
        pytest.skip('packing enables padding_free, which requires a FlashAttention backend (not available)')
    from swift.dev.config import DatasetConfig, TemplateConfig, TrainConfig

    data = write_jsonl(tmp_path / 'pack_sft.jsonl', _tiny_sft_rows())
    train = TrainConfig(per_device_train_batch_size=1, gradient_accumulation_steps=1, num_train_epochs=1)
    template = TemplateConfig(template='qwen2_5', max_length=512)

    base_dir = str(tmp_path / 'out_unpack')
    base = _run_causal_lm(
        text_model_path, base_dir,
        train_config=train,
        dataset_config=DatasetConfig(dataset=[data], dataset_shuffle=False, train_dataloader_shuffle=False),
        template_config=template)

    pack_dir = str(tmp_path / 'out_pack')
    packed = _run_causal_lm(
        text_model_path, pack_dir,
        train_config=train,
        dataset_config=DatasetConfig(
            dataset=[data], dataset_shuffle=False, train_dataloader_shuffle=False, packing=True),
        template_config=template,
        attn_impl='flash_attention_2')

    assert_trained(packed, pack_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)
    assert len(base) == 4, f'unpacked baseline should run 4 steps (4 rows, bs=1, 1 epoch), got {len(base)}'
    assert len(packed) < len(base), (f'packing must merge samples into fewer sequences: packed={len(packed)} '
                                     f'steps vs unpacked={len(base)} -- packing did not concatenate')


def test_padding_free_trains(tmp_path, text_model_path, assert_trained, write_jsonl):
    """padding_free standalone: the InputProcessor emits packed batches and the run still trains.

    ``padding_free`` reaches ``set_processor`` (assembly.py:392) independent of packing, switching the
    collate to the position-id/cu_seqlens representation FlashAttention varlen expects. A finite,
    normalized loss means the padding-free batch really trained (a mismatched mask/position layout would
    corrupt the loss rather than silently pass).
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    if not _flash_attn_available():
        pytest.skip('padding_free requires a FlashAttention backend (not available)')
    from swift.dev.config import TemplateConfig, TrainConfig

    data = write_jsonl(tmp_path / 'padding_free_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_padding_free')
    history = _run_causal_lm(
        text_model_path, out_dir,
        train_config=TrainConfig(
            learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
        data_path=data,
        template_config=TemplateConfig(template='qwen2_5', max_length=256, padding_free=True),
        attn_impl='flash_attention_2')
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


@pytest.mark.parametrize('gradient_checkpointing', [True, False])
def test_gradient_checkpointing_on_and_off_both_train(tmp_path, text_model_path, assert_trained, write_jsonl,
                                                      gradient_checkpointing):
    """gradient_checkpointing is honoured on BOTH ends, not silently forced on.

    twinkle's TransformersModel.__init__ enables gradient checkpointing unconditionally, so dev undoes it
    post-construction when the config says False (``_disable_gradient_checkpointing``, builders/model.py:605)
    -- and the DDP ``find_unused_parameters`` derivation (model.py:538) already assumes the flag is honoured,
    so the two must agree. Both ends train with a finite, normalized loss; the False end is the regression
    guard for the flag having been ignored.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import TrainConfig

    data = write_jsonl(tmp_path / f'gc_{gradient_checkpointing}_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / f'out_gc_{gradient_checkpointing}')
    history = _run_causal_lm(
        text_model_path, out_dir,
        train_config=TrainConfig(
            gradient_checkpointing=gradient_checkpointing, learning_rate=1e-4, per_device_train_batch_size=2,
            gradient_accumulation_steps=1, max_steps=3),
        data_path=data)
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


@pytest.mark.parametrize('strategy', ['left', 'right'])
def test_truncation_strategy_truncates_overlong_rows_and_trains(tmp_path, text_model_path, assert_trained,
                                                                write_jsonl, strategy):
    """'left'/'right' truncate a row that exceeds max_length and keep training.

    The rows in ``_long_sft_rows`` encode to ~130 tokens, well over the 64-token budget. Under the default
    strategy ('delete' -> the template's 'raise') they would refuse to encode; 'left'/'right' instead call
    ``Template._truncate`` (swift/template/base.py:1392) to the budget and train on the kept window. That a
    finite, normalized loss comes out -- rather than a MaxLengthError -- is the differential proving the
    strategy really reached the template encode.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import TemplateConfig, TrainConfig

    data = write_jsonl(tmp_path / f'trunc_{strategy}_sft.jsonl', _long_sft_rows())
    out_dir = str(tmp_path / f'out_trunc_{strategy}')
    history = _run_causal_lm(
        text_model_path, out_dir,
        train_config=TrainConfig(
            learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
        data_path=data,
        template_config=TemplateConfig(
            template='qwen2_5', max_length=_TRUNC_MAX_LENGTH, truncation_strategy=strategy))
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


def test_truncation_strategy_delete_refuses_overlong_rows(tmp_path, text_model_path, write_jsonl):
    """'delete' refuses an over-budget row loudly instead of silently dropping or truncating it.

    dev maps ``truncation_strategy='delete'`` (and the None default) to the template's ``'raise'``
    (builders/template.py:33-35), so a row longer than ``max_length`` raises ``MaxLengthError`` (a ValueError
    whose message names max_length) rather than training on a quietly truncated or substituted row. Every
    row here exceeds the 64-token budget, so the run cannot proceed -- the fail-loudly contract for the
    default strategy.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import TemplateConfig, TrainConfig

    data = write_jsonl(tmp_path / 'trunc_delete_sft.jsonl', _long_sft_rows())
    out_dir = str(tmp_path / 'out_trunc_delete')
    with pytest.raises(ValueError, match='max_length'):
        _run_causal_lm(
            text_model_path, out_dir,
            train_config=TrainConfig(per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
            data_path=data,
            template_config=TemplateConfig(
                template='qwen2_5', max_length=_TRUNC_MAX_LENGTH, truncation_strategy='delete'))


def test_save_total_limit_rotates_old_checkpoints(tmp_path, text_model_path, write_jsonl):
    """save_total_limit prunes all but the newest N checkpoints; None keeps every one.

    ``rotate_checkpoints`` (twinkle base.py:34) runs after each save and deletes ``checkpoints[:-limit]``,
    always protecting the current save. With save_steps=1 over 3 steps plus the final save, an unlimited run
    leaves checkpoint-1/2/3 + checkpoint-final, while limit=1 leaves only checkpoint-final -- a count
    differential that only holds if rotation really deleted the older dirs.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import CheckpointConfig, TrainConfig

    data = write_jsonl(tmp_path / 'rotate_sft.jsonl', _tiny_sft_rows())
    train = TrainConfig(
        learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3)

    def _count_ckpt_dirs(out_dir):
        return len([d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d)) and
                    d.startswith('checkpoint-')])

    unlimited_dir = str(tmp_path / 'out_unlimited')
    _run_causal_lm(
        text_model_path, unlimited_dir, train_config=train, data_path=data,
        checkpoint_config=CheckpointConfig(save_steps=1, save_total_limit=None))
    unlimited = _count_ckpt_dirs(unlimited_dir)

    limited_dir = str(tmp_path / 'out_limited')
    _run_causal_lm(
        text_model_path, limited_dir, train_config=train, data_path=data,
        checkpoint_config=CheckpointConfig(save_steps=1, save_total_limit=1))
    limited = _count_ckpt_dirs(limited_dir)

    assert unlimited >= 4, f'unlimited run should keep checkpoint-1/2/3 + final, found {unlimited}'
    assert limited == 1, f'save_total_limit=1 must retain only the newest checkpoint, found {limited}'
    assert os.path.isdir(os.path.join(limited_dir, 'checkpoint-final')), 'the surviving checkpoint must be final'


def test_save_only_model_omits_optimizer_state(tmp_path, text_model_path, write_jsonl):
    """save_only_model writes weights but no optimizer/scheduler/trainer state; the default writes both.

    ``no_save_optim or save_only_model`` (assembly.py:427) flips ``save_optimizer`` off, so
    ``TransformersModel.save`` skips ``_save_training_state`` (transformers.py:1607) and the checkpoint-final
    carries the .safetensors weights + tokenizer + args.json but NO optimizer.pt / trainer_state.json. The
    default run is the control: it must contain optimizer.pt, so the absence under save_only_model is proven
    meaningful rather than the optimizer never having been written.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import CheckpointConfig, TrainConfig

    data = write_jsonl(tmp_path / 'save_only_sft.jsonl', _tiny_sft_rows())
    train = TrainConfig(
        learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=2)

    default_dir = str(tmp_path / 'out_default')
    _run_causal_lm(text_model_path, default_dir, train_config=train, data_path=data,
                   checkpoint_config=CheckpointConfig(save_only_model=False))
    default_files = set(os.listdir(os.path.join(default_dir, 'checkpoint-final')))

    only_model_dir = str(tmp_path / 'out_only_model')
    _run_causal_lm(text_model_path, only_model_dir, train_config=train, data_path=data,
                   checkpoint_config=CheckpointConfig(save_only_model=True))
    only_model_files = set(os.listdir(os.path.join(only_model_dir, 'checkpoint-final')))

    assert any(f.endswith('.safetensors') for f in only_model_files), \
        f'save_only_model checkpoint must still carry weights: {sorted(only_model_files)}'
    assert 'optimizer.pt' in default_files, \
        f'control (save_only_model=False) should write optimizer.pt, found {sorted(default_files)}'
    assert 'optimizer.pt' not in only_model_files, \
        f'save_only_model=True must omit optimizer state, found {sorted(only_model_files)}'
    assert 'trainer_state.json' not in only_model_files, \
        f'save_only_model=True must omit trainer_state.json, found {sorted(only_model_files)}'


def test_report_to_tensorboard_writes_event_file(tmp_path, text_model_path, assert_trained, write_jsonl):
    """report_to=['tensorboard'] builds a SummaryWriter under output_dir/runs and writes scalars.

    ``RunTracker._setup_tensorboard`` (recipe/tracking.py:110) resolves the log dir to
    ``tensorboard_dir or logging_dir or output_dir/runs`` and ``log`` calls ``add_scalar`` on every step that
    ``should_log`` (logging_steps=1 -> every step). The event file existing after the run is the observable
    that the tracker really wired and flushed; run_sft passes no LoggingConfig by default, so this also pins
    that an explicit one is honoured.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        pytest.skip('torch.utils.tensorboard.SummaryWriter unavailable; report_to=tensorboard cannot be exercised.')
    from swift.dev.config import LoggingConfig, TrainConfig

    data = write_jsonl(tmp_path / 'tb_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_tensorboard')
    history = _run_causal_lm(
        text_model_path, out_dir,
        train_config=TrainConfig(
            learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
        data_path=data,
        logging_config=LoggingConfig(report_to=['tensorboard'], logging_steps=1))
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)
    events = glob.glob(os.path.join(out_dir, 'runs', 'events.out.tfevents.*'))
    assert events, f'report_to=tensorboard wrote no event file under {out_dir}/runs: {os.listdir(out_dir)}'


def test_dataset_mixing_and_num_proc_train(tmp_path, text_model_path, assert_trained, write_jsonl):
    """dataset=[a, b] concatenates the two files and dataset_num_proc=2 drives the multiprocessing encode.

    ``load_dataset`` merges the two paths into one train split, so a 1-epoch bs=2 run over 4+4 rows does 4
    optimizer steps -- double the 2 a single file would give -- which only holds if the files really
    concatenated. ``dataset_num_proc=2`` exercises the multi-worker encode path (builders/dataset.py:203);
    a finite, normalized loss means it produced the same trainable rows the single-process path would.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import DatasetConfig, TrainConfig

    data_a = write_jsonl(tmp_path / 'mix_a.jsonl', _tiny_sft_rows())
    data_b = write_jsonl(tmp_path / 'mix_b.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_mixed')
    history = _run_causal_lm(
        text_model_path, out_dir,
        train_config=TrainConfig(
            learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1,
            num_train_epochs=1),
        dataset_config=DatasetConfig(
            dataset=[data_a, data_b], dataset_shuffle=False, train_dataloader_shuffle=False, dataset_num_proc=2))
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)
    assert len(history) == 4, (f'mixing two 4-row files at bs=2 over 1 epoch should run 4 steps (a single file '
                               f'would run 2), got {len(history)} -- the files did not concatenate')


def test_resume_from_checkpoint_continues_trajectory(tmp_path, text_model_path, write_jsonl):
    """transformers save + resume: global_step and optimizer momentum restore, so the loss continues.

    Phase A trains 2 steps (save_steps=2) and writes checkpoint-2 with optimizer.pt + trainer_state.json.
    Phase B resumes it with max_steps=4: ``build_model`` redirects the model id to the checkpoint (full-param),
    ``resume_from_checkpoint`` restores optimizer/scheduler/RNG/cur_step, and ``loop.resume`` seeds global_step
    -- so B's first record is step 3, not step 1, and its losses continue A's rather than restarting. An
    uninterrupted max_steps=4 run is the control: B's overlapping steps must match it, the only end-to-end
    signal that the optimizer momentum round-tripped to its SAVED values (a zero-reload would offset the
    first resumed step by O(0.1)). lr=1e-6 keeps the trajectory smooth so the diff is attributable.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import CheckpointConfig, DatasetConfig, TrainConfig

    data = write_jsonl(tmp_path / 'resume_sft.jsonl', _tiny_sft_rows())
    dataset_config = DatasetConfig(dataset=[data], dataset_shuffle=False, train_dataloader_shuffle=False)
    lr = 1e-6

    def _train(out_dir, max_steps, *, save_steps=None, resume_from=None):
        # save_strategy='no' is rejected by validate (the twinkle loop only implements 'steps'), so the
        # control / resume runs keep the default 'steps' with a save_steps far beyond max_steps: no periodic
        # checkpoint is written, only checkpoint-final.
        far = 10**6
        if resume_from:
            ckpt = CheckpointConfig(resume_from_checkpoint=resume_from, save_steps=far)
        elif save_steps:
            ckpt = CheckpointConfig(save_steps=save_steps)
        else:
            ckpt = CheckpointConfig(save_steps=far)
        return _run_causal_lm(
            text_model_path, out_dir,
            train_config=TrainConfig(
                learning_rate=lr, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=max_steps),
            dataset_config=dataset_config,
            checkpoint_config=ckpt)

    # Phase A: 2 steps, saving checkpoint-2 with full optimizer/trainer state.
    a_dir = str(tmp_path / 'out_a')
    _train(a_dir, max_steps=2, save_steps=2)
    ckpt2 = os.path.join(a_dir, 'checkpoint-2')
    assert os.path.isfile(os.path.join(ckpt2, 'trainer_state.json')), 'phase A did not write a resumable checkpoint'
    assert os.path.isfile(os.path.join(ckpt2, 'optimizer.pt')), 'phase A checkpoint lacks optimizer state'

    # Control: 4 uninterrupted steps.
    cont = _train(str(tmp_path / 'out_cont'), max_steps=4)
    # Phase B: resume checkpoint-2, train to step 4 (2 more steps).
    resumed = _train(str(tmp_path / 'out_b'), max_steps=4, resume_from=ckpt2)

    assert resumed, 'resumed run produced no steps'
    assert resumed[0]['step'] == 3, (f'resume must continue at global_step 3 (loop.resume seeds it from '
                                     f'cur_step), got {resumed[0]["step"]} -- progress was not restored')
    overlap = [(cont[2 + i]['loss'], resumed[i]['loss']) for i in range(len(resumed)) if 2 + i < len(cont)]
    assert overlap, f'no overlapping steps to compare: cont={cont} resumed={resumed}'
    # Uniform tolerance, deliberately NOT tighter at i==0 (an earlier version had that backwards). Two regimes:
    #   i==0 (step-3 loss) is a forward pass on the restored weights: both runs enter it with the SAME weights
    #     and the SAME Adam moments, so momentum cannot show up here -- only bf16 forward nondeterminism can
    #     (observed up to ~6e-3). A weight-restoration bug would blow this up by O(1), far past the limit.
    #   i>=1 (step-4 loss) is a forward on weights produced by step-3's UPDATE, which consumes the restored
    #     momentum -- THIS is where a zero-moment reload diverges, by O(0.1). The limit sits ~5x below that
    #     signal and ~3x above the bf16 noise, so it catches the real bug without flaking on nondeterminism.
    limit = 2e-2
    for i, (c, r) in enumerate(overlap):
        diff = abs(c - r)
        assert diff < limit, (f'resumed step {i} loss={r:.6f} diverges from continuous={c:.6f} by {diff:.3e} '
                              f'>= {limit} -- the resumed trajectory left the continuous one (weights or '
                              f'optimizer momentum not restored to their saved values)')


# --- documented GAPS: parse + validate but no consumer, so they cannot be asserted honestly. -----------
# Each skips with the precise missing-consumer evidence rather than passing vacuously (a test that sets the
# knob and asserts "it trained" would go green whether or not the knob does anything). Wiring or explicitly
# refusing these is product work tracked separately; see the plan's gap list.


def test_neftune_noise_alpha_is_an_unwired_gap():
    """GAP: neftune_noise_alpha parses and validates but nothing in dev/twinkle consumes it."""
    pytest.skip(
        'GAP: TrainConfig.neftune_noise_alpha has no consumer -- grep finds only the field definition, the '
        'validate._HF_ONLY ledger entry and docs; no code injects NEFTune noise into the embedding (dev drives '
        'training through twinkle, not the HF Trainer that owns activate_neftune). It parses but silently does '
        'nothing on the transformers backend. NOTE the _HF_ONLY entry is itself misleading: under megatron it '
        'raises "only implemented by the transformers backend", but transformers does not implement it either -- '
        'so the honest fix is to WIRE it or fail-loudly refuse it on BOTH backends. Wiring is product work, '
        'tracked separately; not stubbed with a vacuous "it trained" test.')
