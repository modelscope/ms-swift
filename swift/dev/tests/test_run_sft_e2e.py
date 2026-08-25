"""End-to-end SFT happy-path integration test (the last piece of "minimal refactor works").

Unlike the component-level bit tests (configure_loss / resume trajectory / GA equivalence),
this drives the FULL L4 assembly: run_sft(ModelConfig, TemplateConfig, DatasetConfig,
TrainConfig) once, from atomic Configs, with no manual model/template/dataloader wiring — the
green-path that was previously only hand-verified in a throwaway script.

Asserts: optimizer steps ran, loss is finite + normalized (readable per-token, not raw sum),
and a self-describing checkpoint (weights + args.json) is written so `swift infer` can load it.

Marked slow: loads a real Qwen2.5-0.5B. Run with --runslow.
"""
import json
import os
import pytest
import torch

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

# Rendezvous ports, unique to THIS pytest process. The offsets are still hand-assigned per launch
# below (a lingering rendezvous from an earlier phase of the SAME run must not be joinable); only the
# base moves. Fixed bases made two concurrent invocations of this file share a rendezvous, which
# surfaces as a bogus training failure rather than as a port conflict. Kept out of the ephemeral
# range (32768+) so the OS cannot hand the same port to something else mid-run.
_PORT_BASE = 20000 + (os.getpid() % 500) * 20


def _master_port(offset: int) -> int:
    return _PORT_BASE + offset


class _FakeMeta:

    def __init__(self, model_type):
        self.model_type = model_type


class _FakeProcessor:

    def __init__(self, model_type='qwen2_5'):
        self.model_meta = _FakeMeta(model_type)


def test_write_ckpt_args_json_includes_force_load_keys(tmp_path):
    """args.json must carry the force_load keys infer applies unconditionally (task_type,
    tuner_type) so seq_cls / LoRA checkpoints are not silently downgraded to causal_lm/full.

    Mirrors the read side load_args_from_ckpt (base_args.py:246-301): task_type and tuner_type
    are force_load_keys; omitting them leaves infer on its default. Pure function, no model.
    """
    from swift.dev.config import ModelConfig, TemplateConfig, TunerConfig
    from swift.dev.recipe.run_sft import _write_ckpt_args_json

    ckpt = str(tmp_path)
    _write_ckpt_args_json(
        ckpt,
        _FakeProcessor('qwen2_5'),
        ModelConfig(model='/m', task_type='seq_cls', torch_dtype='bfloat16', attn_impl='flash_attn'),
        TemplateConfig(template='qwen2_5', system='sys', truncation_strategy='left'),
        TunerConfig(tuner_type='lora'),
    )
    with open(os.path.join(ckpt, 'args.json')) as f:
        args = json.load(f)
    # force_load keys present (would otherwise silently degrade infer)
    assert args['task_type'] == 'seq_cls'
    assert args['tuner_type'] == 'lora'
    # load_keys dev knows are carried too
    assert args['model_type'] == 'qwen2_5'
    assert args['attn_impl'] == 'flash_attn'
    assert args['truncation_strategy'] == 'left'


def test_write_ckpt_args_json_omits_none_full_param(tmp_path):
    """Full-param causal_lm run: task_type/tuner_type stay None -> filtered out (infer's own
    defaults stand), so we don't write misleading keys."""
    from swift.dev.config import ModelConfig, TemplateConfig
    from swift.dev.recipe.run_sft import _write_ckpt_args_json

    ckpt = str(tmp_path)
    _write_ckpt_args_json(
        ckpt,
        _FakeProcessor('qwen2_5'),
        ModelConfig(model='/m', torch_dtype='bfloat16'),
        TemplateConfig(template='qwen2_5'),
        None,
    )
    with open(os.path.join(ckpt, 'args.json')) as f:
        args = json.load(f)
    assert 'task_type' not in args
    assert 'tuner_type' not in args
    assert args['model_type'] == 'qwen2_5'


def test_initialize_twinkle_sets_a_device_group_on_every_backend():
    """A device group must exist before any model is built -- it is what keeps a DeviceMesh alive.

    remote_class filters every DeviceMesh out of a model's kwargs while twinkle holds no device group
    (infra/__init__.py:548-552), and twinkle.initialize is what sets that group. A run that skips it
    therefore trains with device_mesh=None, which is not cosmetic: calculate_loss then divides its
    token count by 1 instead of the dp world size, so DDP's averaging yields an avg-of-avg instead of
    legacy's global token weighting, and no metric is ever gathered. One GPU cannot see any of that,
    which is exactly why the transformers backend -- whose run_sft path used to return BEFORE
    initialize -- shipped that way until a 4-GPU comparison. Hence this cheap guard on the decision
    itself. The Ray branch is left to the slow tests: asserting it here would boot a Ray cluster.
    """
    import twinkle.infra as infra

    from swift.dev.config import DistributedConfig
    from swift.dev.recipe.run_sft import _initialize_twinkle

    saved = (infra._mode, infra._device_group, infra._device_mesh)
    try:
        # mode defaults to 'local' and the transformers backend has no Ray path at all, so a default
        # DistributedConfig (what every HF caller passes) must still init.
        for dist_config in (DistributedConfig(), DistributedConfig(mode='local'),
                            DistributedConfig(backend='megatron', mode='local', nproc_per_node=1)):
            infra._mode, infra._device_group, infra._device_mesh = None, None, None
            _initialize_twinkle(dist_config)
            assert infra._device_group is not None, f'no device group for {dist_config}'
            assert infra._device_mesh is not None, f'no device mesh for {dist_config}'
            assert infra._mode == 'local', f'unexpected mode {infra._mode} for {dist_config}'
    finally:
        infra._mode, infra._device_group, infra._device_mesh = saved


def _write_toy_dataset(path):
    rows = [
        {
            'messages': [{
                'role': 'user',
                'content': 'What is 2+2?'
            }, {
                'role': 'assistant',
                'content': '2 + 2 equals 4.'
            }]
        },
        {
            'messages': [{
                'role': 'user',
                'content': 'Say hello.'
            }, {
                'role': 'assistant',
                'content': 'Hello! How can I help you today?'
            }]
        },
        {
            'messages': [{
                'role': 'user',
                'content': 'Capital of France?'
            }, {
                'role': 'assistant',
                'content': 'The capital of France is Paris.'
            }]
        },
        {
            'messages': [{
                'role': 'user',
                'content': 'Name a color.'
            }, {
                'role': 'assistant',
                'content': 'Blue is a color.'
            }]
        },
    ]
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _write_identical_dataset(path, n=4):
    """Write n IDENTICAL rows so every DP rank is fed the same sequence (isolates construction
    correctness from the two backends' differing dataloader sharding)."""
    row = {
        'messages': [{
            'role': 'user',
            'content': 'What is 2+2?'
        }, {
            'role': 'assistant',
            'content': '2 + 2 equals 4.'
        }]
    }
    with open(path, 'w') as f:
        for _ in range(n):
            f.write(json.dumps(row) + '\n')


@pytest.mark.slow
def test_run_sft_end_to_end_happy_path(tmp_path):
    """Full run_sft green-path: 4 atomic Configs -> trained checkpoint, from a real 0.5B model."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from modelscope import snapshot_download

    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig)
    from swift.dev.recipe import run_sft

    model_path = snapshot_download(MODEL)
    data_path = str(tmp_path / 'toy_sft.jsonl')
    out_dir = str(tmp_path / 'out')
    _write_toy_dataset(data_path)

    history = run_sft(
        ModelConfig(model=model_path, torch_dtype='bfloat16'),
        TemplateConfig(template='qwen2_5', max_length=512),
        DatasetConfig(dataset=[data_path], dataset_shuffle=True, data_seed=42),
        TrainConfig(
            learning_rate=1e-4,
            optim='adamw',
            lr_scheduler_type='cosine',
            warmup_ratio=0.0,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=4,
            max_grad_norm=1.0),
        DistributedConfig(),
        CheckpointConfig(),
        output_dir=out_dir,
    )

    assert history, 'run_sft produced no optimizer steps'
    losses = [r['loss'] for r in history]
    assert all(loss == loss and abs(loss) != float('inf') for loss in losses), \
        f'non-finite loss: {losses}'
    assert losses[0] < 20, f'first loss {losses[0]:.2f} too large -> not normalized (raw sum?)'

    ckpt = os.path.join(out_dir, 'checkpoint-final')
    assert os.path.isdir(ckpt), f'no checkpoint dir at {ckpt}'
    files = set(os.listdir(ckpt))
    assert any(f.endswith('.safetensors') or f == 'model.safetensors' for f in files), \
        f'no model weights in checkpoint: {sorted(files)}'
    assert 'args.json' in files, f'no args.json (checkpoint not self-describing): {sorted(files)}'
    with open(os.path.join(ckpt, 'args.json')) as f:
        args = json.load(f)
    for k in ('model_type', 'template', 'swift_version'):
        assert k in args, f'args.json missing {k!r} (swift infer would be ambiguous): {sorted(args)}'


@pytest.mark.slow
def test_run_sft_zero_opt_steps_raises(tmp_path):
    """Contract 5 (fail-fast): when N micro-batches <= ga the GA gate never fires, so run_sft
    would train forward/backward but NEVER update the model (0 optimizer steps). run_sft must
    raise ValueError rather than silently run a no-op "green" training.

    Setup: 2-row dataset, per_device_train_batch_size=2 -> 1 micro-batch/epoch; ga=2, 1 epoch ->
    num_optimizer_steps(1, 2) == 0 -> raise. The check runs BEFORE build_model, so this is cheap
    (no weight loading); it only needs the processor + a dataset load, hence gated on the model.
    """
    from modelscope import snapshot_download

    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig)
    from swift.dev.recipe import run_sft

    model_path = snapshot_download(MODEL)
    data_path = str(tmp_path / 'tiny_sft.jsonl')
    # exactly 2 rows -> with batch_size=2 that is a single micro-batch (one GA window)
    with open(data_path, 'w') as f:
        for _ in range(2):
            f.write(
                json.dumps({
                    'messages': [{
                        'role': 'user',
                        'content': 'What is 2+2?'
                    }, {
                        'role': 'assistant',
                        'content': '2 + 2 equals 4.'
                    }]
                }) + '\n')

    with pytest.raises(ValueError, match='computed 0 optimizer steps'):
        run_sft(
            ModelConfig(model=model_path, torch_dtype='bfloat16'),
            TemplateConfig(template='qwen2_5', max_length=512),
            DatasetConfig(dataset=[data_path], dataset_shuffle=False, data_seed=42),
            TrainConfig(
                per_device_train_batch_size=2, gradient_accumulation_steps=2, num_train_epochs=1.0, max_steps=None),
            DistributedConfig(),
            CheckpointConfig(),
            output_dir=str(tmp_path / 'out'),
        )


@pytest.mark.slow
@pytest.mark.parametrize('bridge_backend', ['mcore-bridge', 'megatron-bridge'])
def test_run_sft_megatron_end_to_end(tmp_path, bridge_backend):
    """Full run_sft on the Megatron backend (DP=2, Ray), for BOTH bridge backends.

    Smoke-level e2e: loss finite/normalized + a loadable HF checkpoint written, for each bridge.
    Cross-bridge numeric equality is gated by test_run_sft_megatron_two_bridges_bit_identical; the
    dev-vs-legacy numeric gate is test_run_sft_megatron_vs_legacy_loss.

    Scope: gradient_accumulation_steps=1 (larger GA is gated by test_run_sft_megatron_ga_equivalence).
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')
    # Only the megatron-bridge parametrization needs the megatron.bridge package; the mcore-bridge
    # instance must NOT be skipped just because megatron.bridge is absent in this env.
    try:
        if bridge_backend == 'megatron-bridge':
            from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # noqa: F401
    except Exception:
        pytest.skip('megatron.bridge (Megatron-Bridge) not importable in this env')

    data_path = str(tmp_path / 'toy_sft.jsonl')
    _write_toy_dataset(data_path)
    history = _run_megatron_sft(bridge_backend, data_path, str(tmp_path / f'out_{bridge_backend}'))

    assert history, f'run_sft (megatron/{bridge_backend}) produced no optimizer steps'
    losses = [r['loss'] for r in history]
    assert all(loss == loss and abs(loss) != float('inf') for loss in losses), \
        f'non-finite loss: {losses}'
    assert losses[0] < 20, f'first loss {losses[0]:.2f} too large -> not normalized (raw sum?)'

    ckpt = os.path.join(str(tmp_path / f'out_{bridge_backend}'), 'checkpoint-final')
    assert os.path.isdir(ckpt), f'no checkpoint dir at {ckpt}'
    files = set(os.listdir(ckpt))
    assert any(f.endswith('.safetensors') for f in files), \
        f'no model weights in checkpoint: {sorted(files)}'
    print(f'\nrun_sft megatron/{bridge_backend}: steps={len(history)} losses={losses}')


@pytest.mark.slow
def test_run_sft_megatron_two_bridges_bit_identical(tmp_path):
    """The two bridge backends, driven through the SAME run_sft assembly on the SAME
    model/data/dtype/hparams, must produce a bit-identical loss trajectory.

    Both build the same mcore GPTModel (only the construction library differs), so any per-step
    loss divergence means a real construction/normalization mismatch between the bridges. Each
    backend runs sequentially in its OWN twinkle Ray session (run_sft wraps initialize/shutdown).
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')
    try:
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # noqa: F401
    except Exception:
        pytest.skip('megatron.bridge (Megatron-Bridge) not importable in this env')

    data_path = str(tmp_path / 'toy_sft.jsonl')
    _write_toy_dataset(data_path)

    mcore = _run_megatron_sft('mcore-bridge', data_path, str(tmp_path / 'out_mcore'))
    mbridge = _run_megatron_sft('megatron-bridge', data_path, str(tmp_path / 'out_mbridge'))

    mcore_losses = [r['loss'] for r in mcore]
    mbridge_losses = [r['loss'] for r in mbridge]
    print(f'\ntwo-bridge SFT: mcore={mcore_losses} megatron-bridge={mbridge_losses}')
    assert len(mcore_losses) == len(mbridge_losses) and mcore_losses, \
        f'step count mismatch: mcore={mcore_losses} mbridge={mbridge_losses}'
    assert mcore_losses == mbridge_losses, (f'two bridges diverged (construction/normalization mismatch): '
                                            f'mcore={mcore_losses} megatron-bridge={mbridge_losses}')


@pytest.mark.slow
def test_run_sft_megatron_ga_equivalence(tmp_path):
    """GA>1 gate: Megatron ga=2/bs=2 must match ga=1/bs=4 (same 4 samples per optimizer step).

    Megatron accumulates gradients across the microbatch LIST inside one forward_backward, so
    grouping ga=2 dataloader batches of 2 into one 4-microbatch step must be gradient-equivalent to
    a single ga=1 step over a batch of 4. At step 1 both see identical initial weights over the
    identical first 4 samples (dataset_shuffle=False), so the step-1 loss is bit-identical; the
    grad_norm equality is the real accumulation check (it reflects the summed/normalized gradient).
    Runs mcore-bridge only; max_steps=1 keeps the data window aligned without a larger dataset.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')

    data_path = str(tmp_path / 'toy_sft.jsonl')
    _write_toy_dataset(data_path)  # 4 rows == one optimizer step at effective batch 4

    ga_hist = _run_megatron_sft(
        'mcore-bridge', data_path, str(tmp_path / 'out_ga2'), per_device_batch=2, ga=2, max_steps=1)
    big_hist = _run_megatron_sft(
        'mcore-bridge', data_path, str(tmp_path / 'out_bs4'), per_device_batch=4, ga=1, max_steps=1)

    ga_loss = [r['loss'] for r in ga_hist]
    big_loss = [r['loss'] for r in big_hist]
    ga_gn = [r.get('grad_norm') for r in ga_hist]
    big_gn = [r.get('grad_norm') for r in big_hist]
    print(f'\nGA equiv: ga2xbs2 loss={ga_loss} gn={ga_gn} | bs4 loss={big_loss} gn={big_gn}')
    assert ga_loss and big_loss and len(ga_loss) == len(big_loss) == 1, \
        f'expected 1 optimizer step each: ga2={ga_loss} bs4={big_loss}'
    assert ga_loss == big_loss, (f'GA>1 diverged from equivalent large-batch loss: ga2xbs2={ga_loss} bs4={big_loss}')
    if ga_gn[0] is not None and big_gn[0] is not None:
        assert abs(ga_gn[0] - big_gn[0]) < 1e-4, (
            f'GA>1 grad_norm diverged (accumulation mismatch): ga2xbs2={ga_gn} bs4={big_gn}')


def _run_megatron_sft(bridge_backend: str,
                      data_path: str,
                      out_dir: str,
                      *,
                      per_device_batch: int = 2,
                      ga: int = 1,
                      max_steps: int = 2,
                      eval_steps: int = 0,
                      split_dataset_ratio: float = 0.0):
    """Run one Megatron run_sft (DP=2, fp32) and return its loss/grad_norm history.

    per_device_batch/ga are parametrized so a GA>1 run can be compared against a GA=1 run with a
    proportionally larger batch (same effective samples per optimizer step -> same trajectory).
    eval_steps>0 (with split_dataset_ratio>0 to carve a val split) exercises SFTLoop.evaluate on
    the Megatron backend -- its forward_only + calculate_loss(NotImplementedError) fallback path.
    """
    from modelscope import snapshot_download

    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig)
    from swift.dev.recipe import run_sft

    model_path = snapshot_download(MODEL)
    return run_sft(
        # fp32 (float32) so the loss is comparable across bridges without a wide bf16 band.
        ModelConfig(model=model_path, torch_dtype='float32'),
        TemplateConfig(template='qwen2_5', max_length=512),
        DatasetConfig(dataset=[data_path], dataset_shuffle=False, split_dataset_ratio=split_dataset_ratio),
        # per_device_train_batch_size must be >= dp_size (2): twinkle slice_dp splits each driver
        # batch across the 2 DP ranks, so a batch of 1 would starve a rank. eval batch likewise >=2.
        # optim is deliberately not set: it is a transformers-only knob (rejected by the _HF_ONLY
        # guard on the Megatron backend, which always uses Adam).
        TrainConfig(
            learning_rate=1e-5,
            lr_scheduler_type='constant',
            warmup_ratio=0.0,
            per_device_train_batch_size=per_device_batch,
            per_device_eval_batch_size=2,
            eval_steps=(eval_steps or None),
            gradient_accumulation_steps=ga,
            max_steps=max_steps,
            max_grad_norm=1.0),
        # mode='ray' is explicit: this test relies on the Ray driver scatter (slice_dp splits each
        # driver batch across the 2 DP ranks -- see the per_device_train_batch_size note above), and
        # DistributedConfig.mode defaults to 'local', where slice_dp no-ops and the dataloader shards
        # instead.
        DistributedConfig(backend='megatron', bridge_backend=bridge_backend, nproc_per_node=2, mode='ray'),
        CheckpointConfig(),
        output_dir=out_dir,
    )


@pytest.mark.slow
def test_run_sft_megatron_evaluate_returns_metrics(tmp_path):
    """SFTLoop.evaluate() must work on the Megatron backend (contract 9): eval goes through
    forward_only, and Megatron's calculate_loss raises NotImplementedError (pipeline scheduler
    fuses loss into forward), which evaluate() swallows because the loss is already populated.

    This guards that fallback end-to-end: a run_sft with eval_steps>0 + a val split drives
    evaluate() during training. If the NotImplementedError weren't caught, evaluate() -- and thus
    run_sft -- would raise; a green run is the guard. mcore-bridge only (backend-agnostic eval path;
    the two-bridge equality is gated elsewhere).
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')

    # 8 rows so a 0.25 val split yields a non-empty eval set (>=2 for DP=2 eval batch).
    data_path = str(tmp_path / 'toy_sft.jsonl')
    _write_identical_dataset(data_path, n=8)
    # eval_steps=1 -> evaluate() runs every optimizer step (and a final eval), exercising the
    # Megatron forward_only + calculate_loss-NotImplementedError fallback repeatedly.
    history = _run_megatron_sft(
        'mcore-bridge',
        data_path,
        str(tmp_path / 'out'),
        per_device_batch=2,
        max_steps=2,
        eval_steps=1,
        split_dataset_ratio=0.25)
    # A green run means evaluate() completed (no uncaught NotImplementedError from calculate_loss).
    assert history, 'run_sft (megatron eval) produced no optimizer steps'
    losses = [r['loss'] for r in history]
    assert all(loss == loss and abs(loss) != float('inf') for loss in losses), \
        f'non-finite loss: {losses}'


def _run_torchrun(cmd, *, timeout=1800):
    """Run a torchrun command with a timeout that actually fires; return (stdout, stderr).

    ``subprocess.run(capture_output=True, timeout=...)`` is not enough here: it kills torchrun
    itself, but the worker grandchildren inherit the pipes, so communicate() keeps blocking for EOF
    and the timeout never takes effect (one such run hung for 12 hours instead of 30 minutes).
    Giving torchrun its own process group and killing the whole group is what makes it bounded.
    """
    import signal
    import subprocess

    proc = subprocess.Popen(
        cmd, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        return proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, err = proc.communicate()
        raise AssertionError(f'torchrun exceeded {timeout}s and was killed: {" ".join(cmd)}\n'
                             f'stdout tail:\n{out[-2000:]}\nstderr tail:\n{err[-3000:]}')


def _run_backend_subprocess(backend, data_path, out_dir, result_path, steps, lr):
    """Launch _megatron_sft_runner.py for ONE backend as a fresh torchrun subprocess.

    Each backend runs in its OWN process because a legacy MegatronTrainer and dev-Megatron cannot
    coexist in one process (the first leaves Megatron/TE global state -- parallel_state, TE backend,
    RNG -- that NaN-poisons the second). Returns the per-step loss list read back from result_path.
    """
    import sys

    runner = os.path.join(os.path.dirname(__file__), '_megatron_sft_runner.py')
    # Distinct port per backend so a lingering rendezvous from a previous backend cannot be joined.
    port_offset = {'dev': 0, 'legacy': 1, 'dev_cli': 2}[backend]
    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--nproc_per_node=2',
        f'--master_port={_master_port(port_offset)}',
        runner,
        '--backend',
        backend,
        '--data',
        data_path,
        '--out',
        result_path,
        '--out_dir',
        out_dir,
        '--steps',
        str(steps),
        '--lr',
        str(lr),
    ]
    out, err = _run_torchrun(cmd)
    if not os.path.exists(result_path):
        raise AssertionError(f'{backend} runner produced no result file. stdout tail:\n{out[-2000:]}\n'
                             f'stderr tail:\n{err[-3000:]}')
    with open(result_path) as f:
        return json.load(f)['losses']


@pytest.mark.slow
def test_run_sft_megatron_vs_legacy_loss(tmp_path):
    """dev-Megatron SFT must track legacy megatron_sft_main step-for-step (same model/data/hparams).

    dev (run_sft, mode='local') and legacy (megatron_sft_main) both build the same mcore GPTModel
    and train the same bf16 0.5B at bs=1/ga=1, each in its OWN torchrun subprocess (they cannot share
    a process -- see _run_backend_subprocess). bf16 is megatron's production dtype, so this compares
    the path real users run (dev-Megatron's fp32 construction is separately gated by the two-bridge
    bit test).

    Data: every row is IDENTICAL -- and this is load-bearing. dev and legacy reduce the reported loss
    across DP DIFFERENTLY: dev is avg-of-avg (each rank self-normalizes loss/count, then op=AVG across
    DP; megatron.py:527/550); legacy is token-weighted global mean (op=SUM on [sum,count] then divide;
    trainer.py loss_func). These are equal ONLY when every DP rank has the same token count. With
    DISTINCT rows the ranks get different-length samples and the two reductions diverge for a pure
    LOGGING-convention reason (the 63% gap first observed), NOT a construction bug. Identical rows
    force equal per-rank token counts so the two reductions coincide by construction, leaving a step-1
    mismatch attributable only to a real construction/normalization difference. (Training gradients
    match regardless; only the logged scalar's DP reduction differs.)

    Precision policy (bands set from OBSERVED bf16 noise, not guessed):
      - step-0 loss (no optimizer update yet) must match tightly (rel < 1e-2): pure forward
        agreement, the load-bearing construction/normalization gate. Observed identical-data rel
        ~3e-3; 1e-2 sits just above that noise floor and still catches a ~1% drift.
      - later steps: relative tolerance is meaningless here because identical repeated data overfits
        to near-zero loss within 3 steps (observed dev [0.20, 0.04, 4e-4]); a ~0.01 absolute bf16 gap
        then reads as ~20% relative. So later steps assert TRAJECTORY agreement instead: both backends
        strictly decreasing + small ABSOLUTE per-step gap.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')

    data_path = str(tmp_path / 'toy_sft.jsonl')
    _write_identical_dataset(data_path, n=4)
    steps, lr = 3, 1e-5

    dev_losses = _run_backend_subprocess('dev', data_path, str(tmp_path / 'dev_out'), str(tmp_path / 'dev.json'), steps,
                                         lr)
    legacy_losses = _run_backend_subprocess('legacy', data_path, str(tmp_path / 'legacy_out'),
                                            str(tmp_path / 'legacy.json'), steps, lr)

    print(f'\nmegatron dev-vs-legacy: dev={dev_losses} legacy={legacy_losses}')
    assert dev_losses and legacy_losses, f'empty losses: dev={dev_losses} legacy={legacy_losses}'
    assert all(loss == loss for loss in dev_losses + legacy_losses), \
        f'NaN loss: dev={dev_losses} legacy={legacy_losses}'
    n = min(len(dev_losses), len(legacy_losses))
    assert n >= 1, f'no comparable steps: dev={dev_losses} legacy={legacy_losses}'

    rel0 = abs(dev_losses[0] - legacy_losses[0]) / max(abs(legacy_losses[0]), 1e-8)
    assert rel0 < 1e-2, (f'step-0 dev-vs-legacy construction mismatch: dev={dev_losses[0]:.6f} '
                         f'legacy={legacy_losses[0]:.6f} rel={rel0:.2e} (limit 1e-2)')

    for i in range(1, n):
        assert dev_losses[i] < dev_losses[i - 1], f'dev loss not decreasing: {dev_losses}'
        assert legacy_losses[i] < legacy_losses[i - 1], f'legacy loss not decreasing: {legacy_losses}'
        gap = abs(dev_losses[i] - legacy_losses[i])
        assert gap < 0.05, (f'step {i} dev-vs-legacy absolute loss gap too large: dev={dev_losses[i]:.6f} '
                            f'legacy={legacy_losses[i]:.6f} gap={gap:.4f} (limit 0.05)')


@pytest.mark.slow
def test_megatron_cli_vs_legacy_loss(tmp_path):
    """The dev Megatron CLI must train identically to legacy given the SAME argv.

    Sibling of test_run_sft_megatron_vs_legacy_loss, one layer up: that test drives dev by building
    Configs directly, so it validates the Megatron training path. This one drives dev through
    ``swift.dev.cli.megatron.megatron_sft_main`` on legacy's own ``MegatronSftArguments`` parser, so
    it validates the argv -> Config MAPPING as well -- the renames (lr / train_iters /
    micro_batch_size / adam_eps / lr_decay_style), the gradient_accumulation_steps derivation from
    global_batch_size, and the dev-only backend/mode/nproc_per_node the entry point must set itself.

    Why this is the mapping's real gate: a missed rename does not raise, it silently trains at a dev
    default (e.g. lr falls back to 1e-5), which the fast tests catch only for the fields they name.
    Here any such fallback shows up as a step-0 loss divergence, because both sides parsed the same
    flags. The two dev backends stay separable on purpose: if only dev_cli moves, the mapping is at
    fault; if dev moves too, the Megatron path is.

    Data/precision policy is inherited verbatim from the sibling test -- identical rows to force
    equal per-DP-rank token counts (dev logs avg-of-avg, legacy token-weighted; they coincide only
    then), step-0 rel < 1e-2 as the construction gate, later steps by trajectory + absolute gap.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')

    data_path = str(tmp_path / 'toy_sft.jsonl')
    _write_identical_dataset(data_path, n=4)
    steps, lr = 3, 1e-5

    cli_losses = _run_backend_subprocess('dev_cli', data_path, str(tmp_path / 'cli_out'), str(tmp_path / 'cli.json'),
                                         steps, lr)
    legacy_losses = _run_backend_subprocess('legacy', data_path, str(tmp_path / 'legacy_out'),
                                            str(tmp_path / 'legacy.json'), steps, lr)

    print(f'\nmegatron CLI-vs-legacy: cli={cli_losses} legacy={legacy_losses}')
    assert cli_losses and legacy_losses, f'empty losses: cli={cli_losses} legacy={legacy_losses}'
    assert all(loss == loss for loss in cli_losses + legacy_losses), \
        f'NaN loss: cli={cli_losses} legacy={legacy_losses}'
    n = min(len(cli_losses), len(legacy_losses))
    assert n >= 1, f'no comparable steps: cli={cli_losses} legacy={legacy_losses}'

    rel0 = abs(cli_losses[0] - legacy_losses[0]) / max(abs(legacy_losses[0]), 1e-8)
    assert rel0 < 1e-2, (f'step-0 CLI-vs-legacy mismatch (argv mapped to different hyperparameters?): '
                         f'cli={cli_losses[0]:.6f} legacy={legacy_losses[0]:.6f} rel={rel0:.2e} (limit 1e-2)')

    for i in range(1, n):
        assert cli_losses[i] < cli_losses[i - 1], f'cli loss not decreasing: {cli_losses}'
        assert legacy_losses[i] < legacy_losses[i - 1], \
            f'legacy loss not decreasing: {legacy_losses}'
        gap = abs(cli_losses[i] - legacy_losses[i])
        assert gap < 0.05, (f'step {i} CLI-vs-legacy absolute loss gap too large: cli={cli_losses[i]:.6f} '
                            f'legacy={legacy_losses[i]:.6f} gap={gap:.4f} (limit 0.05)')


def _run_dense_backend(backend, out_dir, result_path, steps, *, model=None, dataset=None):
    """Launch _megatron_dense_runner.py for ONE backend (dense.sh-shaped config) via torchrun.

    Separate from _run_backend_subprocess because this runner takes a different flag set and returns
    an lr trajectory as well; same isolation reason though (legacy and dev-Megatron cannot share a
    process). Returns (losses, lrs).
    """
    import sys

    runner = os.path.join(os.path.dirname(__file__), '_megatron_dense_runner.py')
    port = _master_port(3 + (1 if backend == 'legacy' else 0))
    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--nproc_per_node=2',
        f'--master_port={port}',
        runner,
        '--backend',
        backend,
        '--out',
        result_path,
        '--out_dir',
        out_dir,
        '--steps',
        str(steps),
    ]
    if model:
        cmd += ['--model', model]
    if dataset:
        cmd += ['--dataset'] + list(dataset)
    out, err = _run_torchrun(cmd, timeout=3600)
    if not os.path.exists(result_path):
        raise AssertionError(f'{backend} dense runner produced no result file. stdout tail:\n{out[-2000:]}\n'
                             f'stderr tail:\n{err[-3000:]}')
    with open(result_path) as f:
        data = json.load(f)
    return data['losses'], data['lrs']


@pytest.mark.slow
def test_megatron_dense_cli_vs_legacy_50_steps(tmp_path):
    """dense.sh-shaped 50-step run: lr must match legacy BIT-FOR-BIT, loss within an observed band.

    Config mirrors examples/megatron/lora/dense.sh (tp=2 + sequence_parallel, micro=global=16, LoRA,
    warmup_fraction + cosine + min_lr, recompute full/uniform/1) -- i.e. every argv-mapping branch at
    once, which is what a minimal bs=1/tp=1 comparison cannot reach. Both sides parse the SAME argv.

    Judgement, split by what is physically achievable (bands from OBSERVED values, never guessed):
      - lr, all 50 steps: EXACT equality. lr is a closed-form function of the schedule, so it carries
        no bf16 noise; this is the one bit-level dev-vs-legacy gate. It found two real bugs (warmup
        budget rounded to an int; min_lr not reaching the optimizer, so cosine decayed past the floor
        to 0) -- both now also pinned CPU-only in test_megatron_lr_schedule.py.
      - loss: banded, NOT bit-level, but compared STEP-FOR-STEP across all 50 steps. The band is
        widest at step 0 and narrows after -- the opposite of drift:
            step 0     0.5B abs 0.015 (rel 8.0e-3) / 7B abs 0.094 (rel 6.0e-2)
            steps 1-3  0.5B abs <=0.017          / 7B abs <=0.21 (worst point, step 2)
            steps 4+   0.5B abs <=0.037          / 7B abs <=0.017
        Scale-dependent and front-loaded, which is the signature of per-layer bf16 accumulation in
        the forward, not of a schedule or data-order bug: dataloader_num_workers=0 leaves step 0
        unchanged and the lr sequences are bit-identical.

        This test previously needed a shift-aware comparison past the first epoch boundary, because
        dev trained the trailing partial batch while legacy dropped it (500 samples at
        global_batch_size 16 = 31.25 steps/epoch), putting dev one step behind from step 32 on.
        That was a real defect, now fixed: builders/dataset.py::_drop_last returns True on the
        Megatron backend, matching legacy's train sampler
        (MegatronPretrainingRandomSampler, which has no drop_last knob and discards the remainder
        unconditionally via last_batch_size/active_total_samples --
        swift/megatron/trainers/batch_sampler.py:100,114).
        The fix is confirmed in both directions -- aligned max fell 0.4356 -> 0.0368 while the
        one-step-shifted comparison went from better-than-aligned (0.001-0.03) to far worse (0.68),
        i.e. the offset is genuinely gone rather than merely smaller.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (tp=2 in the dense.sh config)')

    steps = 50
    cli_losses, cli_lrs = _run_dense_backend('dev_cli', str(tmp_path / 'cli_out'), str(tmp_path / 'cli.json'), steps)
    legacy_losses, legacy_lrs = _run_dense_backend('legacy', str(tmp_path / 'legacy_out'),
                                                   str(tmp_path / 'legacy.json'), steps)

    print(f'\ndense cli  lr[:3]={cli_lrs[:3]} loss[:3]={cli_losses[:3]}')
    print(f'dense leg  lr[:3]={legacy_lrs[:3]} loss[:3]={legacy_losses[:3]}')

    # --- lr: bit-level ---
    assert len(cli_lrs) == len(legacy_lrs) == steps, \
        f'step count mismatch: cli={len(cli_lrs)} legacy={len(legacy_lrs)}'
    lr_mismatch = [(i, cli_lrs[i], legacy_lrs[i]) for i in range(steps) if cli_lrs[i] != legacy_lrs[i]]
    assert not lr_mismatch, (f'lr differs from legacy at {len(lr_mismatch)}/{steps} steps (the schedule mapping is '
                             f'wrong, not a precision issue); first 3: {lr_mismatch[:3]}')

    # --- loss ---
    # Absolute gaps, not relative: loss dips toward ~0.1 in places, where a numerically tiny gap
    # reads as a huge percentage.
    n = min(len(cli_losses), len(legacy_losses))
    assert all(x == x for x in cli_losses + legacy_losses), \
        f'NaN loss: cli={cli_losses} legacy={legacy_losses}'

    gaps = [abs(cli_losses[i] - legacy_losses[i]) for i in range(n)]
    for i in range(min(4, n)):
        assert gaps[i] < 0.30, (f'step {i} (forward-dominated) absolute loss gap {gaps[i]:.4f} exceeds 0.30 '
                                f'(cli={cli_losses[i]:.6f} legacy={legacy_losses[i]:.6f})')
    for i in range(4, n):
        assert gaps[i] < 0.08, (f'step {i} absolute loss gap {gaps[i]:.4f} exceeds 0.08 '
                                f'(cli={cli_losses[i]:.6f} legacy={legacy_losses[i]:.6f})')

    mean_gap = sum(gaps) / n
    assert mean_gap < 0.02, (
        f'mean absolute loss gap over {n} steps is {mean_gap:.4f} (limit 0.02) -- the trajectories '
        f'are drifting apart, not just jittering')

    # A step offset would make the shifted comparison the better one; assert it is NOT, so a
    # reintroduced partial-batch mismatch cannot hide inside the bands above.
    if n > 5:
        shifted = [abs(cli_losses[i] - legacy_losses[i - 1]) for i in range(1, n)]
        assert sum(shifted) / len(shifted) > mean_gap, (
            'the one-step-shifted comparison fits better than the aligned one -- dev and legacy are '
            'off by a step again (check drop_last / partial-batch handling)')

    assert max(cli_losses) < 20 and max(legacy_losses) < 20, 'a side diverged'


def _run_resume_phase(phase, out_dir, result_path, *, resume_from=None, steps=3):
    """Launch _megatron_resume_runner.py for ONE phase as a fresh torchrun subprocess (mode='local').

    Each phase runs in its own clean process (no shared Megatron/TE global state). Returns the
    result dict {phase, losses, weights_path, optim_dir} read back from result_path.
    """
    import sys

    runner = os.path.join(os.path.dirname(__file__), '_megatron_resume_runner.py')
    ports = {'A': _master_port(5), 'B': _master_port(6), 'cont': _master_port(7), 'resume_cont': _master_port(8)}
    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--nproc_per_node=2',
        f'--master_port={ports[phase]}',
        runner,
        '--phase',
        phase,
        '--out',
        result_path,
        '--out_dir',
        out_dir,
        '--steps',
        str(steps),
    ]
    if resume_from is not None:
        cmd += ['--resume_from', resume_from]
    out, err = _run_torchrun(cmd)
    if not os.path.exists(result_path):
        raise AssertionError(f'resume phase {phase} produced no result file. stdout tail:\n{out[-2000:]}\n'
                             f'stderr tail:\n{err[-3000:]}')
    with open(result_path) as f:
        return json.load(f)


@pytest.mark.slow
def test_run_sft_megatron_local_save_resume(tmp_path):
    """Megatron mode='local' (torchrun) save + resume closed loop, 3-layer criteria.

    Validates that a mode='local' Megatron checkpoint is not only written but restores correctly.
    (An earlier "local save fails" report was traced to /tmp disk exhaustion -- each fp32 ckpt with
    Adam optimizer state is ~7GB, and 4 accumulated ckpts fill a 30GB /tmp; the save code itself is
    sound. This test guards the restore, which "save didn't crash" does NOT prove: load_state_dict(
    strict=False) at twinkle megatron.py:1204/1209 silently tolerates missing/misplaced shards.)

    Three layers, each catching a distinct failure mode:
      1. MAIN (weights bit-exact, load-bearing): phase A trains from seed 0 and saves; phase B inits
         from a DIFFERENT seed (999), resumes A, trains 0 steps, saves. A complete resume overwrites
         the seed-999 init entirely, so B's weights must equal A's max|diff|==0. Any non-zero diff
         means strict=False silently dropped/misplaced a weight shard.
      2. SECONDARY (optimizer existence): A and B optimizer dist-ckpts must have the same file set
         (both DP shards + metadata). Catches whole-shard loss/misplacement. NOTE: optimizer momentum
         ROUND-TRIP NUMERIC accuracy is NOT directly gated here -- a 0-step resume has no cheap direct
         criterion and the last bits carry known upstream mcore noise (~1.6e-3/step full-param); numeric
         fidelity is covered end-to-end by layer 3 instead.
      3. TRAJECTORY (must, not optional): interrupt-resume vs continuous training. cont trains 2N
         steps straight; resume_cont resumes A (already N steps) and trains N more. Their overlapping
         steps must match -- the ONLY end-to-end signal that optimizer momentum was restored to the
         SAVED values (not silently reloaded as zero, which layers 1-2 cannot see). Bands pinned from a
         SMOOTH-trajectory observation (lr=1e-6 so loss decreases monotonically; a larger lr diverges
         and makes the diff chaotic/unattributable):
           - step right after resume (optimizer-most-sensitive): abs diff < 1e-3 (observed 0.0000;
             a zero/misloaded momentum would offset this step by O(0.1)).
           - later overlapping steps: abs diff < 1e-2 (observed monotonic accumulation up to 4.9e-3
             from full-param momentum round-trip; still an order of magnitude below O(0.1)).
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs (twinkle MegatronModel requires world_size>=2)')

    from safetensors.torch import load_file
    steps = 3
    out = str(tmp_path / 'rz')

    # --- layers 1+2: seed-differentiated weight bit-exact + optimizer existence ---
    a = _run_resume_phase('A', out, str(tmp_path / 'A.json'), steps=steps)
    b = _run_resume_phase('B', out, str(tmp_path / 'B.json'), resume_from=os.path.join(out, 'ckpt-A'), steps=steps)

    wa, wb = load_file(a['weights_path']), load_file(b['weights_path'])
    common = [k for k in wa if k in wb]
    assert common and not [k for k in wa if k not in wb], \
        f'weight key mismatch: A-only={[k for k in wa if k not in wb][:4]}'
    max_diff = max((wa[k].float() - wb[k].float()).abs().max().item() for k in common)
    print(f'\nresume MAIN: weights max|diff|={max_diff:.3e} over {len(common)} params')
    assert max_diff == 0.0, (f'resume did NOT fully restore weights (strict=False silently dropped a shard?): '
                             f'seed-999 phase B != seed-0 phase A, max|diff|={max_diff:.3e} over {len(common)} params')

    a_optim = set(os.listdir(a['optim_dir']))
    b_optim = set(os.listdir(b['optim_dir']))
    print(f'resume SECONDARY: optim files A={sorted(a_optim)} B={sorted(b_optim)}')
    assert a_optim == b_optim and any(f.endswith('.distcp') for f in a_optim), \
        f'optimizer dist-ckpt file set differs / no shards: A={sorted(a_optim)} B={sorted(b_optim)}'

    # --- layer 3 (must): interrupt-resume vs continuous trajectory ---
    cont = _run_resume_phase('cont', out, str(tmp_path / 'cont.json'), steps=steps)['losses']
    # free cont's ckpt before resume_cont writes its own (each fp32+optim ckpt is ~7GB; /tmp is 30GB)
    import shutil
    shutil.rmtree(os.path.join(out, 'ckpt-cont'), ignore_errors=True)
    rc = _run_resume_phase(
        'resume_cont', out, str(tmp_path / 'rc.json'), resume_from=os.path.join(out, 'ckpt-A'), steps=steps)['losses']

    # cont = [s1..s2N]; resume_cont = [s(N+1)..s2N]. Overlap starts at cont[steps].
    overlap = [(cont[steps + i], rc[i]) for i in range(len(rc)) if steps + i < len(cont)]
    print(f'\nresume TRAJECTORY: cont={cont} resume_cont={rc} overlap={overlap}')
    assert overlap, f'no overlapping steps: cont={cont} resume_cont={rc}'
    for i, (c, r) in enumerate(overlap):
        diff = abs(c - r)
        limit = 1e-3 if i == 0 else 1e-2  # i==0 is the resume-immediate step (momentum-most-sensitive)
        assert diff < limit, (f'resume trajectory step {i} (cont={c:.6f} vs resume_cont={r:.6f}) diff={diff:.3e} '
                              f'>= {limit} -> optimizer momentum not restored to saved values (loaded as zero?)')

    # Release the remaining checkpoints (ckpt-A, ckpt-B, ckpt-resume_cont -- ~7GB each). Unlike the
    # other Megatron runners, this cleanup cannot live inside the runner: ckpt-A is phase B's and
    # resume_cont's INPUT, so it must survive until every phase has run. Placed after the assertions
    # on purpose -- if one fails, the checkpoints are the evidence and are worth keeping. pytest only
    # retains the last few basetemp dirs, so a failed run's leftovers still get reclaimed eventually.
    shutil.rmtree(out, ignore_errors=True)


def _write_imbalanced_dataset(path):
    """Two samples differing in BOTH token count and per-token loss -- both differences are needed.

    A token-weighted global mean and an avg-of-avg coincide when every rank holds the same number of
    tokens (identical rows are used elsewhere in this file precisely to make them coincide), and they
    coincide again when the per-token losses match. So the two conventions can only be told apart by a
    short fluent answer against a long implausible one: that separates them by far more than bf16
    noise, while equal-length or equally-surprising rows would let the broken path pass.
    """
    rows = [
        {
            'messages': [{
                'role': 'user',
                'content': 'What is 2+2?'
            }, {
                'role': 'assistant',
                'content': '4.'
            }]
        },
        {
            'messages': [{
                'role': 'user',
                'content': 'Repeat this code exactly.'
            }, {
                'role': 'assistant',
                'content': ' '.join(['qx7', 'zzp', '44b', 'kkv', 'w9q'] * 12)
            }]
        },
    ]
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _run_hf_shape(shape, data_path, out_dir, result_prefix):
    """Launch _hf_dp_runner.py for ONE shape; returns one result dict per rank.

    dp2 goes through torchrun (2 ranks, one sample each); single is a plain process, which is how a
    one-GPU user actually runs it -- WORLD_SIZE unset, so twinkle's default mesh is dp=1 and the
    reported loss is just sum/tokens over the whole batch. Both use _run_torchrun for its
    process-group kill, since a hung rank otherwise blocks on inherited pipes forever.
    """
    import sys

    runner = os.path.join(os.path.dirname(__file__), '_hf_dp_runner.py')
    cmd = [sys.executable]
    if shape == 'dp2':
        cmd += ['-m', 'torch.distributed.run', '--nproc_per_node=2', f'--master_port={_master_port(9)}']
    cmd += [runner, '--shape', shape, '--data', data_path, '--out', result_prefix, '--out_dir', out_dir]
    out, err = _run_torchrun(cmd)

    results = []
    for rank in range(2 if shape == 'dp2' else 1):
        path = f'{result_prefix}.rank{rank}.json'
        if not os.path.exists(path):
            raise AssertionError(f'{shape} runner produced no result for rank {rank}. stdout tail:\n{out[-2000:]}\n'
                                 f'stderr tail:\n{err[-3000:]}')
        with open(path) as f:
            results.append(json.load(f))
    return results


@pytest.mark.slow
def test_run_sft_hf_dp_loss_is_globally_token_weighted(tmp_path):
    """2-GPU transformers SFT must train on (and report) the GLOBAL token-weighted loss.

    This is the multi-GPU counterpart of the single-GPU legacy-vs-dev comparison, and it exists
    because that comparison cannot see the failure: with device_mesh=None every rank normalises by its
    OWN token count, so one GPU looks perfect while two GPUs silently optimise a different objective
    (an avg-of-avg of per-rank means instead of legacy's global_sum_grad / global_num_tokens). dev
    shipped that way -- run_sft returned before twinkle.initialize on this backend, and remote_class
    then dropped the mesh -- and no test noticed, because there was no multi-GPU test on this path.

    Reference is the SAME two samples in one process (bs=2), which by construction reports
    sum_loss/total_tokens -- exactly what legacy computes. A dp2 run that aggregates correctly must
    land on it; a broken one reports whatever its own rank held, and the deliberately imbalanced rows
    (see _write_imbalanced_dataset) put that tens of percent away.

    Three assertions, each pinning a different link:
      - plumbing: mesh present with dp world size 2, dp group built. This is the gradient side --
        calculate_loss reads _get_dp_fsdp_world_size() straight off the mesh, so a right mesh here IS
        the right divisor, which no logged scalar can prove on its own.
      - cross-rank identity: both ranks report one number, i.e. the gather happened.
      - value: that number equals the single-process token-weighted mean (band from bf16 noise).

    Every number below is measured, including the failing side -- the pre-fix behaviour was replayed
    on purpose to check this test can see it. Fixed: both ranks 0.7794 against a single-process 0.7764
    (rel 3.9e-3, pure bf16 grouping noise, so the 2e-2 band has room without going slack). Pre-fix:
    mesh absent, rank0 reported 2.6482 and rank1 0.7321 -- each its own sample, with rank0 off the
    global value by 241%. All three assertion groups fire on that, which is the point of having them.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip('needs >=2 GPUs')

    data_path = str(tmp_path / 'imbalanced.jsonl')
    _write_imbalanced_dataset(data_path)

    dp2 = _run_hf_shape('dp2', data_path, str(tmp_path / 'dp2_out'), str(tmp_path / 'dp2'))
    single = _run_hf_shape('single', data_path, str(tmp_path / 'single_out'), str(tmp_path / 'single'))

    print(f'\nHF dp aggregation: dp2={[r["losses"] for r in dp2]} single={single[0]["losses"]}')
    for r in dp2:
        assert r['device_mesh_present'], (
            f'rank {r["rank"]}: device_mesh is None -- remote_class dropped it, so the gradient '
            'divisor is 1 instead of the dp world size (call twinkle.initialize before build_model)')
        assert r['dp_world_size'] == 2, (
            f'rank {r["rank"]}: mesh reports dp world size {r["dp_world_size"]}, not 2 -- the token '
            'count would be normalised by the wrong divisor')
        assert r['dp_group_present'], (
            f'rank {r["rank"]}: dp process group not built, so LossMetric skips its gather and this '
            "rank only ever reports its own loss (see set_optimizer's _ensure_optimizer_dp_groups)")

    assert dp2[0]['losses'] == dp2[1]['losses'], (
        f'ranks disagree: rank0={dp2[0]["losses"]} rank1={dp2[1]["losses"]} -- each is reporting its '
        'own loss instead of the dp-wide aggregate')

    ref, got = single[0]['losses'][0], dp2[0]['losses'][0]
    rel = abs(got - ref) / max(abs(ref), 1e-8)
    assert rel < 0.02, (f'dp2 loss {got:.6f} is not the global token-weighted mean {ref:.6f} (rel={rel:.2e}, limit '
                        '2e-2): the two shapes trained on the same two samples, so this gap is an aggregation '
                        'convention difference, not noise')
