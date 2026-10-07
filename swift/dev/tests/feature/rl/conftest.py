# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared fixtures for the dev RL feature tests (the end-to-end alignment suite).

Conventions this suite is built on (RL_PLAN basic principles, which override everything below):

* **RL is always ``mode='ray'``.** There is no ``mode='local'`` RL implementation -- the trainer and the
  rollout sampler are separate Ray actors the driver talks to. :func:`run_rl` therefore drives the real
  CLI lifecycle (``parse`` -> force Ray/vLLM placement -> ``process_and_validate_configs`` -> ``run_rlhf``)
  rather than the torchrun runner the SFT suite uses. Every derivation and validation the production
  ``swift rl`` command runs, runs here too -- so a config the CLI would reject is rejected here as well
  (which is exactly what the infeasible-combination tests assert against).
* **Backend equivalence.** ``megatron`` and ``transformers`` are identical except ``generate`` (in-process,
  TransformersModel-only), so no test writes a backend-specific stub; a capability that works on one must
  work on the other. On-policy generation always rides the independent-sampler + weight-sync path both
  backends share.
* **End to end, not degenerate.** A green test must mean real Ray actors x a real sampler x a real model
  trained optimizer steps -- never ``group=None`` / single-rank / empty-input shortcuts that skip the core
  branch. Heavy tests are gated ``@pytest.mark.slow`` (+ ``@pytest.mark.accel(N)`` when multi-GPU); run them
  on a free card with ``CUDA_VISIBLE_DEVICES=<cards> pytest swift/dev/tests/feature/rl -m slow``.

No mocks live in this suite. Post-conditions are read back from the written artifact (checkpoint on disk),
not from in-memory values, and are anchored to independent oracles (hand-computed GAE/value loss in
``test_recipes.py``, legacy per-step loss in ``test_parity_legacy.py``, greedy-determinism for the sampler).
"""
import json
import os
from typing import Any, Dict, List, Optional

import pytest

#: Real cached 0.5B, for the paths that need genuine generation quality (vLLM rollout determinism,
#: legacy loss parity). Wiring-only paths use the tiny random checkpoints below instead.
MODEL_TEXT = 'Qwen/Qwen2.5-0.5B-Instruct'

#: The online family rolls out on an independent vLLM/SGLang sampler and weight-syncs into it; the
#: offline preference family only forwards a fixed chosen/rejected dataset. Mirrors run_rlhf's dispatch.
ONLINE_TYPES = frozenset({'grpo', 'ppo', 'rft', 'gkd', 'opsd', 'mopd'})
OFFLINE_TYPES = frozenset({'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'})


# --- the driver -----------------------------------------------------------------------------


def _force_rl_placement(configs: Dict[str, Any]) -> None:
    """Replicate the CLI's Ray/vLLM placement forcing (``cli/rlhf.py`` ``parse_rlhf_configs``).

    ``run_rlhf_configs`` runs process+validate+dispatch but NOT this forcing -- it lives in the argv
    parser, which a programmatic caller bypasses. Applying it here keeps ``run_rl`` on the exact
    production placement: every online type gets a vLLM sampler (colocate by default) and Ray mode; the
    offline family is Ray too (its policy is a Ray actor) but rolls out nothing, so it sets no sampler.
    """
    rlhf_config = configs['rlhf_config']
    rollout_config = configs['rollout_config']
    distributed_config = configs['distributed_config']
    if rlhf_config.rlhf_type in ONLINE_TYPES:
        rollout_config.use_vllm = True
        rollout_config.vllm_mode = rollout_config.vllm_mode or 'colocate'
        distributed_config.mode = 'ray'
    elif rlhf_config.rlhf_type in OFFLINE_TYPES:
        distributed_config.mode = 'ray'


def run_rl(configs: Dict[str, Any]) -> List[dict]:
    """Drive one RL run through the real CLI lifecycle and return its loss history.

    Tears down any Ray session left by an earlier ``run_rl`` in the SAME test first (the autouse
    ``_isolate_twinkle_runtime`` fixture only resets BETWEEN tests), so a test that runs twice -- backend
    equivalence, parity -- needs no manual reset. The teardown is a cheap no-op when no Ray session is
    live, and dropping it is safe because a loop materializes its history into plain dicts before
    returning (see ``_twinkle_runtime.reset_twinkle_runtime``).
    """
    from swift.dev.cli.rlhf import run_rlhf_configs
    from swift.dev.tests._twinkle_runtime import reset_twinkle_runtime

    reset_twinkle_runtime()
    _force_rl_placement(configs)
    return run_rlhf_configs(configs)


def rl_configs(
    *,
    rlhf_type: str,
    model: str,
    dataset: Any,
    out_dir: str,
    nproc: int = 1,
    backend: Optional[str] = None,
    tuner: Optional[str] = 'lora',
    model_type: Optional[str] = None,
    template: Optional[str] = None,
    max_steps: int = 2,
    model_over: Optional[Dict[str, Any]] = None,
    template_over: Optional[Dict[str, Any]] = None,
    dataset_over: Optional[Dict[str, Any]] = None,
    train_over: Optional[Dict[str, Any]] = None,
    dist_over: Optional[Dict[str, Any]] = None,
    ckpt_over: Optional[Dict[str, Any]] = None,
    tuner_over: Optional[Dict[str, Any]] = None,
    rollout_over: Optional[Dict[str, Any]] = None,
    rlhf_over: Optional[Dict[str, Any]] = None,
    generation_over: Optional[Dict[str, Any]] = None,
    megatron_over: Optional[Dict[str, Any]] = None,
    moe_over: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``run_rlhf_configs`` config mapping with sane tiny-run defaults.

    Each ``*_over`` dict is merged onto that Config's defaults, so a test states only what its dimension
    varies. Defaults are chosen for a 1-2 optimizer-step run on one card: batch 1 / ga 1 (every
    mini-batch is its own step -- see the design-B note in ``examples/v5/rl/rft/rft.sh``), a short
    completion budget, an eager vLLM at low memory utilisation so a colocated sampler shares the card
    with the trainer, greedy generation (deterministic), and ``logging_steps=1`` so every step lands in
    the returned history. ``dataset_shuffle`` is off so parity/resume runs are reproducible.
    """
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        PluginConfig,
        QuantizeConfig,
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

    def _merge(cls, over, **base):
        return cls(**{**base, **(over or {})})

    datasets = [dataset] if isinstance(dataset, str) else list(dataset)
    model_config = _merge(
        ModelConfig, model_over, model=model, model_type=model_type, torch_dtype='bfloat16')
    template_config = _merge(TemplateConfig, template_over, template=template, max_length=1024)
    dataset_config = _merge(
        DatasetConfig, dataset_over, dataset=datasets, dataset_shuffle=False, data_seed=42)
    train_config = _merge(
        TrainConfig,
        train_over,
        learning_rate=1e-4,
        optim='adamw',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        seed=42,
    )
    distributed_config = _merge(
        DistributedConfig, dist_over, backend=backend, mode='ray', nproc_per_node=nproc)
    checkpoint_config = _merge(
        CheckpointConfig, ckpt_over, output_dir=out_dir, save_strategy='steps', save_steps=max_steps)
    tuner_config = None
    if tuner is not None:
        tuner_config = _merge(
            TunerConfig, tuner_over, tuner=tuner, lora_rank=8, lora_alpha=32, target_modules=['all-linear'])
    # vLLM at low utilisation + eager (no CUDA-graph capture) so a colocated sampler fits beside the
    # trainer on one card; a short model_len bounds the KV cache. Overridable per dimension.
    rollout_config = _merge(
        RolloutConfig,
        rollout_over,
        rollout_sampler='vllm',
        vllm_gpu_memory_utilization=0.4,
        vllm_enforce_eager=True,
        vllm_max_model_len=512,
    )
    rlhf_config = _merge(RLHFConfig, rlhf_over, rlhf_type=rlhf_type, max_completion_length=64)
    generation_config = _merge(GenerationConfig, generation_over, max_new_tokens=32, temperature=0.0)
    logging_config = LoggingConfig(report_to=['none'], logging_steps=1)
    megatron_config = _merge(MegatronConfig, megatron_over) if megatron_over else None
    moe_config = _merge(MoEConfig, moe_over) if moe_over else None

    return {
        'model_config': model_config,
        'plugin_config': PluginConfig(),
        'template_config': template_config,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'distributed_config': distributed_config,
        'checkpoint_config': checkpoint_config,
        'logging_config': logging_config,
        'tuner_config': tuner_config,
        'generation_config': generation_config,
        'rollout_config': rollout_config,
        'rlhf_config': rlhf_config,
        'quantize_config': QuantizeConfig(),
        'megatron_config': megatron_config,
        'moe_config': moe_config,
    }


# --- checkpoint read-back helpers (post-conditions read the artifact, not memory) -------------


def read_args_json(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, 'args.json')) as f:
        return json.load(f)


def lora_b_nonzero(ckpt_dir: str) -> bool:
    """True if some ``lora_B`` weight read back from the checkpoint safetensors is non-zero.

    ``lora_B`` is initialised to exactly zero (the adapter starts as the identity), so a non-zero
    ``lora_B`` on disk is direct proof the optimizer really updated the adapter -- a run that silently
    trained nothing (frozen group, detached graph, zero grad) leaves it at zero.
    """
    import glob

    from safetensors.torch import load_file

    for path in glob.glob(os.path.join(ckpt_dir, '*.safetensors')):
        tensors = load_file(path)
        for name, tensor in tensors.items():
            if 'lora_B' in name and float(tensor.abs().max()) > 0.0:
                return True
    return False


# --- fixtures ---------------------------------------------------------------------------------


@pytest.fixture(scope='session')
def text_model_path():
    """Local path to the real cached 0.5B text model, resolved once per session."""
    from modelscope import snapshot_download
    return snapshot_download(MODEL_TEXT)


@pytest.fixture(scope='session')
def tiny_qwen2_5(tmp_path_factory):
    """A 4-layer random-weight dense Qwen2.5 checkpoint (real tokenizer) for wiring-only RL runs."""
    from swift.dev.tests.tiny import TinyModel
    dest = tmp_path_factory.mktemp('tiny_qwen2_5') / 'model'
    return TinyModel.build(dest)


@pytest.fixture(scope='session')
def tiny_qwen2_5_teacher(tmp_path_factory):
    """A SECOND tiny dense checkpoint with DIFFERENT random weights, for a real (non-self) distillation
    teacher.

    GKD/MOPD pull the student toward a frozen teacher; a self teacher (the adapter-disabled student base)
    equals the identity-adapter student at init, so the divergence -- and its gradient -- would be exactly
    zero and ``lora_B`` would never move. A separately built tiny checkpoint draws fresh random weights
    (``TinyModel.build`` advances the global RNG), so student != teacher and the distillation loss is
    non-trivial from step one.
    """
    from swift.dev.tests.tiny import TinyModel
    dest = tmp_path_factory.mktemp('tiny_qwen2_5_teacher') / 'model'
    return TinyModel.build(dest)


@pytest.fixture(scope='session')
def tiny_qwen2_5_teacher2(tmp_path_factory):
    """A THIRD distinct tiny checkpoint, so MOPD's multi-teacher blend mixes two DIFFERENT distributions.

    MOPD fuses K teacher channels by ``teacher_weights`` (``logsumexp`` over the weighted per-teacher
    log-probs). Feeding it the SAME checkpoint twice would still run the fusion, but a bug that dropped or
    duplicated a channel would be invisible (all channels identical). Two separately built checkpoints
    (fresh random weights each) make the blended target a genuine mixture, so the K-channel path is
    observably exercised rather than degenerate.
    """
    from swift.dev.tests.tiny import TinyModel
    dest = tmp_path_factory.mktemp('tiny_qwen2_5_teacher2') / 'model'
    return TinyModel.build(dest)


@pytest.fixture(scope='session')
def tiny_qwen3_moe(tmp_path_factory):
    """A tiny random-weight Qwen3-MoE checkpoint (4 experts) for the R2/R3 + MoE-backend dimensions."""
    from swift.dev.tests.tiny import TinyModel
    from swift.dev.tests.tiny_loader import loader_builder
    dest = tmp_path_factory.mktemp('tiny_qwen3_moe') / 'model'
    return TinyModel.build(
        dest,
        tokenizer_id='Qwen/Qwen3-30B-A3B',
        builder=loader_builder('qwen3_moe'),
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
    )


@pytest.fixture(scope='session')
def tiny_vl(tmp_path_factory):
    """A tiny random-weight Qwen2.5-VL checkpoint (model + image processor) for the multimodal dimension."""
    from swift.dev.tests.tiny_loader import build_tiny_multimodal
    dest = tmp_path_factory.mktemp('tiny_vl') / 'model'
    return build_tiny_multimodal(str(dest), model_type='qwen2_5_vl', model_id='Qwen/Qwen2.5-VL-3B-Instruct')


@pytest.fixture
def write_jsonl():
    """Return a writer that dumps a list of row dicts to a ``.jsonl`` path."""

    def _write(path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        return str(path)

    return _write


@pytest.fixture
def assert_rl_trained():
    """The RL post-condition checker, analogous to the SFT suite's ``assert_trained``.

    Universal (every algorithm): the recipe ran optimizer steps, the loss stayed finite and normalized
    (a raw-sum reduction or a diverged run trips ``max_loss``), and a self-describing checkpoint
    (``*.safetensors`` + ``args.json``) was written and is readable from disk. Dimension-specific extras
    are opt-in so a test asserts only fields its algorithm actually emits:

    * ``require_keys`` -- history fields the dimension needs (PPO's ``reward``/``value_loss``, GRPO's
      ``entropy``); an offline run emits only ``loss``/``grad_norm``.
    * ``lora=True`` -- read ``lora_B`` back from the checkpoint and require it non-zero (real training);
      set False for a full-parameter run (no adapter).
    * ``expect_task_type`` / ``expect_tuner_type`` -- force_load keys ``args.json`` must carry so
      ``swift infer`` does not silently downgrade the checkpoint (rm -> seq_cls, LoRA -> lora).
    """

    def _assert(history,
                configs,
                rlhf_type,
                *,
                max_loss,
                require_keys=(),
                lora=True,
                expect_task_type=None,
                expect_tuner_type='lora'):
        assert history, f'{rlhf_type}: recipe produced no optimizer steps'
        losses = [r['loss'] for r in history]
        # loss == loss is the NaN test (NaN != itself); not a typo.
        assert all(loss == loss and abs(loss) != float('inf') for loss in losses), \
            f'{rlhf_type}: non-finite loss: {losses}'
        assert abs(losses[-1]) < max_loss, \
            f'{rlhf_type}: converged loss out of sane range (<{max_loss}) -- raw sum or diverged: {losses}'
        for key in require_keys:
            assert all(key in r for r in history), f'{rlhf_type}: {key!r} missing from history {history}'

        # The checkpoint lives under the RESOLVED output_dir: process_and_validate_configs mutates
        # checkpoint_config.output_dir in place to the add_version subdir (v0-<ts>), so reading the
        # config back after run_rl is the faithful path (we do NOT disable add_version to make the
        # assertion easier -- that would test a layout production never writes).
        out_dir = configs['checkpoint_config'].output_dir
        ckpt = os.path.join(out_dir, 'checkpoint-final')
        assert os.path.isdir(ckpt), f'{rlhf_type}: no final checkpoint at {ckpt}'
        files = set(os.listdir(ckpt))
        assert any(f.endswith('.safetensors') for f in files), f'{rlhf_type}: no weights in {sorted(files)}'
        assert 'args.json' in files, f'{rlhf_type}: checkpoint not self-describing: {sorted(files)}'
        args = read_args_json(ckpt)
        if expect_task_type is not None:
            assert args.get('task_type') == expect_task_type, \
                f'{rlhf_type}: args.json task_type={args.get("task_type")!r}, expected {expect_task_type!r}'
        if expect_tuner_type is not None:
            assert args.get('tuner_type') == expect_tuner_type, \
                f'{rlhf_type}: args.json tuner_type={args.get("tuner_type")!r}, expected {expect_tuner_type!r}'
        if lora:
            assert lora_b_nonzero(ckpt), \
                f'{rlhf_type}: every lora_B is still zero on disk -- the optimizer never updated the adapter'
        return losses

    return _assert


# --- RL datasets ------------------------------------------------------------------------------
# Preference / prompt-only reuse swift.dev.tests.tiny.TinyData; the RL-specific shapes (KTO's binary
# label, RFT's ground-truth solution, OPSD's privileged teacher prompt, multimodal image packs) are
# written here. Formats match examples/v5/rl/data/*.jsonl and the registered loaders.

_PROMPTS = ('What is 2+2?', 'Name a colour.', 'Say hello.', 'Count to three.')


def kto_rows(n: int = 8) -> List[dict]:
    """KTO is UNPAIRED: one completion per row carrying a binary desirability ``label``.

    ``PreferenceLoop._encode_kto_batch`` reads the last assistant turn as the completion and ``label`` as
    desirable/undesirable, and builds its own mismatched KL batch by rotating completions -- so the row
    needs no ``rejected_response``. Alternating labels keep both the desirable and undesirable branches of
    ``KTOLoss`` live (an all-one-sided batch would exercise only half the loss).
    """
    rows = []
    for i in range(n):
        rows.append({
            'messages': [{
                'role': 'user',
                'content': _PROMPTS[i % len(_PROMPTS)]
            }, {
                'role': 'assistant',
                'content': 'a good answer' if i % 2 == 0 else 'a bad answer'
            }],
            'label': i % 2 == 0,
        })
    return rows


def rft_math_rows(n: int = 4) -> List[dict]:
    """RFT best-of-n rows: a prompt plus a ``solution`` ground truth for the ``accuracy`` ORM.

    ``--orm accuracy`` (MathAccuracy) reads the dataset's ``solution`` column and the completion's
    ``\\boxed{}`` answer, so each row ships a boxed target (mirrors examples/v5/rl/data/rft_math.jsonl).
    """
    rows = []
    for i in range(n):
        answer = 42 + i
        rows.append({
            'messages': [{
                'role': 'user',
                'content': f'Compute the answer and put it in \\boxed{{}}. Target value is {answer}.'
            }],
            'solution': f'The answer is \\boxed{{{answer}}}.',
        })
    return rows


def opsd_rows(n: int = 4) -> List[dict]:
    """OPSD rows: a prompt plus a privileged ``teacher_prompt`` (mirrors examples/v5/rl/data/opsd.jsonl)."""
    rows = []
    for i in range(n):
        rows.append({
            'messages': [{
                'role': 'user',
                'content': _PROMPTS[i % len(_PROMPTS)]
            }],
            'teacher_prompt': f'Reference solution for {_PROMPTS[i % len(_PROMPTS)]}: answer concisely.',
        })
    return rows


def write_image(path, size: int = 56) -> str:
    """A tiny deterministic RGB PNG on disk, big enough for one Qwen2.5-VL vision patch grid."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(str(path))
    return str(path)


def multimodal_prompt_rows(image_paths: List[str]) -> List[dict]:
    """VL GRPO/RFT prompts: an ``<image>`` placeholder in the content plus a parallel ``images`` column.

    This is the sibling-column form ``builders.dataset.to_trajectory`` threads onto the rollout ``Trajectory``
    (it copies the row's ``images``/``videos``/... columns), and the form ``infer/test_multimodal_e2e`` proves
    the VL template expands into ``<image>`` pad tokens + ``pixel_values``. The template validates the
    placeholder/image-count match, so a dropped image fails loudly instead of degrading to a silent text-only
    rollout -- which is what makes a green VL RL run proof the image reached both ``generate`` and the training
    forward. The ``images`` column also stays in ``prompt_extras`` for the reward, so a reward can assert the
    media reference survived to scoring.
    """
    return [{
        'messages': [{
            'role': 'user',
            'content': '<image>Describe this image in one short sentence.'
        }],
        'images': [image_paths[i % len(image_paths)]],
    } for i in range(len(image_paths))]


@pytest.fixture
def rl_data(tmp_path, write_jsonl):
    """No-arg writers for each RL dataset shape; each returns the written ``.jsonl`` path.

    A namespace so a test names only the dataset its algorithm needs. Preference / prompt-only delegate
    to :class:`~swift.dev.tests.tiny.TinyData` (reuse-first); the RL-specific shapes are local.
    """
    from swift.dev.tests.tiny import TinyData

    class _RLData:

        def preference(self, n: int = 8):
            """dpo / cpo / orpo / simpo / rm: a chosen turn plus a ``rejected_response``."""
            return TinyData.preference(tmp_path / 'preference.jsonl', n=n)

        def prompt_only(self, n: int = 4):
            """grpo / ppo: prompts with no assistant turn to learn from."""
            return TinyData.prompt_only(tmp_path / 'prompts.jsonl', n=n)

        def kto(self, n: int = 8):
            return write_jsonl(tmp_path / 'kto.jsonl', kto_rows(n))

        def rft_math(self, n: int = 4):
            return write_jsonl(tmp_path / 'rft_math.jsonl', rft_math_rows(n))

        def opsd(self, n: int = 4):
            return write_jsonl(tmp_path / 'opsd.jsonl', opsd_rows(n))

        def multimodal(self, n_images: int = 2):
            images = [write_image(tmp_path / f'img_{i}.png') for i in range(n_images)]
            return write_jsonl(tmp_path / 'vl_prompts.jsonl', multimodal_prompt_rows(images))

    return _RLData()
