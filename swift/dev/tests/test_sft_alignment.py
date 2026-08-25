"""Regression test: refactored SFT loss == legacy HF shifted CE (step-1, exact in fp32).

Freezes the alignment result. Uses step-1 loss on a fixed batch (no optimizer
compounding) so the check is deterministic and bit-exact in fp32. Skips when the
local model or a CUDA device is unavailable, so it is CI-safe.

Label conventions under test:
- legacy path: aligned labels (labels[i] == input_ids[i]) + HF INTERNAL shift at loss
  time (logits[:, :-1] vs labels[:, 1:]) — what swift Seq2SeqTrainer.compute_loss does.
- dev path: dev Template shifts labels to next-token at ENCODE time, twinkle forward
  computes no-shift logps. The two must produce the same loss.
"""
import os

import pytest

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

# Heavy tests (real weights + GPU) are gated per-test; the lightweight configure_loss
# unit tests below always run (they guard the product-path default).
requires_model = pytest.mark.skipif(not os.path.exists(MODEL), reason='local Qwen2.5-0.5B not available')

MESSAGES = [
    {
        'role': 'user',
        'content': 'What is 2+2?'
    },
    {
        'role': 'assistant',
        'content': '2 + 2 equals 4.'
    },
]


@requires_model
def test_sft_loss_matches_legacy_hf_step1():
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from swift.dev.template import DevMixin
    from swift.model import get_model_processor
    from swift.template import get_template

    _, proc = get_model_processor(MODEL, load_model=False)

    # legacy template: aligned labels (no encode-time shift)
    legacy_tpl = get_template(proc, template_type='qwen2_5', max_length=512)
    legacy_tpl.set_mode('train')
    legacy_feat = legacy_tpl.encode({'messages': MESSAGES})

    # dev template: next-token-shifted labels at encode
    dev_tpl = _dev_template(get_template(proc, template_type='qwen2_5', max_length=512))
    dev_tpl.set_mode('train')
    dev_feat = dev_tpl.encode({'messages': MESSAGES})

    # input_ids must be identical; only the label convention differs
    assert legacy_feat['input_ids'] == dev_feat['input_ids']

    # legacy: HF shifted CE token-mean on ALIGNED labels
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval().cuda()
    ids = torch.tensor(legacy_feat['input_ids']).unsqueeze(0).cuda()
    labs = torch.tensor(legacy_feat['labels']).unsqueeze(0).cuda()
    with torch.no_grad():
        logits = hf(input_ids=ids).logits.float()
    sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    tl = labs[:, 1:].contiguous().view(-1)
    legacy = F.cross_entropy(sl, tl, ignore_index=-100).item()
    del hf
    torch.cuda.empty_cache()

    # dev refactored path (fp32, same weights, pre-shifted labels)
    from swift.dev.loss import CrossEntropyLoss
    from swift.dev.model import TransformersModel
    from swift.dev.processor import InputProcessor
    model = TransformersModel(model_id=MODEL, mixed_precision='no', strategy='accelerate', dtype=torch.float32)
    model.set_loss(CrossEntropyLoss(reduction='mean'))  # scalar comparison: single-sample mean matches legacy
    model.set_processor(InputProcessor())
    model.forward_only(inputs=[dict(dev_feat)])
    dev = model.calculate_loss()

    assert abs(legacy - dev) < 1e-3, f'legacy={legacy:.6f} dev={dev:.6f}'


@requires_model
def test_ga_equivalence_sum_reduction():
    """GA=2/bs=1 accumulated grad == GA=1/bs=2 single-batch grad (reduction='sum').

    With sum loss + twinkle's num_tokens normalization (clip_grad_norm divides by
    total tokens across the GA window), GA=2 is mathematically identical to bs=2.
    Verifies via grad-ratio: the ratio of grad elements must be 1.0 (std<1e-5),
    proving no systematic scaling difference — any residual is fp32 kernel noise.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from swift.dev.loss import CrossEntropyLoss
    from swift.dev.model import TransformersModel
    from swift.dev.processor import InputProcessor
    from swift.dev.template import DevMixin
    from swift.model import get_model_processor
    from swift.template import get_template

    _, proc = get_model_processor(MODEL, load_model=False)
    tpl = _dev_template(get_template(proc, template_type='qwen2_5', max_length=512))
    tpl.set_mode('train')
    # Use two samples (may differ in length; sum reduction handles it correctly)
    feats = [
        tpl.encode({
            'messages': [{
                'role': 'user',
                'content': 'What is 2+2?'
            }, {
                'role': 'assistant',
                'content': '2 + 2 equals 4.'
            }]
        }),
        tpl.encode({
            'messages': [{
                'role': 'user',
                'content': 'Capital?'
            }, {
                'role': 'assistant',
                'content': 'The capital of France is Paris.'
            }]
        }),
    ]

    def get_grads(mode):
        m = TransformersModel(model_id=MODEL, mixed_precision='no', strategy='accelerate', dtype=torch.float32)
        m.set_loss(CrossEntropyLoss(reduction='sum'))
        m.set_processor(InputProcessor())
        m.set_optimizer('SGD', lr=1e-3)
        if mode == 'ga2':
            for f in feats:
                m.forward_backward(inputs=[dict(f)], gradient_accumulation_steps=2)
        else:
            m.forward_backward(inputs=[dict(f) for f in feats], gradient_accumulation_steps=1)
        raw = m.strategy.unwrap_model(m.model)
        return {
            n: p.grad.detach().float().cpu().clone()
            for n, p in raw.named_parameters() if p.requires_grad and p.grad is not None
        }

    g_ga2 = get_grads('ga2')
    torch.cuda.empty_cache()
    g_bs2 = get_grads('bs2')

    # Grad-ratio on large-magnitude elements (above median by abs) should be ~1.0
    # Exclude very large params (embedding) where quantile() overflows on some torch versions.
    ratios = []
    for n in g_ga2:
        a, b = g_ga2[n], g_bs2[n]
        if a.numel() > 2_000_000:  # skip huge embedding/lm_head
            continue
        threshold = b.abs().median()
        big = b.abs() > threshold
        if big.sum() == 0:
            continue
        r = (a[big] / b[big])
        ratios.append(r)
    all_ratios = torch.cat(ratios)
    ratio_std = all_ratios.std().item()
    ratio_mean = all_ratios.mean().item()
    assert abs(ratio_mean - 1.0) < 1e-2, f'ratio mean={ratio_mean:.6f} (expected ~1.0)'
    assert ratio_std < 1e-3, f'ratio std={ratio_std:.2e} (expected <1e-3; mean-reduction gives ~0.6)'


# ----------------------------------------------------------------------
# Product-path default: the dev SFT assembly must yield reduction='sum'
# (lightweight: no GPU / model weights; guards the DEFAULT, not a manual set_loss).
# ----------------------------------------------------------------------


class _FakeModel:
    """Captures the loss instance set via the twinkle set_loss contract."""

    def __init__(self):
        self.loss_instance = None

    def set_loss(self, loss, **kwargs):
        self.loss_instance = loss


def test_configure_loss_defaults_to_sum():
    """The dev SFT loss assembly (configure_loss) must default to reduction='sum'.

    This guards the *product path* default (GA-correctness), independent of any test
    manually passing reduction='sum'. Runs without GPU/model.
    """
    from swift.dev.loss import CrossEntropyLoss, configure_loss

    m = _FakeModel()
    configure_loss(m)
    assert isinstance(m.loss_instance, CrossEntropyLoss)
    assert getattr(m.loss_instance, 'reduction', None) == 'sum'


def test_configure_loss_rejects_unknown_type():
    from swift.dev.loss import configure_loss
    m = _FakeModel()
    with pytest.raises(NotImplementedError):
        configure_loss(m, loss_type='dpo')


# ----------------------------------------------------------------------
# Unified naming layer: swift-style names -> twinkle constructibles.
# Guards the twinkle construct_class gotcha (plain names like 'cross_entropy'
# don't resolve via getattr; the naming layer bridges it). Lightweight: pure
# mapping logic, no GPU/model.
# ----------------------------------------------------------------------


def test_resolve_loss_returns_twinkle_type():
    """resolve_loss reuses twinkle torch_loss_mapping and returns the SAME class
    that swift.dev.loss re-exports (so configure_loss semantics are unchanged)."""
    from twinkle.loss import CrossEntropyLoss as TwinkleCE

    from swift.dev.loss import CrossEntropyLoss
    from swift.dev.naming import resolve_loss

    cls = resolve_loss('cross_entropy')
    assert cls is TwinkleCE
    assert cls is CrossEntropyLoss  # dev re-exports twinkle's -> same object


def test_resolve_optim_aliases_and_case():
    from swift.dev.naming import resolve_optim
    assert resolve_optim('adamw_torch_fused') == 'AdamW'
    assert resolve_optim('AdamW') == 'AdamW'  # case-insensitive
    assert resolve_optim('sgd') == 'SGD'


def test_resolve_optim_target_matches_transformers_trainer():
    """dev must construct the SAME optimizer class + kwargs as transformers' Trainer.

    First principle for the dev refactor is not changing legacy swift's training outcome, and legacy
    goes through HF Trainer. A name-only mapping silently dropped two things:
      - adamw_torch_fused -> fused=True (HF shares the adamw_torch handler and only adds this flag),
        so the dev DEFAULT optim claimed 'fused' while running the unfused kernel;
      - adafactor -> HF uses transformers.optimization.Adafactor with scale_parameter/relative_step
        forced off, NOT torch.optim.Adafactor, whose defaults differ -> a silently DIFFERENT
        training trajectory.
    Compare against the real HF resolver so this cannot drift.
    """
    import torch
    from transformers.trainer import Trainer

    from swift.dev.naming import resolve_optim_target
    from transformers import TrainingArguments

    for name in ('adamw_torch', 'adamw_torch_fused', 'adafactor', 'sgd'):
        args = TrainingArguments(
            output_dir='/tmp/_optim_align',
            optim=name,
            learning_rate=1e-4,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            report_to=[])
        hf_cls, hf_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        target, extra = resolve_optim_target(name)
        dev_cls = getattr(torch.optim, target) if isinstance(target, str) else target
        assert dev_cls is hf_cls, f'{name}: dev builds {dev_cls}, HF builds {hf_cls}'
        # Every non-lr/adam kwarg HF applies must also be applied by dev.
        hf_extra = {k: v for k, v in hf_kwargs.items() if k not in ('lr', 'betas', 'eps')}
        assert extra == hf_extra, f'{name}: dev extra kwargs {extra} != HF {hf_extra}'


def test_parse_optim_args_coerces_types():
    """optim_args was an accepted TrainConfig field that configure_optimizer never read (silently
    ignored). CLI values arrive as strings, so they must be coerced before hitting a constructor."""
    from swift.dev.naming import parse_optim_args
    assert parse_optim_args(None) == {}
    assert parse_optim_args('') == {}
    assert parse_optim_args('fused=True,eps=1e-6,rank=8,proj=std,x=none') == {
        'fused': True,
        'eps': 1e-6,
        'rank': 8,
        'proj': 'std',
        'x': None
    }
    with pytest.raises(ValueError, match='expected "key=value"'):
        parse_optim_args('bogus')


def test_configure_optimizer_applies_extra_and_optim_args():
    """End-to-end kwargs handed to twinkle set_optimizer, incl. the Adam-only betas/eps guard and
    optim_args overriding our defaults (HF merges _parse_optim_args into optimizer_kwargs too)."""
    from unittest.mock import MagicMock

    from swift.dev.config import TrainConfig
    from swift.dev.optimizer import configure_optimizer

    def _kwargs(**cfg_kw):
        m = MagicMock()
        # A plain MagicMock is not the dev MegatronModel, so this takes the transformers branch.
        configure_optimizer(m, TrainConfig(learning_rate=1e-4, weight_decay=0.1, **cfg_kw), num_training_steps=10)
        return m.set_optimizer.call_args[1]

    assert _kwargs(optim='adamw_torch_fused')['fused'] is True
    # Adafactor rejects betas/eps -- they must NOT be forwarded.
    adafactor_kw = _kwargs(optim='adafactor')
    assert adafactor_kw['scale_parameter'] is False and adafactor_kw['relative_step'] is False
    assert 'betas' not in adafactor_kw and 'eps' not in adafactor_kw
    # optim_args wins over the defaults we set.
    assert _kwargs(optim='adamw_torch', optim_args='eps=1e-05')['eps'] == 1e-05


def test_configure_optimizer_does_not_pass_params():
    """twinkle's set_optimizer builds the two weight-decay param groups itself (reusing HF's
    get_decay_parameter_names rules: bias/*norm excluded from decay, plus LoRA adapter filtering).
    Passing an explicit `params` would bypass that and apply weight_decay to norms/biases, i.e.
    diverge from legacy swift -- so dev must leave it out."""
    from unittest.mock import MagicMock

    from swift.dev.config import TrainConfig
    from swift.dev.optimizer import configure_optimizer

    m = MagicMock()
    configure_optimizer(m, TrainConfig(learning_rate=1e-4), num_training_steps=10)
    assert 'params' not in m.set_optimizer.call_args[1]


def test_resolve_scheduler_constant_is_none():
    from swift.dev.naming import resolve_scheduler
    assert resolve_scheduler('cosine') == 'CosineWarmupScheduler'
    assert resolve_scheduler('constant') is None  # constant => no scheduler


def test_resolve_strategy_aliases():
    from swift.dev.naming import resolve_strategy
    assert resolve_strategy('fsdp') == 'native_fsdp'
    assert resolve_strategy('ddp') == 'accelerate'


def test_resolve_rejects_unknown_names():
    from swift.dev.naming import resolve_loss, resolve_optim, resolve_scheduler, resolve_strategy
    for fn, bad in [(resolve_loss, 'no_such_loss'), (resolve_optim, 'no_such_optim'),
                    (resolve_scheduler, 'no_such_sched'), (resolve_strategy, 'no_such_strategy')]:
        with pytest.raises(NotImplementedError):
            fn(bad)


def test_resolve_unified_entry_dispatch():
    from twinkle.loss import CrossEntropyLoss as TwinkleCE

    from swift.dev.naming import resolve
    assert resolve('loss', 'cross_entropy') is TwinkleCE
    assert resolve('optim', 'adamw') == 'AdamW'
    assert resolve('scheduler', 'cosine') == 'CosineWarmupScheduler'
    assert resolve('strategy', 'fsdp') == 'native_fsdp'
    with pytest.raises(ValueError):
        resolve('no_such_category', 'x')


# ----------------------------------------------------------------------
# GA step-count contract: num_optimizer_steps() must match the number of
# optimizer updates twinkle's do_grad_sync actually performs over N micro-batches.
# Guards two contracts: (1) twinkle's GA boundary lags one micro-step, and
# (2) small-data-under-large-GA yields ZERO optimizer steps (silent no-update).
# Lightweight: pure arithmetic, no GPU/model.
# ----------------------------------------------------------------------


def _do_grad_sync(cur_step: int, ga: int) -> bool:
    """Exact copy of twinkle OptimizerGroup.do_grad_sync predicate."""
    return ga == 1 or ((cur_step - 1) % ga == 0 and cur_step > 1)


def _actual_optimizer_steps(num_micro: int, ga: int) -> int:
    """Count optimizer steps a real SFTLoop takes over num_micro micro-batches.

    Mirrors the loop: each micro-step increments cur_step (backward does this) THEN
    do_grad_sync is evaluated (SFTLoop reads it at the GA boundary).
    """
    cur_step = 0
    steps = 0
    for _ in range(num_micro):
        cur_step += 1
        if _do_grad_sync(cur_step, ga):
            steps += 1
    return steps


def test_num_optimizer_steps_matches_do_grad_sync():
    """num_optimizer_steps() must equal the actual do_grad_sync update count.

    Freezes the twinkle-GA-lag arithmetic so a well-meaning simplification
    (e.g. N // ga) can't silently desync the LR scheduler's num_training_steps
    from the steps the loop actually takes.
    """
    from swift.dev.recipe import num_optimizer_steps

    for ga in (1, 2, 3, 4):
        for num_micro in range(0, 13):
            predicted = num_optimizer_steps(num_micro, ga)
            actual = _actual_optimizer_steps(num_micro, ga)
            assert predicted == actual, (f'ga={ga} num_micro={num_micro}: predicted={predicted} actual={actual}')


def test_small_data_large_ga_yields_zero_steps():
    """Contract: N micro-batches <= ga produce ZERO optimizer steps under twinkle GA.

    twinkle's boundary lags one micro-step (first step at cur_step=ga+1), so a dataset
    that fits within a single GA window trains (forward/backward run) but NEVER updates
    the model. run_sft fail-fasts on this (contract 5, see test_run_sft_zero_opt_steps_raises
    in test_run_sft_e2e.py); here we freeze the arithmetic that detects it.
    """
    from swift.dev.recipe import num_optimizer_steps

    assert num_optimizer_steps(2, 2) == 0  # exactly one GA window -> no update
    assert num_optimizer_steps(1, 2) == 0
    assert num_optimizer_steps(3, 2) == 1  # one micro-step past the window -> first update
    assert num_optimizer_steps(3, 4) == 0  # still inside the first (larger) window


# ----------------------------------------------------------------------
# End-to-end resume: stop@odd-phase (mid GA window) then resume must produce a
# BIT-IDENTICAL parameter trajectory to an uninterrupted run. This is the deepest
# resume guarantee (GA phase + optimizer/scheduler/RNG + data-order all realigned).
# Heavy: real weights + GPU; gated. Scenario b (odd-phase) is the hardest case.
# ----------------------------------------------------------------------


@requires_model
def test_resume_param_trajectory_bit_identical_oddphase(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from datasets import Dataset as HfDataset

    from swift.dev.builders.dataset import _encode
    from swift.dev.legacy_dataloader import build_dataloader, identity_collate

    # Self-contained: drive SFTLoop's contract directly on an in-memory dataset so the
    # test doesn't depend on a registered dataset id.
    from swift.dev.loss import configure_loss
    from swift.dev.optimizer import configure_optimizer
    from swift.dev.processor import InputProcessor
    from swift.dev.recipe import num_optimizer_steps
    from swift.dev.template import DevMixin
    from swift.model import get_model_processor
    from swift.template import get_template

    _, proc = get_model_processor(MODEL, load_model=False)
    ga = 2

    def make_dl():
        tpl = _dev_template(get_template(proc, template_type='qwen2_5', max_length=256))
        tpl.set_mode('train')
        raw = HfDataset.from_list([{
            'messages': [{
                'role': 'user',
                'content': f'Q{i}?'
            }, {
                'role': 'assistant',
                'content': f'Answer number {i} here.'
            }]
        } for i in range(8)])
        enc = _encode(raw, tpl, mode='lazy', num_proc=1, strict=False, data_seed=42)
        return build_dataloader(enc, collate_fn=identity_collate, batch_size=1, shuffle=False, resumable=True)

    def build_from(model_path):
        # NOTE: FULL-PARAM resume (方向 X): weights load from the given path (ckpt dir on
        # resume). resume_from_checkpoint does NOT reload full weights, only optim/sched/RNG.
        # Construct TransformersModel directly with mixed_precision='no' (NOT via build_model,
        # which defaults to 'bf16'): accelerate's AcceleratorState is a process-global singleton
        # that can't switch mixed_precision once other fp32/'no' GPU tests in this file have
        # initialized it. fp32 + 'no' is also what a bit-exact param-trajectory check requires.
        from swift.dev.model import TransformersModel
        torch.manual_seed(0)
        m = TransformersModel(model_id=model_path, mixed_precision='no', strategy='accelerate', dtype=torch.float32)
        m.set_processor(InputProcessor())
        configure_loss(m)
        configure_optimizer(m, _TinyTrainCfg(), num_training_steps=num_optimizer_steps(6, ga))
        return m

    def fresh_model():
        return build_from(MODEL)

    def params(m):
        raw = m.strategy.unwrap_model(m.model)
        return {n: p.detach().float().cpu().clone() for n, p in raw.named_parameters()}

    # --- uninterrupted reference: 6 micro-steps (=> 2 optimizer steps under GA lag) ---
    ref = fresh_model()
    _drive_micro(ref, make_dl(), n_micro=6, ga=ga)
    ref_params = params(ref)

    # --- interrupted at odd phase (3 micro-steps => stop mid 2nd window), save, resume ---
    a = fresh_model()
    dl_a = make_dl()
    _drive_micro(a, dl_a, n_micro=3, ga=ga)
    ckpt = a.save(
        'checkpoint-3',
        output_dir=str(tmp_path / 'a'),
        save_optimizer=True,
        consumed_train_samples=dl_a.consumed_samples)

    # FULL-PARAM resume: weights come from the ckpt dir (方向 X), NOT from
    # resume_from_checkpoint (which only restores optim/sched/RNG/cur_step for full-param).
    b = build_from(ckpt)
    state = b.resume_from_checkpoint(ckpt)
    dl_b = make_dl()
    dl_b.skip_consumed_samples(state['consumed_train_samples'])
    _drive_micro(b, dl_b, n_micro=3, ga=ga)  # remaining 3 micro-steps -> total 6
    res_params = params(b)

    max_diff = max((ref_params[n] - res_params[n]).abs().max().item() for n in ref_params)
    assert max_diff < 1e-5, f'resume param trajectory diverged: max|diff|={max_diff:.3e}'


class _TinyTrainCfg:
    learning_rate = 1e-4
    optim = 'adamw'
    weight_decay = 0.0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    lr_scheduler_type = 'cosine'
    warmup_ratio = 0.0


def _drive_micro(model, dataloader, *, n_micro, ga):
    """Drive exactly n_micro micro-steps through the SFTLoop contract (no epoch wrapping)."""
    it = iter(dataloader)
    for _ in range(n_micro):
        batch = next(it)
        model.forward_backward(inputs=batch, gradient_accumulation_steps=ga)
        model.clip_grad_and_step(max_grad_norm=1.0, gradient_accumulation_steps=ga)


class _TinyTunerCfg:
    tuner_type = 'lora'
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.0
    lora_bias = 'none'
    target_modules = ['all-linear']
    target_regex = None
    modules_to_save = None
    use_rslora = False
    use_dora = False


# ----------------------------------------------------------------------
# LoRA resume: the resumed run must bit-align to an uninterrupted run.
# STRONGER than "same-seed align": phase2's apply_tuner uses a DIFFERENT random
# seed than phase1/reference. If resume's adapter load is complete, that seed is
# irrelevant and the LoRA trajectory is still bit-identical. If it only aligned
# when seeds matched, the adapter load would be incomplete (a real product bug).
# Guards the twinkle LoRA resume contract (adapter_name='default', has_adapter
# branch of resume_from_checkpoint). GA odd-phase (stop@3) is the hardest case.
# ----------------------------------------------------------------------
@requires_model
def test_lora_resume_param_trajectory_bit_identical_seed_independent(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from datasets import Dataset as HfDataset

    from swift.dev.adapter import apply_tuner
    from swift.dev.builders.dataset import _encode
    from swift.dev.legacy_dataloader import build_dataloader, identity_collate
    from swift.dev.loss import configure_loss
    from swift.dev.model import TransformersModel
    from swift.dev.optimizer import configure_optimizer
    from swift.dev.processor import InputProcessor
    from swift.dev.recipe import num_optimizer_steps
    from swift.dev.template import DevMixin
    from swift.model import get_model_processor
    from swift.template import get_template

    _, proc = get_model_processor(MODEL, load_model=False)
    ga = 2

    def make_dl():
        tpl = _dev_template(get_template(proc, template_type='qwen2_5', max_length=256))
        tpl.set_mode('train')
        raw = HfDataset.from_list([{
            'messages': [{
                'role': 'user',
                'content': f'Q{i}?'
            }, {
                'role': 'assistant',
                'content': f'Answer number {i} here.'
            }]
        } for i in range(8)])
        enc = _encode(raw, tpl, mode='lazy', num_proc=1, strict=False, data_seed=42)
        return build_dataloader(enc, collate_fn=identity_collate, batch_size=1, shuffle=False, resumable=True)

    def build_lora(model_path, *, tuner_seed):
        m = TransformersModel(model_id=model_path, mixed_precision='no', strategy='accelerate', dtype=torch.float32)
        # LoRA lora_A/lora_B random init happens inside apply_tuner, governed by this seed.
        torch.manual_seed(tuner_seed)
        apply_tuner(m, _TinyTunerCfg(), gradient_accumulation_steps=ga)
        m.set_processor(InputProcessor())
        configure_loss(m)
        configure_optimizer(m, _TinyTrainCfg(), num_training_steps=num_optimizer_steps(6, ga))
        return m

    def lora_params(m):
        raw = m.strategy.unwrap_model(m.model)
        return {n: p.detach().float().cpu().clone() for n, p in raw.named_parameters() if 'lora_' in n}

    # reference: seed A, uninterrupted 6 micro-steps (=> 2 optimizer steps under GA lag).
    ref = build_lora(MODEL, tuner_seed=0)
    _drive_micro(ref, make_dl(), n_micro=6, ga=ga)
    ref_p = lora_params(ref)

    # phase1: seed A, 3 micro-steps (odd phase, mid 2nd GA window), save adapter.
    a = build_lora(MODEL, tuner_seed=0)
    dl_a = make_dl()
    _drive_micro(a, dl_a, n_micro=3, ga=ga)
    ckpt = a.save(
        'ckpt-lora-3',
        output_dir=str(tmp_path / 'a'),
        save_optimizer=True,
        consumed_train_samples=dl_a.consumed_samples)

    # phase2: DIFFERENT seed (999). Resume must overwrite the phase2 random init with the
    # saved adapter weights, so this seed must NOT affect the resumed trajectory.
    b = build_lora(MODEL, tuner_seed=999)
    state = b.resume_from_checkpoint(ckpt, adapter_name='default')
    dl_b = make_dl()
    dl_b.skip_consumed_samples(state['consumed_train_samples'])
    _drive_micro(b, dl_b, n_micro=3, ga=ga)
    res_p = lora_params(b)

    common = [n for n in ref_p if n in res_p]
    assert common, 'no LoRA params captured'
    max_diff = max((ref_p[n] - res_p[n]).abs().max().item() for n in common)
    assert max_diff < 1e-5, (f'LoRA resume diverged with phase2 seed!=phase1 seed: max|diff|={max_diff:.3e} '
                             f'over {len(common)} params -> adapter load is INCOMPLETE (product bug)')


# ----------------------------------------------------------------------
# GRPO GA equivalence: GRPO returns num_tokens=0 -> twinkle PER-TOKEN-MEAN
# branch (grad / num_micro_steps). GRPOLoss is per-seq-mean-then-batch-mean, so each
# seq is weighted 1/(seqs in its aggregation). => GA=k (1 seq/step) == bs=k (k seqs)
# ONLY when every micro-batch has the SAME seq count; sensitive to SEQ COUNT, not
# token count. Must be measured in a STRONG-signal region (full-param fp32, large
# advantages, distinct old_logps): near-zero gradients make rel_norm_diff explode
# from fp32 noise (a scaffold trap, not a product bug). We bypass twinkle's GA gate
# timing by manually dividing the accumulated grad by the micro-step count.
# ----------------------------------------------------------------------
_GRPO_SEQ_LEN = 12
_GRPO_PROMPT_LEN = 4


@requires_model
def test_grpo_ga_equivalence_equal_seq_strong_signal():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from swift.dev.data_format import InputFeature, ModelOutput
    from swift.dev.loss import GRPOLoss
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)
    model.train()
    loss_fn = GRPOLoss(epsilon=0.2, beta=0.0)
    V = model.config.vocab_size
    L, P = _GRPO_SEQ_LEN, _GRPO_PROMPT_LEN

    # per-seq deterministic data (bound to seq seed, NOT batch structure) so the bs and
    # GA paths feed IDENTICAL old_logps/advantages for the SAME sequence.
    def seq_ids(seed):
        return torch.randint(0, V, (L, ), generator=torch.Generator().manual_seed(seed))

    def seq_labels(seed):
        lab = seq_ids(seed).clone()
        lab[:P] = -100
        return lab

    def seq_old(seed):
        return -torch.rand(L, generator=torch.Generator().manual_seed(seed + 7)) * 2.0

    def seq_adv(seed):
        return 2.0 if (seed % 2 == 0) else -2.0  # strong +-2

    def loss_for(seeds):
        ids = torch.stack([seq_ids(s) for s in seeds]).to(device)
        labels = torch.stack([seq_labels(s) for s in seeds]).to(device)
        old = torch.stack([seq_old(s) for s in seeds]).to(device)
        adv = torch.tensor([seq_adv(s) for s in seeds], dtype=torch.float32).to(device)
        logits = model(input_ids=ids).logits.float()
        mask = (labels != -100).bool()
        masked = labels.clone()
        masked[~mask] = 0
        logps = torch.log_softmax(logits, -1).gather(-1, masked.unsqueeze(-1)).squeeze(-1)
        return loss_fn(
            InputFeature({'labels': labels}), ModelOutput({'logps': logps}), old_logps=old, advantages=adv)['loss']

    def grads_scaled(seeds, *, accumulate):
        model.zero_grad(set_to_none=True)
        if accumulate:
            for s in seeds:
                loss_for([s]).backward()
            k = len(seeds)
        else:
            loss_for(seeds).backward()
            k = 1
        return {n: (p.grad.detach().float().cpu() / k) for n, p in model.named_parameters() if p.grad is not None}

    seeds = [0, 1, 2, 3]
    ga = grads_scaled(seeds, accumulate=True)  # GA=4, 1 seq/step, /4
    bs = grads_scaled(seeds, accumulate=False)  # bs=4, one batch

    common = [n for n in ga if n in bs]
    max_diff = max((ga[n] - bs[n]).abs().max().item() for n in common)
    max_grad = max(bs[n].abs().max().item() for n in common)
    # Require a STRONG signal (else the test is meaningless / in the noise region).
    assert max_grad > 1e-3, f'gradient signal too weak ({max_grad:.3e}); test would be noise-bound'
    # Equivalence to fp32 precision (relative to the gradient magnitude).
    assert max_diff / max_grad < 1e-3, (
        f'GRPO GA!=bs under EQUAL seq count: max|diff|={max_diff:.3e} on max|grad|={max_grad:.3e} '
        f'(ratio {max_diff / max_grad:.3e}) -> genuine GRPO GA non-equivalence (product bug)')


def _dev_template(legacy):
    """Derive dev's template from an already-built legacy one, exactly as build_template does."""
    from swift.dev.template import shifted_template_class
    legacy.__class__ = shifted_template_class(type(legacy))
    return legacy
