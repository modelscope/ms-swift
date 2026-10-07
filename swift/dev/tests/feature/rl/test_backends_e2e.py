# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dimension ② -- model type + backend equivalence (RL_PLAN §四/§五 test_backends_e2e).

Basic principle 1 says the ``megatron`` and ``transformers`` training backends are IDENTICAL except for
in-place ``generate`` (a TransformersModel-only capability, and one RL never needs: on-policy generation
always rides the independent-sampler + per-step-weight-sync route both backends share). This file attacks
that claim end to end -- it does NOT trust it. Every test drives the real ``run_rl`` CLI lifecycle on the
megatron backend (real Ray actors, real MegatronModel sharded across ranks, a real vLLM sampler for the
online path) and asserts the SAME contract the transformers backend must satisfy.

Why DPO carries the equivalence oracle: its first loss is an INDEPENDENT analytic value, not a
self-consistency round-trip. With an identity adapter (``lora_B``=0 at init) the policy equals its own
disabled-adapter reference, so the chosen/rejected log-ratio difference is exactly 0 and
``-log(sigmoid(beta*0)) == ln2`` on EITHER backend, for any batch, DP/TP layout, or RNG stream. Both
backends landing on ln2 at step 1 is therefore proof they compute the same reference-contrastive objective
-- a backend-specific branch, a wrong reduction, or a sharding-induced logps error moves one off ln2. This
is exactly the anchor that caught two real megatron-only defects this suite fixed:

* the base ``OptimizerGroup.accumulate_metrics`` forwarded ``gradient_accumulation_steps`` twice (once
  explicitly, once inside ``**forward_kwargs``), so ANY megatron run that threads gradient accumulation
  crashed with ``TypeError: got multiple values`` -- the transformers subclass popped it, megatron did not;
* ``MegatronStrategy.reduce_loss`` reported ``local_loss.detach()`` (a storage-SHARING view) as the metric
  loss, and Megatron-core scales that same tensor in place by ``1/num_microbatches`` for the backward, so
  the logged loss was silently halved whenever ``num_microbatches>1`` for a mean-reduced (``num_tokens=0``)
  loss like DPO. ``.clone()`` snapshots the true value first.

Both defects are invisible at ``num_microbatches==1`` / batch 1 -- which is why the single-card transformers
DPO test never saw them, and why these megatron runs use batch 2 (a real multi-microbatch config).

Gated ``slow`` + ``accel(2)`` (twinkle's MegatronModel requires ``world_size>=2``). Run on two free cards:
``CUDA_VISIBLE_DEVICES=<c1>,<c2> pytest swift/dev/tests/feature/rl/test_backends_e2e.py -m slow``.
"""
import math

import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl

_LN2 = math.log(2.0)

# Megatron needs world_size>=2, so its runs use nproc=2; per_device_train_batch_size must be >= the DP
# size (twinkle slice_dp splits each driver batch across the DP ranks). optim is reset to its default
# (adamw_torch_fused) because rl_configs otherwise sets the transformers-only optim='adamw', which the
# _HF_ONLY backend guard rejects on megatron. fp32 keeps the loss comparable across bridges without a wide
# bf16 band. num_microbatches = per_device_train_batch_size / micro_batch_size(1) = 2 here, which is what
# exercises the multi-microbatch loss-reporting path the docstring describes.
_MEGATRON_NPROC = 2
_MEGATRON_BS = 2


def _megatron_train_over(**extra):
    return {'per_device_train_batch_size': _MEGATRON_BS, 'optim': 'adamw_torch_fused', **extra}


def _varied_content_reward(completions, **kwargs):
    """Deterministic content-keyed ORM reward (char-sum mod 11, scaled to [0,1)).

    Same rationale as the algorithms suite: GRPO's advantage is ``(r - group_mean)/group_std``, so a
    constant reward across a prompt's completions gives std=0 -> zero advantage -> ``lora_B`` never moves.
    Keying on content makes most groups' completions differ, so the megatron online run really trains.
    """
    return [float(sum(ord(ch) for ch in completion) % 11) / 11.0 for completion in completions]


@pytest.mark.slow
@pytest.mark.accel(2)
def test_backend_equivalence_dpo_identity_ln2(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Contract: DPO's step-1 loss is ln2 on BOTH backends, and both really train + persist.

    Runs the identical tiny model / preference data / seed through the transformers backend and the
    megatron backend and anchors each on the identity-adapter ln2 oracle. Because ln2 is independent of
    backend, batch, DP/TP layout and RNG, the two backends agreeing on it (to 1e-3) is direct evidence the
    reference-contrastive DPO objective is computed identically across the sharded megatron path and the
    in-process transformers path -- no backend-specific branch, no reduction divergence, no sharding-induced
    logps error. Step 2 then drifts apart slightly (different optimizer/RNG consumption), which is expected
    and is why deep per-step parity is anchored against legacy in ``test_parity_legacy``, not here.

    Failure it kills: a megatron-only loss corruption (the two defects in the module docstring), or any
    code path that treats the backends as non-equivalent. Reverse-verify: revert ``reduce_loss`` to
    ``local_loss.detach()`` and the megatron step-1 loss reads ln2/2 (num_microbatches=2), tripping both the
    per-backend ln2 assertion and the cross-backend agreement assertion.
    """
    losses_by_backend = {}
    for backend in (None, 'megatron'):
        name = backend or 'transformers'
        is_meg = backend == 'megatron'
        configs = rl_configs(
            rlhf_type='dpo',
            model=tiny_qwen2_5,
            model_type='qwen2',
            template='qwen2_5',
            dataset=rl_data.preference(n=4),
            out_dir=str(tmp_path / f'out_dpo_{name}'),
            backend=backend,
            nproc=_MEGATRON_NPROC if is_meg else 1,
            max_steps=2,
            # fp32 on both so the equivalence is not masked by a wide bf16 tolerance band.
            model_over={'torch_dtype': 'float32'},
            train_over=_megatron_train_over() if is_meg else None,
        )
        history = run_rl(configs)
        losses = assert_rl_trained(history, configs, f'dpo[{name}]', max_loss=2.0)
        assert abs(losses[0] - _LN2) < 1e-3, (
            f'dpo[{name}] step-1 loss {losses[0]:.5f} != ln2 {_LN2:.5f}: the identity-adapter '
            'policy/reference contrast is not cancelling on this backend (backend-specific loss branch, '
            'wrong reduction, or a sharding/microbatch scaling error -- see module docstring)')
        losses_by_backend[name] = losses

    hf, meg = losses_by_backend['transformers'], losses_by_backend['megatron']
    assert abs(hf[0] - meg[0]) < 1e-3, (
        f'backend equivalence violated at step 1: transformers {hf[0]:.5f} vs megatron {meg[0]:.5f} '
        '(both must equal the analytic ln2 for an identity adapter)')


@pytest.mark.slow
@pytest.mark.accel(2)
def test_reward_model_seq_cls_backend_equivalent(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Contract: the ``rm`` (seq_cls, num_labels=1 scalar head) model type trains on BOTH backends.

    A reward model is a different MODEL TYPE from causal_lm: no vocab logits, no next-token logps -- a
    scalar score pooled per sequence, trained with ``RewardLoss``. Basic principle 1 makes it
    backend-agnostic (``forward_only``/``forward_backward``/``calculate_loss`` are the shared
    ``TrainableModel`` interface), so ``rm`` must run identically on megatron and transformers. Both runs
    must persist a checkpoint whose ``args.json`` carries ``task_type='seq_cls'`` -- the force_load key that
    keeps ``swift infer`` from silently downgrading the trained reward head back to causal_lm.

    Failure it kills: a seq_cls head wired only for transformers (a G4 special case), or the megatron
    seq_cls pooling branch (last-valid-token pick) missing so the scalar head reads vocab logits. This is
    the same seq_cls routing the PPO reward-model fix (``task='seq_cls'`` on ``forward_only``) depended on.
    """
    for backend in (None, 'megatron'):
        name = backend or 'transformers'
        is_meg = backend == 'megatron'
        configs = rl_configs(
            rlhf_type='rm',
            model=tiny_qwen2_5,
            model_type='qwen2',
            template='qwen2_5',
            dataset=rl_data.preference(n=4),
            out_dir=str(tmp_path / f'out_rm_{name}'),
            backend=backend,
            nproc=_MEGATRON_NPROC if is_meg else 1,
            max_steps=2,
            model_over={'torch_dtype': 'float32'} if is_meg else None,
            train_over=_megatron_train_over() if is_meg else None,
        )
        history = run_rl(configs)
        assert_rl_trained(
            history, configs, f'rm[{name}]', max_loss=10.0, expect_task_type='seq_cls')


@pytest.mark.slow
@pytest.mark.accel(2)
def test_megatron_moe_dpo_trains(tiny_qwen3_moe, rl_data, tmp_path, assert_rl_trained):
    """Contract: a MoE model type trains on the megatron backend (MoE is NOT transformers-only).

    G4 forbids "this technique supports transformers but not megatron" special cases. A tiny Qwen3-MoE
    (4 experts, top-2) is trained with DPO on megatron, where the experts are sharded/named under
    Megatron's own parameter layout and the router runs mcore-bridge's TopKRouter. It must train (finite,
    normalized loss), move ``lora_B``, and persist a self-describing checkpoint -- proving the MoE model
    type reaches the megatron training path rather than being refused or silently degraded to dense.

    DPO (offline) is the vehicle so the assertion isolates the MoE-on-megatron training path from rollout
    complexity; the router-replay (R2/R3) dimensions are exercised separately in test_router_replay_e2e.
    """
    configs = rl_configs(
        rlhf_type='dpo',
        model=tiny_qwen3_moe,
        model_type='qwen3_moe',
        template='qwen3',
        dataset=rl_data.preference(n=4),
        out_dir=str(tmp_path / 'out_moe_megatron'),
        backend='megatron',
        nproc=_MEGATRON_NPROC,
        max_steps=2,
        model_over={'torch_dtype': 'float32'},
        train_over=_megatron_train_over(),
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'dpo[moe-megatron]', max_loss=2.0)


@pytest.mark.slow
@pytest.mark.accel(2)
def test_megatron_grpo_online_rollout_and_weight_sync(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Contract: the ONLINE path (independent vLLM sampler + per-step weight sync) runs on megatron.

    This is the load-bearing backend-equivalence claim of basic principle 1: on-policy generation never uses
    in-place ``generate`` (which MegatronModel refuses by design); it rolls out on a separate sampler and
    syncs the trained policy weights into it every step -- a route both backends share because both mix in
    ``CheckpointEngineMixin``. A megatron policy (sharded across TP/PP ranks, Megatron parameter names) must
    therefore drive a real colocated vLLM rollout, receive the synced weights, and take real policy-gradient
    steps.

    Post-conditions beyond the universal ones: the history carries the streaming rollout driver's
    fingerprint (``stream_publishes``/``version_span_mean`` -- absent if GRPO degraded to a non-rollout
    loop) and ``lora_B`` moved (a non-zero group-relative advantage back-propagated through the megatron
    policy). The exact loss is NOT compared to the transformers backend here: a temperature>0 rollout is
    sampled, so the two backends' completions differ by construction -- the anchor is that the megatron
    online path trains at all, with the rollout+sync machinery live.

    Failure it kills: a megatron policy that cannot sync weights into the sampler, or an online path
    quietly restricted to transformers (a G4 special case).
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=str(tmp_path / 'out_grpo_megatron'),
        backend='megatron',
        nproc=_MEGATRON_NPROC,
        max_steps=2,
        model_over={'torch_dtype': 'float32'},
        train_over=_megatron_train_over(),
        rlhf_over={'num_generations': 2, 'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(
        history, configs, 'grpo[megatron]', max_loss=20.0,
        require_keys=('stream_publishes', 'version_span_mean'))
