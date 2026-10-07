# Copyright (c) ModelScope Contributors. All rights reserved.
"""Router-replay dimension (RL_PLAN §四/⑦): keeping an MoE policy's importance ratio honest by REPLAYING
the expert routing that actually produced each rollout token, instead of recomputing it under drifted
weights.

An MoE router picks experts from the CURRENT weights, so a training forward that recomputes routing after
the policy has moved selects different experts than the ones that generated the token -- the recomputed
log-prob (and thus the GRPO ratio) is measured against the wrong expert mixture. Routing replay fixes this:
the training forward is driven with ``router_replay_action='replay_forward'`` and each row's
``encoded['routed_experts']`` so it REPLAYS the generating routing. Two modes differ only in WHERE the
routing comes from, then feed the SAME replay path (``grpo.py::_prepare_routing_replay``):

* ``R2`` -- RECORD the training model's OWN routing for the rollout tokens (a ``forward_only`` under the
  just-synced generating weights, a faithful proxy when the sampler did not export routing), then replay it.
* ``R3`` -- replay the routing the SAMPLER exported at generation time (``SampledSequence.routed_experts``,
  lifted by ``rollout._seq_routed_experts``); needs a vLLM build + MoE checkpoint that report routing
  (``enable_return_routed_experts``).

A per-token ``replay_mask`` (== ``completion_mask``) rides alongside ``routed_experts`` so only the causal
rows that produce response log-probs are rewritten; twinkle's ``processor.align_routed_experts`` pads both
to the ``input_ids`` length and clamps the sampler's short-by-one last routing row. ``enable_router_replay``
is threaded to BOTH backends (``builders/model.py`` TransformersModel + MegatronModel), so R2/R3 are
backend-agnostic (basic principle 1). ``validate._check_router_replay`` guards it GRPO-only and CHORD-free;
the MoE-only (R2) and sampler-export (R3) requirements are runtime errors in the loop (validate has no
model/engine to inspect).

PLAN table -- every router-replay corner case, mapped to its owner:

| # | case | input | expected (contract) | where covered | failure it kills |
|---|------|-------|---------------------|---------------|------------------|
| 1 | R2 RECORD + REPLAY (transformers) | tiny Qwen3-MoE, GRPO, ``router_replay_mode='R2'``, colocate vLLM | trains (finite normalized loss, non-zero ``lora_B`` from disk); the RECORD forward returns routing or the loop raises -- so a green run proves routing was captured and replayed | **THIS FILE** ``test_r2_record_transformers_moe_grpo`` | R2 silently recomputes routing (ratio corrupted) / the RECORD wiring is dead (``enable_router_replay`` never set on the model) |
| 1b | R2 RECORD + REPLAY (megatron, backend equivalence) | same on the megatron backend | trains identically -- R2 is NOT transformers-only | **THIS FILE** ``test_r2_record_megatron_moe_grpo`` (accel(2)) | a G4 special case ("routing replay supports transformers but not megatron") |
| 2 | R3 without an exporting sampler -> fail loudly | tiny MoE, GRPO, ``router_replay_mode='R3'``, default vLLM (``enable_return_routed_experts`` off) | ``RuntimeError`` naming the missing ``routed_experts`` and pointing at R2 | **THIS FILE** ``test_r3_requires_exporting_sampler`` | R3 silently recomputes routing when the sampler exported none (the exact corruption replay exists to prevent) |
| 3 | R2 on a DENSE policy -> fail loudly | tiny dense Qwen2.5, GRPO, ``router_replay_mode='R2'`` | ``RuntimeError`` ("a dense policy has no expert routing to record") | **THIS FILE** ``test_r2_requires_moe_policy`` | R2 silently no-ops on a dense policy instead of refusing |
| 4 | R3 replays sampler routing + mask blend | tiny MoE, GRPO, ``router_replay_mode='R3'`` + ``vllm_engine_kwargs={'enable_return_routed_experts': True}`` | trains; the sampler's ``routed_experts`` is carried into ``encoded`` with ``replay_mask == completion_mask`` | **THIS FILE** ``test_r3_replays_sampler_routing`` -- a REAL run: the installed vLLM 0.23.0 DOES export routing for the tiny Qwen3-MoE (verified at runtime, see note) | ``replay_mask`` blend wrong / the exported ``routed_experts`` dropped between sampler and training forward |
| 5 | PP>1 / VPP / CP / SP routing reconstruction | megatron MoE under pipeline / virtual-pipeline / context / sequence parallelism | all-layer routing reconstructed, non-contiguous ``layer_number`` addressed, CP/SP-aware local top-k slice | **N/A in this suite** (see note) | a layout/schedule reorganization addressing the wrong router |

Feasibility / N-A notes (RL_PLAN §九 discipline -- mark N/A with a reason, do not fake coverage):

* Row 4 IS a real end-to-end run in this env -- CONFIRMED at runtime, not assumed. The installed vLLM 0.23.0
  exports ``routed_experts`` for the tiny Qwen3-MoE once ``enable_return_routed_experts`` is on: dev threads the
  kwarg verbatim (``build_engine_args`` spreads ``RolloutConfig.vllm_engine_kwargs`` into the engine args, twinkle
  merges them into ``AsyncEngineArgs``), and vLLM's capture is GENERIC -- ``_bind_routed_experts_capturer`` hooks
  EVERY ``FusedMoE`` layer whose router is a ``BaseRouter`` (``select_experts`` fires the hook on the native
  ``topk_ids``), and Qwen3-MoE builds its experts as a stock ``FusedMoE`` (the capturer's ``get_num_experts`` even
  names Qwen3-MoE). The run log shows the capturer initializing for this checkpoint (``enable_return_routed_
  experts: True`` + ``RoutedExpertsManager ... layers=4, top_k=2``) and the test trains ``lora_B`` in ~90s, so R3
  does NOT raise 'did not export' here (contrast row 2). The R3 CONSUMER mechanism (``align_routed_experts``
  pad/clamp) stays unit-anchored by ``component/processor/test_packing.py::test_align_routed_experts_uses_
  padded_length_not_stale_cache``; row 4 drives it end to end with real sampler-exported routing.
* Row 5 is a multi-GPU megatron PARALLEL-layout reconstruction (PP>1 / VPP / CP / SP). Per RL_PLAN §九 the
  overweight parallel e2e is gated ``accel(N)`` + ``slow`` and "有 UT 即可、先不跑" on a capability machine;
  the CP/SP-aware slice (``get_local_topk_idx_for_current_rank``) and the R2 RECORD-per-microbatch glue were
  already GPU-verified upstream, and the tiny single/dual-card CI env cannot express PP>1/VPP. Marked N/A
  here rather than faked with a degenerate PP=1 run that would not touch the reconstruction.

Reverse-verification: row 1/1b go red if the RECORD wiring is removed (``enable_router_replay`` not set ->
the RECORD forward returns no ``routed_experts`` -> the loop raises) or if training diverges on corrupted
routing. Rows 2/3 go red (i.e. the expected ``RuntimeError`` does NOT fire) if the loud guard is removed and
R3/R2 silently recompute instead. Row 4 was reverse-verified in-session BOTH ways: with the export knob off it
IS row 2 (raises 'did not export'); and with the knob ON but ``rollout._seq_routed_experts`` forced to drop the
routing, row 4 goes red with the SAME 'did not export' raise (2 of 2 samples carry none) -- so its green run
depends on the dev-side lift actually carrying the sampler-exported routing into ``encoded``, not a trivial pass
(restoring the lift returns it to green).

LoRA target scoping (why rows 1/2 name the attention projections instead of ``all-linear``): the R2/R3
mechanism under test is the ROUTER's record/replay, which is independent of which modules carry the LoRA --
the router gate stays native either way, and an attention LoRA still trains the policy and moves ``lora_B``.
Naming ``q/k/v/o_proj`` explicitly (rather than the ``all-linear`` shorthand) is deliberate and keeps the
file portable across peft builds, because ``all-linear`` on this packed-expert Qwen3-MoE is fragile from two
independent directions:

* site-packages peft 0.19.0 mis-resolves MoE ``all-linear`` -- its ``_maybe_include_all_linear_layers`` leaves
  the shorthand unresolved, so the matcher sees ``set('all-linear')`` (the literal characters) and raises
  ``ValueError: Target modules {'a','l',...} not found``, which would mask row 2's expected ``RuntimeError``
  and leave row 1's ``lora_B`` at its zero init;
* the in-repo peft fork (0.20.1.dev0) DOES resolve it, but to attention ``target_modules`` PLUS the packed
  3D expert weights as ``target_parameters`` (peft ``ParamWrapper``); with the experts wrapped, twinkle's
  transformers R2 RECORD setup (``_wrap_all_moe_blocks`` -> ``_get_top_k``) can no longer read ``top_k`` off
  the MoE block and raises ``MoE block must define top_k``.

Neither is a dev/twinkle defect in the routing-replay path (dev passes the standard shorthand; the second is
an expert-param-LoRA x router-replay interaction orthogonal to this dimension). Explicit attention targeting
sidesteps both, so the file runs on stock site-packages peft with no fork prerequisite. Row 1b (megatron)
keeps ``all-linear``: megatron builds LoRA via mcore-bridge (not peft ``get_peft_model``) and replays routing
via mcore's ``moe_enable_routing_replay``, so neither failure mode applies there.

Run with ``CUDA_VISIBLE_DEVICES=<cards> pytest swift/dev/tests/feature/rl/test_router_replay_e2e.py -m slow``.
"""
import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl


@pytest.mark.slow
@pytest.mark.accel(1)
def test_r2_record_transformers_moe_grpo(tiny_qwen3_moe, rl_data, tmp_path, assert_rl_trained):
    """Row 1: R2 RECORDs the training model's own MoE routing and REPLAYs it, on the transformers backend.

    A tiny Qwen3-MoE (4 experts, top-2) runs GRPO with ``router_replay_mode='R2'``. ``assembly`` sets
    ``enable_router_replay`` on the policy (it is a trainable-policy concern), so each window's
    ``_prepare_routing_replay`` -> ``_record_routing`` issues a ``forward_only(router_replay_action=
    'record')`` that MUST return ``routed_experts`` (a dense policy or a dead wiring returns none and the
    loop raises ``RuntimeError``), slices it to each row's ``input_ids`` length and sets
    ``replay_mask = completion_mask``; the training ``forward_backward`` then replays that routing. A green
    run is therefore evidence the RECORD->REPLAY path really executed (it cannot silently skip: the RECORD
    is fail-loudly), and ``assert_rl_trained`` reads a non-zero ``lora_B`` back from disk to confirm the
    replayed-routing update actually moved the adapter.

    ``max_steps=4`` over ``num_generations=2`` spans two windows (two RECORD passes) and two groups, so a
    single zero-variance group (both completions drawing the same content reward) cannot leave ``lora_B``
    untouched. Colocate vLLM is the forced default placement; the sampler only generates (R2 records on the
    training model, so it needs no routing export).
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen3_moe,
        model_type='qwen3_moe',
        template='qwen3',
        dataset=rl_data.prompt_only(n=6),
        out_dir=str(tmp_path / 'out_r2_transformers'),
        nproc=1,
        max_steps=4,
        # Target the attention projections explicitly rather than the 'all-linear' shorthand. Qwen3-MoE
        # stores its experts as PACKED 3D parameters (not nn.Linear), so the peft fork resolves MoE
        # 'all-linear' to attention target_modules PLUS expert target_parameters (peft ParamWrapper); with
        # the experts wrapped, twinkle's transformers R2 RECORD setup (_wrap_all_moe_blocks -> _get_top_k)
        # can no longer read top_k off the MoE block and raises 'MoE block must define top_k'. That is an
        # expert-param-LoRA x router-replay interaction, orthogonal to this dimension -- naming the
        # attention projections keeps the LoRA off the packed experts so the R2 RECORD->REPLAY mechanism
        # (the thing under test) runs against the native router, cleanly and with no background error.
        tuner_over={'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']},
        rlhf_over={'num_generations': 2, 'orm': [_varied_content_reward], 'router_replay_mode': 'R2'},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo[r2-transformers]', max_loss=20.0)


@pytest.mark.slow
@pytest.mark.accel(2)
def test_r2_record_megatron_moe_grpo(tiny_qwen3_moe, rl_data, tmp_path, assert_rl_trained):
    """Row 1b: the SAME R2 RECORD->REPLAY contract on the megatron backend (routing replay is backend-agnostic).

    Basic principle 1 forbids a "supports transformers but not megatron" special case (G4). On megatron the
    RECORD rides ``forward_backward(forward_only=True)`` with mcore's ``moe_enable_routing_replay``
    (translated from ``enable_router_replay`` in ``builders/model.py``), and the replay is driven per
    microbatch (REPLAY_FORWARD, with the backward recomputation switching each router to REPLAY_BACKWARD).
    This drives the real megatron MoE (experts sharded under Megatron's layout, mcore-bridge ``TopKRouter``)
    with a colocate vLLM rollout and asserts the identical post-condition row 1 does. Megatron needs
    ``world_size>=2``, so ``nproc=2`` and ``per_device_train_batch_size=2`` (mirrors ``test_backends_e2e``'s
    MoE-megatron config); fp32 keeps the multi-microbatch loss reporting on the path that suite anchors.
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen3_moe,
        model_type='qwen3_moe',
        template='qwen3',
        dataset=rl_data.prompt_only(n=6),
        out_dir=str(tmp_path / 'out_r2_megatron'),
        backend='megatron',
        nproc=2,
        max_steps=4,
        model_over={'torch_dtype': 'float32'},
        train_over={'per_device_train_batch_size': 2, 'optim': 'adamw_torch_fused'},
        rlhf_over={'num_generations': 2, 'orm': [_varied_content_reward], 'router_replay_mode': 'R2'},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo[r2-megatron]', max_loss=20.0)


@pytest.mark.slow
@pytest.mark.accel(1)
def test_r3_requires_exporting_sampler(tiny_qwen3_moe, rl_data, tmp_path):
    """Row 2: R3 with a sampler that exports no routing fails loudly instead of silently recomputing.

    ``router_replay_mode='R3'`` promises to replay the SAMPLER's generating-pass routing. dev's rollout does
    not turn on ``enable_return_routed_experts`` by default, so the collected samples carry no
    ``routed_experts``; ``_prepare_routing_replay`` must then RAISE (naming the missing routing and pointing
    at R2) rather than fall back to recomputing -- recomputing is exactly the ratio corruption routing
    replay exists to prevent, so a silent fallback would be the worst outcome. The run reaches the raise
    only after a real rollout produced samples, so this drives the genuine R3 consumer seam.
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen3_moe,
        model_type='qwen3_moe',
        template='qwen3',
        dataset=rl_data.prompt_only(n=4),
        out_dir=str(tmp_path / 'out_r3_no_export'),
        nproc=1,
        max_steps=2,
        # Attention-only LoRA (see row 1 note): keeps the packed experts unwrapped so the run reaches the
        # R3 'did not export' guard cleanly instead of tripping the expert-LoRA x router-replay top_k error.
        tuner_over={'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']},
        rlhf_over={'num_generations': 2, 'orm': [_varied_content_reward], 'router_replay_mode': 'R3'},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    with pytest.raises(RuntimeError, match='did not export'):
        run_rl(configs)


@pytest.mark.slow
@pytest.mark.accel(1)
def test_r2_requires_moe_policy(tiny_qwen2_5, rl_data, tmp_path):
    """Row 3: R2 on a DENSE policy fails loudly -- there is no expert routing to RECORD.

    ``router_replay_mode='R2'`` asks the training model to RECORD its MoE routing. A dense Qwen2.5 has no
    experts, so the RECORD forward returns no ``routed_experts`` and ``_record_routing`` must RAISE ("a dense
    policy has no expert routing to record") rather than proceed with a silent no-op replay. This is the
    runtime MoE-only guard ``validate._check_router_replay`` deliberately defers to the loop (validate has no
    model to inspect), so it is only observable end to end.
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=4),
        out_dir=str(tmp_path / 'out_r2_dense'),
        nproc=1,
        max_steps=2,
        rlhf_over={'num_generations': 2, 'orm': [_varied_content_reward], 'router_replay_mode': 'R2'},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    with pytest.raises(RuntimeError, match='RECORD|dense policy|expert routing'):
        run_rl(configs)


@pytest.mark.slow
@pytest.mark.accel(1)
def test_r3_replays_sampler_routing(tiny_qwen3_moe, rl_data, tmp_path, assert_rl_trained):
    """Row 4: R3 REPLAYS the sampler's exported routing end to end -- the positive counterpart of row 2.

    Row 2 pins the loud refusal when the engine exports NO routing (``enable_return_routed_experts`` off ->
    ``_prepare_routing_replay`` counts the missing ``encoded['routed_experts']`` and raises 'did not export').
    This row turns that ONE knob on and is otherwise byte-identical to row 1's R2 transformers run (same tiny
    Qwen3-MoE, same attention-only LoRA, same GRPO colocate recipe), so the delta isolates exactly 'the engine
    exported routing and dev carried it into the training forward'. The export chain, verified against the
    installed vLLM 0.23.0:

    * ``build_engine_args`` spreads ``RolloutConfig.vllm_engine_kwargs`` verbatim into the engine args, which
      twinkle's ``VLLMEngine`` merges into ``AsyncEngineArgs`` (``enable_return_routed_experts`` is a valid
      field, so it survives the signature filter);
    * vLLM's ``_bind_routed_experts_capturer`` attaches a capture hook to EVERY ``FusedMoE`` layer whose router
      is a ``BaseRouter`` (not model-specific), and ``BaseRouter.select_experts`` fires it on the native
      ``topk_ids`` -- Qwen3-MoE builds its experts as a stock ``FusedMoE`` and the capturer's ``get_num_experts``
      names Qwen3-MoE explicitly, so the tiny checkpoint's per-layer routing is captured and returned on the
      ``RequestOutput``;
    * twinkle writes it to ``SampledSequence.routed_experts`` and dev's ``rollout._seq_routed_experts`` lifts it
      into ``encoded['routed_experts']`` alongside ``replay_mask = completion_mask`` (rollout/__init__.py).

    The oracle is row 2's fail-loudly guard read POSITIVELY: ``_prepare_routing_replay`` RAISES if any sample's
    ``encoded['routed_experts']`` is None, so a run that does NOT raise and trains ``lora_B`` (read back from
    ``checkpoint-final``) proves the sampler-exported routing survived the entire lift into ``encoded`` and was
    replayed by the training ``forward_backward`` (``router_replay_action='replay_forward'``, twinkle's
    ``align_routed_experts`` padding it to the ``input_ids`` length and clamping the sampler's short-by-one
    last row). Had the routing been dropped anywhere between vLLM and the training forward, this row would
    raise exactly like row 2 -- so green-here against red-there, differing only in the export knob, is the
    reverse-verification pair. A corrupted ``replay_mask`` blend (wrong frame or length) would instead trip the
    replay forward / GRPO loss length check, so the finite normalized loss + moved ``lora_B`` also pin the blend.

    ``max_steps=4`` over ``num_generations=2`` spans two rollout windows and two groups (mirrors row 1) so a
    single zero-variance group cannot leave ``lora_B`` untouched for a reason unrelated to routing replay.
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen3_moe,
        model_type='qwen3_moe',
        template='qwen3',
        dataset=rl_data.prompt_only(n=6),
        out_dir=str(tmp_path / 'out_r3_transformers'),
        nproc=1,
        max_steps=4,
        # Attention-only LoRA (rows 1/2 note): keeps the packed experts unwrapped so the R3 replay drives the
        # native router cleanly, with no expert-param-LoRA x router-replay top_k interaction in the way.
        tuner_over={'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']},
        # The ONE knob separating this from row 2: ask the vLLM engine to export each MoE layer's routing.
        rollout_over={'vllm_engine_kwargs': {'enable_return_routed_experts': True}},
        rlhf_over={'num_generations': 2, 'orm': [_varied_content_reward], 'router_replay_mode': 'R3'},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo[r3-transformers]', max_loss=20.0)


def _varied_content_reward(completions, **kwargs):
    """Deterministic content-keyed ORM reward (char-sum mod 11, scaled to [0,1)).

    GRPO's advantage is ``(r - group_mean) / group_std``, so a constant reward across a prompt's group gives
    std=0 -> zero advantage -> ``lora_B`` never moves and ``assert_rl_trained`` would fail for a reason
    unrelated to routing replay. Keying on content makes most groups' completions differ (mirrors
    ``test_backends_e2e`` / ``test_placement_e2e``).
    """
    return [float(sum(ord(ch) for ch in completion) % 11) / 11.0 for completion in completions]
