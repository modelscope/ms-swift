"""Isolation layer over twinkle's vLLM sampler for RL rollout.

Purpose (rollout decoupled from the trainer into its own component): the GRPOLoop depends on THIS
interface, never directly on the engine.

Engine base is twinkle's ``vLLMSampler`` (built via :func:`swift.dev.builders.build_sampler`), not
legacy ``swift.infer_engine.GRPOVllmEngine``. twinkle's sampler already speaks the twinkle Template
contract, returns per-token logprobs natively (``sequence.logprobs``), exposes the prompt tokens vLLM
conditioned on (``response.prompt_token_ids``), and owns multimodal placeholder logic + LoRA routing --
so no swift engine / decode shim is needed. This makes dev's rollout twinkle-first like
``run_infer`` / ``run_deploy``; the ``run_grpo`` weight-syncing ``SyncableRollout``
subclasses :class:`RolloutEngine`, so the rollout ``generate`` contract lives in exactly one place.

The rollout needs per-token logprobs for old_logps (contract 15), which the weight-syncable engines
(vLLM and SGLang, both with a ``CheckpointEngineMixin``) provide; the RL recipes pick between them via
``RolloutConfig.rollout_sampler`` (see :func:`swift.dev.recipe.run_grpo._sampler_backend`). This base
engine's own convenience constructor builds vLLM (its default), and deliberately exposes no
multi-backend dispatch shell here -- only ``generate(prompts, num_samples)``. The orthogonal rollout
dimension that DOES vary is placement (colocate / disaggregated), not the sampling contract.

Prompt-half decision (read it back from the sampler, do NOT re-encode): the sampler returns
``response.prompt_token_ids`` -- the exact prompt tokens vLLM conditioned on. We build the training
feature from those instead of encoding the prompt a second time on our side, so the feature cannot
drift from what was actually sampled (a second encode would only be *assumed* identical, and any
per-model/template difference in the generation prompt would silently misalign old_logps against the
training forward).

Multimodal: a prompt row's media columns (``images``/``videos``/``audios``/``objects``/``tools``, kept in
``prompt_extras``) are threaded onto the ``Trajectory`` so the sampler's own template encodes them, and the
resulting vision tensors ride back on ``sequence.new_input_feature``. :func:`samples_from_responses` lifts
those media keys into the hand-built training feature verbatim -- they cannot be rebuilt from token ids --
so the policy / reference / teacher forwards all see exactly the media the sampler conditioned on (this is
the twinkle mm-GRPO cookbook's ``sequence.new_input_feature`` path, not a second encode). A text-only row
carries none and is unaffected.

Weight sync: the base :class:`RolloutEngine` does NOT sync weights (vLLM keeps initial weights =>
behaviour policy is stale => NOT algorithmically-correct GRPO — a known intermediate stage). Its
:meth:`RolloutEngine.sync_weights` / :meth:`RolloutEngine.finish_generate` are therefore warn-once
no-ops, so an on-policy loop wired to the base engine says out loud that it is training against stale
weights instead of skipping the sync silently (W14). ``run_grpo``'s ``SyncableRollout`` overrides them
to actually push the trained policy in.
"""

from __future__ import annotations
import copy
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Marker key written into `encoded` when the training feature's labels are already next-token
# shifted (contract 14). Mirrors dev Template.SHIFTED_KEY so the "who shifted" fact is queryable
# instead of relying on a comment. The RL path hand-builds `encoded` (bypassing Template.encode),
# so it MUST set this to record that the shift was applied here.
SHIFTED_KEY = '_labels_shifted'


@dataclass
class RolloutSample:
    """One on-policy rollout trajectory, pre-collation (dev's RL-sample layer).

    This is the "RL training sample" layer (distinct from the vLLM engine-output layer and the
    model-input InputFeature layer). It stays a dev-owned dataclass rather than importing legacy
    ``swift.rl_core.data.OnPolicySample`` (which eagerly pulls ``RolloutOutput`` /
    ``ChatCompletionResponse`` / legacy template at import time — heavy coupling dev is meant to
    avoid). Field names/semantics are aligned to that RL-sample layer so a later merge is a rename,
    not a redesign.

    Fields:
        encoded: full prompt+completion training feature (fed to model.forward_backward). Its labels
            are next-token shifted (contract 14) and it carries ``SHIFTED_KEY=True`` to record that.
        response_token_ids: the policy-produced completion tokens, flat (``List[int]``). twinkle
            accounts a trajectory in one flat array, so dev keeps it one-dimensional too; used for
            reward / length. Multi-turn per-turn slices, when a consumer needs them (teacher
            distillation), are recovered from ``encoded`` labels by splitting contiguous trainable
            runs -- see ``rollout.multi_turn._messages_with_token_ids``.
        rollout_logprobs: per-token logprob under the SAMPLING policy, flat (``List[float]``, GRPO
            old_logps). One entry per ``response_token_ids`` token, in order.
        prompt_id: which prompt group this sample belongs to (group-relative advantage). String id
            (not an index) to match the RL-sample layer and support dynamic/multi-turn sample counts.
        extra: dataset passthrough columns (solution/target/... for reward), kept out of encode.
        decoded: decoded completion text (logging / rule-based reward).
    """
    encoded: dict
    response_token_ids: List[int]
    rollout_logprobs: List[float]
    prompt_id: str
    extra: Dict[str, Any] = field(default_factory=dict)
    decoded: str = ''
    truncated: bool = False
    response_loss_mask: List[int] = field(default_factory=list)
    messages: Optional[List[dict]] = None
    rollout_infos: Dict[str, Any] = field(default_factory=dict)
    #: The sampler's per-token sampling support set (twinkle ``SamplingMask``), carried only when the engine
    #: ran with sampling replay on. Assembled into the top-level ``sampling_masks`` forward_backward kwarg
    #: (a list parallel to the rows), NOT into ``encoded`` -- the loss reads it separately from the labels.
    sampling_mask: Optional[Any] = None
    #: The policy version this trajectory was generated under, stamped by the streaming rollout layer
    #: (:meth:`RolloutEngine.collect_sample`) from the admission pin. 0 for the path that does not track
    #: versions (the blocking :meth:`RolloutEngine.generate`). The streaming driver's ready
    #: buffer carries the authoritative version on its own records; this field floats it up to the
    #: sample so consumers (metrics, staleness audits) can read it without the buffer.
    policy_version: int = 0
    #: partial rollout 的版本跨度：这条 trajectory 从准入到 collect，策略版本推进了几次。仅在流式
    #: collect 侧（控制面）填充——引擎（twinkle MultiTurnRollout / sampler）始终版本无关。
    #: ``adapter_snapshot`` 下恒为 0（整条 episode pin 在一个 frozen adapter_path，逐轮同版本）；
    #: ``in_place`` 下 = collect 时 current_version − 准入 ``policy_version``（episode 被 abort+resume
    #: 到新权重上，跨了版本）。仅指标用途（partial rollout 触发率 / version-span 分布）；staleness 的
    #: 正确性锚点始终是最老的准入 ``policy_version``，与本字段无关。
    version_span: int = 0

    @property
    def input_feature(self) -> dict:
        """Back-compat alias: the training feature is ``encoded`` (was ``input_feature``)."""
        return self.encoded

    @property
    def old_logps(self) -> List[float]:
        """The sampling-policy logprobs, already flat (one per response token)."""
        return list(self.rollout_logprobs)

    @property
    def prompt_index(self) -> str:
        """Back-compat alias for the group id (was an int index, now the ``prompt_id`` string)."""
        return self.prompt_id


@dataclass
class SampleHandle:
    """One in-flight trajectory admitted by :meth:`RolloutEngine.submit_sample` (streaming path).

    One trajectory per submission, so one handle per trajectory. It carries the submit-time facts collect
    needs -- the prompt's media/reward columns (``prompt_extras``), whether per-token logprobs were forced
    (``require_logprobs``), which prompt / which of its ``num_generations`` trajectories this is (so the
    streaming layer can stamp the group id), and the policy version the trajectory was admitted under
    (``policy_version``, from the driver's pin; stamped onto the collected :class:`RolloutSample`).
    """
    submission_id: str
    prompt_idx: Any
    trajectory_idx: int
    prompt_extras: Optional[Dict[str, Any]] = None
    require_logprobs: bool = True
    policy_version: int = 0


def _sampled_token_logprobs(tokens: List[int], logprobs) -> List[float]:
    """Normalize sampler logprobs to one scalar for each sampled token."""
    if logprobs is None:
        return []
    result: List[float] = []
    for token, position in zip(tokens, logprobs):
        if isinstance(position, (int, float)):
            result.append(float(position))
            continue
        candidates = list(position or [])
        matched = next((value for token_id, value in candidates if int(token_id) == int(token)), None)
        if matched is None:
            raise RuntimeError(f'sampled token {token} is missing from its returned logprobs: {candidates!r}.')
        result.append(float(matched))
    return result


def _seq_routed_experts(seq: Any) -> Any:
    """The sampler-reported MoE expert routing for one sampled sequence, or None.

    Like ``rollout_logprobs``, this is a generation-time artifact the training side cannot recompute (the
    policy has since drifted), so R3 routing replay must carry it. vLLM surfaces it either as a top-level
    ``SampledSequence.routed_experts`` or, after the sampler rewraps its engine output, inside
    ``new_input_feature['routed_experts']``; prefer whichever is present. dev rebuilds the training feature
    from tokens rather than trusting ``new_input_feature``, so this is the one field it must lift out
    explicitly. twinkle's ``processor.align_routed_experts`` pads it to the ``input_ids`` length, so it is
    passed through verbatim here.
    """
    routed = getattr(seq, 'routed_experts', None)
    if routed is None:
        feature = getattr(seq, 'new_input_feature', None)
        if isinstance(feature, dict):
            routed = feature.get('routed_experts')
    return routed


#: Prompt-row media columns threaded onto the ``Trajectory`` so the sampler encodes them (mirrors
#: ``builders.dataset.to_trajectory``'s key set). Everything else in ``prompt_extras`` stays put for rewards.
_MULTIMODAL_PROMPT_KEYS = ('images', 'videos', 'audios', 'objects', 'tools')

#: Per-token / grid multimodal side-channels the training forward reads that are NOT in twinkle's
#: ``VLM_CONCAT_FIELDS`` (see ``processor.base`` ``_extract_keys``): they ride alongside the vision tensors
#: and, like them, cannot be rebuilt from token ids.
_MM_FEATURE_EXTRA_KEYS = ('mm_token_type_ids', 'second_per_grid_ts')


def _vision_feature_keys() -> frozenset:
    """The multimodal feature keys to lift from the sampler's ``new_input_feature`` into ``encoded``."""
    from twinkle.processor.base import InputProcessor
    return frozenset(InputProcessor.VLM_CONCAT_FIELDS) | frozenset(_MM_FEATURE_EXTRA_KEYS)


def _prompt_trajectory(messages: List[dict], extra: Optional[Dict[str, Any]]) -> Any:
    """Build the twinkle ``Trajectory`` for one prompt, threading its media columns through.

    A prompt message list on its own is text-only; a multimodal row carries its ``images``/``videos``/
    ``audios`` (plus ``objects``/``tools``) as sibling dataset columns, preserved in ``prompt_extras``.
    ``Trajectory`` holds those keys directly and the sampler's template turns them into the vision tensors
    that ride back on ``new_input_feature`` -- so a VLM prompt is sampled WITH its media instead of as bare
    text. A row with no media yields a plain text trajectory (unchanged behaviour).
    """
    from twinkle.data_format import Trajectory
    trajectory = Trajectory(messages=list(messages))
    if extra:
        for key in _MULTIMODAL_PROMPT_KEYS:
            if extra.get(key):
                trajectory[key] = extra[key]
    return trajectory


def _merge_multimodal_keys(encoded: Dict[str, Any], feature: Any) -> None:
    """Lift the sampler's vision/audio tensors into a hand-built training feature, in place.

    ``samples_from_responses`` rebuilds ``input_ids``/``labels``/``completion_mask`` from tokens (so old_logps
    line up with the training forward), but the media tensors are prompt-anchored artifacts it cannot
    recompute. They ride on the sampler's ``new_input_feature`` -- ``concat_input_feature`` carries the
    prompt's vision fields through and pads ``mm_token_type_ids`` to the full prompt+response length, so they
    already align with the hand-built ``input_ids``. Copy exactly the multimodal keys over, leaving every
    token-derived field untouched; a text-only sample has none, so this is a no-op there.
    """
    if not isinstance(feature, dict):
        return
    for key in _vision_feature_keys():
        if key in feature and key not in encoded:
            encoded[key] = feature[key]


def samples_from_responses(responses: List[Any],
                          prompt_extras: Optional[List[Dict[str, Any]]] = None,
                          *,
                          allow_message_only: bool = False,
                          require_logprobs: bool = True) -> List[RolloutSample]:
    """Build RolloutSamples from twinkle SampleResponses (one group per response).

    The single-turn counterpart of :func:`rollout.multi_turn.trajectory_to_rollout_sample`: a plain
    ``sampler.sample`` returns one ``SampleResponse`` per prompt (holding ``num_samples`` sequences),
    and this turns each sequence into a ``RolloutSample`` carrying the training feature and old_logps.
    ``run_infer`` reuses it so its single-turn path produces the SAME encoded/logprob payload the
    multi-turn engine does, rather than throwing the tokens away and keeping only decoded text.

    The training feature is rebuilt from ``prompt_token_ids`` + ``sequence.tokens`` rather than
    trusting the sampler's own ``new_input_feature`` labelling, so old_logps (``sequence.logprobs``)
    line up with the training forward. The next-token shift is applied HERE (contract 14): twinkle's
    RL forward computes logps via no-shift ``selective_log_softmax(logits, masked_labels)`` where
    ``logits[i]`` predicts ``token[i+1]``, so the masked labels must be next-token shifted or logps
    are off-by-one vs vLLM old_logps and the whole GRPO importance ratio is wrong.

    ``require_logprobs`` says whether the caller forced per-token logprobs. GRPO leaves it True, so a
    missing/short ``sequence.logprobs`` is fatal (see the alignment check below); a plain inference run
    passes False -- it requests no logprobs, so ``sequence.logprobs`` is None and an empty old_logps is
    the expected, correct result rather than an error.
    """
    out: List[RolloutSample] = []
    for pidx, response in enumerate(responses):
        prompt_tokens = list(response.prompt_token_ids or [])
        if not prompt_tokens:
            if allow_message_only:
                # A text-only backend (the ``client`` teacher) returns decoded text but no prompt tokens,
                # so there is no training feature to rebuild -- carry the text/messages instead of raising.
                out.extend(_message_only_samples(response, pidx, prompt_extras))
                continue
            raise RuntimeError('vLLMSampler returned no prompt_token_ids; cannot build the RL training feature. '
                               'The sampler must run with a template set (set_template) so the prompt is encoded.')
        for seq in response.sequences:
            response_tokens = list(seq.tokens or [])
            aligned = [-100] * len(prompt_tokens) + response_tokens
            labels = list(aligned[1:]) + [-100]
            # completion_mask lives on labels' index space (the next-token/output order), NOT input
            # order. GRPOLoss intersects ``(labels != -100) & completion_mask`` position-by-position
            # without un-shifting either (twinkle loss/grpo.py::_resolve_loss_mask), and dev's own
            # template._invoke_post_pipeline rolls labels and completion_mask together -- so the mask
            # must be in the SAME frame as labels. Deriving it from labels keeps the two in lockstep:
            # every policy-produced (trainable) token is scored. Building it in input order instead
            # (``[0]*prompt + [1]*response``) slips it one position ahead of the shifted labels, so the
            # intersection drops the first response token and old_logps (response-only, R values) no
            # longer matches the R-1 surviving positions -- a fatal length mismatch in the loss.
            completion_mask = [0 if label == -100 else 1 for label in labels]
            encoded = {
                'input_ids': prompt_tokens + response_tokens,
                'labels': labels,
                'completion_mask': completion_mask,
                SHIFTED_KEY: True,
            }
            # R3 routing replay: carry the sampler-reported expert routing (if the engine exported it) so the
            # training forward REPLAYS the routing that actually produced these tokens. Absent for a
            # non-exporting engine, where R2 records it on the training model instead.
            routed_experts = _seq_routed_experts(seq)
            if routed_experts is not None:
                encoded['routed_experts'] = routed_experts
            # Multimodal: lift the sampler's vision/audio tensors (pixel_values / image_grid_thw /
            # mm_token_type_ids / ...) into the hand-built feature. They are prompt-anchored and cannot be
            # rebuilt from token ids, so the training / reference / teacher forwards need them verbatim to
            # reproduce the media the sampler conditioned on. No-op for a text-only sample.
            _merge_multimodal_keys(encoded, getattr(seq, 'new_input_feature', None))
            old_logps = _sampled_token_logprobs(response_tokens, seq.logprobs)
            # Enforce alignment only when logprobs were requested (``require_logprobs``). GRPO forces
            # them, so there a length mismatch is raised, never padded: these values ARE old_logps, and
            # 0.0 is a legal logprob (p=1.0), not a sentinel -- padding it would turn a missing-logprob
            # bug into a silently wrong importance ratio exp(logps - 0). A plain inference run requests
            # none (``force_logprobs=False``), the sampler returns ``logprobs=None``, and old_logps is
            # legitimately empty -- the consumer never reads it, so the check is skipped.
            if require_logprobs and len(old_logps) != len(response_tokens):
                raise RuntimeError(f'rollout logprobs misaligned: {len(old_logps)} logprobs for '
                                   f'{len(response_tokens)} tokens. These are old_logps; a mismatch would '
                                   'silently corrupt the GRPO importance ratio, so it is fatal.')
            out.append(
                RolloutSample(
                    encoded=encoded,
                    response_token_ids=response_tokens,
                    rollout_logprobs=old_logps,
                    prompt_id=str(pidx),
                    extra=dict(prompt_extras[pidx]) if prompt_extras and pidx < len(prompt_extras) else {},
                    decoded=seq.decoded or '',
                    truncated=getattr(seq, 'stop_reason', None) == 'length',
                    sampling_mask=getattr(seq, 'sampling_mask', None)))
    return out


def _message_only_samples(response: Any, pidx: int,
                          prompt_extras: Optional[List[Dict[str, Any]]]) -> List[RolloutSample]:
    """Build token-less RolloutSamples for a text-only backend (the ``client`` teacher).

    The single-turn counterpart of the multi-turn engine's message-only mode: a remote teacher returns
    decoded text (and structured tool calls inside ``new_input_feature['messages']``) but no token ids,
    so ``encoded`` / ``response_token_ids`` / ``rollout_logprobs`` stay empty and only ``decoded`` /
    ``messages`` carry anything -- exactly what ``run_infer`` reads. A failed request degrades to an
    empty ``error`` sequence with no ``new_input_feature``; its empty ``decoded`` is dropped downstream.
    """
    extra = dict(prompt_extras[pidx]) if prompt_extras and pidx < len(prompt_extras) else {}
    out: List[RolloutSample] = []
    for seq in response.sequences:
        feature = getattr(seq, 'new_input_feature', None)
        has_messages = isinstance(feature, dict) and feature.get('messages')
        messages = copy.deepcopy(feature['messages']) if has_messages else None
        out.append(
            RolloutSample(
                encoded={},
                response_token_ids=[],
                rollout_logprobs=[],
                prompt_id=str(pidx),
                extra=extra,
                decoded=seq.decoded or '',
                truncated=getattr(seq, 'stop_reason', None) == 'length',
                messages=messages))
    return out


def _build_sampling_params(num_samples: int, sampling_params: Optional[dict], force_logprobs: bool) -> Any:
    """Assemble the twinkle ``SamplingParams`` shared by EVERY rollout entry point.

    One contract for the blocking :meth:`RolloutEngine.generate`, the admitted
    :meth:`RolloutEngine.submit_sample`, and the multi-turn :meth:`MultiTurnRollout.generate`, so no path
    can silently sample under different params than another (the temperature default, the forced old_logps,
    and the per-submission ``num_samples``). ``num_samples`` is an explicit argument because the multi-turn
    engine drives one trajectory per turn and always submits ``1``, while the single-turn blocking path
    submits the caller's group width.
    """
    from twinkle.data_format import SamplingParams
    sp = dict(sampling_params or {})
    sp.setdefault('temperature', 1.0)
    if force_logprobs:
        # Twinkle uses logprobs=1 for the sampled token in its normalized top-k representation.
        # Contract 15: these logprobs ARE old_logps, so requesting them is forced, not defaulted.
        sp['logprobs'] = max(int(sp.get('logprobs') or 0), 1)
    sp['num_samples'] = num_samples
    return SamplingParams(**sp)


class RolloutEngine:
    """Thin wrapper over twinkle's ``vLLMSampler``: prompts in, RolloutSample (training feature +
    old_logps) out. The sampler owns encoding/decoding; this layer only assembles the RL training
    sample. ``run_grpo``'s weight-syncing ``SyncableRollout`` subclasses this, reusing
    :meth:`generate` / :meth:`_samples_from_responses` verbatim and overriding :meth:`sync_weights` /
    :meth:`finish_generate` to actually push the trained policy into the sampler."""

    def __init__(self,
                 model_id: Optional[str] = None,
                 template: Any = None,
                 *,
                 engine_args: Optional[dict] = None,
                 sampler: Any = None):
        self.model_id = model_id
        self.template = template
        self._init_streaming_state()
        if sampler is not None:
            # Injected: the caller built and placed the sampler (``run_infer`` across backends,
            # ``run_grpo``'s SyncableRollout on its own remote_group). This engine borrows it and does NOT
            # own its lifecycle -- close() releases only the multi-turn env pool; shutdown() also stops it.
            self.sampler = sampler
            return
        if model_id is None:
            raise ValueError('RolloutEngine needs either an injected sampler or a model_id to build one.')
        from swift.dev.builders import build_sampler
        from swift.dev.config import ModelConfig
        # build_sampler sets the template on the sampler so Trajectory (messages) inputs are encoded,
        # and returns the prompt tokens the model conditioned on -- exactly the prompt half we need.
        self.sampler = build_sampler(
            ModelConfig(model=model_id), backend='vllm', engine_args=dict(engine_args or {}), template=template)

    def _init_streaming_state(self) -> None:
        """Initialise the sampler-independent streaming/rollout state every ``RolloutEngine`` carries.

        Factored out of :meth:`__init__` so a subclass that BORROWS an already-built sampler and therefore
        deliberately skips ``RolloutEngine.__init__`` (``run_grpo``'s ``SyncableRollout``) still initialises
        these -- ``poll_completions``/``collect_sample``/``cancel_sample``/``close`` read
        ``_episode_futures`` UNCONDITIONALLY (before any ``_multi_turn`` branch), so a subclass that omits it
        crashes with ``AttributeError`` on the first poll even for a plain single-turn stream. Keeping the
        three attributes in one initializer is what stops that drift recurring when a fourth is added.
        """
        self._multi_turn = None
        # In-flight multi-turn episodes admitted by the streaming path: submission_id -> the Future its
        # background thread resolves with the episode's single RolloutSample. Only the driver thread mutates
        # this dict (submit adds, collect/cancel pop); the episode threads touch only their own Future. See
        # :meth:`_submit_episode` for why a multi-turn admit runs on a thread rather than the sampler's
        # non-blocking submit_generation.
        self._episode_futures: Dict[str, Future] = {}
        # One-shot latch for the base engine's "no weight sync" warning (see sync_weights), so a loop that
        # syncs every step does not spam the log.
        self._warned_no_sync = False

    def configure_multi_turn(self,
                             *,
                             max_turns: Optional[int] = None,
                             max_trajectory_tokens: Optional[int] = None,
                             tool_manager: Any = None,
                             harness: Any = None,
                             followup_fn: Any = None,
                             env_pool: Any = None,
                             tool_plugins: Optional[List[Any]] = None) -> None:
        """Attach twinkle's native multi-turn engine (no scheduler, no gym).

        Per-round length is ``sampling_params.max_tokens``; whole-trajectory length is
        ``max_trajectory_tokens``. The extension points are twinkle's ``tool_manager``
        / ``harness`` / ``followup_fn``, wired by the caller. A sandbox ``env_pool`` plus
        ``tool_plugins`` (see :mod:`swift.dev.rollout.sandbox`) instead bind tools per episode -- each
        trajectory leases its own env and gets the tools for it -- which is how a run exposes tools
        without sharing one workspace across concurrent episodes.
        """
        from .multi_turn import MultiTurnRollout
        self._multi_turn = MultiTurnRollout(
            self.sampler,
            self.template,
            max_turns=max_turns,
            max_trajectory_tokens=max_trajectory_tokens,
            tool_manager=tool_manager,
            harness=harness,
            followup_fn=followup_fn,
            env_pool=env_pool,
            tool_plugins=tool_plugins)

    def generate(self,
                 prompts: List[List[dict]],
                 num_samples: int = 1,
                 sampling_params: Optional[dict] = None,
                 prompt_extras: Optional[List[Dict[str, Any]]] = None,
                 force_logprobs: bool = True,
                 **kwargs) -> List[RolloutSample]:
        """Generate ``num_samples`` completions per prompt as RolloutSample objects (grouped by prompt).

        Args:
            prompts: list of message-lists (each a chat prompt).
            num_samples: completions per prompt (the GRPO group size).
            sampling_params: dict of SamplingParams fields (temperature/max_tokens/top_p/...).
            force_logprobs: force ``logprobs >= 1`` so each sample carries old_logps (contract 15). GRPO
                leaves it True; a caller needing neither old_logps nor stored tokens (plain inference)
                passes False to skip the extra logprob compute.
            kwargs: forwarded to the single-turn ``sampler.sample`` call (e.g. ``strict`` for transformers).
                ``adapter_path`` is honoured by both paths: the single-turn call passes it straight to the
                sampler, and the multi-turn engine threads it into every per-turn ``sampler.sample`` so a
                LoRA is selected there too. Other kwargs are single-turn only -- the multi-turn engine
                drives its own per-turn sampling and ignores them.

        Returns:
            flat list of RolloutSample, grouped by prompt_id (num_samples per prompt).
        """
        if self._multi_turn is not None:
            return self._multi_turn.generate(
                prompts,
                num_samples=num_samples,
                sampling_params=sampling_params,
                prompt_extras=prompt_extras,
                force_logprobs=force_logprobs,
                adapter_path=kwargs.get('adapter_path'))

        params = _build_sampling_params(num_samples, sampling_params, force_logprobs)
        trajectories = self._build_trajectories(prompts, prompt_extras)
        responses = self.sampler.sample(trajectories, params, **kwargs)
        # A text-only backend (template is None, the ``client`` teacher) returns no prompt tokens, so the
        # response is carried as messages instead of a training feature -- the same rule the multi-turn
        # engine uses to pick its message-only mode.
        return self._samples_from_responses(
            responses,
            prompt_extras=prompt_extras,
            allow_message_only=self.template is None,
            require_logprobs=force_logprobs)

    @staticmethod
    def _build_trajectories(prompts: List[List[dict]], prompt_extras: Optional[List[Dict[str, Any]]]) -> List[Any]:
        """Thread each prompt's media columns onto its ``Trajectory`` so a VLM prompt samples WITH media."""
        return [
            _prompt_trajectory(messages, prompt_extras[i] if prompt_extras and i < len(prompt_extras) else None)
            for i, messages in enumerate(prompts)
        ]

    # --- per-sample streaming path (submit_sample / poll_completions / collect_sample) --------------------
    # The streaming driver's data plane: ONE trajectory per sampler submission, so completions arrive
    # as-completed instead of in fixed batches. The sampler layer needs no change -- the non-blocking
    # ``submit_generation`` (twinkle's ``GenerationSubmissionMixin``, mixed into the core vLLM/SGLang
    # samplers) already tracks each ``submission_id`` as an independent future, so a one-trajectory
    # submission is just its smallest case, and the training-sample assembly (``_samples_from_responses``)
    # is shared verbatim with the blocking :meth:`generate`.

    def submit_sample(self,
                      prompt: List[dict],
                      trajectory_idx: int = 0,
                      *,
                      prompt_idx: Any = 0,
                      sampling_params: Optional[dict] = None,
                      prompt_extras: Optional[Dict[str, Any]] = None,
                      force_logprobs: bool = True,
                      adapter_name: str = '',
                      adapter_path: Optional[str] = None,
                      allow_partial_rollout: bool = False,
                      policy_version: int = 0) -> SampleHandle:
        """Admit ONE trajectory on the sampler WITHOUT blocking (the streaming per-sample admit).

        Builds the same ``SamplingParams`` (``num_samples=1``) and media-threaded ``Trajectory`` the
        blocking :meth:`generate` would, pins the same version fields (``adapter_name`` /
        ``adapter_path`` / ``allow_partial_rollout``), and returns a :class:`SampleHandle` for
        :meth:`poll_completions` / :meth:`collect_sample`. ``prompt_idx`` / ``trajectory_idx`` are
        carried for the collect-side group stamping; ``policy_version`` is the admission pin's version,
        stamped onto the collected sample. A single-turn prompt goes straight onto the sampler's
        non-blocking ``submit_generation``; a multi-turn prompt is admitted per-episode on a background
        thread instead (see :meth:`_submit_episode`), because the multi-turn engine drives a whole
        episode of blocking per-turn ``sample()`` calls with no sampler-level admit form.
        """
        if self._multi_turn is not None:
            return self._submit_episode(
                prompt,
                trajectory_idx,
                prompt_idx=prompt_idx,
                sampling_params=sampling_params,
                prompt_extras=prompt_extras,
                force_logprobs=force_logprobs,
                adapter_path=adapter_path,
                allow_partial_rollout=allow_partial_rollout,
                policy_version=policy_version)
        if not callable(getattr(self.sampler, 'submit_generation', None)):
            raise RuntimeError('streaming submit_sample needs a sampler with the non-blocking submit_generation '
                               '(twinkle GenerationSubmissionMixin): the core vLLM/SGLang samplers provide it, '
                               'a transformers/torch or mock sampler does not. Use --sampler vllm or sglang.')
        params = _build_sampling_params(1, sampling_params, force_logprobs)
        trajectory = _prompt_trajectory(prompt, prompt_extras)
        submission_id = uuid.uuid4().hex
        self.sampler.submit_generation(
            submission_id, [trajectory], params,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
            allow_partial_rollout=allow_partial_rollout)
        return SampleHandle(
            submission_id=submission_id,
            prompt_idx=prompt_idx,
            trajectory_idx=trajectory_idx,
            prompt_extras=dict(prompt_extras) if prompt_extras else None,
            require_logprobs=force_logprobs,
            policy_version=policy_version)

    def _submit_episode(self,
                        prompt: List[dict],
                        trajectory_idx: int,
                        *,
                        prompt_idx: Any,
                        sampling_params: Optional[dict],
                        prompt_extras: Optional[Dict[str, Any]],
                        force_logprobs: bool,
                        adapter_path: Optional[str],
                        allow_partial_rollout: bool,
                        policy_version: int) -> SampleHandle:
        """Admit ONE multi-turn episode on a background daemon thread (the per-episode streaming admit).

        A multi-turn episode is a whole conversation -- a sequence of blocking per-turn ``sample()`` calls
        interleaved with tool execution -- so it has no sampler-level non-blocking form the way a
        single-turn ``submit_generation`` does: the engine's own pool only parallelizes a BATCH handed to
        one ``generate`` call, and the streaming driver admits one trajectory at a time. Per-episode
        admission therefore runs the ONE-episode blocking :meth:`MultiTurnRollout.generate` on its own
        daemon thread and returns a handle immediately; :meth:`poll_completions` reports it done when the
        thread's Future resolves and :meth:`collect_sample` takes the result.

        Concurrency is bounded by the streaming driver's backpressure -- it admits at most its in-flight
        budget of handles -- which is the single source of admission control, so this layer adds no pool of
        its own and an episode is never queued behind another here. The threads are parked on GPU/tool I/O
        almost all the time, so one per in-flight episode is cheap; the sampler's continuous batching and
        concurrency cap (and a sandbox env pool's lease) are the real throughput limits.

        ``allow_partial_rollout`` (set only under ``in_place``) is threaded into every per-turn ``sample()``
        so a publish's abort-and-resume continues a turn mid-episode on the fresh weights; ``adapter_path``
        (set only under ``adapter_snapshot``) pins the whole episode to one frozen LoRA snapshot. The
        admission ``policy_version`` is stamped onto the collected sample -- the conservative staleness
        anchor (the oldest version any of its turns was generated under).
        """
        submission_id = uuid.uuid4().hex
        future: Future = Future()
        self._episode_futures[submission_id] = future

        def _work() -> None:
            try:
                samples = self._multi_turn.generate(
                    [prompt],
                    num_samples=1,
                    sampling_params=sampling_params,
                    prompt_extras=[prompt_extras] if prompt_extras else None,
                    force_logprobs=force_logprobs,
                    adapter_path=adapter_path,
                    allow_partial_rollout=allow_partial_rollout)
                future.set_result(samples)
            except Exception as exc:  # surfaced at poll/collect; the thread must not die without resolving
                future.set_exception(exc)

        threading.Thread(target=_work, name=f'episode-{submission_id[:8]}', daemon=True).start()
        return SampleHandle(
            submission_id=submission_id,
            prompt_idx=prompt_idx,
            trajectory_idx=trajectory_idx,
            prompt_extras=dict(prompt_extras) if prompt_extras else None,
            require_logprobs=force_logprobs,
            policy_version=policy_version)

    def poll_completions(self, handles: List[SampleHandle]) -> List[SampleHandle]:
        """Non-blocking as-completed query: the subset of ``handles`` whose generation has completed.

        One status probe per handle (no sleeping, no waiting), so the streaming driver can poll between
        admission and training passes. A submission that FAILED raises immediately -- the same
        fail-loudly contract :meth:`_await_generation` enforces on the blocking path. Handles with no
        recorded status (a probe race) are treated as still running.
        """
        if not callable(getattr(self.sampler, 'get_generation_status', None)):
            raise RuntimeError('streaming poll_completions needs a sampler with get_generation_status '
                               '(twinkle GenerationSubmissionMixin): the core vLLM/SGLang samplers provide '
                               'it, a transformers/torch or mock sampler does not.')
        completed: List[SampleHandle] = []
        for handle in handles:
            future = self._episode_futures.get(handle.submission_id)
            if future is not None:
                # Multi-turn episode: its completion lives on the background thread's Future, not the
                # sampler's submission tracker (the episode drives blocking per-turn ``sample()`` calls,
                # never ``submit_generation``). Fail loudly if the thread raised, mirroring the sampler
                # contract below.
                if not future.done():
                    continue
                error = future.exception()
                if error is not None:
                    raise RuntimeError(
                        f'streaming multi-turn episode {handle.submission_id} failed: {error}') from error
                completed.append(handle)
                continue
            states = self.sampler.get_generation_status(handle.submission_id)
            states = states if isinstance(states, list) else [states]
            failed = next((state for state in states if state.get('status') not in ('running', 'completed')), None)
            if failed is not None:
                error = failed.get('error') or failed.get('status', 'unknown failure')
                raise RuntimeError(f'streaming generation {handle.submission_id} failed: {error}')
            if states and all(state.get('status') == 'completed' for state in states):
                completed.append(handle)
        return completed

    def collect_sample(self, handle: SampleHandle) -> RolloutSample:
        """Collect ONE completed trajectory as a single :class:`RolloutSample` (stamped with its version).

        The per-sample collect of the streaming path (the as-completed counterpart of the blocking
        :meth:`generate`'s tail). A single-turn submission blocks until the
        sampler reports it completed (via the shared :meth:`_await_generation` poll), then lifts the
        responses with the same training-feature assembly; a multi-turn episode takes the background
        thread's already-assembled :class:`RolloutSample` off its Future instead. Either way the submission
        held exactly one trajectory, so exactly one sample must come back -- anything else is a contract
        violation and fails loudly. The sample's group id is stamped to the GLOBAL prompt index (the same
        stamping ``GRPOLoop._finalize_samples`` does on the batch path, so group-relative advantage keys
        match), and ``policy_version`` records the admission pin's version.
        """
        future = self._episode_futures.pop(handle.submission_id, None)
        if future is not None:
            # Multi-turn episode: the background thread already assembled the single RolloutSample (the
            # multi-turn engine owns per-turn encoding/logprobs), so collect just takes the Future's result
            # -- which re-raises the episode's exception here if it failed (fail-loudly at collect) -- and
            # stamps the same group id / admission version as the single-turn path below.
            samples = future.result()
            if len(samples) != 1:
                raise RuntimeError(f'streaming collect_sample expected exactly 1 sample for one multi-turn '
                                   f'episode (submission {handle.submission_id}), got {len(samples)}.')
            sample = samples[0]
            sample.prompt_id = str(handle.prompt_idx)
            sample.policy_version = handle.policy_version
            return sample
        responses = self._await_generation(handle.submission_id)
        samples = self._samples_from_responses(
            responses,
            prompt_extras=[handle.prompt_extras] if handle.prompt_extras else None,
            allow_message_only=self.template is None,
            require_logprobs=handle.require_logprobs)
        if len(samples) != 1:
            raise RuntimeError(f'streaming collect_sample expected exactly 1 sample for one trajectory '
                               f'(submission {handle.submission_id}), got {len(samples)}.')
        sample = samples[0]
        sample.prompt_id = str(handle.prompt_idx)
        sample.policy_version = handle.policy_version
        return sample

    def cancel_sample(self, handle: Optional[SampleHandle]) -> None:
        """Drop an admitted-but-uncollected per-sample generation (budget hit, drain, or stale cancel).

        A single-turn submission is cancelled on the sampler. A multi-turn episode has no sampler
        submission to cancel (it runs blocking per-turn ``sample()`` on its own daemon thread), so it is
        ABANDONED: its Future is dropped and the thread finishes on its own with its result discarded. A
        ``concurrent.futures.Future`` silently drops an unretrieved exception (unlike ``asyncio.Future``,
        which logs "never retrieved" at GC), so an abandoned episode that later fails needs no drain here.
        """
        if handle is None:
            return
        if self._episode_futures.pop(handle.submission_id, None) is not None:
            return
        self.sampler.cancel_generation(handle.submission_id)

    def abort_all_inflight(self) -> Any:
        """Abort every in-flight generation on the sampler so an in-place weight republish can resume them.

        The abort-on-publish trigger for a deep-buffered ``in_place`` sync: ``InPlaceWeightSync`` calls this
        BEFORE overwriting the sampler's single live weight copy, so no generation decodes across the update
        -- each returns the tokens it produced so far with ``stop_reason='abort'`` and, when its submission
        had ``allow_partial_rollout``, resumes from that point on the fresh weights (see
        :class:`~twinkle.sampler.partial_rollout.PartialRolloutMixin`). Distinct from :meth:`cancel_sample`,
        which DROPS a submission: abort keeps it alive and resumable. Delegates to the sampler's
        ``abort_all_inflight`` (mixed into the core vLLM/SGLang samplers).
        """
        abort = getattr(self.sampler, 'abort_all_inflight', None)
        if not callable(abort):
            raise RuntimeError('abort-on-publish needs a sampler with abort_all_inflight (twinkle '
                               'PartialRolloutMixin): the core vLLM/SGLang samplers provide it, a '
                               'transformers/torch or mock sampler does not. Use --sampler vllm or sglang.')
        return abort()

    def _await_generation(self, submission_id: str) -> List[Any]:
        """Poll an admitted generation to completion, then consume its responses from every DP worker."""
        poll_interval = 0.01
        while True:
            states = self.sampler.get_generation_status(submission_id)
            states = states if isinstance(states, list) else [states]
            failed = next((state for state in states if state.get('status') not in ('running', 'completed')), None)
            if failed is not None:
                error = failed.get('error') or failed.get('status', 'unknown failure')
                raise RuntimeError(f'async generation {submission_id} failed: {error}')
            if states and all(state.get('status') == 'completed' for state in states):
                return self.sampler.collect_generation(submission_id)
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 0.25)

    def sync_weights(self) -> None:
        """No-op on the base engine, with a one-time warning (W14).

        The base :class:`RolloutEngine` holds no ``CheckpointEngineManager``, so it cannot push the trained
        policy into its sampler: an on-policy loop (GRPO/PPO/RFT/distill) wired to it keeps sampling from the
        sampler's INITIAL weights, i.e. a stale behaviour policy -- a pipeline smoke, NOT
        algorithmically-correct on-policy RL. The loop calls this every step, so warn once rather than
        silently skipping (which is how the base used to behave, and how a half-wired run went unnoticed).
        :class:`~swift.dev.recipe.run_grpo.SyncableRollout` overrides this to actually sync.
        """
        if not self._warned_no_sync:
            logger.warning(
                'This rollout engine does not sync weights: the sampler keeps its INITIAL weights, so the '
                'behaviour policy stays stale and on-policy training is NOT algorithmically correct (a '
                'pipeline smoke only). Wire a weight-syncable rollout (run_grpo.SyncableRollout, over a '
                'vLLM/SGLang sampler + CheckpointEngineManager) for a real GRPO/PPO/RFT/distill run.')
            self._warned_no_sync = True

    def finish_generate(self) -> None:
        """Counterpart of :meth:`sync_weights`; a no-op on the base engine (no device hand-over to reverse)."""

    @staticmethod
    def _samples_from_responses(responses: List[Any],
                                prompt_extras: Optional[List[Dict[str, Any]]] = None,
                                *,
                                allow_message_only: bool = False,
                                require_logprobs: bool = True) -> List[RolloutSample]:
        """Delegate to :func:`samples_from_responses`.

        Kept as a staticmethod on the engine because ``run_grpo``'s ``SyncableRollout`` inherits
        :meth:`generate` (which calls ``self._samples_from_responses``) and the rollout unit test
        drives ``RolloutEngine._samples_from_responses`` directly.
        """
        return samples_from_responses(
            responses,
            prompt_extras=prompt_extras,
            allow_message_only=allow_message_only,
            require_logprobs=require_logprobs)

    def close(self) -> None:
        """Release the multi-turn sandbox env pool (its workspaces / microVMs) WITHOUT touching the sampler.

        A caller that injected its own sampler (``run_infer``) owns that sampler's lifecycle and calls
        close(); closing a rollout with no env pool is a no-op.
        """
        if self._multi_turn is not None:
            close = getattr(self._multi_turn, 'close', None)
            if close is not None:
                close()
        # Drop references to any in-flight multi-turn episodes; their daemon threads exit with the process.
        # A ``concurrent.futures.Future`` silently discards an unretrieved exception (unlike asyncio.Future),
        # so clearing the map needs no per-future exception drain.
        self._episode_futures.clear()

    def shutdown(self) -> None:
        """Release the env pool AND the sampler's GPU memory (twinkle's sampler owns its own teardown).

        For an engine that built its own sampler from a model_id; an injected-sampler caller uses
        :meth:`close` and shuts the sampler down itself.
        """
        self.close()
        self.sampler.shutdown()


def blocking_generate(rollout: Any, prompts: List[List[dict]], **generate_kwargs) -> List[RolloutSample]:
    """One blocking on-policy rollout: sync the policy, generate to completion, hand the device back.

    Brackets :meth:`RolloutEngine.generate` with the ``sync_weights`` / ``finish_generate`` device
    hand-over, keeping ``finish_generate`` in a ``finally`` so a generation failure still reverses the
    hand-over -- a bare ``sync_weights`` / ``generate`` pair would otherwise leave the sampler awake and
    the trainer's weights offloaded on an exception. Online recipes (GRPO/PPO/RFT/distill, and GRPO's
    ReMax greedy pass) call this instead of hand-writing the bracket, so the hand-over stays a
    rollout-layer concern in one place with identical semantics everywhere.

    Takes the rollout as an argument rather than being a method, so it works with any object exposing the
    ``sync_weights`` / ``generate`` / ``finish_generate`` contract (including test doubles), and forwards
    ``generate_kwargs`` verbatim -- the hand-over is orthogonal to what a particular call samples.
    """
    rollout.sync_weights()
    try:
        return rollout.generate(prompts, **generate_kwargs)
    finally:
        rollout.finish_generate()
