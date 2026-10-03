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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Marker key written into `encoded` when the training feature's labels are already next-token
# shifted (contract 14). Mirrors dev Template.SHIFTED_KEY so the "who shifted" fact is queryable
# instead of relying on a comment. The RL path hand-builds `encoded` (bypassing Template.encode),
# so it MUST set this to record that the shift was applied here.
SHIFTED_KEY = '_labels_shifted'


# TODO: not implemented yet
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
class GenerationHandle:
    """An in-flight async rollout generation admitted by :meth:`RolloutEngine.submit_generate`.

    The token between the non-blocking admit and the later :meth:`RolloutEngine.collect_generate`: it
    carries the sampler-side ``submission_id`` plus the two things collect needs to rebuild the training
    samples but which are known only at submit time -- the prompt media/reward columns (``prompt_extras``)
    and whether per-token logprobs were forced (``require_logprobs``). Opaque to the caller, which just
    hands it back to collect.
    """
    submission_id: str
    prompt_extras: Optional[List[Dict[str, Any]]] = None
    require_logprobs: bool = True


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
        self._multi_turn = None
        # One-shot latch for the base engine's "no weight sync" warning (see sync_weights), so a loop that
        # syncs every step does not spam the log.
        self._warned_no_sync = False
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

        params = self._build_sampling_params(num_samples, sampling_params, force_logprobs)
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
    def _build_sampling_params(num_samples: int, sampling_params: Optional[dict],
                               force_logprobs: bool) -> Any:
        """Assemble the twinkle ``SamplingParams`` shared by the blocking and the admitted generation.

        Factored out so :meth:`generate` and :meth:`submit_generate` request identical sampling -- an
        overlapped rollout must not silently sample under different params than the blocking one.
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

    @staticmethod
    def _build_trajectories(prompts: List[List[dict]], prompt_extras: Optional[List[Dict[str, Any]]]) -> List[Any]:
        """Thread each prompt's media columns onto its ``Trajectory`` so a VLM prompt samples WITH media."""
        return [
            _prompt_trajectory(messages, prompt_extras[i] if prompt_extras and i < len(prompt_extras) else None)
            for i, messages in enumerate(prompts)
        ]

    def submit_generate(self,
                        prompts: List[List[dict]],
                        num_samples: int = 1,
                        sampling_params: Optional[dict] = None,
                        prompt_extras: Optional[List[Dict[str, Any]]] = None,
                        force_logprobs: bool = True,
                        adapter_name: str = '',
                        adapter_path: Optional[str] = None) -> GenerationHandle:
        """Admit a rollout generation on the sampler WITHOUT blocking; collect it with :meth:`collect_generate`.

        The async (1-batch-lookahead) counterpart of :meth:`generate`: it builds the exact same
        trajectories and ``SamplingParams``, then schedules them on the sampler's background event loop via
        the non-blocking ``submit_generation`` (twinkle's ``GenerationSubmissionMixin``, mixed into the core
        vLLM/SGLang samplers) and returns a handle immediately, so the driver can train the previous batch
        while this one generates. Only the single-turn path is supported -- the multi-turn engine drives its
        own per-turn sampling loop with no admit-without-blocking form -- so async GRPO rejects multi-turn
        (see ``validate._check_async_generate``).
        """
        if self._multi_turn is not None:
            raise RuntimeError('async_generate does not support the multi-turn rollout: the multi-turn engine '
                               'drives its own per-turn sampling loop and cannot be admitted without blocking. '
                               'Disable async_generate or the multi-turn rollout.')
        if not callable(getattr(self.sampler, 'submit_generation', None)):
            raise RuntimeError('async_generate needs a sampler with the non-blocking submit_generation '
                               '(twinkle GenerationSubmissionMixin): the core vLLM/SGLang samplers provide it, '
                               'a transformers/torch or mock sampler does not. Use --sampler vllm or sglang.')
        params = self._build_sampling_params(num_samples, sampling_params, force_logprobs)
        trajectories = self._build_trajectories(prompts, prompt_extras)
        submission_id = uuid.uuid4().hex
        self.sampler.submit_generation(
            submission_id, trajectories, params, adapter_name=adapter_name, adapter_path=adapter_path)
        return GenerationHandle(
            submission_id=submission_id, prompt_extras=prompt_extras, require_logprobs=force_logprobs)

    def collect_generate(self, handle: GenerationHandle) -> List[RolloutSample]:
        """Block until an admitted generation finishes, then build its RolloutSamples.

        The tail of :meth:`generate` for the async path: waits for the sampler to report the submission
        completed on every DP worker, then lifts the responses into training features with the same
        next-token label shift. ``collect_generation`` raises if the work is still running, so the status is
        polled to completion first -- the identical admit/poll/collect contract twinkle's server data plane
        uses (``server.sampler.twinkle_handlers._await_generation``), here on the synchronous driver thread.
        """
        responses = self._await_generation(handle.submission_id)
        return self._samples_from_responses(
            responses,
            prompt_extras=handle.prompt_extras,
            allow_message_only=self.template is None,
            require_logprobs=handle.require_logprobs)

    def cancel_generate(self, handle: Optional[GenerationHandle]) -> None:
        """Drop an admitted-but-uncollected generation (e.g. the loop hit max_steps with one in flight)."""
        if handle is None:
            return
        self.sampler.cancel_generation(handle.submission_id)

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

    def shutdown(self) -> None:
        """Release the env pool AND the sampler's GPU memory (twinkle's sampler owns its own teardown).

        For an engine that built its own sampler from a model_id; an injected-sampler caller uses
        :meth:`close` and shuts the sampler down itself.
        """
        self.close()
        self.sampler.shutdown()
