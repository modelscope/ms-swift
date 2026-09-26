"""Isolation layer over twinkle's vLLM sampler for RL rollout.

Purpose (rollout decoupled from the trainer into its own component): the GRPOLoop depends on THIS
interface, never directly on the engine.

Engine base is twinkle's ``vLLMSampler`` (built via :func:`swift.dev.builders.build_sampler`), not
legacy ``swift.infer_engine.GRPOVllmEngine``. twinkle's sampler already speaks the twinkle Template
contract, returns per-token logprobs natively (``sequence.logprobs``), exposes the prompt tokens vLLM
conditioned on (``response.prompt_token_ids``), and owns multimodal placeholder logic + LoRA routing --
so no swift engine / decode shim is needed. This makes dev's rollout twinkle-first like
``run_infer`` / ``run_deploy`` / ``run_sampling``; the ``run_grpo`` weight-syncing ``SamplerRollout``
subclasses :class:`RolloutEngine`, so the rollout ``generate`` contract lives in exactly one place.

Backend is vLLM-only, by design, NOT a temporary YAGNI (see design.md 5.2.1): RL rollout needs
per-token logprobs for old_logps (contract 15), and only vLLM provides them reliably. That constraint
is about logprobs, not about where sampling lives. So we deliberately do NOT build a multi-backend
dispatch shell here -- only ``generate(prompts, num_samples)``. The orthogonal rollout dimension that
DOES vary is placement (colocate / separate-server), not the sampling backend.

Prompt-half decision (read it back from the sampler, do NOT re-encode): the sampler returns
``response.prompt_token_ids`` -- the exact prompt tokens vLLM conditioned on. We build the training
feature from those instead of encoding the prompt a second time on our side, so the feature cannot
drift from what was actually sampled (a second encode would only be *assumed* identical, and any
per-model/template difference in the generation prompt would silently misalign old_logps against the
training forward).

Scope: text-only; the base :class:`RolloutEngine` does NOT sync weights (vLLM keeps initial weights =>
behaviour policy is stale => NOT algorithmically-correct GRPO — a known intermediate stage).
``run_grpo``'s ``SamplerRollout`` adds weight sync.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


def samples_from_responses(responses: List[Any],
                          prompt_extras: Optional[List[Dict[str, Any]]] = None,
                          *,
                          allow_message_only: bool = False) -> List[RolloutSample]:
    """Build RolloutSamples from twinkle SampleResponses (one group per response).

    The single-turn counterpart of :func:`rollout.multi_turn.trajectory_to_rollout_sample`: a plain
    ``sampler.sample`` returns one ``SampleResponse`` per prompt (holding ``num_samples`` sequences),
    and this turns each sequence into a ``RolloutSample`` carrying the training feature and old_logps.
    ``run_sampling`` reuses it so its single-turn path produces the SAME encoded/logprob payload the
    multi-turn engine does, rather than throwing the tokens away and keeping only decoded text.

    The training feature is rebuilt from ``prompt_token_ids`` + ``sequence.tokens`` rather than
    trusting the sampler's own ``new_input_feature`` labelling, so old_logps (``sequence.logprobs``)
    line up with the training forward. The next-token shift is applied HERE (contract 14): twinkle's
    RL forward computes logps via no-shift ``selective_log_softmax(logits, masked_labels)`` where
    ``logits[i]`` predicts ``token[i+1]``, so the masked labels must be next-token shifted or logps
    are off-by-one vs vLLM old_logps and the whole GRPO importance ratio is wrong.
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
            encoded = {'input_ids': prompt_tokens + response_tokens, 'labels': labels, SHIFTED_KEY: True}
            old_logps = _sampled_token_logprobs(response_tokens, seq.logprobs)
            # A length mismatch is raised, never padded: these values ARE old_logps, and 0.0 is a
            # legal logprob (p=1.0), not a sentinel -- padding it would turn a missing-logprob bug
            # into a silently wrong importance ratio exp(logps - 0).
            if len(old_logps) != len(response_tokens):
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
                    truncated=getattr(seq, 'stop_reason', None) == 'length'))
    return out


def _message_only_samples(response: Any, pidx: int,
                          prompt_extras: Optional[List[Dict[str, Any]]]) -> List[RolloutSample]:
    """Build token-less RolloutSamples for a text-only backend (the ``client`` teacher).

    The single-turn counterpart of the multi-turn engine's message-only mode: a remote teacher returns
    decoded text (and structured tool calls inside ``new_input_feature['messages']``) but no token ids,
    so ``encoded`` / ``response_token_ids`` / ``rollout_logprobs`` stay empty and only ``decoded`` /
    ``messages`` carry anything -- exactly what ``run_sampling`` reads. A failed request degrades to an
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
    sample. ``run_grpo``'s weight-syncing ``SamplerRollout`` subclasses this, reusing
    :meth:`generate` / :meth:`_samples_from_responses` verbatim and adding a ``sync_weights`` hook."""

    def __init__(self,
                 model_id: Optional[str] = None,
                 template: Any = None,
                 *,
                 engine_args: Optional[dict] = None,
                 sampler: Any = None):
        self.model_id = model_id
        self.template = template
        self._multi_turn = None
        if sampler is not None:
            # Injected: the caller built and placed the sampler (``run_sampling`` across backends,
            # ``run_grpo``'s SamplerRollout on its own remote_group). This engine borrows it and does NOT
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
                The multi-turn engine drives its own per-turn sampling, so they are accepted but unused.

        Returns:
            flat list of RolloutSample, grouped by prompt_id (num_samples per prompt).
        """
        if self._multi_turn is not None:
            return self._multi_turn.generate(
                prompts,
                num_samples=num_samples,
                sampling_params=sampling_params,
                prompt_extras=prompt_extras,
                force_logprobs=force_logprobs)

        from twinkle.data_format import SamplingParams, Trajectory

        sp = dict(sampling_params or {})
        sp.setdefault('temperature', 1.0)
        sp.setdefault('max_tokens', 32)
        if force_logprobs:
            # Twinkle uses logprobs=1 for the sampled token in its normalized top-k representation.
            # Contract 15: these logprobs ARE old_logps, so requesting them is forced, not defaulted.
            sp['logprobs'] = max(int(sp.get('logprobs') or 0), 1)
        sp['num_samples'] = num_samples
        params = SamplingParams(**sp)

        trajectories = [Trajectory(messages=list(messages)) for messages in prompts]
        responses = self.sampler.sample(trajectories, params, **kwargs)
        # A text-only backend (template is None, the ``client`` teacher) returns no prompt tokens, so the
        # response is carried as messages instead of a training feature -- the same rule the multi-turn
        # engine uses to pick its message-only mode.
        return self._samples_from_responses(
            responses, prompt_extras=prompt_extras, allow_message_only=self.template is None)

    @staticmethod
    def _samples_from_responses(responses: List[Any],
                                prompt_extras: Optional[List[Dict[str, Any]]] = None,
                                *,
                                allow_message_only: bool = False) -> List[RolloutSample]:
        """Delegate to :func:`samples_from_responses`.

        Kept as a staticmethod on the engine because ``run_grpo``'s ``SamplerRollout`` inherits
        :meth:`generate` (which calls ``self._samples_from_responses``) and the rollout unit test
        drives ``RolloutEngine._samples_from_responses`` directly.
        """
        return samples_from_responses(responses, prompt_extras=prompt_extras, allow_message_only=allow_message_only)

    def close(self) -> None:
        """Release the multi-turn sandbox env pool (its workspaces / microVMs) WITHOUT touching the sampler.

        A caller that injected its own sampler (``run_sampling``) owns that sampler's lifecycle and calls
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
