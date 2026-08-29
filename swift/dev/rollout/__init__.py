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
        response_token_ids: per-turn completion tokens (``List[List[int]]``, 2D to allow multi-turn;
            single-turn is one inner list). Used for reward / length.
        rollout_logprobs: per-turn per-token logprob under the SAMPLING policy (``List[List[float]]``,
            GRPO old_logps). 2D mirrors response_token_ids; GRPOLoss flattens for a single-turn window.
        prompt_id: which prompt group this sample belongs to (group-relative advantage). String id
            (not an index) to match the RL-sample layer and support dynamic/multi-turn sample counts.
        extra: dataset passthrough columns (solution/target/... for reward), kept out of encode.
        decoded: decoded completion text (logging / rule-based reward).
    """
    encoded: dict
    response_token_ids: List[List[int]]
    rollout_logprobs: List[List[float]]
    prompt_id: str
    extra: Dict[str, Any] = field(default_factory=dict)
    decoded: str = ''

    @property
    def input_feature(self) -> dict:
        """Back-compat alias: the training feature is ``encoded`` (was ``input_feature``)."""
        return self.encoded

    @property
    def old_logps(self) -> List[float]:
        """Flatten per-turn rollout_logprobs to a single list (single-turn GRPO window)."""
        return [lp for turn in self.rollout_logprobs for lp in turn]

    @property
    def prompt_index(self) -> str:
        """Back-compat alias for the group id (was an int index, now the ``prompt_id`` string)."""
        return self.prompt_id


class RolloutEngine:
    """Thin wrapper over twinkle's ``vLLMSampler``: prompts in, RolloutSample (training feature +
    old_logps) out. The sampler owns encoding/decoding; this layer only assembles the RL training
    sample. ``run_grpo``'s weight-syncing ``SamplerRollout`` subclasses this, reusing
    :meth:`generate` / :meth:`_samples_from_responses` verbatim and adding a ``sync_weights`` hook."""

    def __init__(self, model_id: str, template: Any, *, engine_args: Optional[dict] = None):
        from swift.dev.builders import build_sampler
        from swift.dev.config import ModelConfig
        self.model_id = model_id
        self.template = template
        # build_sampler sets the template on the sampler so Trajectory (messages) inputs are encoded,
        # and returns the prompt tokens the model conditioned on -- exactly the prompt half we need.
        self.sampler = build_sampler(
            ModelConfig(model=model_id), backend='vllm', engine_args=dict(engine_args or {}), template=template)

    def generate(self,
                 prompts: List[List[dict]],
                 num_samples: int = 1,
                 sampling_params: Optional[dict] = None) -> List[RolloutSample]:
        """Generate ``num_samples`` completions per prompt as RolloutSample objects (grouped by prompt).

        Args:
            prompts: list of message-lists (each a chat prompt).
            num_samples: completions per prompt (the GRPO group size).
            sampling_params: dict of SamplingParams fields (temperature/max_tokens/top_p/...).

        Returns:
            flat list of RolloutSample, grouped by prompt_id (num_samples per prompt).
        """
        from twinkle.data_format import SamplingParams, Trajectory

        sp = dict(sampling_params or {})
        sp.setdefault('temperature', 1.0)
        sp.setdefault('max_tokens', 32)
        # logprobs=0 -> the sampled token's own logprob (== old_logps); num_samples -> the GRPO group.
        # Contract 15: these logprobs ARE old_logps, so requesting them is forced, not defaulted.
        sp['logprobs'] = sp.get('logprobs', 0)
        sp['num_samples'] = num_samples
        params = SamplingParams(**sp)

        trajectories = [Trajectory(messages=list(messages)) for messages in prompts]
        responses = self.sampler.sample(trajectories, params)
        return self._samples_from_responses(responses)

    @staticmethod
    def _samples_from_responses(responses: List[Any]) -> List[RolloutSample]:
        """Build RolloutSamples from twinkle SampleResponses (one group per response).

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
                raise RuntimeError('vLLMSampler returned no prompt_token_ids; cannot build the RL training feature. '
                                   'The sampler must run with a template set (set_template) so the prompt is encoded.')
            for seq in response.sequences:
                response_tokens = list(seq.tokens or [])
                aligned = [-100] * len(prompt_tokens) + response_tokens
                labels = list(aligned[1:]) + [-100]
                encoded = {'input_ids': prompt_tokens + response_tokens, 'labels': labels, SHIFTED_KEY: True}
                old_logps = [float(lp) for lp in (seq.logprobs or [])]
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
                        # single-turn: wrap in a length-1 outer list (2D per-turn contract).
                        response_token_ids=[response_tokens],
                        rollout_logprobs=[old_logps],
                        prompt_id=str(pidx),
                        decoded=seq.decoded or ''))
        return out

    def shutdown(self) -> None:
        """Release the sampler and its GPU memory (twinkle's sampler owns its own teardown)."""
        self.sampler.shutdown()
