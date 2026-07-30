"""Thin isolation layer over swift's vLLM inference engine (GRPOVllmEngine).

Purpose (rollout decoupled from the trainer into its own component): the GRPOLoop depends on
THIS interface, never directly on the engine.

Engine base is swift's own `GRPOVllmEngine` (NOT twinkle's vLLMSampler): swift's engine already
owns the per-model vLLM multimodal placeholder logic, LoRA adapter routing and vLLM version
patches, and it speaks the swift Template contract -- so no decode/`concat_input_feature` shim is
needed. This is also the target direction of the twinkle merge (twinkle delegates to swift, not
the reverse).

Backend is vLLM-only, by design, NOT a temporary YAGNI (see design.md 5.2.1): RL rollout needs
per-token logprobs for old_logps (contract 15), and only vLLM provides them reliably -- swift's
SGLang engine returns logprobs=None (`# TODO: logprobs`), so it physically cannot do RL rollout.
Multi-backend sampling (vLLM/SGLang/LMDeploy/Transformers) belongs to `swift infer` / `swift deploy`
via InferEngine, which does NOT need logprobs. So we deliberately do NOT build a multi-backend
dispatch shell here -- only generate(prompts, num_samples). The orthogonal rollout dimension that
DOES vary is placement (colocate / separate-server), not the sampling backend.

Prompt-half decision (read it back from the engine, do NOT re-encode): with
`RequestConfig.return_details=True` the engine returns `response.prompt_token_ids` -- the exact
prompt tokens vLLM conditioned on. We build the training feature from those instead of encoding the
prompt a second time on our side, so the feature cannot drift from what was actually sampled (a
second encode would only be *assumed* identical, and any per-model/template difference in the
generation prompt would silently misalign old_logps against the training forward).

Scope: text-only; weight-sync NOT wired (vLLM keeps initial weights => behavior policy is stale
=> NOT an algorithmically-correct GRPO state — a known intermediate stage).
"""
from __future__ import annotations

import torch
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
    """Thin wrapper over swift's GRPOVllmEngine: prompts in, RolloutSample (training feature +
    old_logps) out. The engine owns encoding; this layer only assembles the RL training sample."""

    def __init__(self, model_id: str, template: Any, *, engine_args: Optional[dict] = None):
        from swift.infer_engine import GRPOVllmEngine
        self.model_id = model_id
        self.template = template
        engine_args = dict(engine_args or {})
        # The engine owns the vLLM-mode template (it calls set_mode('vllm') internally and uses
        # per-model multimodal placeholder logic + decode_generate_ids), so no decode shim is
        # needed -- unlike twinkle's sampler, which expects a twinkle Template and forced us to
        # patch a shim onto it.
        self._engine = GRPOVllmEngine(
            model_id,
            template=template,
            use_async_engine=False,
            **engine_args,
        )

    @staticmethod
    def _extract_chosen_logps(choice_logprobs: Optional[dict], num_tokens: int) -> List[float]:
        """Read the sampled-token logprob per position from a swift `choice.logprobs`.

        swift's InferEngine._get_logprobs already indexes the SAMPLED token
        (`logprobs[token_id]`) and returns `{'content': [{'token','logprob','bytes'}, ...]}`, so
        there is no top-k disambiguation to do here (contract 15 is satisfied upstream by
        RequestConfig.logprobs=True).

        A length mismatch is raised, never padded: these values ARE old_logps, and 0.0 is a legal
        logprob (p=1.0), not a sentinel -- padding it would turn a missing-logprob bug into a
        silently wrong importance ratio exp(logps - 0). The common cause is logprobs not being
        requested at all, which yields content=[] and is caught loudly here.
        """
        content = (choice_logprobs or {}).get('content') or []
        if len(content) != num_tokens:
            raise RuntimeError(f'rollout logprobs misaligned: got {len(content)} logprobs for {num_tokens} '
                               f'sampled tokens. These are old_logps -- a short/empty list would silently corrupt '
                               f'the GRPO importance ratio, so this is fatal rather than padded. '
                               f'(content=[] usually means RequestConfig.logprobs was not honored.)')
        return [float(item['logprob']) for item in content]

    def generate(self,
                 prompts: List[List[dict]],
                 num_samples: int = 1,
                 sampling_params: Optional[dict] = None) -> List[RolloutSample]:
        """Generate num_samples completions per prompt.

        Args:
            prompts: list of message-lists (each a chat prompt).
            num_samples: completions per prompt (the GRPO group size).
            sampling_params: dict of RequestConfig fields (temperature/max_tokens/top_p/...).

        Returns:
            flat list of RolloutSample, grouped by prompt_id (num_samples per prompt).
        """
        from swift.infer_engine import InferRequest, RequestConfig

        sp = dict(sampling_params or {})
        sp.setdefault('temperature', 1.0)
        sp.setdefault('max_tokens', 32)
        # Contract 15: logprobs ARE old_logps, so this is forced rather than defaulted -- a caller
        # passing logprobs=False would otherwise produce an empty logprob list and a garbage
        # importance ratio. swift maps RequestConfig.logprobs=True (top_logprobs=None) to vLLM
        # logprobs=0, i.e. the SAMPLED token's logprob only, which is exactly what old_logps needs.
        # (top_logprobs is left alone: it only adds top-k alongside the sampled-token logprob.)
        sp['logprobs'] = True
        # return_details=True is REQUIRED for BOTH choice.token_ids (completion tokens) and
        # response.prompt_token_ids (the prompt half); without it both come back None.
        sp['return_details'] = True
        # num_samples is expressed by replicating the request, not via n: the async path asserts
        # n == 1 and a single choice per response keeps the per-sample mapping trivial. Group
        # membership is tracked by prompt_id below.
        sp['n'] = 1
        request_config = RequestConfig(**sp)

        # Replicate each prompt num_samples times; keep a parallel index back to its group.
        infer_requests, group_of = [], []
        for pidx, messages in enumerate(prompts):
            for _ in range(num_samples):
                infer_requests.append(InferRequest(messages=list(messages)))
                group_of.append(pidx)

        outputs = self._engine.infer(infer_requests, request_config, use_tqdm=False)

        out: List[RolloutSample] = []
        for req_idx, rollout_output in enumerate(outputs):
            pidx = group_of[req_idx]
            response = rollout_output.response
            choice = response.choices[0]
            # The prompt half comes from the ENGINE (the tokens vLLM actually conditioned on),
            # not from a second encode on our side -- see the module docstring.
            prompt_tokens = response.prompt_token_ids
            if prompt_tokens is None:
                raise RuntimeError('response.prompt_token_ids is None; RequestConfig.return_details must be True '
                                   'for the rollout feature to carry the prompt half.')
            prompt_tokens = list(prompt_tokens)
            response_tokens = list(choice.token_ids or [])
            full_ids = prompt_tokens + response_tokens
            # Aligned labels: -100 over prompt, response tokens as targets.
            aligned = [-100] * len(prompt_tokens) + response_tokens
            # NEXT-TOKEN SHIFT: twinkle's RL forward computes logps via no-shift
            # selective_log_softmax(logits, masked_labels), where logits[i] predicts token[i+1].
            # So masked_labels must be next-token shifted, exactly like dev Template.encode does
            # in training mode. We hand-build this feature (we don't route through
            # Template.encode here), so we MUST apply the same shift, or logps are off-by-one
            # vs vLLM old_logps -> the whole GRPO importance ratio is wrong. SHIFTED_KEY records
            # that the shift was applied HERE (contract 14), so downstream never re-shifts.
            labels = list(aligned[1:]) + [-100]
            encoded = {'input_ids': full_ids, 'labels': labels, SHIFTED_KEY: True}
            old_logps = self._extract_chosen_logps(choice.logprobs, len(response_tokens))
            out.append(
                RolloutSample(
                    encoded=encoded,
                    # single-turn: wrap in a length-1 outer list (2D per-turn contract).
                    response_token_ids=[response_tokens],
                    rollout_logprobs=[old_logps],
                    prompt_id=str(pidx),
                    decoded=choice.message.content or '',
                ))
        return out

    def shutdown(self):
        """Release the vLLM engine and its GPU memory.

        Neither swift's engine wrapper nor vLLM's ``LLM`` exposes a ``shutdown``, so "release"
        means dropping the last strong references and forcing collection -- probing for a
        ``shutdown`` attribute (as an earlier version did) is a guaranteed no-op and leaks the
        whole engine, which makes several vLLM tests in one process OOM.

        Idempotent: safe to call twice (the second call finds ``_engine`` already None).
        """
        import gc

        if self._engine is None:
            return
        # vLLM keeps process-global parallel state alive independently of the LLM object; without
        # tearing it down a subsequent engine in the same process re-initializes into a dirty world.
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:  # not initialized / API moved -- teardown stays best-effort here only
            pass
        self._engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
