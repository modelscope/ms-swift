"""Dev integration layer over twinkle's native multi-turn rollout engine.

There is no scheduler and no gym here any more: twinkle's ``MultiTurnRollout``
drives the turn loop, and its ``tool_manager`` / ``harness`` / ``followup_fn`` are
the extension points. This module does exactly two jobs -- build the ``Trajectory``
batch and run the engine (:class:`MultiTurnRollout`), and translate each flat
output ``Trajectory`` into dev's one-dimensional :class:`RolloutSample`
(:func:`trajectory_to_rollout_sample`).

The engine runs in one of two modes, decided by whether a local ``template`` is
passed. With one, the trajectory is trainable and twinkle accounts it in tokens
(``input_ids`` / ``labels`` / ``logprobs``); without one -- a text-only backend
such as the ``client`` teacher -- it accounts in messages only and emits no token
fields, so the translated sample carries an empty ``encoded`` and empty response
arrays (which is all ``run_infer`` reads).

Labels are taken from twinkle verbatim. They are already next-token shifted
(``append_ids`` rolls them into output order on the way out), and twinkle's GRPO
loss aligns them by position without a further shift -- so dev must NOT shift
again. ``SHIFTED_KEY`` records that the shift already happened.
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional

from . import SHIFTED_KEY, RolloutSample, _sampled_token_logprobs


def _input_order_labels(labels: List[int]) -> List[int]:
    """Undo twinkle's output-order roll to get labels aligned with ``input_ids``.

    ``append_ids`` stores labels shifted (``_roll_labels`` is ``np.roll(-1)``), so
    ``labels[i]`` is the target for position ``i + 1``. Shifting right by one
    recovers input order, where a trainable position ``j`` holds
    ``in_labels[j] == input_ids[j]``.
    """
    return (labels[-1:] + labels[:-1]) if labels else []


def _trainable_positions(input_ids: List[int], labels: List[int], completion_mask: Any) -> List[int]:
    """Input-order positions of the policy-produced tokens, in order.

    Matches the ledger's own ``audit`` selection -- ``labels != -100`` intersected
    with ``completion_mask`` when one is present -- so the count equals the number
    of logprobs the sampler returned. In the twinkle multi-turn path a whole
    generation is recorded trainable and every observation masked, so this is also
    ``completion_mask == 1``.
    """
    in_labels = _input_order_labels(list(labels or []))
    mask = (list(completion_mask)
            if completion_mask is not None and len(completion_mask) == len(in_labels) else None)
    positions: List[int] = []
    for j, lab in enumerate(in_labels):
        if lab == -100:
            continue
        if mask is not None and not mask[j]:
            continue
        positions.append(j)
    return positions


def _messages_with_token_ids(messages: List[dict], encoded: Dict[str, Any]) -> List[dict]:
    """Splice each assistant turn's exact sampled tokens back into the messages.

    The teacher-distillation view re-encodes the conversation under a different
    prompt but must score the *same* tokens the policy produced, so each assistant
    turn carries its ids verbatim rather than its text (re-encoding text can drift
    to different ids and misalign every logprob after it).

    ``encoded`` is twinkle's flat feature. Per-turn slices are recovered by
    splitting the trainable positions (``labels != -100``) into contiguous runs:
    the ledger records a whole generation as trainable and every observation as
    masked, so the runs alternate with the assistant turns one-to-one. Newest run
    maps to newest assistant message.
    """
    input_ids = list(encoded.get('input_ids') or [])
    labels = list(encoded.get('labels') or [])
    completion_mask = encoded.get('completion_mask')
    positions = _trainable_positions(input_ids, labels, completion_mask)
    runs: List[List[int]] = []
    for j in positions:
        if runs and j == runs[-1][-1] + 1:
            runs[-1].append(j)
        else:
            runs.append([j])
    turn_token_ids = [[input_ids[j] for j in run] for run in runs]

    result = copy.deepcopy(messages)
    completion_index = 0
    for message in reversed(result):
        if message.get('role') != 'assistant':
            continue
        if completion_index >= len(turn_token_ids):
            break
        ids = turn_token_ids[-1 - completion_index]
        message['content'] = {'token_ids': ids, 'loss_scale': [1] * len(ids)}
        completion_index += 1
    if completion_index != len(turn_token_ids):
        raise RuntimeError(
            f'multi-turn messages contain {completion_index} generated assistant turns, but the '
            f'trajectory has {len(turn_token_ids)} trainable token runs. A run-splitting mismatch '
            'means an assistant turn produced no trainable token (an empty generation) or a '
            'completion_mask split one turn in two.')
    return result


def _last_assistant_text(messages: Optional[List[dict]]) -> str:
    """The newest assistant turn's text, for logging / rule-based reward."""
    for message in reversed(messages or []):
        if message.get('role') == 'assistant':
            content = message.get('content')
            return content if isinstance(content, str) else ''
    return ''


def trajectory_to_rollout_sample(traj: Dict[str, Any], prompt_id: str, extra: Dict[str, Any]) -> RolloutSample:
    """Translate one flat twinkle output ``Trajectory`` into a dev ``RolloutSample``.

    Token-level mode fills ``encoded`` (twinkle's labels verbatim, already shifted)
    and the one-dimensional response arrays; message-only mode (a text-only backend)
    has no token fields, so those stay empty and only ``messages`` / ``decoded`` /
    ``truncated`` / ``rollout_infos`` carry anything -- exactly what ``run_infer``
    consumes.
    """
    input_ids = list(traj.get('input_ids') or [])
    labels = list(traj.get('labels') or [])
    completion_mask = traj.get('completion_mask')
    messages = copy.deepcopy(traj.get('messages'))
    logprobs = traj.get('logprobs') or []

    encoded: Dict[str, Any] = {}
    response_token_ids: List[int] = []
    response_loss_mask: List[int] = []
    rollout_logprobs: List[float] = []
    if input_ids and labels:
        encoded = {'input_ids': input_ids, 'labels': labels, SHIFTED_KEY: True}
        if completion_mask is not None:
            encoded['completion_mask'] = list(completion_mask)
        positions = _trainable_positions(input_ids, labels, completion_mask)
        response_token_ids = [input_ids[j] for j in positions]
        response_loss_mask = [1] * len(response_token_ids)
        if logprobs:
            rollout_logprobs = _sampled_token_logprobs(response_token_ids, logprobs)
            # A length mismatch is raised, never padded: these values ARE old_logps,
            # and 0.0 is a legal logprob (p=1.0), not a sentinel.
            if len(rollout_logprobs) != len(response_token_ids):
                raise RuntimeError(f'rollout logprobs misaligned: {len(rollout_logprobs)} logprobs for '
                                   f'{len(response_token_ids)} response tokens. These are old_logps; a '
                                   'mismatch would silently corrupt the GRPO importance ratio, so it is fatal.')

    rollout_infos = {
        key: traj[key]
        for key in ('turns', 'stop_reason', 'truncated', 'stuck_stop', 'tool_stop', 'followups') if key in traj
    }
    if traj.get('error') is not None:
        rollout_infos['error'] = traj['error']

    return RolloutSample(
        encoded=encoded,
        response_token_ids=response_token_ids,
        rollout_logprobs=rollout_logprobs,
        response_loss_mask=response_loss_mask,
        prompt_id=prompt_id,
        extra=extra,
        decoded=_last_assistant_text(messages),
        truncated=bool(traj.get('truncated')),
        messages=messages,
        rollout_infos=rollout_infos)


class MultiTurnRollout:
    """Thin dev wrapper over :class:`twinkle_agentic.rollout.MultiTurnRollout`.

    Builds the ``Trajectory`` batch (``num_samples`` copies per prompt, prompt-major
    so a caller slices the result identically to the single-turn path), runs the
    engine, and translates each flat output. Pass ``template=None`` for a text-only
    backend (the ``client`` teacher) and the engine accounts in messages instead of
    tokens. Per-round length is ``sampling_params.max_tokens``; whole-trajectory
    length is ``max_trajectory_tokens``.
    """

    def __init__(self,
                 sampler: Any,
                 template: Any,
                 *,
                 max_turns: Optional[int] = None,
                 max_trajectory_tokens: Optional[int] = None,
                 tool_manager: Any = None,
                 harness: Any = None,
                 followup_fn: Any = None,
                 env_pool: Any = None,
                 tool_plugins: Optional[List[Any]] = None):
        from twinkle.data_format import SamplingParams
        from twinkle_agentic.rollout import MultiTurnRollout as TwinkleMultiTurnRollout
        self.env_pool = env_pool
        self.tool_plugins = list(tool_plugins or [])
        # A sandbox pool binds tools per episode (each trajectory leases its own env and gets the tools
        # for that env), so the engine is built with no shared tool_manager and each call passes its own.
        # A directly-passed tool_manager is the twinkle default -- one manager shared across the batch --
        # and is kept only when there is no env pool to lease from.
        self._per_episode_tools = env_pool is not None and bool(self.tool_plugins)
        self.rollout = TwinkleMultiTurnRollout(
            sampler,
            template,
            sampling_params=SamplingParams(),
            max_turns=max_turns if max_turns is not None else 6,
            max_trajectory_tokens=max_trajectory_tokens,
            tool_manager=None if self._per_episode_tools else tool_manager,
            harness=harness,
            followup_fn=followup_fn)

    def generate(self,
                 prompts: List[List[dict]],
                 num_samples: int = 1,
                 sampling_params: Optional[dict] = None,
                 prompt_extras: Optional[List[Dict[str, Any]]] = None,
                 force_logprobs: bool = True,
                 adapter_path: Optional[str] = None) -> List[RolloutSample]:
        from twinkle.data_format import SamplingParams
        if num_samples < 1:
            raise ValueError('num_samples must be >= 1.')
        if prompt_extras is not None and len(prompt_extras) != len(prompts):
            raise ValueError('prompt_extras must contain exactly one mapping per prompt.')
        extras = prompt_extras or [{} for _ in prompts]
        trajectories, metadata = [], []
        for prompt_index, prompt in enumerate(prompts):
            for _ in range(num_samples):
                extra = copy.deepcopy(extras[prompt_index])
                trajectories.append({'messages': copy.deepcopy(prompt)})
                metadata.append((str(prompt_index), extra))

        params = dict(sampling_params or {})
        params.setdefault('temperature', 1.0)
        if force_logprobs:
            # Contract 15: the token-level path's logprobs ARE old_logps, so requesting
            # them is forced. A message-only backend ignores this (it returns none).
            params['logprobs'] = max(int(params.get('logprobs') or 0), 1)
        params['num_samples'] = 1
        sp = SamplingParams(**params)
        # twinkle's MultiTurnRollout.__call__ reads adapter_path out of its kwargs and threads it into
        # every per-turn sampler.sample, so a LoRA reserved at engine build is selected here too --
        # without this a multi-turn run silently samples from the base model.
        rollout_kwargs = {'adapter_path': adapter_path} if adapter_path else {}
        if self._per_episode_tools:
            outputs = self._generate_with_envs(trajectories, sp, **rollout_kwargs)
        else:
            outputs = self.rollout(trajectories, sampling_params=sp, **rollout_kwargs)
        if len(outputs) != len(metadata):
            raise RuntimeError(f'multi-turn rollout returned {len(outputs)} trajectories for {len(metadata)} inputs.')
        return [
            trajectory_to_rollout_sample(output, prompt_id, extra)
            for output, (prompt_id, extra) in zip(outputs, metadata)
        ]

    def _generate_with_envs(self, trajectories: List[Dict[str, Any]],
                            sampling_params: Any,
                            adapter_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Roll out each trajectory alone in its own leased env, concurrently up to the pool size.

        A sandbox env holds one workspace, so a trajectory must be driven by itself with the tools bound
        to the env it leased: twinkle broadcasts a call-time ``tool_manager`` across the whole batch,
        which cannot express a per-episode lease (and a per-prompt list longer than the batch would be
        misrouted episode-to-workspace). This is the challenger's per-episode pattern -- every worker
        takes an env, builds its ``ToolManager``, and runs a single-trajectory rollout on the one shared
        engine, which is safe to call concurrently.
        """
        from concurrent.futures import ThreadPoolExecutor

        from .sandbox import tool_manager_for

        def _run(index: int) -> Dict[str, Any]:
            with self.env_pool.lease() as env:
                outputs = self.rollout(
                    [trajectories[index]],
                    sampling_params=sampling_params,
                    tool_manager=tool_manager_for(env, self.tool_plugins),
                    **({'adapter_path': adapter_path} if adapter_path else {}))
                if not outputs:
                    raise RuntimeError('multi-turn rollout returned no trajectory for a leased-env episode.')
                return outputs[0]

        workers = max(1, len(self.env_pool))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_run, range(len(trajectories))))

    def close(self) -> None:
        """Release the sandbox env pool (its workspaces / microVMs). The sampler is not owned here."""
        if self.env_pool is not None:
            self.env_pool.close()
