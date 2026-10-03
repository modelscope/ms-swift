"""Shared primitives for the on-policy distillation recipes (GKD / OPSD / MOPD).

The three distillation recipes share one loop shape -- a student rollout (or a dataset completion, per
``lmbda``) is scored by a frozen teacher and the student is pulled toward that signal -- and differ only
in the teacher's CONTEXT, COUNT and SIGNAL:

  * GKD: one teacher scored on the SAME prompt+response as the student; the signal is full-vocab
    ``teacher_logits`` (a β-JSD target).
  * OPSD: one teacher scored on a PRIVILEGED prompt (``teacher_prompt`` replaces the user turn) while
    sharing the student's response tokens; the signal is response-only ``teacher_logps``.
  * MOPD: K domain teachers, each scoring the student's own on-policy prompt+response directly; the
    signal is a list of per-teacher ``teacher_logps`` channels fused by weighted mixture.

This module holds what all three reuse and nothing else: the uniform per-row payload the loop splits
into mini-batches (:class:`DistillRow`), the frozen-teacher construction (:func:`build_frozen_teacher`),
and the privileged-view builders OPSD needs. The teacher FORWARD itself (chunked ``forward_only``, so it
is correct at dp_size>1) is :meth:`GRPOLoop._response_logps`, which the distillation loops inherit -- so
there is no distillation-local scoring helper to drift out of sync with GRPO's.

There is deliberately NO HTTP teacher server here (no-server / all-Ray premise, RL_PLAN §2.H): a teacher
is always a :class:`~swift.dev.recipe._teacher.Teacher` -- a frozen twinkle model actor scored with
``forward_only``, or the student's own adapter-disabled base -- never a server.
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from swift.dev.recipe._teacher import FrozenModelTeacher, encode_privileged_view


class DistillRow(NamedTuple):
    """One distillation training row: the student feature plus the context a teacher view needs.

    Uniform across both ``lmbda`` sources -- an on-policy rollout sample and an off-policy dataset
    completion both reduce to ``(feature, messages, extra)`` -- so the loop's mini-batch split and the
    teacher scoring never branch on where a row came from.

    Fields:
        feature: the student's trainable feature (``input_ids`` + next-token-shifted, response-only
            ``labels``); fed verbatim to ``forward_backward`` and, for GKD, to the teacher forward.
        messages: the conversation the teacher's privileged view is rebuilt from -- prompt-only for an
            on-policy row, the full dataset conversation for an off-policy one. GKD ignores it (its
            teacher sees ``feature`` directly); OPSD replaces its user turn with ``teacher_prompt``.
        extra: dataset passthrough columns (``teacher_prompt``, multimodal inputs, routing tags).
    """
    feature: Dict[str, Any]
    messages: List[dict]
    extra: Dict[str, Any]


def build_frozen_teacher(rlhf_config: Any,
                         template: Any,
                         model_id: str,
                         *,
                         distributed_config: Any,
                         remote_group: str,
                         teacher_world_size: int,
                         adapters: Optional[List[str]] = None,
                         role: str = 'teacher',
                         offload: bool = False) -> FrozenModelTeacher:
    """Build one frozen twinkle teacher as a Ray actor on its own DeviceGroup (a ``forward_only`` scorer).

    A distillation teacher is always a frozen model actor -- never an HTTP server (no-server / all-Ray
    premise, RL_PLAN §2.H/§3.3). It is built from :func:`frozen_auxiliary_distributed_config`, which
    inherits the run's ``backend`` (basic principle 1: a teacher only ``forward_only``s, which megatron and
    transformers implement identically, so a megatron student gets a megatron teacher) and forces
    ``mode='ray'`` (basic principle 2), then ``remote_group`` makes ``_apply_ray_placement`` place it on the
    planned teacher DeviceGroup -- a ``mode='local'`` build would be a driver-process model that is not a
    Ray actor at all, and the driver holds no GPU under Ray. ``teacher_world_size`` sizes that group's mesh
    (== the ranks ``plan_rl_device_groups`` allocated); ``teacher_parallel_spec`` overrides the mesh layout
    for a sharded teacher. ``model_id`` is one entry of ``rlhf_config.teacher_model``; ``teacher_model_type``
    / ``teacher_model_revision`` / ``teacher_deepspeed`` are read off ``rlhf_config`` so a teacher may differ
    in architecture from the student. The template is bound so the teacher encodes a batch identically to
    the student (response tokens align one-to-one). Returned wrapped in a ``FrozenModelTeacher``; ``offload``
    (GKD's full-vocab memory hook) keeps the teacher on CPU between forwards.
    """
    from swift.dev.builders import build_model, frozen_auxiliary_distributed_config
    from swift.dev.config import ModelConfig
    from swift.dev.recipe.assembly import configure_frozen_adapter

    teacher_cfg = ModelConfig(model=model_id)
    teacher_cfg.model_type = rlhf_config.teacher_model_type
    teacher_cfg.model_revision = rlhf_config.teacher_model_revision
    teacher_dist = frozen_auxiliary_distributed_config(
        distributed_config,
        teacher_world_size,
        parallel_spec=rlhf_config.teacher_parallel_spec,
        deepspeed=rlhf_config.teacher_deepspeed)
    device_mesh = None
    if rlhf_config.teacher_parallel_spec:
        from twinkle import DeviceMesh
        device_mesh = DeviceMesh.from_spec(rlhf_config.teacher_parallel_spec)
    teacher = build_model(teacher_cfg, teacher_dist, device_mesh=device_mesh, remote_group=remote_group)
    configured = configure_frozen_adapter(teacher, template, list(adapters or []), role=role)
    return FrozenModelTeacher(configured, offload=offload)


def response_ids_from_feature(feature: Dict[str, Any]) -> List[int]:
    """The response token ids of an encoded feature, recovered from its (next-token shifted) labels.

    Labels hold the response token ids at trainable positions and ``-100`` elsewhere, so filtering
    ``-100`` yields exactly the sampled response -- the tokens the teacher must re-score.
    """
    labels = feature.get('labels')
    if labels is None:
        raise ValueError('OPSD/MOPD teacher scoring requires a labelled feature to recover the response token ids.')
    response_ids = [int(label) for label in labels if int(label) != -100]
    if not response_ids:
        raise ValueError('OPSD/MOPD feature contains no response tokens to score.')
    return response_ids


def teacher_view_messages(prompt_messages: List[dict], teacher_prompt: Any, response_ids: List[int]) -> List[dict]:
    """Build the privileged teacher message list: ``teacher_prompt`` in the user turn + shared response.

    Replaces the FIRST user turn's content with the privileged ``teacher_prompt`` (the dev-layer OPSD
    convention; for a single-turn sample first == last) and appends the assistant turn carrying the
    student's response token ids verbatim, so the teacher scores exactly the same response tokens.
    """
    messages = [dict(message) for message in prompt_messages]
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    for message in messages:
        if message.get('role') == 'user':
            message['content'] = teacher_prompt
            break
    else:
        raise ValueError('teacher_prompt requires a user turn in the prompt to carry the privileged context.')
    messages.append({'role': 'assistant', 'content': list(response_ids)})
    return messages


def build_privileged_teacher_feature(
    template: Any,
    prompt_messages: List[dict],
    teacher_prompt: Any,
    response_ids: List[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Encode the OPSD privileged teacher view (``teacher_prompt`` in the user turn + the shared response).

    Builds the privileged message list (``teacher_view_messages``) then delegates the encode-and-align-check
    tail to :func:`~swift.dev.recipe._teacher.encode_privileged_view`, which GRPO's RLSD/SDAR teacher view
    shares -- the two differ only in how the messages are assembled, not in the encoding contract.
    """
    messages = teacher_view_messages(prompt_messages, teacher_prompt, response_ids)
    return encode_privileged_view(template, messages, response_ids, extra)


def distill_sampling_params(rlhf_config: Any, generation_config: Any) -> Dict[str, Any]:
    """SamplingParams dict for the on-policy distillation rollout (shared by GKD/OPSD/MOPD).

    The rollout uses ``sampling_temperature``, NOT the distillation ``temperature``: those are separate
    roles (W16). ``sampling_temperature=None`` falls back to ``temperature`` so pre-split runs are
    byte-identical; setting it decouples how hot the student samples from how softly the divergence scores.
    """
    sampling_temperature = rlhf_config.sampling_temperature
    if sampling_temperature is None:
        sampling_temperature = rlhf_config.temperature
    params: Dict[str, Any] = {
        'max_tokens': rlhf_config.max_completion_length,
        'temperature': sampling_temperature,
    }
    if generation_config is not None:
        if generation_config.top_p is not None:
            params['top_p'] = generation_config.top_p
        if generation_config.top_k is not None:
            params['top_k'] = generation_config.top_k
    return params


def distill_rows_from_dataset(
    dataset_config: Any,
    template: Any,
) -> Tuple[List[List[dict]], List[dict], List[DistillRow]]:
    """Load rollout prompts, their passthrough extras, and the off-policy dataset completion rows.

    Shared by GKD/OPSD/MOPD. ``prompts`` / ``prompt_extras`` are parallel (one entry per dataset row) and
    drive the on-policy rollout; ``dataset_rows`` holds only the rows that carry an assistant completion,
    encoded as training features, for the off-policy rounds ``lmbda < 1`` selects.
    """
    from swift.dev.builders import load_prompt_rows

    rows = load_prompt_rows(dataset_config, None, split_dataset_ratio=0.0)
    prompts: List[List[dict]] = []
    prompt_extras: List[dict] = []
    dataset_rows: List[DistillRow] = []
    for row in rows:
        messages = copy.deepcopy(row.get('messages') or [])
        if not messages:
            continue
        extra = {key: copy.deepcopy(value) for key, value in row.items() if key != 'messages'}
        has_completion = messages[-1].get('role') == 'assistant'
        prompts.append(copy.deepcopy(messages[:-1] if has_completion else messages))
        prompt_extras.append(extra)
        if has_completion:
            dataset_rows.append(DistillRow(template.encode(row), messages, extra))
    if not prompts:
        raise ValueError('the distillation dataset provided no prompt messages.')
    return prompts, prompt_extras, dataset_rows
