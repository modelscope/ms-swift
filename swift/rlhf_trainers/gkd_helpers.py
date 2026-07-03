# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-agnostic GKD teacher / encoding helpers.

These functions are stateless: they operate on ``GKDSample`` objects and a
``Template`` and return encoded dicts / teacher requests / assembled teacher
outputs.  Shared by the HF and Megatron GKD trainers.
"""
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from swift.rl_core.data import GKDSample, OnPolicySample
from swift.template.base import Template
from swift.utils import get_cu_seqlens_from_position_ids, get_logger, json_parse_to_dict
from .gkd_loss import TeacherOutput
from .utils import (assemble_teacher_topk_logprobs, encode_sample, get_response_prefix_ids,
                    replace_assistant_response_with_ids)

logger = get_logger()


def encode_teacher_view(
    sample: OnPolicySample,
    template: Template,
) -> Dict[str, Any]:
    """Encode the OPSD teacher view (teacher_prompt + the shared on-policy response).

    Does NOT mutate ``sample.teacher_messages`` — works on copied message dicts
    so the sample's teacher view can be re-encoded across steps_per_generation.
    Caller must have invoked ``sample.build_teacher_view()`` first.
    """
    teacher_row = sample.to_teacher_template_dict()
    msgs = teacher_row.get('messages')
    if msgs is not None:
        teacher_row['messages'] = [m.copy() for m in msgs]
    if teacher_row.get('response_token_ids'):
        loss_mask = teacher_row.get('response_loss_mask') or None
        ctk = sample.extra.get('chat_template_kwargs') or {}
        sample_et = ctk.get('enable_thinking')
        prefix_ids = get_response_prefix_ids(template, sample_enable_thinking=sample_et)
        teacher_row['messages'] = replace_assistant_response_with_ids(
            teacher_row['messages'],
            teacher_row['response_token_ids'],
            loss_mask=loss_mask,
            non_thinking_prefix_ids=prefix_ids)
    teacher_encoded = template.encode(teacher_row, return_length=True)
    teacher_encoded.pop('_extra_kwargs', None)
    return teacher_encoded


def encode_gkd_samples(
    samples: List[GKDSample],
    template: Template,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Encode student and (OPSD) teacher views for a list of GKDSample objects.

    Returns ``(student_encoded_list, teacher_encoded_list, has_opsd)``.
    Trainers are responsible for collation and device placement.
    """
    student_encoded_list: List[Dict[str, Any]] = []
    teacher_encoded_list: List[Dict[str, Any]] = []
    # Cannot use any() here because it short-circuits: build_teacher_view()
    # must be called for ALL samples to populate teacher_messages on each.
    has_opsd = False
    for s in samples:
        if s.build_teacher_view():
            has_opsd = True

    for s in samples:
        encoded = encode_sample(s, template)
        encoded.pop('_extra_kwargs', None)
        student_encoded_list.append(encoded)

        if has_opsd:
            teacher_encoded = encode_teacher_view(s, template)
            teacher_encoded_list.append(teacher_encoded)
        else:
            teacher_encoded_list.append(encoded.copy())

    return student_encoded_list, teacher_encoded_list, has_opsd


def build_teacher_requests(samples: List[OnPolicySample], template: Optional[Template] = None) -> List[Any]:
    """Build teacher API requests from samples (GKD or GRPO/OPD-RL).

    For GKD OPSD samples, ``teacher_messages`` replaces the request messages;
    GRPO samples have no OPSD view, so the on-policy request is used as-is.

    When ``template`` is given and the sample carries ``response_token_ids``, the assistant
    response is sent as raw token ids (mirroring :func:`encode_sample`) so the teacher server
    forwards the *same* tokens the student sampled instead of re-tokenizing the decoded text.
    Re-tokenization can split the response into a different number of tokens, breaking the
    token-in-token-out alignment the teacher KL relies on. Teacher and student must share the
    tokenizer/vocabulary (a prerequisite of distillation) for the ids to be meaningful.
    """
    requests = []
    for s in samples:
        req = s.to_infer_request()
        # OPSD: score the teacher on its privileged prompt instead of the student prompt.
        teacher_messages = getattr(s, 'teacher_messages', None)
        messages = teacher_messages if teacher_messages else req.messages
        if template is not None and s.response_token_ids:
            # Send the sampled response as raw token ids (token-in-token-out) so the teacher server
            # forwards the *same* tokens the student sampled instead of re-tokenizing the decoded text.
            # Re-tokenization can split the response into a different number of tokens, breaking the
            # alignment the teacher KL relies on. Applies to both the on-policy prompt (GRPO/OPD-RL)
            # and the OPSD teacher prompt (which shares the same response tokens).
            loss_mask = s.response_loss_mask or None
            ctk = s.extra.get('chat_template_kwargs') or {}
            prefix_ids = get_response_prefix_ids(template, sample_enable_thinking=ctk.get('enable_thinking'))
            messages = replace_assistant_response_with_ids([m.copy() for m in messages],
                                                           s.response_token_ids,
                                                           loss_mask,
                                                           non_thinking_prefix_ids=prefix_ids)
        req.messages = messages
        requests.append(req)
    return requests


def assemble_teacher_output(
    parsed: List[Tuple],
    teacher_model_inputs: Dict[str, torch.Tensor],
    topk: Optional[int],
    template_padding_free: bool,
    device: torch.device,
) -> TeacherOutput:
    """Build a full-sequence ``TeacherOutput`` (GKD) from parsed teacher prompt logprobs.

    Produces a ``[B, S, K]`` top-k frame aligned to the teacher's full input sequence,
    masked later by ``labels`` in :func:`gkd_loss`. Handles both packed (padding_free)
    and non-packed layouts, including mrope / position-id based cu_seqlens detection.
    GRPO/OPD-RL uses :func:`assemble_teacher_completion_logprobs` instead (completion frame).
    """
    K = topk or 1
    input_ids = teacher_model_inputs['input_ids']
    batch_size, seq_len = input_ids.shape

    cu_seqlens = None
    offsets = None
    if template_padding_free:
        cu_seqlens = [0]
        for lps, ixs in parsed:
            cu_seqlens.append(cu_seqlens[-1] + len(lps) + 1)
        trainer_seq_lens = teacher_model_inputs.get('cu_seq_lens_q')
        if trainer_seq_lens is None:
            position_ids = teacher_model_inputs.get('text_position_ids')
            if position_ids is None:
                position_ids = teacher_model_inputs.get('position_ids')
            if position_ids is not None:
                trainer_seq_lens = get_cu_seqlens_from_position_ids(position_ids)
        if trainer_seq_lens is not None and cu_seqlens[-1] != int(trainer_seq_lens[-1]):
            logger.warning('The number of tokens returned by the teacher server differs from that of the trainer. '
                           'This may be caused by non-aligned processing.')
    else:
        attention_mask = teacher_model_inputs.get('attention_mask')
        if attention_mask is not None:
            offsets = []
            for i in range(attention_mask.shape[0]):
                nz = attention_mask[i].nonzero().flatten()
                offsets.append(int(nz[0]) if nz.numel() else 0)
        else:
            offsets = None

    topk_logprobs, topk_indices = assemble_teacher_topk_logprobs(
        parsed,
        batch_size=batch_size,
        seq_len=seq_len,
        cu_seqlens=cu_seqlens,
        topk=K,
        device=device,
        offsets=offsets,
    )

    teacher_out = TeacherOutput(topk_logprobs=topk_logprobs, topk_indices=topk_indices)
    if 'labels' in teacher_model_inputs:
        teacher_out.labels = teacher_model_inputs['labels']
    return teacher_out


def assemble_teacher_completion_logprobs(
    parsed: List[Tuple],
    completion_mask: torch.Tensor,
    device: torch.device,
    response_token_ids: Optional[List[List[int]]] = None,
) -> TeacherOutput:
    """Align parsed teacher *sampled-token* prompt logprobs to a ``[B, T, 1]`` completion frame (GRPO/OPD-RL).

    ``parsed[i] = (lps, ixs)`` come from :func:`parse_prompt_logprobs` with ``topk=0``,
    i.e. the sampled token's logp/id at each position. They are placed on
    ``completion_mask``'s response region.

    When the teacher API returns logprobs for the **full sequence** (prompt + response
    + end tokens), ``response_token_ids`` is used to locate the response portion by
    matching token IDs from the end of the sequence (the response is the last
    ``count`` tokens before the end tokens).
    """
    batch_size, seq_len = completion_mask.shape
    out_lp = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32, device=device)
    out_ix = torch.zeros(batch_size, seq_len, 1, dtype=torch.long, device=device)
    for i, (lps, ixs) in enumerate(parsed):
        completion_indices = completion_mask[i].nonzero(as_tuple=True)[0]
        count = completion_indices.numel()
        if count == 0:
            continue
        if len(lps) == count + 1:
            lps, ixs = lps[:count], ixs[:count]
        elif len(lps) > count + 1 and response_token_ids is not None:
            # Teacher returned logprobs for the full sequence (prompt + response + end tokens).
            # Locate the response portion by matching token IDs from the end.
            rti = response_token_ids[i]
            if isinstance(rti[0], list):
                rti = rti[0]  # flatten [[1,2,3]] -> [1,2,3]
            flat_ixs = [ix[0] for ix in ixs]  # each ix is a single-element list [token_id]
            # Search from the end: find the starting index where the last `count` tokens
            # match the response token IDs (allowing 0-2 end tokens after the response).
            start = None
            for offset in range(min(3, len(lps) - count + 1)):  # try 0, 1, 2 end tokens
                candidate_start = len(flat_ixs) - count - offset
                if candidate_start < 0:
                    continue
                if flat_ixs[candidate_start:candidate_start + count] == rti[:count]:
                    start = candidate_start
                    break
            if start is not None:
                lps = lps[start:start + count]
                ixs = ixs[start:start + count]
            else:
                # The response token ids did not align anywhere in the returned sequence: the teacher
                # tokenized differently than the student, so any slice would yield a meaningless
                # per-token KL. Fail loudly rather than silently mis-aligning.
                raise ValueError(
                    f'Teacher returned {len(lps)} logprobs but could not locate the {count} sampled response '
                    'tokens by id. The teacher server must use the same tokenizer as the student '
                    '(token-in-token-out).')
        assert len(lps) == count, (f'Teacher logp count {len(lps)} != sampled tokens {count}. The teacher server '
                                   'must use the same tokenizer as the student (token-in-token-out).')
        # lps/ixs are per-position single-element lists ([[lp], ...]) -> [count, 1].
        out_lp[i, completion_indices] = torch.tensor(lps, dtype=torch.float32, device=device)
        out_ix[i, completion_indices] = torch.tensor(ixs, dtype=torch.long, device=device)
    return TeacherOutput(topk_logprobs=out_lp, topk_indices=out_ix)


@dataclass
class TeacherServerConfig:
    """Configuration for a single teacher model server.

    Args:
        url: Teacher server URL (e.g., 'http://localhost:8000').
        tags: Tags this teacher handles. With multiple teachers each sample is routed to
            exactly one teacher by tag. Empty only for the single-teacher case (one teacher
            handles all samples). All teachers share the global ``--teacher_kl_coef``.
    """
    url: str
    tags: List[str] = field(default_factory=list)


def parse_teacher_model_server(val: Optional[str]) -> Optional[List[TeacherServerConfig]]:
    """Parse ``teacher_model_server``: single URL string or JSON multi-teacher config.

    Accepts two formats:
    - Single URL (backward compatible): ``'http://localhost:8000'`` -> one teacher, no tags,
      handles all samples.
    - Multi-teacher JSON: ``'[{"url":"http://localhost:8000","tags":["math"]}, ...]'``.
      With multiple teachers each entry MUST have non-empty ``tags`` and tags MUST NOT overlap
      across teachers (each sample routes to exactly one teacher).

    Returns ``None`` when *val* is ``None``. Otherwise returns a list of
    :class:`TeacherServerConfig` (always a list, even for single teacher).
    """
    if val is None:
        return None
    val = val.strip()
    if not val.startswith('['):
        return [TeacherServerConfig(url=val, tags=[])]

    # JSON list of teachers: reuse swift's shared JSON parser (repairs malformed JSON), matching
    # how other JSON-valued args (e.g. model_kwargs) are parsed rather than a bespoke json.loads.
    configs_raw = json_parse_to_dict(val)
    if not configs_raw:
        raise ValueError('teacher_model_server JSON list is empty.')
    configs = []
    for i, c in enumerate(configs_raw):
        url = c.get('url')
        if not url:
            raise ValueError(f'teacher_model_server[{i}]: "url" is required.')
        tags = c.get('tags', [])
        if not isinstance(tags, list):
            raise ValueError(f'teacher_model_server[{i}]: "tags" must be a list, got {type(tags)}.')
        configs.append(TeacherServerConfig(url=url, tags=tags))

    if len(configs) > 1:
        seen = {}
        for i, cfg in enumerate(configs):
            if not cfg.tags:
                raise ValueError(f'teacher_model_server[{i}]: "tags" must be non-empty with multiple teachers '
                                 '(each sample routes to exactly one teacher).')
            for tag in cfg.tags:
                if tag in seen:
                    raise ValueError(f'teacher_model_server: tag "{tag}" appears in both teacher[{seen[tag]}] '
                                     f'and teacher[{i}]; tags must not overlap.')
                seen[tag] = i
    return configs


def route_samples_to_teachers(
    samples: List[OnPolicySample],
    teacher_configs: List[TeacherServerConfig],
    tag_key: str = 'dataset',
) -> Dict[int, List[int]]:
    """Route each sample to exactly one teacher, returning ``{teacher_index: [sample_indices]}``.

    Single teacher: all samples route to it (tags ignored). Multiple teachers: route by tag;
    ``parse_teacher_model_server`` guarantees tags are non-empty and non-overlapping, so a tag
    maps to at most one teacher. Fails fast when a sample's tag matches no teacher.
    """
    routing: Dict[int, List[int]] = {i: [] for i in range(len(teacher_configs))}
    if len(teacher_configs) == 1:
        routing[0] = list(range(len(samples)))
        return routing

    tag_to_teacher = {tag: t_idx for t_idx, cfg in enumerate(teacher_configs) for tag in cfg.tags}
    for s_idx, sample in enumerate(samples):
        tag = sample.get_tag(tag_key)
        t_idx = tag_to_teacher.get(tag)
        if t_idx is None:
            raise ValueError(f'sample[{s_idx}] tag {tag!r} (from "{tag_key}") matches no teacher; '
                             f'available tags: {sorted(tag_to_teacher)}')
        routing[t_idx].append(s_idx)
    return routing


def fetch_teacher_parsed_by_routing(
    samples: List[OnPolicySample],
    requests: List[Any],
    teacher_configs: List[TeacherServerConfig],
    teacher_clients: List[Any],
    fetch_fn=None,
    tag_key: str = 'dataset',
    *,
    gather_fn=None,
    infer_fn=None,
    scatter_fn=None,
    is_main_process: bool = False,
) -> List[Any]:
    routing = route_samples_to_teachers(samples, teacher_configs, tag_key=tag_key)
    num_teachers = len(teacher_configs)

    def subset_requests(t_idx):
        return [requests[i] for i in routing[t_idx]]

    def client_for(t_idx):
        # teacher_clients is empty on non-main ranks (client is only used during infer on the main
        # process); pass None there and let the fetcher resolve its default.
        return teacher_clients[t_idx] if t_idx < len(teacher_clients) else None

    parsed: List[Any] = [None] * len(requests)

    def scatter_back(t_idx, subset):
        for local_i, s_idx in enumerate(routing[t_idx]):
            parsed[s_idx] = subset[local_i]

    if gather_fn is not None:
        # Phase A: gather every teacher's requests (serial collectives, same order on all ranks).
        handles = [gather_fn(subset_requests(t_idx)) for t_idx in range(num_teachers)]
        # Phase B: infer concurrently on the main process (HTTP to distinct teacher servers).
        parsed_globals: List[Any] = [None] * num_teachers
        if is_main_process:

            def _handle_requests(handle):
                if isinstance(handle, dict):
                    return handle.get('all_requests', handle)
                return handle

            def _infer_or_empty(t_idx):
                handle = handles[t_idx]
                if not _handle_requests(handle):
                    return []
                return infer_fn(handle, client_for(t_idx))

            if num_teachers == 1:
                parsed_globals[0] = _infer_or_empty(0)
            else:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=num_teachers) as pool:
                    futures = {
                        pool.submit(_infer_or_empty, t): t
                        for t in range(num_teachers) if _handle_requests(handles[t])
                    }
                    for t in range(num_teachers):
                        if not _handle_requests(handles[t]):
                            parsed_globals[t] = []
                    for fut in futures:
                        parsed_globals[futures[fut]] = fut.result()
        # Phase C: broadcast + slice each teacher's result back (serial collectives, rank-ordered).
        for t_idx in range(num_teachers):
            scatter_back(t_idx, scatter_fn(handles[t_idx], parsed_globals[t_idx]))
        return parsed

    for t_idx in range(num_teachers):
        scatter_back(t_idx, fetch_fn(subset_requests(t_idx), client_for(t_idx)))
    return parsed


def resolve_dynamic_opd_self_distillation(
    *,
    has_teacher_explicit: bool,
    is_self_distillation: bool,
) -> bool:
    if has_teacher_explicit or not is_self_distillation:
        return False
    return True


def should_compute_local_teacher_logps(
    *,
    has_teacher_explicit: bool,
    is_dynamic_self_distillation: bool,
    use_teacher_api: bool,
    has_opsd_batch: bool,
) -> bool:
    """Per-step gate for a local (non-API) teacher forward in OPD-RL."""
    if use_teacher_api:
        return False
    if has_teacher_explicit:
        return True
    return is_dynamic_self_distillation and has_opsd_batch


def build_opsd_samples(samples: List[OnPolicySample]) -> bool:
    """Build each sample's OPSD teacher view; return True if ANY sample is OPSD.

    ``build_teacher_view`` must run for every sample (not short-circuited by ``any()``)
    so ``teacher_messages`` is populated wherever ``teacher_prompt`` is set.
    """
    has_opsd = False
    for s in samples:
        if s.build_teacher_view():
            has_opsd = True
    return has_opsd


def remap_teacher_logps_to_student_frame(
    teacher_logps: torch.Tensor,
    teacher_completion_mask: torch.Tensor,
    student_completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Move per-token teacher logps from the OPSD teacher frame to the student frame.

    OPSD teacher and student share the *same* response tokens (same ids, same order)
    but different prompts, so their completion regions sit at different positions and
    the sequences differ in length. The OPD-RL advantage compares teacher vs. student
    logps position-by-position on the student ``completion_mask`` frame, so the teacher
    logps (returned aligned to the teacher frame) must be scattered onto the student
    frame: for each sample, the ``count`` valid teacher-completion values are placed at
    the student-completion positions in order. Returns ``[B, T_student]``.
    """
    out = torch.zeros_like(student_completion_mask, dtype=teacher_logps.dtype)
    for i in range(student_completion_mask.shape[0]):
        s_idx = student_completion_mask[i].nonzero(as_tuple=True)[0]
        t_idx = teacher_completion_mask[i].nonzero(as_tuple=True)[0]
        assert s_idx.numel() == t_idx.numel(), (
            f'OPSD response length mismatch at sample {i}: student={s_idx.numel()} teacher={t_idx.numel()}. '
            'Teacher and student must share the same response tokens.')
        out[i, s_idx] = teacher_logps[i, t_idx]
    return out
