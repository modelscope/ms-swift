# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-agnostic GKD teacher / encoding helpers.

These functions are stateless: they operate on ``GKDSample`` objects and a
``Template`` and return encoded dicts / teacher requests / assembled teacher
outputs.  Shared by the HF and Megatron GKD trainers.
"""
import torch
from typing import Any, Dict, List, Optional, Tuple

from swift.rl_core.data import GKDSample, OnPolicySample
from swift.template.base import Template
from swift.utils import get_cu_seqlens_from_position_ids, get_logger
from .gkd_loss import TeacherOutput
from .utils import (assemble_teacher_topk_logprobs, encode_sample, get_response_prefix_ids,
                    replace_assistant_response_with_ids)

logger = get_logger()


def encode_teacher_view(
    sample: GKDSample,
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
        teacher_messages = getattr(s, 'teacher_messages', None)
        if teacher_messages:
            req.messages = teacher_messages
        # DEBUG: check if response_token_ids is populated
        if template is not None and s.response_token_ids:
            loss_mask = s.response_loss_mask or None
            ctk = s.extra.get('chat_template_kwargs') or {}
            prefix_ids = get_response_prefix_ids(template, sample_enable_thinking=ctk.get('enable_thinking'))
            req.messages = replace_assistant_response_with_ids([m.copy() for m in req.messages],
                                                               s.response_token_ids,
                                                               loss_mask,
                                                               non_thinking_prefix_ids=prefix_ids)
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
                # Fallback: take the last `count` entries (best effort).
                lps = lps[-count:]
                ixs = ixs[-count:]
        assert len(lps) == count, (f'Teacher logp count {len(lps)} != sampled tokens {count}. The teacher server '
                                   'must use the same tokenizer as the student (token-in-token-out).')
        # lps/ixs are per-position single-element lists ([[lp], ...]) -> [count, 1].
        out_lp[i, completion_indices] = torch.tensor(lps, dtype=torch.float32, device=device)
        out_ix[i, completion_indices] = torch.tensor(ixs, dtype=torch.long, device=device)
    return TeacherOutput(topk_logprobs=out_lp, topk_indices=out_ix)
