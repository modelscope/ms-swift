# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-agnostic GKD teacher / encoding helpers.

These functions are stateless: they operate on ``GKDSample`` objects and a
``Template`` and return encoded dicts / teacher requests / assembled teacher
outputs.  Shared by the HF and Megatron GKD trainers.
"""
import torch
from typing import Any, Dict, List, Optional, Tuple

from swift.rl_core.data import GKDSample
from swift.template.base import Template
from swift.utils import get_cu_seqlens_from_position_ids, get_logger
from .gkd_loss import TeacherOutput
from .utils import (assemble_teacher_topk_logprobs, encode_sample, get_non_thinking_prefix_ids,
                    replace_assistant_response_with_ids)

logger = get_logger()


def encode_teacher_view(
    sample: GKDSample,
    template: Template,
    *,
    non_thinking_prefix_ids: Optional[List[int]] = None,
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
        teacher_row['messages'] = replace_assistant_response_with_ids(
            teacher_row['messages'],
            teacher_row['response_token_ids'],
            loss_mask=loss_mask,
            non_thinking_prefix_ids=non_thinking_prefix_ids)
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
    non_thinking_prefix_ids = get_non_thinking_prefix_ids(template)

    student_encoded_list: List[Dict[str, Any]] = []
    teacher_encoded_list: List[Dict[str, Any]] = []
    # Cannot use any() here because it short-circuits: build_teacher_view()
    # must be called for ALL samples to populate teacher_messages on each.
    has_opsd = False
    for s in samples:
        if s.build_teacher_view():
            has_opsd = True

    for s in samples:
        encoded = encode_sample(s, template, non_thinking_prefix_ids=non_thinking_prefix_ids)
        encoded.pop('_extra_kwargs', None)
        student_encoded_list.append(encoded)

        if has_opsd:
            teacher_encoded = encode_teacher_view(s, template, non_thinking_prefix_ids=non_thinking_prefix_ids)
            teacher_encoded_list.append(teacher_encoded)
        else:
            teacher_encoded_list.append(encoded.copy())

    return student_encoded_list, teacher_encoded_list, has_opsd


def build_teacher_requests(samples: List[GKDSample]) -> List[Any]:
    """Build teacher API requests from GKDSample objects.

    For OPSD samples, ``teacher_messages`` replaces the request messages.
    """
    requests = []
    for s in samples:
        req = s.to_infer_request()
        if s.teacher_messages:
            req.messages = s.teacher_messages
        requests.append(req)
    return requests


def assemble_teacher_output(
    parsed: List[Tuple],
    teacher_model_inputs: Dict[str, torch.Tensor],
    topk: Optional[int],
    template_padding_free: bool,
    device: torch.device,
) -> TeacherOutput:
    """Build a ``TeacherOutput`` from parsed teacher prompt logprobs.

    Handles both packed (padding_free) and non-packed layouts, including
    mrope / position-id based cu_seqlens detection.
    """
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
        topk=topk or 1,
        device=device,
        offsets=offsets,
    )

    teacher_out = TeacherOutput(topk_logprobs=topk_logprobs, topk_indices=topk_indices)
    if 'labels' in teacher_model_inputs:
        teacher_out.labels = teacher_model_inputs['labels']
    return teacher_out
