# Copyright (c) ModelScope Contributors. All rights reserved.
# The SGLang backend of the ``decision`` (System-1 scoring) forward pass.

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import ModelConfig, TemplateConfig

#: Decision template families SGLang serves natively. Every other decision family
#: (jev / omnijev / thx02 / startlux / laya / generic) has no SGLang pooler and
#: stays on the transformers forward.
SGLANG_DECISION_TEMPLATES = ('kev', 'clef')

_KEV_TEXT_ARCH = 'Qwen3_5TextModel'
_CLEF_ARCH = 'Qwen3_5ForConditionalGeneration'


def is_sglang_decision_template(template_name: Optional[str]) -> bool:
    """Whether the decision template family is one SGLang serves natively."""
    return str(template_name or '') in SGLANG_DECISION_TEMPLATES


def _config_architectures(model_path: str) -> Optional[List[str]]:
    path = os.path.join(model_path, 'config.json')
    if not os.path.isfile(path):
        return None
    import json

    with open(path) as f:
        return json.load(f).get('architectures')


def validate_sglang_decision_checkpoint(model_path: str, family: str) -> None:
    """Check the checkpoint is in the format SGLang's decision loaders read.
    Raises ``ValueError`` with the exact gap otherwise
    """
    if model_path is None or not os.path.isdir(model_path):
        raise ValueError(
            f'sampler=sglang with template {family!r} serves a local decision checkpoint '
            f'directory, got {model_path!r}. Point --model at a {family} checkpoint in the '
            'SGLang decision format.')

    if family == 'kev':
        # A Kev LoRA release (no config.json) is served through the base its head.pt
        # pins; a full release carries its own Qwen3_5TextModel config.
        if not os.path.isfile(os.path.join(model_path, 'head.pt')):
            raise ValueError(
                f'{model_path!r} has no head.pt: SGLang recognises a kev checkpoint by its '
                'pointer head file. Point --model at a Kev release (backbone saved without '
                'an LM head plus head.pt, or a LoRA release whose head.pt pins its base).')
        archs = _config_architectures(model_path)
        if archs is not None and archs != [_KEV_TEXT_ARCH]:
            raise ValueError(
                f'{model_path!r} config architectures {archs} is not [{_KEV_TEXT_ARCH!r}]: '
                'SGLang detects a full kev checkpoint by the bare text-model architecture '
                '(backbone without an LM head). Re-export the merged checkpoint with '
                f'architectures [{_KEV_TEXT_ARCH!r}] and head.pt beside the weights.')
        if archs is None:
            import torch

            meta = torch.load(os.path.join(model_path, 'head.pt'), map_location='cpu',
                              weights_only=True)
            if 'base' not in meta:
                raise ValueError(
                    f'{model_path!r} is an adapter-only kev checkpoint whose head.pt does '
                    'not pin a base (no "base" key), which SGLang cannot resolve. Serve a '
                    'full kev release, or merge the adapter into its base first.')
        return

    if family == 'clef':
        missing = [name for name in ('joint_head_config.json', 'joint_head.safetensors')
                   if not os.path.isfile(os.path.join(model_path, name))]
        if missing:
            raise ValueError(
                f'{model_path!r} has no {"/".join(missing)}: SGLang recognises a clef '
                'checkpoint by its joint schema head files (joint_head_config.json + '
                'joint_head.safetensors beside the backbone weights).')
        archs = _config_architectures(model_path)
        if archs is not None and archs != [_CLEF_ARCH]:
            raise ValueError(
                f'{model_path!r} config architectures {archs} is not [{_CLEF_ARCH!r}]: '
                'SGLang detects a clef checkpoint by the Qwen3.5 conditional-generation '
                'architecture. Re-export the merged checkpoint with architectures '
                f'[{_CLEF_ARCH!r}] and the joint head files beside the weights.')
        return

    raise ValueError(f'unknown SGLang decision family {family!r}')


def _kev_decision_layout(input_ids: List[int], meta: Dict[str, Any]) -> List[int]:
    """KevTemplate's decision_meta -> SGLang's packed layout ints."""
    kind = int(meta['kinds'][0])
    decide = int(meta['decide_idx'][0])
    positions = [int(p) for p in meta['opt_idx'][0]]
    if decide != len(input_ids) - 1 or any(p >= decide for p in positions):
        raise ValueError(
            f'kev decision_meta is inconsistent: decide_idx={decide} for a '
            f'{len(input_ids)}-token prompt with option positions {positions}.')
    layout: List[int] = [len(input_ids), 1, kind, decide, decide + 1, len(positions)]
    for position in positions:
        layout += [position, position + 1]
    return layout


def _clef_decision_layout(input_ids: List[int], meta: Dict[str, Any]) -> List[int]:
    """ClefTemplate's decision_meta -> SGLang's packed layout ints"""
    kinds = meta['kinds']
    question_spans = meta['question_spans']
    option_spans = meta['option_spans']
    if not (len(kinds) == len(question_spans) == len(option_spans)):
        raise ValueError(
            f'clef decision_meta disagrees in length: kinds={len(kinds)}, '
            f'question_spans={len(question_spans)}, option_spans={len(option_spans)}.')
    layout: List[int] = [len(input_ids), len(kinds)]
    for kind, (start, end), spans in zip(kinds, question_spans, option_spans):
        layout += [int(kind), int(start), int(end), len(spans)]
        for option_start, option_end in spans:
            layout += [int(option_start), int(option_end)]
    return layout


def _softmax(logits: List[float]) -> List[float]:
    top = max(logits)
    weights = [math.exp(value - top) for value in logits]
    total = math.fsum(weights)
    if total <= 0 or not all(math.isfinite(value) for value in [*weights, total]):
        raise RuntimeError(f'decision logits are not finite: {logits!r}')
    return [weight / total for weight in weights]


def _row_predictions(logits: List[float], n_options: List[int]) -> List[Dict[str, Any]]:
    """Flat per-option logits -> one ``{'index', 'probs'}`` dict per question.

    The shape matches ``Template.decode_decision``'s per-record predictions, so the
    sglang forward lands in the same output rows / metrics as the transformers one.
    """
    if len(logits) != sum(n_options):
        raise RuntimeError(
            f'the decision head scored {len(logits)} options, but the record has '
            f'{sum(n_options)} ({n_options}); the backend dropped part of the layout.')
    predictions: List[Dict[str, Any]] = []
    offset = 0
    for count in n_options:
        scores = [float(value) for value in logits[offset:offset + count]]
        offset += count
        probs = _softmax(scores)
        predictions.append({'index': probs.index(max(probs)), 'probs': probs})
    return predictions


def _engine_kwargs(engine_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The offline Engine kwargs from the CLI's engine args"""
    kwargs: Dict[str, Any] = {}
    for name, value in (engine_args or {}).items():
        if name in ('tensor_parallel_size', 'data_parallel_size', 'tp_size', 'dp_size',
                    'pipeline_parallel_size', 'pp_size', 'max_batch_size'):
            if name != 'max_batch_size' and value not in (None, 1):
                raise ValueError(
                    f'Kev/Clef checkpoints are served on a single GPU by SGLang; got {name}='
                    f'{value}. Serve the decision checkpoint with all parallel sizes at 1.')
            continue
        kwargs[name] = value
    return kwargs


def decision_via_sglang(
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    rows: List[Dict[str, Any]],
    *,
    adapters: Optional[List[str]] = None,
    engine_args: Optional[Dict[str, Any]] = None,
    shutdown: bool = True,
) -> List[Any]:
    """Run the decision forward on SGLang's offline Engine, one prediction list per row.

    Encodes every row through swift's own decision template (the same prompts the
    transformers forward scores), converts each row's ``decision_meta`` into
    SGLang's packed ``decision_layout``, submits ``EmbeddingReqInput`` batches the
    way SGLang's /v1/systemone route does, and softmaxes the returned per-option
    logits into ``decode_decision``'s per-record prediction shape.
    """
    from swift.dev.builders import build_template, load_model_processor

    family = str(template_config.template or '')
    if family not in SGLANG_DECISION_TEMPLATES:
        raise ValueError(
            f'template {family!r} has no SGLang decision head; only '
            f'{list(SGLANG_DECISION_TEMPLATES)} are served natively.')
    if adapters:
        raise ValueError(
            'sampler=sglang decision serving takes no --adapters: SGLang scores the '
            'checkpoint weights it loads and supports no per-request LoRA on this route. '
            'Merge the adapter into the base first (swift export --merge_lora, then place '
            'the head files beside the merged weights).')

    model_path = str(model_config.model or '')
    validate_sglang_decision_checkpoint(model_path, family)

    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor, task_type='decision')
    template.set_mode('transformers')

    encoded_rows = []
    for index, row in enumerate(rows):
        encoded = template.encode(row)
        input_ids = list(encoded['input_ids'])
        meta = encoded.get('decision_meta')
        if meta is None:
            raise ValueError(
                f'row {index} encoded with no decision_meta: the {family} template '
                'did not see the decision columns (kinds/questions/options).')
        if family == 'kev':
            layout = _kev_decision_layout(input_ids, meta)
        else:
            layout = _clef_decision_layout(input_ids, meta)
        n_options = [int(n) for n in (meta.get('n_options') or [])]
        encoded_rows.append((input_ids, layout, n_options))
    if not encoded_rows:
        return []

    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import EmbeddingReqInput

    kwargs = _engine_kwargs(engine_args)
    engine = Engine(model_path=model_path, **kwargs)
    try:
        batch_size = (engine_args or {}).get('max_batch_size') or len(encoded_rows)
        per_row: List[Any] = []
        for start in range(0, len(encoded_rows), batch_size):
            chunk = encoded_rows[start:start + batch_size]
            request = EmbeddingReqInput(
                input_ids=[input_ids for input_ids, _, _ in chunk],
                decision_layout=[layout for _, layout, _ in chunk],
            )
            generator = engine.tokenizer_manager.generate_request(request, None)
            rets = engine.loop.run_until_complete(generator.__anext__())
            if isinstance(rets, dict):
                rets = [rets]
            if len(rets) != len(chunk):
                raise RuntimeError(
                    f'the SGLang engine returned {len(rets)} results for a '
                    f'{len(chunk)}-request batch.')
            for (_input_ids, _layout, n_options), ret in zip(chunk, rets):
                logits = ret.get('embedding')
                if not logits:
                    raise RuntimeError(
                        'the SGLang decision pooler scored no options for this request; '
                        'the decision layout did not match the prompt it read.')
                per_row.append(_row_predictions(list(logits), n_options))
        return per_row
    finally:
        if shutdown:
            engine.shutdown()
