"""run_infer: offline inference over a dataset, on twinkle's Sampler.

dev counterpart of legacy ``swift infer`` (``swift/pipelines/infer/infer.py::SwiftInfer`` plus
``infer/utils.py``). It covers the same ground as legacy:

- three generation backends (vllm / sglang / transformers; lmdeploy is deliberately dropped),
- LoRA, either applied at request time or merged in first,
- the pooling task types (seq_cls / embedding / reranker), which run a forward pass rather than
  generation: on vLLM/SGLang through the sampler's ``encode``, on transformers through a HF forward,
- a generative reranker, which is a decoder-only causal LM scored off generation (the yes/no logprob
  difference of its first token) on vLLM/SGLang, and through a HF forward on transformers,
- streaming to the terminal, incremental result writing with cross-process gathering, and the
  acc/rouge metrics. The interactive REPL now lives in :mod:`swift.dev.recipe.infer_tui`.

What is structured differently from legacy, and why:

- generation and pooling are two functions with one dispatcher, instead of ``task_type`` branches
  threaded through a single ``_batch_infer``. They share the sampler surface but not the call -- one
  decodes tokens via ``sample``, the other runs one forward via ``encode`` (or a HF forward on
  transformers) -- so keeping them apart is what stops each from carrying the other's cases.
- there is no ``__getattr__`` proxy onto the engine. Legacy's ``SwiftInfer.infer`` was actually the
  engine's method, which made the public surface depend on the backend; here the recipe owns it.
- the sampler is built once and shut down in a ``finally``, so a crash mid-run still frees the GPU.
"""
from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import json

if TYPE_CHECKING:
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        TemplateConfig,
        TunerConfig,
    )

logger = logging.getLogger(__name__)

#: How many top logprobs to request when scoring a generative reranker off generation. The positive and
#: negative tokens must both land in this window or the score cannot be read; 20 is within the default
#: ``max_logprobs`` of both vLLM and sglang and comfortably covers the two tokens a trained reranker
#: puts its mass on.
_GENERATIVE_RERANKER_TOP_LOGPROBS = 20


def run_infer(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    backend: Literal['vllm', 'sglang', 'transformers'] = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    distributed_config: Optional[DistributedConfig] = None,
    tuner_config: Optional[TunerConfig] = None,
    adapters: Optional[List[str]] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    plugin_config: Optional[PluginConfig] = None,
    merge_lora: bool = False,
    num_samples: int = 1,
    max_rows: Optional[int] = None,
    split_dataset_ratio: float = 0.01,
    output_path: Optional[str] = None,
    write_batch_size: Optional[int] = None,
    metric: Optional[Literal['acc', 'rouge']] = None,
    strict: bool = True,
    _shutdown: bool = True,
) -> List[Dict[str, Any]]:
    """Infer over a dataset and return one result row per input.

    Args:
        model_config: model id/path, dtype, ``task_type``. A pooling ``task_type`` (see
            ``builders.is_pooling_task``) switches to the forward path, which still honours ``backend``:
            vLLM/SGLang serve it through ``sampler.encode``, transformers through a HF forward. A
            ``generative_reranker`` also returns one value per row but scores off generation.
        template_config: chat template, and the ``system`` that overrides the dataset's.
        dataset_config: what to infer over. See ``split_dataset_ratio`` for how the split is chosen.
        generation_config: decoding knobs. ``stream=True`` prints tokens as they arrive.
        backend: generation engine. ``transformers`` is the one that runs anything and the only one
            that can degrade per row (see ``strict``).
        engine_args: forwarded verbatim to the engine.
        distributed_config: when it declares DP > 1, a DeviceMesh is built and ``sample`` slices the
            inputs across ranks by itself.
        tuner_config: source of ``adapters`` when not given explicitly.
        adapters: LoRA checkpoints. One adapter is applied to the whole run; the engine is configured
            for LoRA at construction because vLLM cannot enable it later.
        merge_lora: fold the adapters into the base weights first and infer on the merged model.
            Slower to start and needs disk, but then costs nothing per request -- which is the right
            trade for a long offline run, and the reason legacy defaulted to it for export.
        num_samples: completions per prompt. The first one is what ``response`` and the metric see.
        max_rows: stop after this many rows, for smoke tests.
        split_dataset_ratio: how much of ``dataset`` becomes the eval split when
            ``DatasetConfig.val_dataset`` is not set. Legacy's semantics, kept because a config that
            names one ``--dataset`` means "infer over the eval slice", not "over everything".
        output_path: jsonl destination.
        write_batch_size: rows per incremental flush. Without it results are written once at the end,
            so a crash loses the run; with it the file grows as inference proceeds. Under DP the
            batches are gathered across ranks before writing, so the file stays whole.
        metric: 'acc' or 'rouge', computed against the reference answers.
        strict: transformers backend only -- when False a row that fails to encode or generate is
            recorded with an empty response instead of aborting the run.
        _shutdown: leave True. False keeps the engine alive for tests that reuse it.

    Returns:
        Result rows, each with ``response`` / ``responses`` / ``labels`` / ``messages`` plus every
        column the dataset row already had.
    """
    from swift.dev.builders import is_pooling_task, load_prompt_rows
    from swift.dev.plugin import PluginRegistry
    from swift.dev.recipe.assembly import TrainAssembly

    # Inference has no Configs to cross-validate, but it still needs the run's plugin files imported:
    # a custom model or dataset lives in one, and its registration must precede the first name lookup.
    PluginRegistry.load_configured(plugin_config)
    TrainAssembly.initialize_twinkle(distributed_config)
    adapters = _resolve_adapters(adapters, tuner_config)
    if merge_lora and adapters:
        model_config, adapters = _merge_adapters(model_config, template_config, adapters)

    rows = load_prompt_rows(dataset_config, max_rows, split_dataset_ratio)
    if not rows:
        raise ValueError('run_infer got an empty dataset. Set DatasetConfig.dataset or .val_dataset.')

    task_type = model_config.task_type or 'causal_lm'
    if is_pooling_task(task_type):
        return _run_pooling(
            model_config,
            template_config,
            distributed_config,
            rows,
            adapters,
            quantize_config,
            output_path,
            metric,
            backend=backend,
            engine_args=engine_args,
            shutdown=_shutdown,
        )
    if task_type == 'generative_reranker':
        return _run_generative_reranker(
            model_config,
            template_config,
            generation_config,
            distributed_config,
            rows,
            adapters,
            quantize_config,
            output_path,
            metric,
            backend=backend,
            engine_args=engine_args,
            shutdown=_shutdown,
        )

    return _run_generative(
        model_config,
        template_config,
        generation_config,
        rows,
        backend=backend,
        engine_args=engine_args,
        distributed_config=distributed_config,
        adapters=adapters,
        quantize_config=quantize_config,
        num_samples=num_samples,
        output_path=output_path,
        write_batch_size=write_batch_size,
        metric=metric,
        strict=strict,
        shutdown=_shutdown,
    )


def _run_generative(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig],
    rows: List[Dict[str, Any]],
    *,
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    distributed_config: Optional[DistributedConfig],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    num_samples: int,
    output_path: Optional[str],
    write_batch_size: Optional[int],
    metric: Optional[str],
    strict: bool,
    shutdown: bool,
) -> List[Dict[str, Any]]:
    """The causal-LM path: encode prompts, sample, write."""
    from swift.dev.builders import (build_device_mesh_if_dp, build_sampler, build_template, load_model_processor,
                                    split_prompt_and_reference, to_sampling_params)

    logger.info(f'run_infer: {len(rows)} prompts, backend={backend}, num_samples={num_samples}')
    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    device_mesh = build_device_mesh_if_dp(distributed_config)
    adapter_path = adapters[0] if adapters else None

    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=engine_args,
        device_mesh=device_mesh,
        template=template,
        adapters=adapters,
        quantize_config=quantize_config)
    try:
        params = to_sampling_params(generation_config, num_samples=num_samples)
        streaming = bool(generation_config is not None and generation_config.stream)
        writer = _IncrementalWriter(output_path, write_batch_size)
        results: List[Dict[str, Any]] = []

        for batch in _batches(rows, write_batch_size or len(rows)):
            trajectories, labels = split_prompt_and_reference(batch, template_config)
            if streaming:
                texts = _sample_streaming(sampler, trajectories, params, adapter_path)
            else:
                texts = _sample_batch(sampler, trajectories, params, adapter_path, backend, strict)
            batch_results = _assemble_results(batch, trajectories, labels, texts)
            results.extend(batch_results)
            writer.write(batch_results)
    finally:
        if shutdown:
            sampler.shutdown()

    writer.finish(results)
    if metric:
        logger.info(f'run_infer metric: {compute_metric(results, metric)}')
    return results


def _sample_batch(sampler, trajectories: List[Dict[str, Any]], params: Any, adapter_path: Optional[str], backend: str,
                  strict: bool) -> List[List[str]]:
    """One ``sample()`` call for the batch. ``strict`` only reaches the transformers backend."""
    from swift.dev.builders import sampled_texts

    kwargs: Dict[str, Any] = {}
    if adapter_path is not None:
        kwargs['adapter_path'] = adapter_path
    if backend == 'transformers':
        kwargs['strict'] = strict
    elif not strict:
        logger.warning(f'strict=False is only honoured by the transformers backend; the {backend} engine fails '
                       'the whole batch on a bad row. Switch backend if per-row tolerance matters.')
    return sampled_texts(sampler.sample(trajectories, params, **kwargs))


def _sample_streaming(sampler, trajectories: List[Dict[str, Any]], params: Any,
                      adapter_path: Optional[str]) -> List[List[str]]:
    """Stream each prompt to stdout, and return the accumulated texts.

    One prompt at a time by necessity: ``sample_stream`` is a single-request API, and interleaving
    several streams onto one terminal would produce unreadable output. This is a display mode, not a
    throughput mode -- for many rows leave ``stream`` off.
    """
    texts: List[List[str]] = []
    for index, trajectory in enumerate(trajectories):
        print(f'[{index}] ', end='', flush=True)
        pieces: List[str] = []
        for delta, _finish_reason in sampler.sample_stream(trajectory, params, adapter_path=adapter_path):
            if delta:
                print(delta, end='', flush=True)
                pieces.append(delta)
        print(flush=True)
        texts.append([''.join(pieces)])
    return texts


def _run_pooling(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    rows: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    output_path: Optional[str],
    metric: Optional[str],
    *,
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
) -> List[Dict[str, Any]]:
    """The forward-pass path for seq_cls / embedding / reranker.

    Two ways to run the same pooling forward, chosen by ``backend``:

    - vLLM / SGLang have a pooling head, so the model is built as a pooling engine and served through
      ``sampler.encode`` -- the same sampler surface the generative path uses, which is what lets one
      backend serve both. This is the throughput path.
    - transformers has no pooling head, so it falls back to ``build_model`` + ``forward_only(task=..)``,
      a plain HF forward. This is the reach path.

    Either way each row yields one plain-Python value (a vector, class logits, or a score), so the
    result assembly (:func:`_finalize_forward_results`) is shared with the generative-reranker path. The
    label comes from the row's own ``label`` column rather than from a trailing assistant turn, because
    there is no completion to strip.
    """
    from swift.dev.builders import to_trajectory

    task_type = model_config.task_type
    trajectories = [to_trajectory(row, list(row['messages']), template_config) for row in rows]

    if backend in ('vllm', 'sglang'):
        per_row = _pooling_via_sampler(
            model_config, template_config, distributed_config, trajectories, adapters, quantize_config, backend,
            engine_args, shutdown)
    else:
        per_row = _pooling_via_forward(
            model_config, template_config, distributed_config, rows, trajectories, adapters, quantize_config)

    return _finalize_forward_results(rows, per_row, task_type, backend, output_path, metric)


def _finalize_forward_results(
    rows: List[Dict[str, Any]],
    per_row: List[Any],
    task_type: Optional[str],
    backend: str,
    output_path: Optional[str],
    metric: Optional[str],
) -> List[Dict[str, Any]]:
    """Assemble, write and score the one-value-per-row output of a forward-pass path.

    Shared by :func:`_run_pooling` and :func:`_run_generative_reranker`: both produce exactly one plain
    value per row (a vector, class logits, or a relevance score), so the row shape is the same. The label
    comes from the row's own ``label`` column rather than a trailing assistant turn, because there is no
    completion to strip.
    """
    logger.info(f'run_infer: {len(rows)} rows, task_type={task_type}, backend={backend}')
    results = []
    for row, output in zip(rows, per_row):
        passthrough = {key: value for key, value in row.items() if key != 'messages'}
        results.append({
            'response': output,
            'responses': [output],
            'labels': row.get('label'),
            'messages': list(row['messages']),
            **passthrough
        })
    if output_path:
        _write_jsonl(output_path, results)
        logger.info(f'run_infer: wrote {len(results)} rows to {output_path}')
    if metric:
        logger.info(f'run_infer metric: {compute_metric(results, metric)}')
    return results


def _pooling_via_sampler(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
) -> List[Any]:
    """Run the pooling forward on a vLLM/SGLang sampler's ``encode``, one plain value per row."""
    from swift.dev.builders import (build_device_mesh_if_dp, build_sampler, build_template, load_model_processor,
                                    pooled_data, to_pooling_params)

    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    device_mesh = build_device_mesh_if_dp(distributed_config)
    adapter_path = adapters[0] if adapters else None

    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=engine_args,
        device_mesh=device_mesh,
        template=template,
        adapters=adapters,
        quantize_config=quantize_config)
    try:
        pooling_params = to_pooling_params(model_config.task_type)
        kwargs: Dict[str, Any] = {}
        if adapter_path is not None:
            kwargs['adapter_path'] = adapter_path
        return pooled_data(sampler.encode(trajectories, pooling_params, **kwargs))
    finally:
        if shutdown:
            sampler.shutdown()


def _pooling_via_forward(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    rows: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
) -> List[Any]:
    """Run the pooling forward as a plain HF forward (transformers backend), one plain value per row."""
    from swift.dev.builders import build_model, build_template, load_model_processor
    from swift.dev.config import DistributedConfig

    task_type = model_config.task_type
    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    model = build_model(
        model_config, distributed_config or DistributedConfig(), quantize_config=quantize_config)
    if adapters:
        # add_adapter_to_model takes a checkpoint directory as well as a PeftConfig, so a trained
        # adapter is loaded here rather than built -- apply_tuner would create a fresh, untrained one.
        for index, adapter in enumerate(adapters):
            model.add_adapter_to_model(f'adapter_{index}' if index else 'default', adapter)

    encoded = [template.encode(trajectory) for trajectory in trajectories]
    outputs = model.forward_only(inputs=encoded, task=task_type, return_logits=True)
    return _per_row_outputs(outputs, len(rows))


def _per_row_outputs(outputs: Any, num_rows: int) -> List[Any]:
    """Unpack ``forward_only``'s output into one plain-Python value per row.

    The shape depends on the head: seq_cls gives logits per row, embedding gives a vector. Both are
    returned as lists so the result is jsonl-serialisable, which is the whole point of this path.
    """
    tensor = outputs
    if isinstance(outputs, dict):
        for key in ('logits', 'embedding', 'last_hidden_state'):
            if key in outputs:
                tensor = outputs[key]
                break
    if hasattr(tensor, 'tolist'):
        listed = tensor.tolist()
        if isinstance(listed, list) and len(listed) == num_rows:
            return listed
        return [listed]
    return list(tensor) if isinstance(tensor, (list, tuple)) else [tensor] * num_rows


def _run_generative_reranker(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig],
    distributed_config: Optional[DistributedConfig],
    rows: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    output_path: Optional[str],
    metric: Optional[str],
    *,
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
) -> List[Dict[str, Any]]:
    """Score a decoder-only (generative) reranker, one relevance score per row.

    A generative reranker is a causal LM, not a pooling model: its score is the difference between the
    logprob of a positive token (``yes``) and a negative one (``no``) at the first generated position.
    That is exactly what ``swift.utils.torch_utils.get_generative_reranker_logits`` computes from the
    lm_head for the HF path, and what the generation path reproduces here from logprobs -- the two agree
    because the log-softmax normaliser cancels in the difference.

    So the backend split differs from pooling: vLLM/SGLang run it through *generation* (``sample``),
    since neither serves a decoder-only reranker through ``encode`` (sglang's docs are explicit that it
    must not be launched with ``--is-embedding``); transformers runs the same HF forward the pooling
    path uses, with ``task='generative_reranker'``. Either way each row yields one score, so the result
    assembly is shared.
    """
    from swift.dev.builders import to_trajectory

    task_type = model_config.task_type
    trajectories = [to_trajectory(row, list(row['messages']), template_config) for row in rows]

    if backend in ('vllm', 'sglang'):
        per_row = _generative_reranker_via_sampler(
            model_config, template_config, generation_config, distributed_config, trajectories, adapters,
            quantize_config, backend, engine_args, shutdown)
    else:
        per_row = _pooling_via_forward(
            model_config, template_config, distributed_config, rows, trajectories, adapters, quantize_config)

    return _finalize_forward_results(rows, per_row, task_type, backend, output_path, metric)


def _generative_reranker_via_sampler(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig],
    distributed_config: Optional[DistributedConfig],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
) -> List[float]:
    """Score a generative reranker on a vLLM/SGLang generation engine, one score per row."""
    from swift.dev.builders import (build_device_mesh_if_dp, build_sampler, build_template, load_model_processor,
                                    to_sampling_params)

    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    device_mesh = build_device_mesh_if_dp(distributed_config)
    adapter_path = adapters[0] if adapters else None

    # Built as a generation engine, not a pooling runner: build_sampler treats ``generative_reranker`` as
    # a generative task_type, so no ``runner='pooling'``/``is_embedding`` is injected.
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=engine_args,
        device_mesh=device_mesh,
        template=template,
        adapters=adapters,
        quantize_config=quantize_config)
    try:
        tokenizer = _tokenizer_of(processor)
        positive_id, negative_id = _generative_reranker_token_ids(tokenizer)
        # temperature is forced to 1.0 so the reported logprobs are the log-softmax of the raw logits:
        # only then does logprob(yes) - logprob(no) equal the logit difference the HF path returns (any
        # other temperature scales the logits before the softmax and changes the gap). max_tokens=1 --
        # not 0 -- because twinkle's max_tokens==0 logprobs-only path drops logprobs, and we need the
        # first generated token's.
        params = to_sampling_params(
            generation_config, max_tokens=1, temperature=1.0, logprobs=_GENERATIVE_RERANKER_TOP_LOGPROBS)
        kwargs: Dict[str, Any] = {}
        if adapter_path is not None:
            kwargs['adapter_path'] = adapter_path
        responses = sampler.sample(trajectories, params, **kwargs)
        return [_score_from_logprobs(response, positive_id, negative_id) for response in responses]
    finally:
        if shutdown:
            sampler.shutdown()


def _score_from_logprobs(response: Any, positive_id: int, negative_id: int) -> float:
    """``logprob(yes) - logprob(no)`` at the first generated position of one SampleResponse."""
    sequence = response.sequences[0]
    if not sequence.logprobs:
        raise RuntimeError('generative reranker scoring requested logprobs but the sampler returned none; '
                           'the backend may not honour logprobs on this path.')
    topk = dict(sequence.logprobs[0])
    if positive_id not in topk or negative_id not in topk:
        raise RuntimeError(
            f'generative reranker: the positive/negative tokens ({positive_id}/{negative_id}) are not both '
            f'in the top-{len(topk)} logprobs of the first generated token. Raise '
            '_GENERATIVE_RERANKER_TOP_LOGPROBS (or check GENERATIVE_RERANKER_POSITIVE_TOKEN/'
            'GENERATIVE_RERANKER_NEGATIVE_TOKEN) so both are returned.')
    return topk[positive_id] - topk[negative_id]


def _generative_reranker_token_ids(tokenizer: Any) -> Tuple[int, int]:
    """The (positive, negative) token ids, from the same env vars the HF scoring path reads."""
    positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
    return tokenizer.convert_tokens_to_ids(positive_token), tokenizer.convert_tokens_to_ids(negative_token)


def _tokenizer_of(processor: Any) -> Any:
    """A tokenizer from whatever ``load_model_processor`` returned (a tokenizer, or a processor wrapping one)."""
    return getattr(processor, 'tokenizer', processor)


class _IncrementalWriter:
    """Append results to jsonl as batches complete, gathering across DP ranks first.

    Why gather: with DP > 1 each rank holds only its own slice, so a naive per-rank write either
    interleaves partial files or has every rank overwrite the same path. Legacy solved this with
    ``JsonlWriter(gather_obj=True)``; twinkle's ``gather_object`` is the same idea.

    When ``batch_size`` is None this does nothing until :meth:`finish`, which writes everything at
    once -- the simple case stays simple.
    """

    def __init__(self, output_path: Optional[str], batch_size: Optional[int]):
        self.output_path = output_path
        self.incremental = bool(output_path and batch_size)
        self._started = bool(output_path and os.path.exists(output_path))

    def write(self, batch: List[Dict[str, Any]]) -> None:
        if not self.incremental:
            return
        rows = _gather_rows(batch)
        if rows is None:
            return  # not the writing rank
        _write_jsonl(self.output_path, rows, append=self._started)
        self._started = True
        logger.info(f'run_infer: flushed {len(rows)} rows to {self.output_path}')

    def finish(self, results: List[Dict[str, Any]]) -> None:
        if self.incremental or not self.output_path:
            return
        rows = _gather_rows(results)
        if rows is None:
            return
        _write_jsonl(self.output_path, rows, append=os.path.exists(self.output_path))
        logger.info(f'run_infer: wrote {len(rows)} rows to {self.output_path}')


def _gather_rows(rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """All ranks' rows on the writing rank, None elsewhere. A no-op without torch.distributed."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return rows

    from twinkle.utils import framework_util, is_master

    gathered = framework_util.gather_object(rows, device_mesh=None)
    return gathered if is_master() else None


def _assemble_results(
    rows: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
    labels_list: List[Optional[str]],
    texts: List[List[str]],
) -> List[Dict[str, Any]]:
    """Zip prompts, labels and completions into the output rows.

    ``messages`` comes from the trajectory actually sampled -- prompt-only (the reference answer was
    popped) and carrying any system substitution -- so appending the response yields one assistant
    turn, not the reference followed by the model's.
    """
    results = []
    for row, trajectory, label, candidates in zip(rows, trajectories, labels_list, texts):
        messages = list(trajectory['messages'])
        if candidates:
            messages = messages + [{'role': 'assistant', 'content': candidates[0]}]
        passthrough = {key: value for key, value in row.items() if key != 'messages'}
        results.append({
            'response': candidates[0] if candidates else None,
            'responses': candidates,
            'labels': label,
            'messages': messages,
            **passthrough
        })
    return results


def _resolve_adapters(adapters: Optional[List[str]], tuner_config: Optional[TunerConfig]) -> Optional[List[str]]:
    if adapters:
        return list(adapters)
    if tuner_config is not None and getattr(tuner_config, 'adapters', None):
        return list(tuner_config.adapters)
    return None


def _merge_adapters(model_config: ModelConfig, template_config: TemplateConfig,
                    adapters: List[str]) -> Tuple[ModelConfig, None]:
    """Merge the adapters into the base weights and return a ModelConfig pointing at the result.

    The returned config is a copy: mutating the caller's would make a second run with the same object
    silently infer on the merged directory. Adapters come back as None because they are now part of
    the weights -- passing them again would apply the same delta twice.
    """
    import dataclasses

    from swift.dev.config import TunerConfig
    from swift.dev.recipe.merge_lora import run_merge_lora

    merged = run_merge_lora(
        model_config, TunerConfig(adapters=list(adapters)), template_config=template_config, device_map='cpu')
    logger.info(f'run_infer: merged {len(adapters)} adapter(s) into {merged}')
    return dataclasses.replace(model_config, model=merged), None


def _batches(rows: List[Dict[str, Any]], size: int):
    for start in range(0, len(rows), max(1, size)):
        yield rows[start:start + max(1, size)]


def _write_jsonl(path: str, rows: List[Dict[str, Any]], append: bool = False) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'a' if append else 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def compute_metric(results: List[Dict[str, Any]], metric: Literal['acc', 'rouge']) -> Dict[str, float]:
    """Score the first completion of each row against its reference answer.

    ``acc`` is exact string equality, which is what legacy's ``--metric acc`` measured -- NOT the
    token-level ``twinkle.metric.Accuracy``, which scores logits against label ids and would report a
    different (and much higher) number for the same run.

    Only the first completion is scored: with ``num_samples > 1`` a best-of-n score is a different
    measurement (pass@n) and reporting it as accuracy would overstate the model. Rows without a
    reference are skipped rather than counted as wrong.
    """
    from twinkle.metric import ExactMatch, RougeBleu

    pairs = [(r['response'], r['labels']) for r in results if r.get('labels') is not None and r.get('response')]
    if not pairs:
        logger.warning('metric requested but no row has both a response and a reference answer.')
        return {}
    predictions, references = zip(*pairs)
    scorer = ExactMatch() if metric == 'acc' else RougeBleu()
    scorer.accumulate(predictions=list(predictions), references=list(references))
    return scorer.calculate()
