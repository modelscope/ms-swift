"""In-process evaluation: build a twinkle sampler and drive it through EvalScope's Native runner.

``swift eval`` scores a model by constructing a twinkle sampler in this process and handing it to
``twinkle_agentic.evaluator.Evaluator``, which adapts the sampler to EvalScope's Native (custom-model)
runner. There is no HTTP deployment and no remote service: the sampler *is* the model under test, so a
LoRA adapter loads live into the engine and generation is driven per trajectory by EvalScope's own
concurrency -- a continuous-batching backend (vLLM/SGLang) keeps the engine saturated without one
trajectory waiting on another.
"""
from __future__ import annotations
import datetime as dt
import os
from typing import Any, Dict, List, Optional

from swift.dev.utils.logger import get_logger

logger = get_logger()


def _model_name(model_config) -> str:
    """A short, report-friendly name for the model: the last path segment of its id/path."""
    return os.path.basename((model_config.model or 'model').rstrip('/'))


def _guard_backend(backend: str) -> None:
    """Reject the message-only backends: eval builds and scores a local model, it cannot score a remote one."""
    if backend in {'client', 'no'}:
        raise ValueError(
            f'swift eval builds a local sampler to score, but sampler={backend!r} loads no local model. '
            'Use a local backend (vllm, sglang or transformers); to score an already-served model, evaluate it '
            'out of process against that service.')


def _validate_eval_datasets(eval_config) -> None:
    """Normalize ``--eval_dataset`` names against EvalScope's Native benchmark registry, rejecting unknowns."""
    from evalscope.api.registry import BENCHMARK_REGISTRY

    supported = sorted(BENCHMARK_REGISTRY)
    mapping = {name.lower(): name for name in supported}
    invalid = [name for name in eval_config.eval_dataset if name.lower() not in mapping]
    if invalid:
        raise ValueError(f'eval_dataset {invalid} is not supported by the Native backend; '
                         f'supported datasets: {supported}')
    eval_config.eval_dataset = [mapping[name.lower()] for name in eval_config.eval_dataset]


def _build_task_config(eval_config) -> Dict[str, Any]:
    """dev ``EvalConfig`` -> the EvalScope ``TaskConfig`` kwargs the Evaluator does not own.

    The Evaluator pins ``model`` / ``datasets`` / ``eval_type`` / ``eval_backend`` / ``model_task`` itself,
    so none of those may appear here. ``extra_eval_args`` is the escape hatch for any other EvalScope
    ``TaskConfig`` field and is merged last; an owned key smuggled through it is rejected loudly by the
    Evaluator rather than silently dropped. ``eval_num_proc`` becomes ``eval_batch_size``, which is both
    EvalScope's request concurrency and (for a non-continuous backend) the sampler micro-batch width.
    """
    task_config: Dict[str, Any] = {
        'work_dir': eval_config.eval_output_dir,
        'limit': eval_config.eval_limit,
        'eval_batch_size': eval_config.eval_num_proc,
        'dataset_args': eval_config.eval_dataset_args,
        'generation_config': eval_config.eval_generation_config,
    }
    task_config.update(eval_config.extra_eval_args or {})
    return task_config


def _summarize(task_config):
    """EvalScope report rows for the finished Native task, in the shape ``result_jsonl`` records."""
    from evalscope.summarizer import Summarizer

    return Summarizer.get_report_from_cfg(task_cfg=task_config)


def run_eval(model_config, template_config, eval_config, *, backend: str = 'vllm',
             engine_args: Optional[Dict[str, Any]] = None, adapters: Optional[List[str]] = None,
             quantize_config=None) -> Dict[str, Any]:
    """Score a model by building a twinkle sampler in-process and running EvalScope's Native runner on it.

    Args:
        model_config: model id/path plus the engine knobs ``build_sampler`` reads (dtype, max_model_len).
        template_config: chat template; the Evaluator uses it to decode and to parse tool calls.
        eval_config: datasets, limit, generation config, output dir, result jsonl.
        backend: 'vllm' / 'sglang' / 'transformers' -- the local engine that serves the model under test.
        engine_args: forwarded verbatim to the engine (tensor parallelism, memory fraction, ...).
        adapters: LoRA checkpoints loaded live into the engine; the first is selected for every request.
        quantize_config: load-time quantization for the transformers backend (see ``build_sampler``).

    Returns:
        The report dict that is also appended to ``eval_config.result_jsonl`` when set.
    """
    import twinkle
    from twinkle_agentic.evaluator import Evaluator

    from swift.dev.builders import build_sampler, build_template, load_model_processor
    from swift.utils import append_to_jsonl

    if not eval_config.eval_dataset:
        raise ValueError('At least one --eval_dataset is required.')
    _guard_backend(backend)
    _validate_eval_datasets(eval_config)

    # One local engine, no Ray placement and no data-parallel mesh: multi-GPU tensor parallelism still works
    # through engine_args (e.g. vllm_tensor_parallel_size), but DP-across-replicas is not wired for eval.
    twinkle.initialize(mode='local')
    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    sampler = build_sampler(
        model_config, backend=backend, engine_args=engine_args, template=template,
        adapters=adapters or None, quantize_config=quantize_config)

    # build_sampler only reserves the engine's LoRA slots; the adapter is selected per request, so pin the
    # one being scored. eval scores a single model, hence adapters[0] (more than one is ambiguous).
    sampler_kwargs: Optional[Dict[str, Any]] = None
    if adapters:
        if len(adapters) > 1:
            logger.warning(f'eval scores a single model; using adapters[0]={adapters[0]!r} and ignoring the rest.')
        sampler_kwargs = {'adapter_path': adapters[0]}

    model_name = _model_name(model_config)
    evaluator = Evaluator(
        sampler=sampler,
        datasets=eval_config.eval_dataset,
        template=template,
        model_id=model_name,
        sampler_kwargs=sampler_kwargs,
        task_config=_build_task_config(eval_config))
    try:
        evaluator.run()
        summary = _summarize(evaluator.resolved_task_config)
    finally:
        sampler.shutdown()

    report: Dict[str, Any] = {
        'Native': summary,
        'time': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
        'model': model_config.model,
        'adapters': list(adapters or []),
        'eval_output_dir': eval_config.eval_output_dir,
        'eval_limit': eval_config.eval_limit,
    }
    if eval_config.result_jsonl:
        append_to_jsonl(eval_config.result_jsonl, report)
    return report
