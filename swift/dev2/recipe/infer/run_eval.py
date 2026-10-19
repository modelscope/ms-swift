# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import datetime as dt
from typing import Any, Dict, List, Optional

from swift.dev.utils.logger import get_logger  # TODO(dev2): bridge until the utils batch
from swift.dev2.eval import build_task_config, model_name, run_evaluator, validate_eval_datasets

logger = get_logger()


def _guard_backend(backend: str) -> None:
    """Reject the message-only backends: eval builds and scores a local model, it cannot score a remote one."""
    if backend in {'client', 'no'}:
        raise ValueError(
            f'swift eval builds a local sampler to score, but sampler={backend!r} loads no local model. '
            'Use a local backend (vllm, sglang or transformers); to score an already-served model, evaluate it '
            'out of process against that service.')


def run_eval(model_config, template_config, eval_config, *, backend: str = 'vllm',
             engine_args: Optional[Dict[str, Any]] = None, adapters: Optional[List[str]] = None,
             quantize_config=None) -> Dict[str, Any]:
    """Score a model by building a twinkle sampler in-process and running EvalScope's Native runner on it.

    ``model_config`` carries the engine knobs ``build_sampler`` reads; ``adapters`` load live into the
    engine with the first pinned for every request (eval scores a single model). Returns the report
    dict that is also appended to ``eval_config.result_jsonl`` when set.
    """
    import twinkle

    from swift.dev2.builders import build_sampler, build_template, load_model_processor
    from swift.utils import append_to_jsonl

    if not eval_config.eval_dataset:
        raise ValueError('At least one --eval_dataset is required.')
    _guard_backend(backend)
    datasets = validate_eval_datasets(eval_config.eval_dataset)

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

    task_config = build_task_config(
        work_dir=eval_config.eval_output_dir,
        limit=eval_config.eval_limit,
        eval_batch_size=eval_config.eval_num_proc,
        dataset_args=eval_config.eval_dataset_args,
        generation_config=eval_config.eval_generation_config,
        extra_eval_args=eval_config.extra_eval_args)
    try:
        summary = run_evaluator(
            sampler, template, datasets=datasets, model_id=model_name(model_config),
            task_config=task_config, sampler_kwargs=sampler_kwargs)
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
