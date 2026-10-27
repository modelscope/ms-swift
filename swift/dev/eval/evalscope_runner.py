# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def model_name(model_config) -> str:
    """A short, report-friendly name for the model: the last path segment of its id/path."""
    return os.path.basename((model_config.model or 'model').rstrip('/'))


def validate_eval_datasets(eval_dataset: List[str]) -> List[str]:
    """Normalize ``--eval_dataset`` names against EvalScope's Native benchmark registry, rejecting unknowns.

    Returns the registry-cased names. Takes and returns a list rather than mutating a config object,
    so the standalone and in-training callers share one implementation.
    """
    from evalscope.api.registry import BENCHMARK_REGISTRY

    supported = sorted(BENCHMARK_REGISTRY)
    mapping = {name.lower(): name for name in supported}
    invalid = [name for name in eval_dataset if name.lower() not in mapping]
    if invalid:
        raise ValueError(f'eval_dataset {invalid} is not supported by the Native backend; '
                         f'supported datasets: {supported}')
    return [mapping[name.lower()] for name in eval_dataset]


def build_task_config(*,
                      work_dir: str,
                      limit: Optional[int] = None,
                      eval_batch_size: Optional[int] = None,
                      dataset_args: Optional[Dict[str, Any]] = None,
                      generation_config: Optional[Dict[str, Any]] = None,
                      extra_eval_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The EvalScope ``TaskConfig`` kwargs the Evaluator does not own.

    The Evaluator pins ``model`` / ``datasets`` / ``eval_type`` / ``eval_backend`` / ``model_task``
    itself; ``extra_eval_args`` is the escape hatch for any other TaskConfig field, merged last.
    """
    task_config: Dict[str, Any] = {'work_dir': work_dir}
    # Optional EvalScope fields are written only when set: a present None would override a
    # downstream default with an invalid value (generation_config=None; ABSENT batch size = 8).
    optional = {
        'limit': limit,
        'eval_batch_size': eval_batch_size,
        'dataset_args': dataset_args,
        'generation_config': generation_config,
    }
    task_config.update({key: value for key, value in optional.items() if value is not None})
    task_config.update(extra_eval_args or {})
    return task_config


def summarize(task_config) -> Any:
    """EvalScope report rows for the finished Native task, in the shape ``result_jsonl`` records."""
    from evalscope.summarizer import Summarizer

    return Summarizer.get_report_from_cfg(task_cfg=task_config)


def run_evaluator(sampler,
                  template,
                  *,
                  datasets: List[str],
                  model_id: str,
                  task_config: Dict[str, Any],
                  sampler_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """Drive twinkle_agentic's Evaluator over an already-built sampler and return the Native report rows.

    The sampler is NOT shut down here -- its lifecycle belongs to the caller. ``sampler_kwargs``
    carries per-request options such as the LoRA ``adapter_path`` to select.
    """
    from twinkle_agentic.evaluator import Evaluator

    evaluator = Evaluator(
        sampler=sampler,
        datasets=datasets,
        template=template,
        model_id=model_id,
        sampler_kwargs=sampler_kwargs,
        task_config=task_config)
    evaluator.run()
    return summarize(evaluator.resolved_task_config)


def run_evaluator_api(api,
                      *,
                      datasets: List[str],
                      model_id: str,
                      task_config: Dict[str, Any]) -> Any:
    """Drive twinkle_agentic's Evaluator over an OpenAI-compatible API client.

    The template lives server-side (the endpoint already renders chat templates), so none is
    passed here -- the Evaluator rejects a template alongside an api. Returns the same Native
    report-row shape as :func:`run_evaluator`.
    """
    from twinkle_agentic.evaluator import Evaluator

    evaluator = Evaluator(
        api=api,
        datasets=datasets,
        model_id=model_id,
        task_config=task_config)
    evaluator.run()
    return summarize(evaluator.resolved_task_config)
