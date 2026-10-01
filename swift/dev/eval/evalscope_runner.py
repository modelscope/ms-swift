"""Shared EvalScope Native-runner adaptation.

``swift eval`` (standalone) and the in-training generative eval (``predict_with_generate``) both score a
model by handing a twinkle sampler to ``twinkle_agentic.evaluator.Evaluator``, which adapts it to
EvalScope's Native (custom-model) runner. This module is the single place that knows EvalScope's contract
-- benchmark-name normalization, the ``TaskConfig`` kwargs the Evaluator does not own, driving the run, and
reading back the report -- so the adaptation is written once and called from both places.

What is deliberately NOT here: the sampler lifecycle and the report/metric shape. ``swift eval`` builds a
sampler per run and shuts it down; training holds one persistent sampler across evals (see
``TrainAssembly.build_loop`` / ``SFTLoop``) and folds the summary into its own ``eval_history`` / tracker.
Each caller owns those, and calls :func:`run_evalscope` for the EvalScope part in between.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def model_name(model_config) -> str:
    """A short, report-friendly name for the model: the last path segment of its id/path."""
    return os.path.basename((model_config.model or 'model').rstrip('/'))


def validate_eval_datasets(eval_dataset: List[str]) -> List[str]:
    """Normalize ``--eval_dataset`` names against EvalScope's Native benchmark registry, rejecting unknowns.

    Returns the registry-cased names. Takes and returns a list rather than mutating a config object, so the
    standalone caller (``EvalConfig.eval_dataset``) and the in-training caller (``TrainConfig.eval_dataset``)
    -- which carry the list on different objects -- share one implementation.
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

    The Evaluator pins ``model`` / ``datasets`` / ``eval_type`` / ``eval_backend`` / ``model_task`` itself,
    so none of those may appear here. ``extra_eval_args`` is the escape hatch for any other EvalScope
    ``TaskConfig`` field and is merged last; an owned key smuggled through it is rejected loudly by the
    Evaluator rather than silently dropped. ``eval_batch_size`` is both EvalScope's request concurrency and
    (for a non-continuous backend) the sampler micro-batch width.

    Explicit keyword arguments rather than one config object: the standalone path reads these off
    ``EvalConfig`` (``eval_output_dir`` / ``eval_num_proc``), the in-training path off ``TrainConfig`` plus
    the run's output dir, and the two do not share field names.
    """
    task_config: Dict[str, Any] = {'work_dir': work_dir}
    # Optional EvalScope fields are written only when the caller sets them. Emitting a present None
    # instead would override a downstream default with an invalid value: TaskConfig rejects
    # generation_config=None (it wants a dict / GenerateConfig), and the Evaluator rejects
    # eval_batch_size=None -- it sizes its sampler batcher with ``get('eval_batch_size', 8)`` and
    # ``setdefault('eval_batch_size', 8)``, both of which read an ABSENT key as "use 8". The in-training
    # path sets none of these (TrainConfig carries no eval concurrency / generation override), so it must
    # fall through to those defaults rather than poison them.
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


def run_evalscope(sampler,
                  template,
                  *,
                  datasets: List[str],
                  model_id: str,
                  task_config: Dict[str, Any],
                  sampler_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """Drive twinkle_agentic's Evaluator over an already-built sampler and return the Native report rows.

    The sampler is NOT shut down here -- its lifecycle belongs to the caller (see the module docstring):
    ``swift eval`` builds and closes one per run, training reuses a persistent one across evals.
    ``sampler_kwargs`` carries per-request options such as the LoRA ``adapter_path`` to select.
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
