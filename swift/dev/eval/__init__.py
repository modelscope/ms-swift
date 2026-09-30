"""Evaluation helpers shared by the standalone ``swift eval`` recipe and the in-training eval path."""
from .evalscope_runner import (build_task_config, model_name, run_evalscope, summarize,
                               validate_eval_datasets)

__all__ = ['build_task_config', 'model_name', 'run_evalscope', 'summarize', 'validate_eval_datasets']
