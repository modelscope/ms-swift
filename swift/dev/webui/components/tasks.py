# Copyright (c) ModelScope Contributors. All rights reserved.
"""Task bindings: one-shot actions over training products.

Each binding reuses the REAL entity component / runner -- construction here,
behaviour stays native. Tasks write real artifacts; the artifact port value
is the path and format the writer actually produced, never a placeholder.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from ..component import ComponentError, TaskComponent

logger = logging.getLogger(__name__)


class MergeTask(TaskComponent):
    """merge: fold adapters into base weights through the real MergeComponent."""

    type = 'merge'

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from swift.dev.components.weight_transform.merge import MergeComponent
        artifact = inputs.get('artifact') or {}
        base_path = artifact.get('path') if isinstance(artifact, dict) else None
        if not base_path:
            raise ComponentError(f'{self.node_id}: merge needs a wired model ARTIFACT (its path)')
        merge_cfg = dict(self.params.get('merge_config') or {})
        adapters = merge_cfg.pop('adapters', None) or []
        if isinstance(adapters, str):
            adapters = [part for part in adapters.split(',') if part.strip()]
        if not adapters:
            raise ComponentError(f'{self.node_id}: merge_config.adapters lists the checkpoints to fold')
        from swift.dev.config import ModelConfig, TunerConfig
        model_config = ModelConfig(model=str(base_path))
        tuner_config = TunerConfig(adapters=[str(path) for path in adapters])
        output_dir = str(self.params.get('output_dir') or '') or None
        component = MergeComponent(model_config, tuner_config, output_dir=output_dir, **merge_cfg)
        path = component.run(component.default_output_dir() if output_dir is None else output_dir)
        if not os.path.exists(path):
            raise ComponentError(f'{self.node_id}: merge reported no artifact at {path}')
        return {'model': {'path': path, 'format': 'model'}}


class QuantizeTask(TaskComponent):
    """quantize: pack weights through the real QuantizeComponent.

    The calibration dataset (optional) is the wired DATASET resource's own
    dataset id; every other knob rides the typed ``quantize_config`` dict
    (``quant_method`` required, ``template_config`` overrides included)."""

    type = 'quantize'

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from swift.dev.components.weight_transform.quantize import QuantizeComponent
        artifact = inputs.get('artifact') or {}
        base_path = artifact.get('path') if isinstance(artifact, dict) else None
        if not base_path:
            raise ComponentError(f'{self.node_id}: quantize needs a wired model ARTIFACT (its path)')
        quant_cfg = dict(self.params.get('quantize_config') or {})
        quant_method = quant_cfg.pop('quant_method', None)
        if not quant_method:
            raise ComponentError(f'{self.node_id}: quantize_config.quant_method is required')
        calib = inputs.get('calib')
        dataset_id = None
        if isinstance(calib, dict) and calib.get('config') is not None:
            dataset_id = list(getattr(calib['config'], 'dataset', None) or [])
        from swift.dev.config import DatasetConfig, ModelConfig, QuantizeConfig, TemplateConfig
        template_config = TemplateConfig(**dict(quant_cfg.pop('template_config', None) or {}))
        quantize_config = QuantizeConfig(quant_method=str(quant_method), **quant_cfg)
        dataset_config = DatasetConfig(dataset=dataset_id) if dataset_id else None
        output_dir = str(self.params.get('output_dir') or '') or None
        component = QuantizeComponent(
            ModelConfig(model=str(base_path)), template_config, quantize_config,
            dataset_config, output_dir=output_dir)
        path = component.run(component.default_output_dir() if output_dir is None else output_dir)
        if not os.path.exists(path):
            raise ComponentError(f'{self.node_id}: quantize reported no artifact at {path}')
        return {'model': {'path': path, 'format': 'model'}}


class EvalTask(TaskComponent):
    """eval: drive the real Evaluator over the session's own sampler or an API.

    The sampler's lifecycle stays the caller's (it is a session resident);
    the report rows land as the metrics summary and a written report file."""

    type = 'eval'

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from swift.dev.eval.evalscope_runner import build_task_config, run_evaluator, run_evaluator_api
        from .resources import build_template_from
        sampler = inputs.get('sampler')
        endpoint = inputs.get('endpoint')
        dataset = inputs.get('data')
        if (sampler is None) == (endpoint is None):
            raise ComponentError(
                f'{self.node_id}: eval takes exactly one source -- a sampler OR an endpoint')
        template = None
        if sampler is not None:
            template = build_template_from(
                str(self.params.get('tokenizer') or ''), None, self.node_id)
        if not isinstance(dataset, dict) or not getattr(dataset.get('config'), 'dataset', None):
            raise ComponentError(f'{self.node_id}: eval needs a wired DATASET (its dataset id)')
        datasets = [str(name) for name in dataset['config'].dataset]
        task_cfg = dict(self.params.get('task_config') or {})
        model_id = str(task_cfg.pop('model_id', '') or '')
        if not model_id:
            raise ComponentError(f'{self.node_id}: task_config.model_id names the evaluated model')
        work_dir = str(task_cfg.pop('work_dir', '') or '') or os.path.join('output', 'eval')
        task_config = build_task_config(work_dir=work_dir, **task_cfg)
        if sampler is not None:
            rows = run_evaluator(sampler, template, datasets=datasets,
                                 model_id=model_id, task_config=task_config)
        else:
            rows = run_evaluator_api(endpoint, datasets=datasets,
                                     model_id=model_id, task_config=task_config)
        metrics: Dict[str, float] = {}
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict):
                for key, value in row.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[str(key)] = float(value)
        report_path = os.path.join(work_dir, 'report.json')
        os.makedirs(work_dir, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as stream:
            json.dump({'rows': rows, 'metrics': metrics}, stream, ensure_ascii=False, default=str)
        return {'metrics': {'values': metrics},
                'artifact': {'path': report_path, 'format': 'report'}}


class InferTask(TaskComponent):
    """infer: rebuild the sampling engine over a weight source, through the real InferComponent.

    The weight source is a model ARTIFACT (its path) or the session's MODEL
    (read back to the origin it loaded); the dataset is the wired DATASET's
    own config. The engine and every other knob ride the typed
    ``infer_config`` dict (``sampler`` names the engine; ``template_config``
    / ``generation_config`` / ``adapters`` / ``result_path`` pop out of it)."""

    type = 'infer'

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from swift.dev.components.infer.component import InferComponent
        artifact = inputs.get('artifact') or {}
        path = artifact.get('path') if isinstance(artifact, dict) else None
        if path is None:
            model_obj = inputs.get('model')
            path = getattr(getattr(model_obj, 'model', None), 'name_or_path', None)
        if not path:
            raise ComponentError(
                f'{self.node_id}: infer needs a weight source -- a model ARTIFACT (its path) '
                'or a wired MODEL (read back to the origin it loaded)')
        dataset = inputs.get('data')
        if not isinstance(dataset, dict) or dataset.get('config') is None:
            raise ComponentError(f'{self.node_id}: infer needs a wired DATASET (its dataset config)')
        cfg = dict(self.params.get('infer_config') or {})
        backend = str(cfg.get('sampler') or '')
        if not backend:
            raise ComponentError(
                f'{self.node_id}: infer_config.sampler names the engine (vllm / sglang / transformers)')
        template_cfg = dict(cfg.pop('template_config', None) or {})
        generation_cfg = dict(cfg.pop('generation_config', None) or {})
        adapters = cfg.pop('adapters', None)
        output_path = str(cfg.pop('result_path', '') or '') or None
        max_rows = cfg.pop('max_rows', None)
        from swift.dev.config import GenerationConfig, InferConfig, ModelConfig, TemplateConfig
        try:
            template_config = TemplateConfig(**template_cfg)
            generation_config = GenerationConfig(**generation_cfg) if generation_cfg else None
            infer_config = InferConfig(**cfg)
        except TypeError as exc:
            raise ComponentError(f'{self.node_id}: infer_config: {exc}') from None
        rows = InferComponent(
            ModelConfig(model=str(path)), template_config, dataset['config'], infer_config,
            generation_config, backend=backend, engine_args=None, adapters=adapters,
            quantize_config=None, max_rows=max_rows, split_dataset_ratio=0.0,
            output_path=output_path, shutdown=True)()
        artifact_out = {'path': output_path, 'format': 'dataset'} if output_path else None
        return {'predictions': {'rows': rows}, 'artifact': artifact_out}


class DeployTask(TaskComponent):
    """deploy: serve a weight source through the real run_deploy (detached).

    The deployment is DETACHED -- the click never blocks; the apps stay on
    the Ray cluster and ``deploy_config.stop`` removes them instead. The
    endpoint port value is the live OpenAI-shaped client over the served
    model, same shape as an external API connection."""

    type = 'deploy'

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from swift.dev.recipe.infer import run_deploy, stop_deployment
        artifact = inputs.get('model') or {}
        path = artifact.get('path') if isinstance(artifact, dict) else None
        if not path:
            raise ComponentError(f'{self.node_id}: deploy needs a wired model ARTIFACT (its path)')
        cfg = dict(self.params.get('deploy_config') or {})
        stop = bool(cfg.pop('stop', False))
        backend = str(cfg.pop('backend', 'vllm') or 'vllm')
        template_cfg = dict(cfg.pop('template_config', None) or {})
        if cfg.get('detach') is False:
            raise ComponentError(
                f'{self.node_id}: deploy is click-driven and never blocks; detach=False '
                'has no meaning here (use deploy_config.stop to remove a deployment)')
        cfg.pop('detach', None)
        from swift.dev.config import DeployConfig, ModelConfig, TemplateConfig
        try:
            deploy_config = DeployConfig(**cfg, detach=True)
        except TypeError as exc:
            raise ComponentError(f'{self.node_id}: deploy_config: {exc}') from None
        model_config = ModelConfig(model=str(path))
        if stop:
            stop_deployment(deploy_config, model_config)
            return {'endpoint': None}
        run_deploy(model_config, TemplateConfig(**template_cfg),
                   backend=backend, deploy_config=deploy_config)
        host = '127.0.0.1' if deploy_config.host in ('', '0.0.0.0') else deploy_config.host
        base_url = f'http://{host}:{deploy_config.port}{deploy_config.route_prefix}'
        name = deploy_config.served_model_name or os.path.basename(str(path).rstrip('/'))
        from twinkle_agentic.protocol.openai import OpenAI
        return {'endpoint': OpenAI(model=name, base_url=base_url,
                                   api_key=deploy_config.api_key or 'EMPTY')}


class HubPushTask(TaskComponent):
    """hub_push: publish a model folder through the public Hub API.

    Credentials NEVER ride the graph: ``token_env`` names the environment
    variable the push reads at run time (unset -> refuse, never push
    anonymously by accident). The artifact is what the hub actually got --
    the repo id and its address."""

    type = 'hub_push'

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from swift.hub.hub import get_hub
        artifact = inputs.get('model') or {}
        folder = artifact.get('path') if isinstance(artifact, dict) else None
        if not folder:
            raise ComponentError(f'{self.node_id}: hub_push needs a wired model ARTIFACT (its folder)')
        repo_id = str(self.params.get('hub_model_id') or '')
        if not repo_id:
            raise ComponentError(f'{self.node_id}: hub_model_id names the target repository')
        hub_name = str(self.params.get('hub', 'ModelScope') or 'ModelScope')
        use_hf = {'ModelScope': False, 'HuggingFace': True}.get(hub_name)
        if use_hf is None:
            raise ComponentError(f'{self.node_id}: hub {hub_name!r} is neither ModelScope nor HuggingFace')
        token_env = str(self.params.get('token_env') or '')
        token = os.environ.get(token_env) if token_env else None
        if token_env and not token:
            raise ComponentError(
                f'{self.node_id}: token env var {token_env!r} is unset; export it before pushing')
        get_hub(use_hf).push_to_hub(
            repo_id, folder,
            commit_message=str(self.params.get('commit_message') or '') or None,
            private=bool(self.params.get('private', True)),
            revision=str(self.params.get('revision', 'main') or 'main'),
            token=token)
        url = f'https://huggingface.co/{repo_id}' if use_hf else f'https://modelscope.cn/models/{repo_id}'
        return {'artifact': {'path': repo_id, 'format': 'publication', 'url': url}}
