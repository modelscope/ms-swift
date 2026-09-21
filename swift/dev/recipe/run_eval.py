"""EvalScope orchestration over a remote or temporary dev deployment."""
from __future__ import annotations
import datetime as dt
import os
from contextlib import nullcontext
from typing import Any, Dict, Optional


def _model_name(model_config, deploy_config) -> str:
    return deploy_config.served_model_name or os.path.basename((model_config.model or 'model').rstrip('/'))


def _validate_eval_datasets(eval_config) -> None:
    from evalscope.api.registry import BENCHMARK_REGISTRY
    from evalscope.backend.opencompass import OpenCompassBackendManager

    supported = {
        'Native': sorted(BENCHMARK_REGISTRY),
        'OpenCompass': sorted(OpenCompassBackendManager.list_datasets()),
    }
    if eval_config.eval_backend == 'VLMEvalKit':
        from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
        supported['VLMEvalKit'] = sorted(VLMEvalKitBackendManager.list_supported_datasets())
    mapping = {name.lower(): name for name in supported[eval_config.eval_backend]}
    invalid = [name for name in eval_config.eval_dataset if name.lower() not in mapping]
    if invalid:
        raise ValueError(f'eval_dataset {invalid} is not supported by {eval_config.eval_backend}; '
                         f'supported datasets: {supported[eval_config.eval_backend]}')
    eval_config.eval_dataset = [mapping[name.lower()] for name in eval_config.eval_dataset]


def _prepare_opencompass_data() -> None:
    if os.path.exists('data'):
        if not os.path.exists(os.path.join('data', 'CMB')):
            raise RuntimeError('OpenCompass requires its own `data` folder, but an unrelated path already exists.')
        return
    from swift.dataset import MediaResource
    local_dir = MediaResource.download(
        'https://modelscope.cn/datasets/opencompass/OpenCompassDataComplete/'
        'resolve/master/OpenCompassData-complete-20240207.zip', 'OpenCompassData')
    os.symlink(os.path.join(local_dir, 'data'), 'data')


def build_eval_task(eval_config, deploy_config, model_name: str, base_url: str):
    """Build an EvalScope TaskConfig without running it."""
    from evalscope.constants import EvalBackend, EvalType
    from evalscope.run import TaskConfig

    datasets = eval_config.eval_dataset
    api_key = deploy_config.api_key or 'EMPTY'
    if eval_config.eval_backend == 'OpenCompass':
        work_dir = os.path.join(eval_config.eval_output_dir, 'opencompass')
        return TaskConfig(
            eval_backend=EvalBackend.OPEN_COMPASS,
            eval_config={
                'datasets': datasets,
                'batch_size': eval_config.eval_num_proc,
                'work_dir': work_dir,
                'models': [{
                    'path': model_name,
                    'openai_api_base': f"{base_url.rstrip('/')}/chat/completions",
                    'key': api_key,
                    'is_chat': eval_config.use_chat_template,
                }],
                'limit': eval_config.eval_limit,
            },
            work_dir=work_dir)
    if eval_config.eval_backend == 'VLMEvalKit':
        work_dir = os.path.join(eval_config.eval_output_dir, 'vlmeval')
        return TaskConfig(
            eval_backend=EvalBackend.VLM_EVAL_KIT,
            eval_config={
                'data': datasets,
                'model': [{
                    'type': model_name,
                    'name': 'CustomAPIModel',
                    'api_base': f"{base_url.rstrip('/')}/chat/completions",
                    'key': api_key,
                    **(eval_config.eval_generation_config or {}),
                }],
                'nproc': eval_config.eval_num_proc,
                'limit': eval_config.eval_limit,
            },
            work_dir=work_dir)
    work_dir = os.path.join(eval_config.eval_output_dir, 'native')
    return TaskConfig(
        model=model_name,
        eval_type=EvalType.SERVICE,
        api_url=base_url,
        api_key=api_key,
        datasets=datasets,
        work_dir=work_dir,
        limit=eval_config.eval_limit,
        eval_batch_size=eval_config.eval_num_proc,
        dataset_args=eval_config.eval_dataset_args,
        generation_config=eval_config.eval_generation_config,
        **(eval_config.extra_eval_args or {}))


def _summarize(task_config, backend: str, model_name: str):
    from evalscope.constants import EvalBackend
    from evalscope.summarizer import Summarizer

    reports = Summarizer.get_report_from_cfg(task_cfg=task_config)
    if backend == 'OpenCompass':
        result = {}
        for report in reports:
            if report[model_name] != '-':
                result[report['dataset']] = {report['metric']: report[model_name]}
        return result
    if backend == 'VLMEvalKit':
        result = {}
        for report in reports:
            parts = next(iter(report)).rsplit('_', 2)
            dataset, metric = (parts[1], parts[2]) if len(parts) == 3 else ('-', '-')
            result[dataset] = {metric: next(iter(report.values()))}
        return result
    return reports


def run_eval(model_config, template_config, generation_config, infer_config, rollout_config, deploy_config,
             eval_config, *, adapter_mapping: Optional[Dict[str, str]] = None,
             merge_lora: bool = False) -> Dict[str, Any]:
    """Run EvalScope, starting a temporary dev deployment when eval_url is absent."""
    from evalscope.run import run_task

    from swift.dev.cli.infer import _engine_args
    from swift.dev.recipe.run_deploy import run_deploy_process
    from swift.utils import append_to_jsonl

    if not eval_config.eval_dataset:
        raise ValueError('At least one --eval_dataset is required.')
    _validate_eval_datasets(eval_config)
    if eval_config.local_dataset and eval_config.eval_backend == 'OpenCompass':
        _prepare_opencompass_data()
    base_url = eval_config.eval_url
    if base_url and '/chat/completions' in base_url:
        base_url = base_url.split('/chat/completions', 1)[0]
    deploy_context = nullcontext(base_url) if base_url else run_deploy_process(
        model_config,
        template_config,
        generation_config,
        backend=infer_config.infer_backend,
        engine_args=_engine_args(infer_config.infer_backend, infer_config, rollout_config),
        adapter_mapping=adapter_mapping,
        merge_lora=merge_lora,
        host=deploy_config.host,
        port=deploy_config.port,
        served_model_name=deploy_config.served_model_name,
        owned_by=deploy_config.owned_by,
        api_key=deploy_config.api_key,
        max_logprobs=deploy_config.max_logprobs,
        max_concurrency=deploy_config.max_concurrency,
        log_interval=deploy_config.log_interval,
        request_log_path=deploy_config.request_log_path,
        verbose=deploy_config.verbose,
        ssl_keyfile=deploy_config.ssl_keyfile,
        ssl_certfile=deploy_config.ssl_certfile,
        log_level=deploy_config.log_level)
    model_name = _model_name(model_config, deploy_config)
    with deploy_context as active_url:
        task_config = build_eval_task(eval_config, deploy_config, model_name, active_url)
        run_task(task_cfg=task_config)
        report = {eval_config.eval_backend: _summarize(task_config, eval_config.eval_backend, model_name)}
    report.update({
        'time': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
        'model': model_config.model,
        'adapters': list((adapter_mapping or {}).values()),
        'eval_output_dir': eval_config.eval_output_dir,
        'eval_limit': eval_config.eval_limit,
    })
    if eval_config.result_jsonl:
        append_to_jsonl(eval_config.result_jsonl, report)
    return report
