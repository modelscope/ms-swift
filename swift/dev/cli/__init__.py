from __future__ import annotations

from .parser import parse_configs

__all__ = [
    'parse_configs', 'parse_sft_configs', 'sft_main', 'parse_export_configs', 'export_main', 'pt_main',
    'parse_rlhf_configs', 'rlhf_main', 'parse_infer_configs', 'infer_main', 'parse_deploy_configs', 'deploy_main',
    'parse_sample_configs', 'sample_main', 'parse_eval_configs', 'eval_main',
    'parse_merge_lora_configs', 'merge_lora_main'
]


def __getattr__(name):
    if name in {'parse_sft_configs', 'sft_main'}:
        from .sft import parse_sft_configs, sft_main
        return {'parse_sft_configs': parse_sft_configs, 'sft_main': sft_main}[name]
    modules = {
        'parse_export_configs': ('export', 'parse_export_configs'),
        'export_main': ('export', 'export_main'),
        'pt_main': ('pt', 'pt_main'),
        'parse_rlhf_configs': ('rlhf', 'parse_rlhf_configs'),
        'rlhf_main': ('rlhf', 'rlhf_main'),
        'parse_infer_configs': ('infer', 'parse_infer_configs'),
        'infer_main': ('infer', 'infer_main'),
        'parse_deploy_configs': ('deploy', 'parse_deploy_configs'),
        'deploy_main': ('deploy', 'deploy_main'),
        'parse_sample_configs': ('sample', 'parse_sample_configs'),
        'sample_main': ('sample', 'sample_main'),
        'parse_eval_configs': ('eval', 'parse_eval_configs'),
        'eval_main': ('eval', 'eval_main'),
        'parse_merge_lora_configs': ('merge_lora', 'parse_merge_lora_configs'),
        'merge_lora_main': ('merge_lora', 'merge_lora_main'),
    }
    if name in modules:
        from importlib import import_module
        module_name, attr = modules[name]
        return getattr(import_module(f'{__name__}.{module_name}'), attr)
    raise AttributeError(name)
