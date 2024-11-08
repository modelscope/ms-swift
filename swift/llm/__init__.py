# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .infer import (VllmEngine, RequestConfig, InferStats, LmdeployEngine, PtEngine, infer_main, deploy_main,
                        PtLoRARequest, InferClient, SwiftInfer, SwiftDeploy)
    from .export import export_main, merge_lora, quantize_model, export_to_ollama
    from .eval import eval_main
    from .train import sft_main, pt_main, rlhf_main
    from .argument import (EvalArguments, InferArguments, SftArguments, ExportArguments, DeployArguments, RLHFArguments,
                           WebUIArguments, BaseArguments)
    from .template import (TEMPLATE_MAPPING, Template, Word, get_template, TemplateType, register_template,
                           TemplateInputs, Messages, TemplateMeta, get_template_meta, InferRequest)
    from .model import (MODEL_MAPPING, ModelType, get_model_tokenizer, safe_snapshot_download, HfConfigFactory,
                        ModelInfo, ModelMeta, ModelKeys, register_model_arch, MultiModelKeys, ModelArch, get_model_arch,
                        MODEL_ARCH_MAPPING)
    from .dataset import (AlpacaPreprocessor, MessagesPreprocessor, AutoPreprocessor, DatasetName, DATASET_MAPPING,
                          MediaResource, register_dataset, register_dataset_info, dataset_map, stat_dataset, LLMDataset,
                          LLMIterableDataset, LazyLLMDataset, ConstantLengthDataset, print_example, sort_by_max_length,
                          standard_keys, load_dataset, DATASET_TYPE, HfDataset, sample_dataset)
    from .utils import deep_getattr, to_device, History, decode_base64, history_to_messages, messages_to_history
    from .base import SwiftPipeline
else:
    _extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}
    _import_structure = {
        'rlhf': ['rlhf_main'],
        'infer': [
            'deploy_main', 'VllmEngine', 'RequestConfig', 'InferStats', 'LmdeployEngine', 'PtEngine', 'infer_main',
            'PtLoRARequest', 'InferClient', 'SwiftInfer', 'SwiftDeploy'
        ],
        'export': ['export_main', 'merge_lora', 'quantize_model', 'export_to_ollama'],
        'eval': ['eval_main'],
        'train': ['sft_main', 'pt_main', 'rlhf_main'],
        'argument': [
            'EvalArguments', 'InferArguments', 'SftArguments', 'ExportArguments', 'WebUIArguments', 'DeployArguments',
            'RLHFArguments', 'BaseArguments'
        ],
        'template': [
            'TEMPLATE_MAPPING', 'Template', 'Word', 'get_template', 'TemplateType', 'register_template',
            'TemplateInputs', 'Messages', 'TemplateMeta', 'get_template_meta', 'InferRequest'
        ],
        'model': [
            'MODEL_MAPPING', 'ModelType', 'get_model_tokenizer', 'safe_snapshot_download', 'HfConfigFactory',
            'ModelInfo', 'ModelMeta', 'ModelKeys', 'register_model_arch', 'MultiModelKeys', 'ModelArch',
            'MODEL_ARCH_MAPPING', 'get_model_arch'
        ],
        'dataset': [
            'AlpacaPreprocessor', 'ClsPreprocessor', 'ComposePreprocessor', 'MessagesPreprocessor', 'DatasetName',
            'DATASET_MAPPING', 'MediaResource', 'register_dataset', 'register_dataset_info', 'dataset_map',
            'stat_dataset', 'LLMDataset', 'LLMIterableDataset', 'LazyLLMDataset', 'ConstantLengthDataset',
            'print_example', 'sort_by_max_length', 'standard_keys', 'load_dataset', 'DATASET_TYPE', 'HfDataset',
            'sample_dataset'
        ],
        'utils':
        ['deep_getattr', 'to_device', 'History', 'decode_base64', 'history_to_messages', 'messages_to_history'],
        'base': ['SwiftPipeline']
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
