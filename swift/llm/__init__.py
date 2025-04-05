# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .infer import (VllmEngine, RequestConfig, LmdeployEngine, PtEngine, InferEngine, infer_main, deploy_main,
                        InferClient, run_deploy, AdapterRequest, prepare_model_template, BaseInferEngine)
    from .export import (export_main, merge_lora, quantize_model, export_to_ollama)
    from .eval import eval_main
    from .app import app_main
    from .train import sft_main, pt_main, rlhf_main, get_multimodal_target_regex
    from .sampling import sampling_main
    from .argument import (EvalArguments, InferArguments, TrainArguments, ExportArguments, DeployArguments,
                           RLHFArguments, WebUIArguments, BaseArguments, AppArguments, SamplingArguments)
    from .template import (TEMPLATE_MAPPING, Template, Word, get_template, TemplateType, register_template,
                           TemplateInputs, TemplateMeta, get_template_meta, InferRequest, load_image, MaxLengthError,
                           load_file, draw_bbox)
    from .model import (register_model, MODEL_MAPPING, ModelType, get_model_tokenizer, safe_snapshot_download,
                        HfConfigFactory, ModelInfo, ModelMeta, ModelKeys, register_model_arch, MultiModelKeys,
                        ModelArch, get_model_arch, MODEL_ARCH_MAPPING, get_model_info_meta, get_model_name, ModelGroup,
                        Model, get_model_tokenizer_with_flash_attn, get_model_tokenizer_multimodal, load_by_unsloth,
                        git_clone_github, get_matched_model_meta)
    from .dataset import (AlpacaPreprocessor, ResponsePreprocessor, MessagesPreprocessor, AutoPreprocessor,
                          DATASET_MAPPING, MediaResource, register_dataset, register_dataset_info, EncodePreprocessor,
                          LazyLLMDataset, load_dataset, DATASET_TYPE, sample_dataset, RowPreprocessor, DatasetMeta,
                          HfDataset, SubsetDataset)
    from .utils import (deep_getattr, to_float_dtype, to_device, History, Messages, history_to_messages,
                        messages_to_history, Processor, save_checkpoint, ProcessorMixin,
                        get_temporary_cache_files_directory, get_cache_dir, is_moe_model)
    from .base import SwiftPipeline
    from .data_loader import DataLoaderDispatcher, DataLoaderShard, BatchSamplerShard
else:
    _import_structure = {
        'rlhf': ['rlhf_main'],
        'infer': [
            'deploy_main', 'VllmEngine', 'RequestConfig', 'LmdeployEngine', 'PtEngine', 'infer_main', 'InferClient',
            'run_deploy', 'InferEngine', 'AdapterRequest', 'prepare_model_template', 'BaseInferEngine'
        ],
        'export': ['export_main', 'merge_lora', 'quantize_model', 'export_to_ollama'],
        'app': ['app_main'],
        'eval': ['eval_main'],
        'train': ['sft_main', 'pt_main', 'rlhf_main', 'get_multimodal_target_regex'],
        'sampling': ['sampling_main'],
        'argument': [
            'EvalArguments', 'InferArguments', 'TrainArguments', 'ExportArguments', 'WebUIArguments', 'DeployArguments',
            'RLHFArguments', 'BaseArguments', 'AppArguments', 'SamplingArguments'
        ],
        'template': [
            'TEMPLATE_MAPPING', 'Template', 'Word', 'get_template', 'TemplateType', 'register_template',
            'TemplateInputs', 'TemplateMeta', 'get_template_meta', 'InferRequest', 'load_image', 'MaxLengthError',
            'load_file', 'draw_bbox'
        ],
        'model': [
            'MODEL_MAPPING', 'ModelType', 'get_model_tokenizer', 'safe_snapshot_download', 'HfConfigFactory',
            'ModelInfo', 'ModelMeta', 'ModelKeys', 'register_model_arch', 'MultiModelKeys', 'ModelArch',
            'MODEL_ARCH_MAPPING', 'get_model_arch', 'get_model_info_meta', 'get_model_name', 'register_model',
            'ModelGroup', 'Model', 'get_model_tokenizer_with_flash_attn', 'get_model_tokenizer_multimodal',
            'load_by_unsloth', 'git_clone_github', 'get_matched_model_meta'
        ],
        'dataset': [
            'AlpacaPreprocessor', 'MessagesPreprocessor', 'AutoPreprocessor', 'DATASET_MAPPING', 'MediaResource',
            'register_dataset', 'register_dataset_info', 'EncodePreprocessor', 'LazyLLMDataset', 'load_dataset',
            'DATASET_TYPE', 'sample_dataset', 'RowPreprocessor', 'ResponsePreprocessor', 'DatasetMeta', 'HfDataset',
            'SubsetDataset'
        ],
        'utils': [
            'deep_getattr', 'to_device', 'to_float_dtype', 'History', 'Messages', 'history_to_messages',
            'messages_to_history', 'Processor', 'save_checkpoint', 'ProcessorMixin',
            'get_temporary_cache_files_directory', 'get_cache_dir', 'is_moe_model'
        ],
        'base': ['SwiftPipeline'],
        'data_loader': ['DataLoaderDispatcher', 'DataLoaderShard', 'BatchSamplerShard'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
