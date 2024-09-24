# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .app_ui import gradio_chat_demo, gradio_generation_demo, app_ui_main
    from .infer import deploy_main, infer_main, merge_lora_main, merge_lora
    from .export import export_main
    from .eval import eval_main
    from .train import sft_main, pt_main, rlhf_main
    from .argument import EvalArguments, InferArguments, SftArguments, ExportArguments, WebuiArguments, DeployArguments, RLHFArguments
    from .template import TEMPLATE_MAPPING, Template, StopWords, InferTemplate, get_template
    from .model import MODEL_MAPPING, ModelType, get_model_tokenizer
    from .dataset import AlpacaPreprocessor, ClsPreprocessor, ComposePreprocessor, \
        ConversationsPreprocessor, ListPreprocessor, PreprocessFunc, RenameColumnsPreprocessor, \
        SmartPreprocessor, TextGenerationPreprocessor, \
        DatasetLoader, HubDatasetLoader, LocalDatasetLoader, dataset_name_exists, parse_dataset_name, DATASET_MAPPING, \
        MediaResource, register_dataset, register_local_dataset, register_dataset_info_file, register_single_dataset, \
        dataset_map, stat_dataset, LLMDataset, LLMIterableDataset, LazyLLMDataset, ConstantLengthDataset, print_example, \
        sort_by_max_length, standard_keys, standard_tags
else:
    _extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}
    _import_structure = {
        'app_ui': ['gradio_chat_demo', 'gradio_generation_demo', 'app_ui_main'],
        'rlhf': ['rlhf_main'],
        'infer': ['deploy_main', 'merge_lora', 'infer_main', 'merge_lora_main'],
        'export': ['export_main'],
        'eval': ['eval_main'],
        'train': ['sft_main', 'pt_main', 'rlhf_main'],
        "argument": ['EvalArguments', 'InferArguments', 'SftArguments', 'ExportArguments', 'WebuiArguments', 'DeployArguments', 'RLHFArguments'],
        'template': ['TEMPLATE_MAPPING', 'Template', 'StopWords', 'InferTemplate', 'get_template'],
        'model': ['MODEL_MAPPING', 'ModelType', 'get_model_tokenizer'],
        'dataset': ['AlpacaPreprocessor', 'ClsPreprocessor', 'ComposePreprocessor',
        'ConversationsPreprocessor', 'ListPreprocessor', 'PreprocessFunc', 'RenameColumnsPreprocessor',
        'SmartPreprocessor', 'TextGenerationPreprocessor',
        'DatasetLoader', 'HubDatasetLoader', 'LocalDatasetLoader', 'dataset_name_exists', 'parse_dataset_name', 'DATASET_MAPPING',
        'MediaResource', 'register_dataset', 'register_local_dataset', 'register_dataset_info_file', 'register_single_dataset',
        'dataset_map', 'stat_dataset', 'LLMDataset', 'LLMIterableDataset', 'LazyLLMDataset', 'ConstantLengthDataset', 'print_example',
        'sort_by_max_length', 'standard_keys', 'standard_tags']
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
