# Copyright (c) Alibaba, Inc. and its affiliates.
"""
swift.llm 子包初始化模块。

功能说明：
- 在类型检查阶段（TYPE_CHECKING 为 True）直接导入并暴露子模块中的公共符号，便于 IDE 补全与类型提示；
- 在运行时采用懒加载，仅在首次访问符号时再导入对应子模块，降低首包导入开销并避免不必要依赖加载。
"""

from typing import TYPE_CHECKING  # 类型检查哨兵：区分静态类型检查与运行时代码路径

from swift.utils.import_utils import _LazyModule  # 懒加载模块封装：按需导入子模块/符号

if TYPE_CHECKING:  # 仅在静态类型检查中执行：为类型工具与 IDE 提供可见符号
    # Recommend using `xxx_main`  # 建议使用 `xxx_main` 入口函数
    from .infer import (VllmEngine, RequestConfig, LmdeployEngine, PtEngine, InferEngine, infer_main, deploy_main,  # 推理/部署相关符号
                        InferClient, run_deploy, AdapterRequest, prepare_model_template, BaseInferEngine, SglangEngine,  # 推理客户端/适配请求/引擎
                        rollout_main)  # 强化学习/rollout 入口
    from .export import (export_main, merge_lora, quantize_model, export_to_ollama)  # 导出/合并/量化/导出到 Ollama
    from .eval import eval_main  # 评测入口
    from .app import app_main  # Web/UI 应用入口
    from .train import sft_main, pt_main, rlhf_main, get_multimodal_target_regex  # 训练入口与多模态正则
    from .sampling import sampling_main  # 采样脚本入口
    from .argument import (EvalArguments, InferArguments, TrainArguments, ExportArguments, DeployArguments,  # 各任务参数类型
                           RolloutArguments, GymRolloutArguments, RLHFArguments, WebUIArguments, BaseArguments,  # 强化学习/网页/基础参数
                           AppArguments, SamplingArguments)  # 应用/采样参数
    from .template import (TEMPLATE_MAPPING, Template, Word, get_template, TemplateType, register_template,  # 模板系统与注册
                           TemplateInputs, TemplateMeta, get_template_meta, InferRequest, load_image, MaxLengthError,  # 模板输入/元信息/推理请求
                           load_file, draw_bbox, RolloutInferRequest)  # 文件加载/标注工具/rollout 请求
    from .model import (register_model, MODEL_MAPPING, ModelType, get_model_tokenizer, safe_snapshot_download,  # 模型注册/映射/类型
                        HfConfigFactory, ModelInfo, ModelMeta, ModelKeys, register_model_arch, MultiModelKeys,  # 配置工厂/元信息/键
                        ModelArch, get_model_arch, MODEL_ARCH_MAPPING, get_model_info_meta, get_model_name, ModelGroup,  # 架构/获取函数
                        Model, get_model_tokenizer_with_flash_attn, get_model_tokenizer_multimodal, load_by_unsloth,  # 模型/分词器辅助
                        git_clone_github, get_matched_model_meta, get_llm_model)  # 代码获取/匹配/加载 LLM
    from .dataset import (AlpacaPreprocessor, ResponsePreprocessor, MessagesPreprocessor, AutoPreprocessor,  # 数据集预处理器
                          DATASET_MAPPING, MediaResource, register_dataset, register_dataset_info, EncodePreprocessor,  # 数据集映射/注册
                          LazyLLMDataset, load_dataset, DATASET_TYPE, sample_dataset, RowPreprocessor, DatasetMeta,  # 数据加载与采样
                          HfDataset, SubsetDataset)  # HF 数据集封装/子集
    from .utils import (deep_getattr, to_float_dtype, to_device, History, Messages, history_to_messages,  # 工具与历史消息
                        messages_to_history, Processor, save_checkpoint, ProcessorMixin,  # 处理器/保存检查点
                        get_temporary_cache_files_directory, get_cache_dir, dynamic_gradient_checkpointing,  # 缓存/梯度检查点
                        get_packed_seq_params)  # 打包序列参数
    from .base import SwiftPipeline  # 统一推理/训练 Pipeline 抽象
    from .data_loader import DataLoaderDispatcher, DataLoaderShard, BatchSamplerShard  # 数据加载调度/分片/采样器分片
else:  # 运行时：使用懒加载导出结构
    _import_structure = {  # 懒加载导出结构：模块名 -> 需导出的符号列表
        'rlhf': ['rlhf_main'],  # 强化学习相关入口
        'infer': [  # 推理与部署模块导出
            'deploy_main', 'VllmEngine', 'RequestConfig', 'LmdeployEngine', 'PtEngine', 'infer_main', 'InferClient',
            'run_deploy', 'InferEngine', 'AdapterRequest', 'prepare_model_template', 'BaseInferEngine', 'rollout_main',
            'SglangEngine'
        ],
        'export': ['export_main', 'merge_lora', 'quantize_model', 'export_to_ollama'],  # 导出/量化/合并工具
        'app': ['app_main'],  # 应用入口
        'eval': ['eval_main'],  # 评测入口
        'train': ['sft_main', 'pt_main', 'rlhf_main', 'get_multimodal_target_regex'],  # 训练入口与工具
        'sampling': ['sampling_main'],  # 采样入口
        'argument': [  # 各任务参数类型导出
            'EvalArguments', 'InferArguments', 'TrainArguments', 'ExportArguments', 'WebUIArguments', 'DeployArguments',
            'RolloutArguments', 'RLHFArguments', 'BaseArguments', 'AppArguments', 'SamplingArguments'
        ],
        'template': [  # 模板系统导出
            'TEMPLATE_MAPPING', 'Template', 'Word', 'get_template', 'TemplateType', 'register_template',
            'TemplateInputs', 'TemplateMeta', 'get_template_meta', 'InferRequest', 'load_image', 'MaxLengthError',
            'load_file', 'draw_bbox', 'RolloutInferRequest'
        ],
        'model': [  # 模型系统导出
            'MODEL_MAPPING', 'ModelType', 'get_model_tokenizer', 'safe_snapshot_download', 'HfConfigFactory',
            'ModelInfo', 'ModelMeta', 'ModelKeys', 'register_model_arch', 'MultiModelKeys', 'ModelArch',
            'MODEL_ARCH_MAPPING', 'get_model_arch', 'get_model_info_meta', 'get_model_name', 'register_model',
            'ModelGroup', 'Model', 'get_model_tokenizer_with_flash_attn', 'get_model_tokenizer_multimodal',
            'load_by_unsloth', 'git_clone_github', 'get_matched_model_meta', 'get_llm_model'
        ],
        'dataset': [  # 数据集与预处理导出
            'AlpacaPreprocessor', 'MessagesPreprocessor', 'AutoPreprocessor', 'DATASET_MAPPING', 'MediaResource',
            'register_dataset', 'register_dataset_info', 'EncodePreprocessor', 'LazyLLMDataset', 'load_dataset',
            'DATASET_TYPE', 'sample_dataset', 'RowPreprocessor', 'ResponsePreprocessor', 'DatasetMeta', 'HfDataset',
            'SubsetDataset'
        ],
        'utils': [  # 通用工具导出
            'deep_getattr', 'to_device', 'to_float_dtype', 'History', 'Messages', 'history_to_messages',
            'messages_to_history', 'Processor', 'save_checkpoint', 'ProcessorMixin',
            'get_temporary_cache_files_directory', 'get_cache_dir', 'dynamic_gradient_checkpointing',
            'get_packed_seq_params'
        ],
        'base': ['SwiftPipeline'],  # Pipeline 抽象
        'data_loader': ['DataLoaderDispatcher', 'DataLoaderShard', 'BatchSamplerShard'],  # 数据加载组件
    }

    import sys  # 访问 sys.modules 以替换当前模块为懒加载实现

    sys.modules[__name__] = _LazyModule(  # 将当前包对象替换为懒加载模块
        __name__,  # 当前包名
        globals()['__file__'],  # 当前模块文件路径
        _import_structure,  # 懒加载导出映射
        module_spec=__spec__,  # 模块规格（供 importlib 使用）
        extra_objects={},  # 额外注入对象（此处为空）
    )
