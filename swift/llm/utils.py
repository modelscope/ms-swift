# Copyright (c) Alibaba, Inc. and its affiliates.
"""
swift.llm.utils 模块：提供 LLM 相关的通用工具与类型。

功能概述：
- 统一 Processor 类型定义、tokenizer 并行控制与日志器创建；
- 提供张量/容器的设备与 dtype 迁移工具（to_device / to_float_dtype）；
- 生成配置合并工具（set_generation_config）；
- 动态梯度检查点功能（dynamic_gradient_checkpointing），含模型模块查找与 forward 注入；
- 历史消息与对话消息互转（history_to_messages / messages_to_history）；
- 保存检查点（save_checkpoint）与临时缓存目录管理；
- 生成配置的 EOS token 自动补全与打包序列参数生成。
"""
import inspect
import os  # 读取与设置环境变量
import shutil
import tempfile
from types import MethodType  # 用于将函数绑定为实例方法
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union  # 常用类型注解

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
from modelscope.hub.utils.utils import get_cache_dir  # 获取 ModelScope 缓存目录
from peft import PeftModel  # PEFT 模型封装
from transformers import FeatureExtractionMixin, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase  # HF 类型
from transformers import ProcessorMixin as HfProcessorMixin  # HF 的 Processor 混入

from swift.utils import deep_getattr, get_logger  # 深度 getattr 与日志工具

try:
    from transformers import BaseImageProcessor  # 新版图像处理器基类（可能不存在）
    Processor = Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, HfProcessorMixin]  # 统一 Processor 类型
except ImportError:
    Processor = Union[PreTrainedTokenizerBase, FeatureExtractionMixin, HfProcessorMixin]  # 回退定义（无图像处理器）

if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 关闭 tokenizer 并行以减少多进程环境下的警告

logger = get_logger()  # 模块级日志器

Tool = Dict[str, Union[str, Dict]]  # 工具定义：名称/描述/参数等
History = List[Union[Tuple[str, str], List[str]]]  # 历史对话：[(用户, 助手)] 或 [用户, 助手]
Message = Dict[str, Union[str, List[Dict[str, Any]]]]  # 单条消息：{'role': 'user'|'assistant'|'system'|[{'text': 'Hi'}, {'image': 'base64'}], 'content': ...}
Messages = List[Message]  # 消息序列


class ProcessorMixin:
    """
    Processor 混入：提供 tokenizer 与 processor 的统一访问与赋值语义。
    - tokenizer 属性优先从 processor 派生，若 processor 内部包裹了 tokenizer，则展开后返回。
    - 设置 tokenizer 时会同步更新 processor；若 processor 与 tokenizer 非同一对象禁止交叉赋值。
    """

    @property
    def tokenizer(self):
        """
        获取 tokenizer 对象。

        返回:
            PreTrainedTokenizerBase: 若 processor 包裹了 tokenizer，则返回内部 tokenizer，否则返回 processor 本身。
        """
        tokenizer = self.processor  # 默认从 processor 获取
        if not isinstance(tokenizer, PreTrainedTokenizerBase) and hasattr(tokenizer, 'tokenizer'):  # 处理器可能包含 tokenizer 属性
            tokenizer = tokenizer.tokenizer  # 解包得到真实 tokenizer
        return tokenizer  # 返回 tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        """
        设置 tokenizer，并与 processor 保持一致。

        参数:
            value: 新的 tokenizer。
        """
        if self.processor is self.tokenizer:  # 当 processor 与 tokenizer 指向同一对象
            self.processor = value  # 直接替换为新的 tokenizer
        elif self.tokenizer is not value:  # 二者非同一引用时禁止交叉赋值
            raise AttributeError('Please use `self.processor` for assignment.')  # 提示应通过 processor 赋值


def to_float_dtype(data: Any, dtype: torch.dtype) -> Any:
    """
    将浮点类型的张量/容器中的元素转换为指定 dtype。

    参数:
        data: 任意对象（可为 Mapping/序列/张量/标量等）。
        dtype: 目标 torch.dtype。

    返回:
        与输入同结构的对象，所有浮点张量元素被转换到目标 dtype。
    """
    if isinstance(data, Mapping):  # 字典/映射：递归处理其值
        return type(data)({k: to_float_dtype(v, dtype) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):  # 序列：递归处理每个元素
        return type(data)(to_float_dtype(v, dtype) for v in data)
    elif isinstance(data, torch.Tensor) and torch.is_floating_point(data):  # 浮点张量：转换 dtype
        return data.to(dtype=dtype)
    else:  # 其他类型：原样返回
        return data


def to_device(data: Any, device: Union[str, torch.device, int]) -> Any:
    """
    将输入对象（张量或容器）移动到目标设备。

    参数:
        data: 任意对象（可为 Mapping/序列/张量/标量等）。
        device: 目标设备（字符串/torch.device/设备序号）。

    返回:
        与输入同结构的对象，所有张量被移动到目标设备。
    """
    if isinstance(data, Mapping):  # 字典/映射：递归处理值
        return type(data)({k: to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):  # 序列：递归处理元素
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):  # 张量：直接 .to(device)
        return data.to(device=device)
    else:  # 其他类型：原样返回
        return data


def set_generation_config(model: nn.Module, generation_config: GenerationConfig) -> None:
    """
    根据模型已有的 generation_config 合并/覆盖新配置，保留必要优先项。

    规则:
    - 若模型已有配置，优先保留 no_repeat_ngram_size/num_beams；
    - 若旧配置字段非 None 且新配置字段为 None，则沿用旧值。
    """
    old_generation_config = getattr(model, 'generation_config', None)  # 读取旧配置
    old_generation_priority_config = ['no_repeat_ngram_size', 'num_beams']  # 优先保留字段
    if old_generation_config is not None:  # 存在旧配置
        for k, old_v in vars(old_generation_config).items():  # 遍历旧配置的公开属性
            if k.startswith('_'):  # 跳过私有属性
                continue
            v = getattr(generation_config, k, None)  # 新配置中的值
            if k in old_generation_priority_config or (old_v is not None and v is None):  # 覆盖条件
                setattr(generation_config, k, old_v)  # 使用旧值
    model.generation_config = generation_config  # 回写配置到模型


def find_module_list(model) -> Optional[nn.ModuleList]:
    """
    在模型中查找适合插入梯度检查点的长序列模块（ModuleList/Sequential）。

    返回:
        最长的符合条件的模块列表；若检测到已包装 checkpoint 或不可用，则返回 None。
    """
    module_lists = []  # 备选列表
    for m in model.modules():  # 遍历所有子模块
        if hasattr(m, 'gradient_checkpointing') or m.__class__.__name__ == 'CheckpointWrapper':  # 已支持/已包装
            return  # 提前返回，表示不处理
        if (isinstance(m, (nn.ModuleList, nn.Sequential)) and len(m) >= 10  # 足够长的序列
                and 'mlp' not in m[0].__class__.__name__.lower()):  # 排除 MOE/MLP 类序列
            module_lists.append(m)  # 记录备选
    if module_lists:  # 选择最长的一个
        return max(module_lists, key=lambda x: len(x))


def _kwargs_to_args(func, args, kwargs) -> Optional[List[Any]]:
    """
    将关键字参数按函数签名顺序补齐到位置参数列表，用于 checkpoint 包裹时保持调用兼容。

    返回:
        新的参数列表；若存在无法补齐的必选参数则返回 None。
    """
    parameters = inspect.signature(func).parameters  # 获取函数形参签名
    args = list(args)  # 复制位置参数
    parameters = list(parameters.items())[len(args):]  # 仅处理尚未提供的位置参数部分
    for key, param in parameters:  # 依次尝试从 kwargs 或默认值补齐
        if key in kwargs:  # 有关键字参数
            args.append(kwargs[key])  # 追加对应值
        elif param.default != param.empty:  # 使用默认值
            args.append(param.default)
        else:  # 必选参数缺失
            return  # 返回 None
    return args  # 返回补齐后的参数列表


def _add_gradient_checkpointing(module_list):
    """
    为给定的模块序列动态注入 gradient_checkpointing 功能：
    - 为每个模块替换 forward（或 _old_forward）为带 checkpoint 的新 forward；
    - 按需打开第一个输入张量的 requires_grad，确保反向传播正常。
    """

    requires_grad = None  # 用于缓存参数梯度需求，减少开销

    def _new_forward(self, *args, **kwargs):
        nonlocal requires_grad  # 使用外部变量缓存
        if requires_grad is None:  # 首次进入时计算是否需要梯度
            requires_grad = any(p.requires_grad for p in self.parameters())  # 任一参数需要梯度则为 True

        new_args = _kwargs_to_args(self.__old_forward, args, kwargs)  # 将 kwargs 对齐到位置参数
        if new_args is not None and self.gradient_checkpointing and self.training:  # 条件满足则启用 ckpt
            if new_args and isinstance(new_args[0], torch.Tensor) and requires_grad and not new_args[0].requires_grad:
                new_args[0].requires_grad_(True)  # 确保首个张量参与反向
            layer_ret = self._gradient_checkpointing_func(self.__old_forward, *new_args)  # 执行带 ckpt 的前向
            logger.info_once('Successfully using dynamic gradient checkpointing.')  # 仅提示一次
        else:  # 不满足条件时回退到原始 forward
            layer_ret = self.__old_forward(*args, **kwargs)
        return layer_ret  # 返回前向结果

    for module in module_list:  # 遍历序列内各层
        module.gradient_checkpointing = False  # 默认关闭，由外部启用
        if hasattr(module, '_old_forward'):  # 处理 device_map 包装过的 forward
            __old_forward = module._old_forward  # 记录旧 forward
            module._old_forward = MethodType(_new_forward, module)  # 替换为新 forward
        else:
            __old_forward = module.forward  # 记录旧 forward
            module.forward = MethodType(_new_forward, module)  # 替换为新 forward
        module.__old_forward = __old_forward  # 备份到统一属性


def dynamic_gradient_checkpointing(model, including_vit: bool = False) -> None:
    """
    为模型动态注入梯度检查点功能，支持多模态模型按塔（language/vision）分别处理。

    参数:
        model: 目标模型（可为 PEFT 包装模型）。
        including_vit: 是否包含视觉塔（vision tower）。
    """
    from .model import ModelMeta  # 延迟导入避免循环依赖
    if isinstance(model, PeftModel):  # 解包 PEFT 外壳
        model = model.model
    model_meta: ModelMeta = model.model_meta  # 读取元信息
    model_arch = model_meta.model_arch  # 架构信息（含塔名）
    if model_meta.is_multimodal and model_arch:  # 多模态模型
        tower_names = model_arch.language_model.copy()  # 语言塔名称列表
        if including_vit:  # 可选包含视觉塔
            tower_names += model_arch.vision_tower
    else:  # 单塔模型
        tower_names = [None]

    model.supports_gradient_checkpointing = True  # 标记支持级别
    for tower_name in tower_names:  # 遍历各塔
        if tower_name is None:  # 单塔或根模型
            model_tower = model
        else:  # 根据名称获取子模块
            model_tower = deep_getattr(model, tower_name)
        model_tower.supports_gradient_checkpointing = True  # 标记子模块支持
        module_list = find_module_list(model_tower)  # 查找可注入的层序列
        if module_list is None:  # 未找到则跳过
            continue
        _add_gradient_checkpointing(module_list)  # 注入功能
        logger.info(f'Automatically add gradient_checkpointing to {model_tower.__class__}.')  # 提示信息


def history_to_messages(history: History,
                        system: Optional[str] = None,
                        roles: Optional[List[List[str]]] = None) -> 'Messages':
    """
    将历史对话列表转换为 OpenAI 风格的 messages 列表。

    参数:
        history: 形如 [[query, response], ...] 或 [[query, None], ...] 的历史列表。
        system: 可选，系统提示词。
        roles: 可选，逐轮的角色对列表（如 [['user','assistant'], ...]）。

    返回:
        messages: [{'role': 'system'|'user'|'assistant', 'content': str}, ...]
    """
    messages = []  # 输出消息列表
    if not roles:  # 未指定角色对则使用默认 user/assistant
        roles = [['user', 'assistant']] * len(history)
    else:  # 指定角色对时校验长度一致
        assert len(roles) == len(history), f'len(roles): {len(roles)}, len(history): {len(history)}'
    if system is not None:  # 前置系统消息
        messages.append({'role': 'system', 'content': system})

    for role, h in zip(roles, history):  # 逐轮展开为两条消息（若不为 None）
        assert isinstance(h, (list, tuple))  # 基本格式校验
        if h[0] is not None:  # 用户消息
            messages.append({'role': role[0], 'content': h[0]})
        if h[1] is not None:  # 助手消息
            messages.append({'role': role[1], 'content': h[1]})
    return messages  # 返回结果


def messages_to_history(messages: 'Messages') -> Dict[str, Any]:
    """
    将 messages 列表还原为历史对话结构，并取出当前 query/response。

    返回:
        包含 history、history_roles、query、query_role、response、system 的字典。
    """
    system = None  # 系统消息内容
    messages = messages.copy()  # 复制以避免原地修改
    if messages and messages[0]['role'] == 'system':  # 若首条为系统消息
        system = messages[0]['content']  # 保存系统内容
        messages = messages[1::]  # 去除系统消息
    if len(messages) % 2 == 1:  # 若消息为奇数条，补一个空 assistant
        messages.append({'role': 'assistant', 'content': None})
    history = []  # [[query, response], ...]
    history_roles = []  # [[query_role, response_role], ...]
    for user_message, assistant_message in zip(messages[::2], messages[1::2]):  # 按对成对遍历
        assert user_message['role'] in {'tool', 'user'}, f'user_message {user_message}'  # 用户侧允许 tool/user
        assert assistant_message['role'] == 'assistant', f'assistant_message: {assistant_message}'  # 助手侧校验
        history.append([user_message['content'], assistant_message['content']])  # 收集内容
        history_roles.append([user_message['role'], assistant_message['role']])  # 收集角色
    query, response = history.pop() if history else (None, None)  # 取出最后一轮作为当前 query/response
    query_role = history_roles.pop()[0] if history_roles else None  # 对应的 query 角色
    return {
        'history': history,  # 历史内容（不含当前轮）
        'history_roles': history_roles,  # 历史角色
        'query': query,  # 当前用户输入
        'query_role': query_role,  # 当前用户角色
        'response': response,  # 当前助手输出
        'system': system,  # 系统提示
    }


def save_checkpoint(model: Optional[PreTrainedModel],
                    processor: 'Processor',
                    output_dir: str,
                    *,
                    safe_serialization: bool = True,
                    max_shard_size: Union[int, str] = '5GB',
                    model_dirs: List[str] = None,
                    additional_saved_files: Optional[List[str]] = None) -> None:
    """
    保存模型与处理器到 output_dir，并额外拷贝预处理配置与 args.json 等文件。

    参数:
        model: 可选的 HF 模型；SentenceTransformer 需特殊处理。
        processor: tokenizer/processor。
        output_dir: 输出目录。
        safe_serialization: 是否使用 safetensors。
        max_shard_size: 分片大小上限。
        model_dirs: 附加的模型目录（优先从中复制文件）。
        additional_saved_files: 需要一并复制的额外文件列表。
    """
    if model is not None:  # 保存模型权重
        if model.__class__.__name__ != 'SentenceTransformer':  # 普通 HF 模型
            model.save_pretrained(output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
        else:  # SentenceTransformer 模型：不支持 max_shard_size
            model.save_pretrained(output_dir, safe_serialization=safe_serialization)
            # copy sentencetransformers files  # 复制额外源码与配置
            from swift.utils import copy_files_by_pattern  # 延迟导入
            copy_files_by_pattern(model.model_dir, output_dir, '*.py')
            copy_files_by_pattern(model.model_dir, output_dir, '*.json')
    processor.save_pretrained(output_dir)  # 保存处理器

    if model_dirs is None:  # 统一模型目录列表
        model_dirs = []
    else:
        model_dirs = model_dirs.copy()
    if model and getattr(model, 'model_dir', None) and model.model_dir not in model_dirs:  # 追加当前模型目录
        model_dirs.append(model.model_dir)
    for src_file in (additional_saved_files or []) + ['preprocessor_config.json', 'args.json']:  # 逐文件复制
        tgt_path = os.path.join(output_dir, src_file)  # 目标路径
        if os.path.exists(tgt_path) and src_file == 'args.json':  # 避免覆盖已存在的 args.json
            continue
        for model_dir in model_dirs:  # 在模型目录中查找
            src_path: str = os.path.join(model_dir, src_file)
            if os.path.isfile(src_path):  # 普通文件
                shutil.copy(src_path, tgt_path)
                break
            elif os.path.isdir(src_path):  # 目录
                shutil.copytree(src_path, tgt_path)
                break


TEMP_DIR_POOL = {}  # 前缀 -> TemporaryDirectory 对象 的缓存池


def get_temporary_cache_files_directory(prefix=None):
    """
    获取（并创建）临时缓存文件目录，按前缀复用 TemporaryDirectory 对象。

    参数:
        prefix: 临时目录名前缀；默认使用 datasets.config.TEMP_CACHE_DIR_PREFIX。

    返回:
        临时目录的绝对路径字符串。
    """
    if prefix is None:  # 默认从 datasets 读取前缀
        import datasets.config
        prefix = datasets.config.TEMP_CACHE_DIR_PREFIX
    global TEMP_DIR_POOL  # 使用模块级缓存池
    if prefix in TEMP_DIR_POOL:  # 已存在则复用
        TEMP_DIR = TEMP_DIR_POOL[prefix]
    else:  # 不存在则创建
        tmp_dir = os.path.join(get_cache_dir(), 'tmp')  # 放置于 modelscope 缓存目录下
        os.makedirs(tmp_dir, exist_ok=True)  # 确保父目录存在
        kwargs = {}  # 临时目录构造参数
        parameters = inspect.signature(tempfile.TemporaryDirectory.__init__).parameters  # 反射构造参数
        if 'ignore_cleanup_errors' in parameters:  # 高版本支持忽略清理错误
            kwargs['ignore_cleanup_errors'] = True
        TEMP_DIR = tempfile.TemporaryDirectory(prefix=prefix, dir=tmp_dir, **kwargs)  # 创建临时目录对象
        logger.info(f'create tmp_dir: {TEMP_DIR.name}')  # 记录创建信息
        TEMP_DIR_POOL[prefix] = TEMP_DIR  # 放入缓存池

    return TEMP_DIR.name  # 返回目录路径


def get_ckpt_dir(model_dir: str, adapters_dir: Optional[List[str]]) -> str:
    """
    依据是否存在 args.json 判断 checkpoint 根目录；若存在多个目录，适配器目录优先。

    参数:
        model_dir: 模型主目录。
        adapters_dir: 适配器目录列表，优先级更高。

    返回:
        ckpt_dir: 命中的目录路径；若均未命中则为 None。
    """
    model_dirs = (adapters_dir or []).copy()  # 从适配器目录开始（优先）
    if model_dir:  # 追加模型主目录
        model_dirs.append(model_dir)
    ckpt_dir = None  # 默认未命中
    for model_dir in model_dirs:  # 依次检查
        if os.path.exists(os.path.join(model_dir, 'args.json')):  # args.json 视为标志文件
            ckpt_dir = model_dir  # 命中
            break
    return ckpt_dir  # 返回结果


def update_generation_config_eos_token(generation_config, template):
    """
    根据模板中的停止词，补全 generation_config 的 eos_token_id（仅追加单 token 的停止词）。
    """
    if generation_config is None:  # 无配置则直接返回
        return
    stop_words = template.template_meta.stop_words  # 停止词表
    eos_token_id = generation_config.eos_token_id  # 现有 eos 列表/单值
    if eos_token_id is None:  # 规范化为列表
        eos_token_id = []
    elif isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    modified = False  # 记录是否有新增
    for stop_word in stop_words:  # 遍历停止词
        if stop_word is None:
            continue
        if isinstance(stop_word, str):  # 需要先分词
            stop_word = template._tokenize(stop_word)
        if isinstance(stop_word, (list, tuple)) and len(stop_word) == 1 and stop_word[0] not in eos_token_id:
            eos_token_id.append(stop_word[0])  # 追加单 token 的停止词 id
            modified = True
    if modified:  # 若有变化则回写
        generation_config.eos_token_id = eos_token_id


def get_packed_seq_params(position_ids: torch.Tensor):
    """
    基于 position_ids 计算打包注意力所需的累计长度与最大长度参数。

    参数:
        position_ids: 形如 [B, L] 的位置 id 张量。

    返回:
        dict: 包含 cumulative_seqlens_{q,k} 与 max_length_{q,k}。
    """
    position_ids_f = position_ids.flatten()  # 展平成一维
    indices_q = torch.arange(position_ids_f.shape[0], device=position_ids_f.device, dtype=torch.int32)  # 索引序列

    cu_seqlens = torch.cat([  # 将每句开头位置索引与总长度拼接
        indices_q[position_ids_f == 0],
        torch.tensor(position_ids_f.shape, device=position_ids_f.device, dtype=torch.int32),
    ])

    max_length = position_ids_f.max() + 1  # 位置 id 最大值 + 1 即为长度
    return {
        'cumulative_seqlens_q': cu_seqlens,  # Query 累计长度
        'cumulative_seqlens_k': cu_seqlens,  # Key 累计长度（等同）
        'max_length_q': max_length,  # Query 最大长度
        'max_length_k': max_length,  # Key 最大长度
    }
