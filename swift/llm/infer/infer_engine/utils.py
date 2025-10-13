"""模块功能概述：
该模块实现推理引擎的工具类和辅助函数，包括：
- 流式推理辅助类（InferStreamer、StreamerMixin、TokensIteratorStreamer、LogitsStreamer）；
- 生成配置准备和默认值设置；
- LMDeploy 和 vLLM 引擎的补丁函数，用于修复已知问题和扩展功能；
- 适配器请求数据结构。

核心功能：
1. 流式输出处理：支持中文字符检测和智能文本打印（避免不完整单词输出）；
2. 生成配置管理：从 RequestConfig 创建 GenerationConfig，处理温度、top_k、top_p 等参数；
3. LMDeploy 补丁：支持自定义设备选择和权重加载；
4. vLLM 补丁：修复 NPU 设备支持和内存泄漏问题。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者

import os  # 引入 os 模块，用于路径和环境变量操作
import re  # 引入正则表达式模块，用于模式匹配和字符串解析
from collections import OrderedDict  # 引入有序字典，用于保持插入顺序的字典结构
from concurrent.futures import ThreadPoolExecutor  # 引入线程池执行器，用于并发任务执行
from contextlib import contextmanager, nullcontext  # 引入上下文管理器装饰器和空上下文管理器
from dataclasses import dataclass  # 引入 dataclass 装饰器，用于简化数据类定义
from functools import partial  # 引入 partial 函数，用于固定函数的部分参数
from itertools import repeat  # 引入 repeat 迭代器，用于生成重复元素
from queue import Queue  # 引入队列，用于线程安全的数据传递
from typing import List, Optional, Union  # 引入类型注解，用于参数和返回值的类型提示

import torch  # 引入 PyTorch 深度学习框架
import torch.distributed as dist  # 引入 PyTorch 分布式训练模块（当前未直接使用，保留以便扩展）
from packaging import version  # 引入版本解析模块，用于比较软件包版本号
from transformers import GenerationConfig, LogitsProcessor  # 引入 Transformers 的生成配置和 logits 处理器
from transformers.generation.streamers import BaseStreamer  # 引入 Transformers 的基础流式输出器

from swift.llm.model.register import fix_do_sample_warning  # 引入修复 do_sample 警告的函数
from swift.utils import get_current_device, get_device, get_device_count, get_node_setting, set_device  # 引入设备管理工具函数
from ..protocol import RequestConfig  # 引入请求配置协议类


@dataclass  # 使用 dataclass 装饰器，自动生成 __init__、__repr__ 等方法
class AdapterRequest:  # 定义适配器请求数据类
    """类功能：
    `AdapterRequest` 数据类，用于封装 LoRA 适配器的请求信息。
    
    属性：
        - name: 适配器名称，用于标识适配器；
        - path: 适配器权重文件的路径。
    
    应用场景：
        在推理时动态加载和使用 LoRA 适配器，支持多适配器切换。
    """
    name: str  # 适配器名称（如 "lora_adapter_1"）
    path: str  # 适配器权重文件路径（如 "/path/to/adapter"）


class InferTools:
    """类功能
    定义推理工具类，InferTools 提供推理相关的静态工具方法，主要用于文本处理。
    
    方法：
        - _is_chinese_char: 检测给定 Unicode 码点是否为 CJK（中日韩）字符。
    
    应用场景：
        在流式推理中判断字符类型，优化输出显示逻辑。
    """

    @staticmethod  # 静态方法装饰器，不需要实例即可调用
    def _is_chinese_char(cp: int) -> bool:
        """方法功能：
        检测给定的 Unicode 码点 (codepoint) 是否为 CJK（中日韩）字符。
        
        参数：
            cp (int): Unicode 码点值（如 ord('中') = 20013）
        
        返回：
            bool: 如果是 CJK 字符返回 True，否则返回 False
        
        实现逻辑：
            检查码点是否落在以下 Unicode CJK 字符区间：
            - 0x4E00-0x9FFF: CJK 统一表意文字（常用汉字）
            - 0x3400-0x4DBF: CJK 扩展 A
            - 0x20000-0x2A6DF: CJK 扩展 B
            - 0x2A700-0x2B73F: CJK 扩展 C
            - 0x2B740-0x2B81F: CJK 扩展 D
            - 0x2B820-0x2CEAF: CJK 扩展 E
            - 0xF900-0xFAFF: CJK 兼容表意文字
            - 0x2F800-0x2FA1F: CJK 兼容表意文字补充
        
        示例：
            >>> InferTools._is_chinese_char(ord('中'))  # 20013P
            True
            >>> InferTools._is_chinese_char(ord('A'))  # 65
            False
        
        来源：
            从 transformers.generation.streamers.TextStreamer 复制而来。
        """
        # copy from transformers.generation.streamers.TextStreamer  # 注释：代码来源
        if ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)  # 检查 CJK 统一表意文字、扩展 A、扩展 B
                or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x81F) or (0x2B820 <= cp <= 0x2CEAF)  # 检查 CJK 扩展 C、D、E
                or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)):  # 检查 CJK 兼容表意文字和补充
            return True  # 如果码点在任意一个 CJK 区间内，返回 True

        return False  # 否则返回 False


class InferStreamer(InferTools):
    """类功能：
    定义推理流式输出器类，继承 InferTools。
    负责流式推理过程中的文本输出处理，支持智能打印策略。
    
    核心功能：
        - 智能文本打印：避免打印不完整的单词（通过空格检测）；
        - CJK 字符检测：对中日韩字符立即打印（因为它们不需要空格分隔）；
        - 空格对齐：处理首个 token 前的空格，避免重复单词问题；
        - 缓存管理：跟踪已解码和已打印的 token 位置。
    
    属性：
        - template: 对话模板实例，用于 token 解码；
        - tokenizer: 分词器实例；
        - cache_idx: 缓存的 token 索引（已解码的 token 数量）；
        - print_idx: 打印的字符索引（response 字符串中已打印的位置）；
        - decode_kwargs: 解码时的额外参数；
        - first_num_space: 首个 token 前的空格数量（用于对齐）；
        - first_token: 是否为首个 token（首个 token 直接打印）。
    
    应用场景：
        在流式推理中逐步输出生成的文本，提升用户体验。
    """

    def __init__(self, template, **decode_kwargs):
        """初始化流式输出器。
        
        参数：
            template: 对话模板实例（包含 tokenizer 和 decode 方法）
            **decode_kwargs: 解码时的额外参数（如 skip_special_tokens=True）
        """
        self.template = template  # 保存对话模板实例
        self.tokenizer = template.tokenizer  # 从模板中获取分词器

        self.cache_idx = 0  # token idx  # 初始化缓存索引为 0（表示从第 0 个 token 开始解码）
        self.print_idx = 0  # 初始化打印索引为 0（表示从 response 的第 0 个字符开始打印）
        self.decode_kwargs = decode_kwargs  # 保存解码参数
        self.first_num_space = -1  # The number of whitespace characters before the first token.  # 初始化首个 token 前的空格数为 -1（表示尚未确定）
        self.first_token = True  # 标记当前是否为首个 token

    def _align_blank_suffix(self, response: str) -> str:
        """空格对齐方法：对齐 response 的空格前缀，避免重复单词问题。
        
        参数：
            response (str): 解码后的响应文本
        
        返回：
            str: 对齐后的响应文本
        
        实现逻辑：
            1. 计算当前 response 前面的空格数量；
            2. 如果是第一次调用，记录空格数量作为基准；
            3. 如果当前空格数少于基准，在前面补齐空格；
            4. 如果当前空格数多于基准，去掉多余的空格。
        
        目的：
            避免解码过程中因 token 边界问题导致的重复单词输出。
        """
        # Avoid the occurrence of repeated words in sentence.  # 注释：避免句子中出现重复单词
        cur_num_space = len(response) - len(response.lstrip(' '))  # 计算 response 前缀的空格数量
        if self.first_num_space == -1:  # 如果是第一次调用（first_num_space 为 -1）
            self.first_num_space = cur_num_space  # 记录当前空格数作为基准
        elif cur_num_space < self.first_num_space:  # 如果当前空格数少于基准
            response = ' ' * (self.first_num_space - cur_num_space) + response  # 在前面补齐空格
        elif cur_num_space > self.first_num_space:  # 如果当前空格数多于基准
            response = response[cur_num_space - self.first_num_space:]  # 去掉多余的空格
        return response  # 返回对齐后的 response

    def _get_response(self, response: str, is_finished: bool, token_len: int) -> str:
        """
        私有方法，从 response 中提取可打印的文本部分。
        
        参数：
            response (str): 完整的响应文本
            is_finished (bool): 是否生成完成
            token_len (int): 本次解码的 token 数量
        
        返回：
            str: 可以安全打印的文本部分
        
        实现逻辑：
            1. 如果是首个 token，直接返回整个 response；
            2. 如果 response 以换行符结尾或生成完成，返回所有未打印的文本并重置缓存；
            3. 如果最后一个字符是 CJK 字符，返回所有未打印的文本（CJK 不需要等待空格）；
            4. 否则，只返回到最后一个空格的文本（避免打印不完整的单词）。
        """
        # After the symbol for a new line, we flush the cache.  # 注释：遇到换行符时清空缓存
        if self.first_token:  # 如果是首个 token
            printable_text = response  # 直接返回整个 response
            self.first_token = False  # 标记首个 token 已处理
        elif response.endswith('\n') or is_finished:  # 如果 response 以换行符结尾或生成完成
            printable_text = response[self.print_idx:]  # 返回所有未打印的文本
            self.cache_idx += token_len  # 更新缓存索引（已解码的 token 数量）
            self.first_num_space = -1  # 重置空格基准（准备处理下一行）
            self.print_idx = 0  # 重置打印索引
        # If the last token is a CJK character, we print the characters.  # 注释：如果最后一个 token 是 CJK 字符，立即打印
        elif len(response) > 0 and self._is_chinese_char(ord(response[-1])):  # 如果最后一个字符是 CJK 字符
            printable_text = response[self.print_idx:]  # 返回所有未打印的文本
            self.print_idx += len(printable_text)  # 更新打印索引
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)  # 注释：否则，只打印到最后一个空格
        else:  # 否则（最后一个字符不是 CJK 字符）
            printable_text = response[self.print_idx:response.rfind(' ') + 1]  # 返回到最后一个空格的文本（避免不完整单词）
            self.print_idx += len(printable_text)  # 更新打印索引
        return printable_text  # 返回可打印的文本

    def get_printable_text(self, raw_tokens: List[int], is_finished: bool) -> str:
        """
        公有方法，将 token ID 列表解码为可打印的文本。
        
        参数：
            raw_tokens (List[int]): 原始 token ID 列表，形状为 [seq_len]
            is_finished (bool): 是否生成完成
        
        返回：
            str: 可打印的文本部分
        
        实现逻辑：
            1. 截取尚未解码的 token（从 cache_idx 开始）；
            2. 如果是首个 token，跳过解码（避免重复）；
            3. 使用 template.decode 解码 token；
            4. 对齐空格前缀；
            5. 提取可打印的文本部分。
        """
        raw_tokens = raw_tokens[self.cache_idx:]  # 截取尚未解码的 token（从 cache_idx 开始）
        if self.first_token:  # 如果是首个 token
            raw_tokens = []  # 跳过解码（首个 token 会在 _get_response 中直接返回）
        response = self.template.decode(  # 使用 template 解码 token
            raw_tokens, is_finished=is_finished, tokenizer_kwargs=self.decode_kwargs, first_token=self.first_token)  # 传入解码参数
        response = self._align_blank_suffix(response)  # 对齐空格前缀
        return self._get_response(response, is_finished, len(raw_tokens))  # 返回可打印的文本部分


class StreamerMixin:
    """类功能：
    定义流式输出混入类 StreamerMixin，提供基于队列的迭代器功能，用于流式数据传递。
    
    核心功能：
        - 实现迭代器协议（__iter__ 和 __next__）；
        - 使用队列（Queue）在生产者和消费者之间传递数据；
        - 支持终止信号（None）来结束迭代。
    
    属性：
        - queue: 线程安全的队列，用于存储流式数据。
    
    应用场景：
        作为混入类，为其他流式输出器提供迭代器功能。
    """

    def __init__(self):
        """初始化流式混入类，创建队列。"""
        self.queue = Queue()  # 创建线程安全的队列，用于存储流式数据

    def __iter__(self):
        """定义迭代器协议：返回自身作为迭代器对象（self）。"""
        return self  # 返回自身作为迭代器

    def __next__(self) -> torch.Tensor:
        """定义迭代器协议：获取队列中的下一个元素。
        
        返回：
            torch.Tensor: 队列中的下一个张量
        
        异常：
            StopIteration: 当队列中获取到 None 时，表示迭代结束
        
        实现逻辑：
            1. 从队列中阻塞获取下一个值；
            2. 如果值为 None，抛出 StopIteration 异常；
            3. 否则返回该值。
        """
        value = self.queue.get()  # 从队列中阻塞获取下一个值
        if value is None:  # 如果值为 None（终止信号）
            raise StopIteration()  # 抛出 StopIteration 异常，结束迭代
        else:  # 否则
            return value  # 返回该值


class TokensIteratorStreamer(StreamerMixin, BaseStreamer):  # 定义 token 迭代器流式输出器，继承 StreamerMixin 和 BaseStreamer
    """类功能：
    `TokensIteratorStreamer` 用于流式输出生成的 token 张量。
    
    核心功能：
        - 实现 Transformers 的 BaseStreamer 接口；
        - 通过队列传递生成的 token 张量；
        - 支持迭代器协议，可用于 for 循环。
    
    方法：
        - put: 将生成的 token 张量放入队列；
        - end: 发送终止信号，结束迭代。
    
    应用场景：
        在流式推理中逐步获取生成的 token，供后续处理使用。
    """

    def put(self, value: torch.Tensor) -> None:  # 定义 put 方法：将 token 放入队列
        """将生成的 token 张量放入队列。
        
        参数：
            value (torch.Tensor): 生成的 token 张量，形状通常为 [batch_size, seq_len]
        """
        self.queue.put(value)  # 将 token 张量放入队列

    def end(self) -> None:  # 定义 end 方法：发送终止信号
        """发送终止信号（None），结束迭代。"""
        self.queue.put(None)  # 将 None 放入队列，作为终止信号


class LogitsStreamer(LogitsProcessor):  # 定义 logits 流式输出器，继承 LogitsProcessor
    """类功能：
    `LogitsStreamer` 用于在生成过程中捕获每一步的 logits（未归一化的分数）。
    
    核心功能：
        - 实现 LogitsProcessor 接口，可插入到生成流程中；
        - 通过队列捕获每一步的 logits，供外部分析使用；
        - 不修改 logits 值，仅作为监控和记录工具。
    
    属性：
        - queue: 线程安全的队列，用于存储每一步的 logits。
    
    应用场景：
        在推理过程中记录 logits，用于调试、分析或计算 logprobs。
    """

    def __init__(self):  # 定义初始化方法
        """初始化 logits 流式输出器，创建队列。"""
        self.queue = Queue()  # 创建线程安全的队列，用于存储 logits

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:  # 定义 __call__ 方法：处理 logits
        """在生成每一步时被调用，捕获当前步的 logits。
        
        参数：
            input_ids (torch.LongTensor): 当前的输入 token IDs，形状为 [batch_size, seq_len]
            scores (torch.FloatTensor): 当前步的 logits，形状为 [batch_size, vocab_size]
        
        返回：
            torch.FloatTensor: 未修改的 logits，形状为 [batch_size, vocab_size]
        
        实现逻辑：
            1. 将 logits 放入队列；
            2. 返回未修改的 logits（不影响生成过程）。
        """
        self.queue.put(scores)  # 将 logits 放入队列
        return scores  # 返回未修改的 logits


def _set_generation_config_default_value(model_generation_config: GenerationConfig,  # 定义设置生成配置默认值的内部函数
                                         generation_config: GenerationConfig) -> GenerationConfig:
    """从模型的默认生成配置中复制缺失的参数到新的生成配置。
    
    参数：
        model_generation_config (GenerationConfig): 模型的默认生成配置
        generation_config (GenerationConfig): 新创建的生成配置（可能缺少某些参数）
    
    返回：
        GenerationConfig: 补全后的生成配置
    
    实现逻辑：
        遍历模型配置的所有参数，如果新配置中缺失或为 None，则从模型配置中复制。
        特殊处理：
        - 跳过 'max_length'（使用 max_new_tokens 代替）；
        - 强制复制 'no_repeat_ngram_size'（避免重复 n-gram）。
    """
    for k, v in model_generation_config.to_dict().items():  # 遍历模型配置的所有参数
        new_v = getattr(generation_config, k, None)  # 获取新配置中对应的参数值
        if k in ['max_length']:  # 如果是 'max_length'
            continue  # 跳过（使用 max_new_tokens 代替）
        if k in ['no_repeat_ngram_size'] or v is not None and new_v is None:  # 如果是 'no_repeat_ngram_size' 或模型配置有值而新配置为 None
            setattr(generation_config, k, v)  # 将模型配置的值设置到新配置中
    return generation_config  # 返回补全后的生成配置


def prepare_generation_config(model_generation_config: Optional[GenerationConfig], request_config: RequestConfig,  # 定义准备生成配置的函数
                              tokenizer) -> Optional[GenerationConfig]:
    """从请求配置创建生成配置，并处理温度、采样等参数。
    
    参数：
        model_generation_config (Optional[GenerationConfig]): 模型的默认生成配置（可为 None）
        request_config (RequestConfig): 用户请求的配置（包含 temperature、top_k、top_p 等）
        tokenizer: 分词器实例（用于获取 eos_token_id 和 pad_token_id）
    
    返回：
        Optional[GenerationConfig]: 准备好的生成配置（如果输入配置为 None 则返回 None）
    
    实现逻辑：
        1. 如果模型配置或请求配置为 None，直接返回模型配置；
        2. 从请求配置中提取参数（max_new_tokens、length_penalty、temperature、top_k、top_p 等）；
        3. 处理温度参数：temperature=0 表示确定性采样，设置 do_sample=False；
        4. 从模型配置中补全缺失的默认值；
        5. 设置 eos_token_id 和 pad_token_id。
    """
    if model_generation_config is None or request_config is None:  # 如果模型配置或请求配置为 None
        return model_generation_config  # 直接返回模型配置
    kwargs = {'max_new_tokens': request_config.max_tokens}  # 设置最大生成 token 数
    # not use: 'n', 'best_of', 'frequency_penalty', 'presence_penalty'  # 注释：这些参数暂不使用
    for key in ['length_penalty']:  # 遍历必须设置的参数
        kwargs[key] = getattr(request_config, key)  # 从请求配置中获取参数值
    for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty', 'num_beams']:  # 遍历可选参数
        new_value = getattr(request_config, key)  # 从请求配置中获取参数值
        if new_value is None:  # 如果请求配置中该参数为 None
            kwargs[key] = getattr(model_generation_config, key)  # 使用模型配置的默认值
        else:  # 否则
            kwargs[key] = new_value  # 使用请求配置的值

    if not model_generation_config.do_sample and request_config.temperature in {0, None}:  # 如果模型默认不采样且请求温度为 0 或 None
        kwargs['temperature'] = 0  # 设置温度为 0（确定性采样）
    if kwargs['temperature'] == 0:  # 如果温度为 0（确定性采样）
        kwargs['do_sample'] = False  # 禁用采样
        kwargs['temperature'] = 1  # 设置温度为 1（避免除零错误）
        kwargs['top_p'] = 1  # 设置 top_p 为 1（禁用核采样）
        kwargs['top_k'] = 50  # 设置 top_k 为 50（默认值）
    else:  # 否则（温度大于 0）
        kwargs['do_sample'] = True  # 启用采样
    generation_config = GenerationConfig(**kwargs)  # 创建生成配置对象
    generation_config = _set_generation_config_default_value(model_generation_config, generation_config)  # 从模型配置中补全默认值
    fix_do_sample_warning(generation_config)  # 修复 do_sample 相关的警告

    if generation_config.eos_token_id is None:  # 如果 eos_token_id 未设置
        generation_config.eos_token_id = tokenizer.eos_token_id  # 从分词器中获取 eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id  # 设置 pad_token_id
    return generation_config  # 返回准备好的生成配置


def patch_lmdeploy(load_weights=False):  # 定义 LMDeploy 补丁函数
    """函数功能：
    对 LMDeploy 库进行猴子补丁（Monkey Patch），扩展其功能。
    
    核心功能：
        1. 允许 LMDeploy 支持自定义设备选择（而不是固定使用 GPU 0）；
        2. 支持从内存中的 state_dict 加载权重（而不仅是从文件加载）；
        3. 修复 TurboMind 引擎的初始化流程，支持多 GPU 并行加载。
    
    参数：
        load_weights (bool): 是否加载权重（默认 False，表示只创建引擎不加载权重）
    
    实现原理：
        通过替换 LMDeploy 内部的方法和类属性，动态修改其行为。
    
    应用场景：
        在使用 LMDeploy 作为推理引擎时，需要自定义设备或从内存加载权重。
    """
    # This patch allows lmdeploy selects device and reload state_dict  # 注释：此补丁允许 lmdeploy 选择设备并重新加载 state_dict
    import lmdeploy  # 导入 lmdeploy 库
    assert version.parse(lmdeploy.__version__) >= version.parse('0.7.0')  # 断言 lmdeploy 版本 >= 0.7.0（早期版本不支持此补丁）
    from lmdeploy.messages import TurbomindEngineConfig  # 导入 TurboMind 引擎配置类
    from lmdeploy.turbomind.deploy import loader  # 导入加载器模块
    from lmdeploy.turbomind.deploy.loader import create_loader  # 导入原始的 create_loader 函数
    from lmdeploy.turbomind.deploy.source_model import llama  # 导入 llama 模型部署模块

    def _create_loader(model_path: str, pattern: str):  # 定义自定义的加载器创建函数
        """创建模型权重加载器，支持从文件或内存加载。
        
        参数：
            model_path (str | dict): 模型路径（文件路径或 state_dict 字典）
            pattern (str): 正则表达式模式，用于匹配层名称（如 'layers.(\d+).'）
        
        返回：
            OrderedDict | Loader: 如果 model_path 是字典，返回按层分组的生成器；否则调用原始 create_loader
        
        实现逻辑：
            1. 如果 model_path 不是文件路径，说明是内存中的权重字典；
            2. 将权重按层编号分组（通过正则表达式匹配层号）；
            3. 未匹配到层号的权重放入特殊键 -1 中；
            4. 返回按层分组的有序字典。
        """
        if not isinstance(model_path, (str, os.PathLike)):  # 如果 model_path 不是文件路径（说明是 state_dict 字典或迭代器）

            def generate():  # 定义生成器函数，将权重按层分组
                """将 state_dict 按层编号分组。"""
                generator = OrderedDict()  # 创建有序字典，按层号存储权重
                model_dict = {}  # 初始化模型字典
                if not isinstance(model_path, dict):  # 如果 model_path 是迭代器（如权重生成器）
                    for key, value in list(model_path):  # 遍历迭代器，转换为列表
                        model_dict[key] = value  # 将权重添加到字典中
                else:  # 否则（model_path 已经是字典）
                    model_dict = model_path  # 直接使用该字典
                for key, value in model_dict.items():  # 遍历模型权重字典
                    match = re.findall(pattern, key)  # 使用正则表达式匹配层编号（如 'layers.12.weight' 中的 '12'）
                    if not match:  # 如果没有匹配到层编号（说明是全局权重，如 embedding、lm_head）
                        if -1 not in generator:  # 如果 -1 键不存在
                            generator[-1] = {}  # 创建 -1 键，用于存储全局权重
                        generator[-1][key] = value  # 将权重存入 -1 键
                    else:  # 否则（匹配到了层编号）
                        layer = int(match[0])  # 提取层编号（如 '12'）
                        if layer not in generator:  # 如果该层还没有键
                            generator[layer] = {}  # 创建该层的键
                        generator[layer][key] = value  # 将权重存入对应层
                return generator  # 返回按层分组的有序字典

            return generate()  # 调用生成器函数并返回结果
        else:  # 否则（model_path 是文件路径）
            return create_loader(model_path, pattern)  # 调用原始的 create_loader 函数

    loader.create_loader = _create_loader  # 替换 loader 模块的 create_loader 函数为自定义版本
    llama.create_loader = _create_loader  # 替换 llama 模块的 create_loader 函数为自定义版本

    TurbomindEngineConfig.devices = [0]  # 设置默认设备为 [0]（后续会被实际配置覆盖）

    from lmdeploy.turbomind.turbomind import TurboMind  # 导入 TurboMind 推理引擎类
    from lmdeploy.turbomind.utils import ModelSource  # 导入模型来源枚举

    @contextmanager  # 上下文管理器装饰器
    def patch_threadpool_map():  # 定义线程池 map 方法的补丁上下文管理器
        """临时禁用 ThreadPoolExecutor.map 方法的上下文管理器。
        
        目的：
            在初始化过程中禁用 ThreadPoolExecutor.map，避免与自定义的并行加载逻辑冲突。
        
        实现逻辑：
            1. 保存原始的 map 方法；
            2. 将 map 替换为空函数（返回空列表）；
            3. 退出上下文时恢复原始 map 方法。
        """
        ThreadPoolExecutor.map_origin = ThreadPoolExecutor.map  # 保存原始的 map 方法
        ThreadPoolExecutor.map = lambda *args, **kwargs: []  # 将 map 替换为空函数
        yield  # 执行上下文管理器包裹的代码块
        ThreadPoolExecutor.map = ThreadPoolExecutor.map_origin  # 恢复原始的 map 方法
        del ThreadPoolExecutor.map_origin  # 删除临时保存的原始方法

    @contextmanager  # 上下文管理器装饰器
    def tm_model_context(self):  # 定义 TurboMind 模型上下文管理器
        """捕获 get_tm_model 创建的 tm_model 实例的上下文管理器。
        
        参数：
            self: TurboMind 实例（用于保存 tm_model）
        
        目的：
            在模型转换过程中捕获 tm_model 实例，以便后续加载权重使用。
        
        实现逻辑：
            1. 替换 get_tm_model 函数为自定义版本；
            2. 在自定义版本中保存 tm_model 到 self.tm_model；
            3. 退出上下文时恢复原始 get_tm_model 函数。
        """

        def _get_tm_model(model_path,  # 定义自定义的 get_tm_model 函数
                          model_name,
                          chat_template_name,
                          engine_config: TurbomindEngineConfig,
                          group_size: int = None,
                          out_dir: str = None):
            """自定义的 get_tm_model 函数，用于捕获 tm_model 实例。"""
            from lmdeploy.turbomind.deploy.converter import get_tm_model_origin  # 导入原始的 get_tm_model 函数
            tm_model = get_tm_model_origin(model_path, model_name, chat_template_name, engine_config, group_size,  # 调用原始函数创建 tm_model
                                           out_dir)
            self.tm_model = tm_model  # 将 tm_model 保存到 self.tm_model（用于后续加载权重）
            return tm_model  # 返回 tm_model

        from lmdeploy.turbomind.deploy import converter  # 导入转换器模块
        converter.get_tm_model_origin = converter.get_tm_model  # 保存原始的 get_tm_model 函数
        converter.get_tm_model = _get_tm_model  # 替换为自定义版本
        yield  # 执行上下文管理器包裹的代码块
        converter.get_tm_model = converter.get_tm_model_origin  # 恢复原始的 get_tm_model 函数
        del converter.get_tm_model_origin  # 删除临时保存的原始函数

    def __init__(self,  # 定义 TurboMind 的自定义初始化方法
                 model_path: str,
                 tokenizer: object,
                 model_name: str = None,
                 chat_template_name: str = None,
                 engine_config: TurbomindEngineConfig = None,
                 model_source: ModelSource = ModelSource.WORKSPACE,
                 **kwargs):
        """自定义的 TurboMind.__init__ 方法，支持多 GPU 并行加载和自定义设备。
        
        参数：
            model_path (str): 模型路径或 state_dict
            tokenizer (object): 分词器实例
            model_name (str): 模型名称
            chat_template_name (str): 对话模板名称
            engine_config (TurbomindEngineConfig): 引擎配置（包含 devices 列表）
            model_source (ModelSource): 模型来源（WORKSPACE 或 HF_MODEL）
            **kwargs: 其他参数
        """
        self.gpu_list = engine_config.devices  # 从配置中获取设备列表（如 [0, 1, 2, 3]）
        with patch_threadpool_map(), tm_model_context(self):  # 使用上下文管理器禁用 ThreadPoolExecutor.map 并捕获 tm_model
            self.__origin_init__(model_path, tokenizer, model_name, chat_template_name, engine_config, model_source,  # 调用原始的 __init__ 方法
                                 **kwargs)

        with ThreadPoolExecutor(max_workers=self.gpu_count) as e:  # 创建线程池，最大工作线程数为 GPU 数量
            ranks = [self.node_id * self.gpu_count + device_id for device_id in range(self.gpu_count)]  # 计算每个 GPU 的全局 rank（支持多节点）
            if not load_weights:  # 如果不加载权重（默认行为）
                for _ in e.map(self.model_comm.process_weight, self.gpu_list, ranks):  # 并行处理权重（初始化权重缓冲区）
                    pass  # 空循环，等待所有任务完成
            if version.parse(lmdeploy.__version__) < version.parse('0.7.2'):  # 如果 lmdeploy 版本 < 0.7.2
                for _ in e.map(self.model_comm.create_engine, self.gpu_list, ranks, repeat(self.nccl_params)):  # 并行创建引擎（需要 nccl_params）
                    pass  # 空循环，等待所有任务完成
            else:  # 否则（lmdeploy 版本 >= 0.7.2）
                for _ in e.map(self.model_comm.create_engine, self.gpu_list, ranks):  # 并行创建引擎（不需要 nccl_params）
                    pass  # 空循环，等待所有任务完成

    def _create_weight(self, model_comm):  # 定义创建权重缓冲区的方法
        """Allocate weight buffer, load params if from_workspace."""  # 原始注释：分配权重缓冲区，如果来自工作区则加载参数
        """分配权重缓冲区，并在多 GPU 上创建共享权重。
        
        参数：
            model_comm: 模型通信对象（用于多 GPU 通信）
        """

        # TODO: support mpi  # 待办：支持 MPI（多进程接口，用于多节点训练）
        self.node_id = 0  # 设置节点 ID 为 0（当前仅支持单节点）
        self.node_num = 1  # 设置节点数量为 1（当前仅支持单节点）
        if version.parse(lmdeploy.__version__) < version.parse('0.7.2'):  # 如果 lmdeploy 版本 < 0.7.2
            self.nccl_params = model_comm.create_nccl_params(self.node_id)  # 创建 NCCL 参数（用于多 GPU 通信）
        torch.cuda.synchronize()  # 同步所有 CUDA 操作（确保前面的操作完成）

        # create weight  # 注释：创建权重
        def _create_weight_func(index, device_id):  # 定义创建权重的内部函数
            """在指定设备上创建共享权重。"""
            rank = self.node_id * self.gpu_count + index  # 计算当前 GPU 的全局 rank
            model_comm.create_shared_weights(device_id, rank)  # 在指定设备上创建共享权重

        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:  # 创建线程池，最大工作线程数为 GPU 数量
            futures = []  # 初始化 Future 列表
            for idx, device_id in enumerate(self.gpu_list):  # 遍历所有 GPU 设备
                futures.append(executor.submit(_create_weight_func, idx, device_id))  # 提交任务到线程池
            for future in futures:  # 遍历所有 Future
                future.result()  # 等待任务完成（阻塞直到任务结束）

    def _get_model_params(self, model_comm, tm_params):  # 定义获取模型参数的方法
        """Get turbomind model params when loading from hf."""  # 原始注释：从 HF 加载时获取 turbomind 模型参数
        """从多个 GPU 上获取模型参数并聚合。
        
        参数：
            model_comm: 模型通信对象
            tm_params (dict): TurboMind 参数字典（用于存储聚合后的参数）
        """

        def _get_params(idx, device_id, que):  # 定义获取参数的内部函数
            """从指定设备获取参数并放入队列。"""
            rank = self.node_id * self.gpu_count + idx  # 计算当前 GPU 的全局 rank
            out = model_comm.get_params(device_id, rank)  # 从指定设备获取参数
            que.put(out)  # 将参数放入队列

        que = Queue()  # 创建队列，用于存储各 GPU 的参数
        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:  # 创建线程池
            futures = []  # 初始化 Future 列表
            for idx, device_id in enumerate(self.gpu_list):  # 遍历所有 GPU 设备
                futures.append(executor.submit(_get_params, idx, device_id, que))  # 提交任务到线程池
            for future in futures:  # 遍历所有 Future
                future.result()  # 等待任务完成

        for _ in range(self.gpu_count):  # 遍历所有 GPU（从队列中获取参数）
            tensor_map = que.get()  # 从队列中获取一个 GPU 的参数字典
            for k, v in tensor_map.items():  # 遍历参数字典
                if k not in tm_params:  # 如果参数名不在 tm_params 中
                    tm_params[k] = []  # 初始化为空列表
                tm_params[k].append(v)  # 将当前 GPU 的参数添加到列表中（用于聚合）

    def _load_weights(self, state_dict):  # 定义加载权重的方法
        """从 state_dict 加载权重到 TurboMind 模型。
        
        参数：
            state_dict (dict): 权重字典（PyTorch state_dict 格式）
        """
        tm_params = self.tm_model.tm_params  # 获取 TurboMind 模型的参数字典
        self._get_model_params(self.model_comm, tm_params)  # 从各 GPU 获取当前参数（用于后续更新）
        input_model = self.tm_model.input_model  # 获取输入模型对象
        model_path = input_model.model_path  # 保存原始模型路径
        input_model.model_path = state_dict  # 临时将 model_path 设置为 state_dict（用于加载权重）
        self.tm_model.export()  # 导出模型（实际执行权重加载）
        input_model.model_path = model_path  # 恢复原始模型路径

    from lmdeploy.turbomind.turbomind import TurboMindInstance  # 导入 TurboMindInstance 类

    def create_instance(self, cuda_stream_id=0):  # 定义创建推理实例的方法
        """创建 TurboMindInstance 推理实例。
        
        参数：
            cuda_stream_id (int): CUDA 流 ID（默认 0）
        
        返回：
            TurboMindInstance: 推理实例
        """
        return TurboMindInstance(self, self.config, cuda_stream_id, self.gpu_list)  # 创建并返回推理实例（传入 gpu_list）

    TurboMind.__origin_init__ = TurboMind.__init__  # 保存原始的 __init__ 方法
    TurboMind.__init__ = __init__  # 替换为自定义的 __init__ 方法
    TurboMind._create_weight = _create_weight  # 添加 _create_weight 方法
    TurboMind._get_model_params = _get_model_params  # 添加 _get_model_params 方法
    TurboMind.create_instance = create_instance  # 替换 create_instance 方法
    if load_weights:  # 如果需要加载权重
        TurboMind.load_weights = _load_weights  # 添加 load_weights 方法

    def __init_ins__(self, tm_model, config, cuda_stream_id=0, gpu_list=None):  # 定义 TurboMindInstance 的自定义初始化方法
        """自定义的 TurboMindInstance.__init__ 方法，支持自定义 GPU 列表。
        
        参数：
            tm_model: TurboMind 模型实例
            config: 配置对象
            cuda_stream_id (int): CUDA 流 ID
            gpu_list (list): GPU 设备列表（默认 [0]）
        """
        if gpu_list is None:  # 如果 gpu_list 未指定
            gpu_list = [0]  # 默认使用 GPU 0
        self.gpu_list = gpu_list  # 保存 GPU 列表
        self.__origin_init__(tm_model, config, cuda_stream_id)  # 调用原始的 __init__ 方法

    def _create_model_instance(self, device_id):  # 定义创建模型实例的方法
        """创建模型推理实例。
        
        参数：
            device_id (int): 设备 ID（此参数未使用，实际使用 self.gpu_list[0]）
        
        返回：
            ModelInstance: 模型实例
        """
        model_inst = self.tm_model.model_comm.create_model_instance(self.gpu_list[0])  # 在第一个 GPU 上创建模型实例
        return model_inst  # 返回模型实例

    TurboMindInstance.__origin_init__ = TurboMindInstance.__init__  # 保存原始的 __init__ 方法
    TurboMindInstance.__init__ = __init_ins__  # 替换为自定义的 __init__ 方法
    TurboMindInstance._create_model_instance = _create_model_instance  # 替换 _create_model_instance 方法


def patch_npu_vllm(vllm_device: str):  # 定义 NPU 设备的 vLLM 补丁函数
    """函数功能：
    为 vLLM 在 NPU（华为昇腾 AI 处理器）设备上运行提供补丁。
    
    核心功能：
        1. 修复 vLLM 在 NPU 设备上的分布式通信问题；
        2. 固定内存查询函数到指定 NPU 设备；
        3. 为 GPU 设备返回空上下文（不需要补丁）。
    
    参数：
        vllm_device (str | int): vLLM 使用的设备（如 'npu:0' 或 0）
    
    返回：
        ContextManager: 如果是 NPU 设备，返回补丁上下文管理器；否则返回空上下文
    
    应用场景：
        在华为昇腾 NPU 上运行 vLLM 推理引擎时使用。
    """
    if isinstance(vllm_device, int):  # 如果 vllm_device 是整数（如 0）
        vllm_device = get_device(vllm_device)  # 转换为设备字符串（如 'npu:0'）
    device_type = vllm_device.split(':')[0]  # 提取设备类型（'npu' 或 'cuda'）

    @contextmanager  # 上下文管理器装饰器
    def new_group_context():  # 定义新的分布式通信组上下文管理器
        """为 NPU 设备创建分布式通信组的补丁上下文管理器。
        
        实现逻辑：
            1. 替换 torch.distributed.new_group 为使用本地同步的版本；
            2. 固定 torch.npu.mem_get_info 到指定 NPU 设备；
            3. 退出上下文时恢复原始函数。
        """
        original_new_group = torch.distributed.new_group  # 保存原始的 new_group 函数
        try:  # 异常处理
            torch.distributed.new_group = partial(original_new_group, use_local_synchronization=True)  # 替换为使用本地同步的版本（修复 NPU 分布式通信问题）
            torch.npu.mem_get_info = partial(torch.npu.mem_get_info, device=vllm_device)  # 固定 mem_get_info 到指定 NPU 设备（修复内存查询问题）
            yield  # 执行上下文管理器包裹的代码块
        finally:  # 确保恢复原始函数（无论是否发生异常）
            torch.distributed.new_group = original_new_group  # 恢复原始的 new_group 函数

    return new_group_context() if device_type == 'npu' else nullcontext()  # 如果是 NPU 设备，返回补丁上下文；否则返回空上下文


def patch_vllm_memory_leak():  # 定义 vLLM 内存泄漏修复补丁函数
    """函数功能：
    修复 vLLM 0.7.3 版本的内存泄漏问题。
    
    核心问题：
        vLLM 0.7.3 在处理 n>1（并行采样）时，seq_id_to_seq_group 字典没有正确清理，
        导致已完成的请求仍然占用内存。
    
    修复方案：
        1. 修复 Scheduler.abort_seq_group 方法，正确删除 seq_id_to_seq_group 中的条目；
        2. 修复 LLMEngine.abort_request 方法，传递 seq_id_to_seq_group 参数；
        3. 修复 LLMEngine.step 方法，在请求完成时清理 seq_id_to_seq_group。
    
    参考：
        https://github.com/vllm-project/vllm/pull/14326
    
    应用场景：
        仅在 vLLM 版本为 0.7.3 时应用此补丁。
    """
    # fix vllm 0.7.3 memory leak  # 注释：修复 vLLM 0.7.3 内存泄漏
    # https://github.com/vllm-project/vllm/pull/14326  # 注释：参考 GitHub PR
    import vllm  # 导入 vllm 库
    try:  # 尝试解析版本号
        vllm_version = version.parse(vllm.__version__)  # 解析 vllm 版本号
        needs_patch = (vllm_version == version.parse('0.7.3'))  # 判断是否为 0.7.3 版本（需要补丁）
    except version.InvalidVersion:  # 如果版本号无效
        needs_patch = False  # 不需要补丁

    if not needs_patch:  # 如果不需要补丁
        return  # 直接返回（不应用任何补丁）

    def patch_vllm_abort_seq_group():  # 定义修复 Scheduler.abort_seq_group 的子函数
        """修复 Scheduler.abort_seq_group 方法的内存泄漏。
        
        核心修复：
            在中止序列组时，正确删除 seq_id_to_seq_group 中的条目。
        """
        from vllm.core.scheduler import Scheduler  # 导入 Scheduler 类
        from typing import Iterable, Dict  # 导入类型注解
        from vllm.sequence import SequenceGroupBase, SequenceGroup, SequenceStatus  # 导入序列相关类

        def new_abort_seq_group(  # 定义新的 abort_seq_group 方法
            self,
            request_id: Union[str, Iterable[str]],
            seq_id_to_seq_group: Optional[Dict[str, SequenceGroupBase]] = None,
        ) -> None:
            """中止指定的序列组并清理资源。
            
            参数：
                request_id (str | Iterable[str]): 要中止的请求 ID（可以是单个或多个）
                seq_id_to_seq_group (dict): 序列 ID 到序列组的映射（用于内存清理）
            """
            if isinstance(request_id, str):  # 如果 request_id 是字符串（单个 ID）
                request_id = (request_id, )  # 转换为元组（统一处理）
            request_ids = set(request_id)  # 转换为集合（用于快速查找）
            seq_id_to_seq_group = seq_id_to_seq_group or {}  # 如果未提供，使用空字典
            for state_queue in [self.waiting, self.running, self.swapped]:  # 遍历所有状态队列（等待、运行、交换）
                aborted_groups: List[SequenceGroup] = []  # 初始化中止的序列组列表
                for seq_group in state_queue:  # 遍历当前状态队列中的序列组
                    # When n>1, seq_group.request_id looks like  # 注释：当 n>1 时，seq_group.request_id 类似于
                    # foo_parallel_sample_0, while request_ids is just foo, and we  # 注释：foo_parallel_sample_0，而 request_ids 只是 foo
                    # should resolve it as real_request_id to match.  # 注释：我们应该解析为 real_request_id 来匹配
                    if seq_group.request_id in seq_id_to_seq_group:  # 如果序列组 ID 在映射中
                        real_request_id = seq_id_to_seq_group[seq_group.request_id].group_id  # 获取真实的请求 ID（去掉并行采样后缀）
                    else:  # 否则（没有并行采样）
                        real_request_id = seq_group.request_id  # 直接使用序列组 ID
                    if real_request_id in request_ids:  # 如果真实请求 ID 在要中止的列表中
                        # Appending aborted group into pending list.  # 注释：将中止的组添加到待处理列表
                        aborted_groups.append(seq_group)  # 添加到中止列表
                        # We can't remove real_request_id in request_ids here,  # 注释：我们不能在这里删除 real_request_id
                        # because there may be other seq groups sharing the same  # 注释：因为可能有其他序列组共享相同的
                        # real_request_id  # 注释：real_request_id
                for aborted_group in aborted_groups:  # 遍历所有要中止的序列组
                    # Remove the sequence group from the state queue.  # 注释：从状态队列中删除序列组
                    state_queue.remove(aborted_group)  # 从队列中删除
                    # Remove the aborted request from the Mamba cache.  # 注释：从 Mamba 缓存中删除中止的请求
                    self._finished_requests_ids.append(aborted_group.request_id)  # 将请求 ID 添加到已完成列表
                    for seq in aborted_group.get_seqs():  # 遍历序列组中的所有序列
                        if seq.is_finished():  # 如果序列已完成
                            continue  # 跳过
                        seq.status = SequenceStatus.FINISHED_ABORTED  # 设置状态为已中止
                        self.free_seq(seq)  # 释放序列资源
                    if aborted_group.request_id in seq_id_to_seq_group:  # 如果请求 ID 在映射中（关键修复！）
                        del seq_id_to_seq_group[aborted_group.request_id]  # 删除映射条目（防止内存泄漏）

                    self._free_seq_group_cross_attn_blocks(aborted_group)  # 释放交叉注意力块

        origin_method = Scheduler.abort_seq_group  # 保存原始方法
        Scheduler._old_abort_seq_group = origin_method  # 保存原始方法的备份
        Scheduler.abort_seq_group = new_abort_seq_group  # 替换为新方法

    def patch_vllm_engine():  # 定义修复 LLMEngine 的子函数
        """修复 LLMEngine 的内存泄漏问题。
        
        核心修复：
            1. 修复 abort_request 方法，传递 seq_id_to_seq_group 参数；
            2. 修复 step 方法，在请求完成时删除 seq_id_to_seq_group 中的条目。
        """
        from vllm.engine.llm_engine import LLMEngine, SchedulerOutputState  # 导入 LLMEngine 和 SchedulerOutputState
        from vllm.outputs import PoolingRequestOutput, RequestOutput  # 导入输出类型
        from vllm.sequence import ExecuteModelRequest  # 导入执行模型请求类

        def new_abort_request(self, request_id) -> None:  # 定义新的 abort_request 方法
            """中止指定的请求（修复版本）。
            
            参数：
                request_id: 要中止的请求 ID
            
            核心修复：
                传递 seq_id_to_seq_group 参数给 abort_seq_group，确保正确清理内存。
            """
            for scheduler in self.scheduler:  # 遍历所有调度器
                scheduler.abort_seq_group(request_id, seq_id_to_seq_group=self.seq_id_to_seq_group)  # 中止序列组并传递映射字典（关键修复！）

        origin_method = LLMEngine.abort_request  # 保存原始方法
        LLMEngine._old_abort_request = origin_method  # 保存原始方法的备份
        LLMEngine.abort_request = new_abort_request  # 替换为新方法

        def new_step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:  # 定义新的 step 方法
            """执行一步推理（修复版本）。
            
            返回：
                List[Union[RequestOutput, PoolingRequestOutput]]: 本步的输出列表
            
            核心修复：
                在请求完成时，删除 seq_id_to_seq_group 中的条目，防止内存泄漏。
            """
            if self.parallel_config.pipeline_parallel_size > 1:  # 如果启用了流水线并行
                raise NotImplementedError('Pipeline parallelism is only supported through AsyncLLMEngine '  # 抛出异常（流水线并行仅在 AsyncLLMEngine 中支持）
                                          'as performance will be severely degraded otherwise.')

            # For llm_engine, there is no pipeline parallel support, so the engine  # 注释：对于 llm_engine，没有流水线并行支持
            # used is always 0.  # 注释：所以引擎总是 0
            virtual_engine = 0  # 虚拟引擎索引为 0

            # These are cached outputs from previous iterations. None if on first  # 注释：这些是前一次迭代缓存的输出
            # iteration  # 注释：第一次迭代时为 None
            cached_outputs = self.cached_scheduler_outputs[virtual_engine]  # 获取虚拟引擎的缓存调度器输出
            seq_group_metadata_list = cached_outputs.seq_group_metadata_list  # 从缓存中提取序列组元数据列表
            scheduler_outputs = cached_outputs.scheduler_outputs  # 从缓存中提取调度器输出
            allow_async_output_proc = cached_outputs.allow_async_output_proc  # 从缓存中提取是否允许异步输出处理标志

            ctx = self.scheduler_contexts[virtual_engine]  # 获取虚拟引擎的调度器上下文

            # Clear outputs for each new scheduler iteration  # 注释：在每次新的调度器迭代时清空输出
            ctx.request_outputs.clear()  # 清空上下文的请求输出列表

            # Skip the scheduler if there are any remaining steps in the seq groups.  # 注释：如果序列组中还有剩余步骤，跳过调度器
            # This ensures that the scheduler is only called again when the current  # 注释：这确保调度器只在当前批次完成后才被再次调用
            # batch has completed.  # 注释：批次已完成
            # The scheduler is also skipped if a single request caused the last  # 注释：如果单个请求导致上一次引擎步骤失败，也会跳过调度器
            # engine step to fail, and the previous schedule needs to be rerun.  # 注释：需要重新运行前一次的调度
            if not self._has_remaining_steps(seq_group_metadata_list):  # 如果序列组中没有剩余步骤（需要新的调度）
                # Schedule iteration  # 注释：执行调度迭代
                (seq_group_metadata_list, scheduler_outputs,  # 从调度器获取新的序列组元数据列表、调度器输出
                 allow_async_output_proc) = self.scheduler[virtual_engine].schedule()  # 和异步输出处理标志

                ctx.seq_group_metadata_list = seq_group_metadata_list  # 保存序列组元数据列表到上下文
                ctx.scheduler_outputs = scheduler_outputs  # 保存调度器输出到上下文

                finished_requests_ids = self.scheduler[virtual_engine].get_and_reset_finished_requests_ids()  # 获取已完成的请求 ID 列表
                # When n>1, elements in self.seq_id_to_seq_group should be deleted  # 注释：当 n>1 时，self.seq_id_to_seq_group 中的元素应该被删除
                # here, otherwise memory leaks.  # 注释：否则会导致内存泄漏
                for finished_request_id in finished_requests_ids:  # 遍历所有已完成的请求 ID
                    if finished_request_id in self.seq_id_to_seq_group:  # 如果请求 ID 在映射中（关键修复！）
                        del self.seq_id_to_seq_group[finished_request_id]  # 删除映射条目（防止内存泄漏）

                # Maybe switch from async mode to sync mode  # 注释：可能需要从异步模式切换到同步模式
                if not allow_async_output_proc and len(ctx.output_queue) > 0:  # 如果不允许异步输出处理且输出队列不为空
                    self._process_model_outputs(ctx=ctx)  # 立即处理模型输出（同步模式）

                if (self.scheduler_config.is_multi_step and scheduler_outputs.num_lookahead_slots > 0):  # 如果启用了多步调度且有前瞻槽位
                    # cache the scheduler outputs for the next iteration if we have  # 注释：如果有前瞻槽位，缓存调度器输出以供下次迭代使用
                    # lookahead slots  # 注释：前瞻槽位
                    self._cache_scheduler_outputs_for_multi_step(virtual_engine, seq_group_metadata_list,  # 缓存多步调度的调度器输出
                                                                 scheduler_outputs, allow_async_output_proc)  # 传入所有相关参数
            else:  # 否则（有剩余步骤，跳过了调度）
                finished_requests_ids = list()  # 已完成请求 ID 列表为空

            assert seq_group_metadata_list is not None  # 断言序列组元数据列表不为 None
            assert scheduler_outputs is not None  # 断言调度器输出不为 None

            if not scheduler_outputs.is_empty():  # 如果调度器输出不为空（有任务需要执行）

                # Check if we have a cached last_output from the previous iteration.  # 注释：检查是否有前一次迭代缓存的 last_output
                # For supporting PP this is probably the best way to pass the  # 注释：为了支持流水线并行（PP），这可能是传递的最佳方式
                # sampled_token_ids, as a separate broadcast over all the PP stages  # 注释：sampled_token_ids，因为在所有 PP 阶段单独广播
                # will cause one virtual engine's microbatch to block the pipeline.  # 注释：会导致一个虚拟引擎的微批次阻塞流水线
                # 获取上一步采样的 token IDs（用于流水线并行传递）
                last_sampled_token_ids = self._get_last_sampled_token_ids(virtual_engine)  # 从虚拟引擎获取上一步采样的 token IDs

                execute_model_req = ExecuteModelRequest(  # 创建执行模型请求对象
                    seq_group_metadata_list=seq_group_metadata_list,  # 传入序列组元数据列表
                    blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,  # 传入需要换入内存的块列表
                    blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,  # 传入需要换出内存的块列表
                    blocks_to_copy=scheduler_outputs.blocks_to_copy,  # 传入需要复制的块列表
                    num_lookahead_slots=scheduler_outputs.num_lookahead_slots,  # 传入前瞻槽位数量
                    running_queue_size=scheduler_outputs.running_queue_size,  # 传入运行队列大小
                    finished_requests_ids=finished_requests_ids,  # 传入已完成请求 ID 列表
                    # We use ExecuteModelRequest to pass the last sampled_token_ids  # 注释：我们使用 ExecuteModelRequest 传递 last sampled_token_ids
                    # to each of the non-last PP stages for in-place prepare_input.  # 注释：给每个非最后的 PP 阶段，用于就地准备输入
                    last_sampled_token_ids=last_sampled_token_ids)  # 传入上一步采样的 token IDs

                if allow_async_output_proc:  # 如果允许异步输出处理
                    execute_model_req.async_callback = self.async_callbacks[virtual_engine]  # 设置异步回调函数

                outputs = self.model_executor.execute_model(execute_model_req=execute_model_req)  # 执行模型前向推理，获取输出

                # We need to do this here so that last step's sampled_token_ids can  # 注释：我们需要在这里做这件事，以便上一步的 sampled_token_ids
                # be passed to the next iteration for PP.  # 注释：能够传递给下一次迭代（用于流水线并行）
                if self.scheduler_config.is_multi_step:  # 如果启用了多步调度
                    self._update_cached_scheduler_output(virtual_engine, outputs)  # 更新缓存的调度器输出
            else:  # 否则（调度器输出为空，没有任务需要执行）
                # Nothing scheduled => If there is pending async postprocessor,  # 注释：没有任务调度 => 如果有待处理的异步后处理器
                # then finish it here.  # 注释：则在这里完成它
                if len(ctx.output_queue) > 0:  # 如果输出队列不为空
                    self._process_model_outputs(ctx=ctx)  # 处理模型输出
                # No outputs in this case  # 注释：这种情况下没有输出
                outputs = []  # 输出为空列表

            # Finish the current step for all the sequence groups.  # 注释：完成所有序列组的当前步骤
            if self.scheduler_config.is_multi_step:  # 如果启用了多步调度
                for seq_group in seq_group_metadata_list:  # 遍历所有序列组
                    seq_group.finish_step()  # 标记该序列组的当前步骤已完成

            if not self._has_remaining_steps(seq_group_metadata_list):  # 如果所有序列组都没有剩余步骤（当前批次已完成）
                # clear the cache if we have finished all the steps.  # 注释：如果所有步骤都已完成，清空缓存
                if self.scheduler_config.is_multi_step:  # 如果启用了多步调度
                    self.cached_scheduler_outputs[0] = SchedulerOutputState()  # 重置缓存的调度器输出为空状态

                # is_first_step_output is True only when the num_steps of all  # 注释：is_first_step_output 仅在所有序列的 num_steps
                # the sequences are 1. When the num_steps > 1,  # 注释：都为 1 时为 True。当 num_steps > 1 时，
                # multi_step_model_runner does the first-step output append.  # 注释：multi_step_model_runner 会追加第一步输出
                # 判断是否为第一步输出（仅当所有序列的步数都为 1 时）
                is_first_step_output: bool = (False if not seq_group_metadata_list
                                              else seq_group_metadata_list[0].state.num_steps == 1)  # 检查第一个序列组的步数

                # Add results to the output_queue  # 注释：将结果添加到输出队列
                ctx.append_output(  # 向上下文追加输出
                    outputs=outputs,  # 传入模型输出
                    seq_group_metadata_list=seq_group_metadata_list,  # 传入序列组元数据列表
                    scheduler_outputs=scheduler_outputs,  # 传入调度器输出
                    is_async=allow_async_output_proc,  # 传入是否异步处理标志
                    is_last_step=True,  # 标记这是最后一步
                    is_first_step_output=is_first_step_output)  # 传入是否为第一步输出标志

                if outputs and allow_async_output_proc:  # 如果有输出且允许异步输出处理
                    assert len(outputs) == 1, ('Async postprocessor expects only a single output set')  # 断言输出数量为 1（异步后处理器只接受单个输出集）

                    self._advance_to_next_step(outputs[0], seq_group_metadata_list,  # 推进到下一步
                                               scheduler_outputs.scheduled_seq_groups)  # 传入调度的序列组

                # Check if need to run the usual non-async path  # 注释：检查是否需要运行常规的非异步路径
                if not allow_async_output_proc:  # 如果不允许异步输出处理
                    self._process_model_outputs(ctx=ctx)  # 同步处理模型输出

                    # Log stats.  # 注释：记录统计信息
                    self.do_log_stats(scheduler_outputs, outputs)  # 记录调度器输出和模型输出的统计信息

                    # Tracing  # 注释：跟踪
                    self.do_tracing(scheduler_outputs)  # 记录调度器输出的跟踪信息
            else:  # 否则（有剩余步骤，多步情况）
                # Multi-step case  # 注释：多步情况
                return ctx.request_outputs  # 返回当前的请求输出（不是最后一步）

            if not self.has_unfinished_requests():  # 如果没有未完成的请求
                # Drain async postprocessor (if exists)  # 注释：排空异步后处理器（如果存在）
                if len(ctx.output_queue) > 0:  # 如果输出队列不为空
                    self._process_model_outputs(ctx=ctx)  # 处理剩余的模型输出
                assert len(ctx.output_queue) == 0  # 断言输出队列已为空

                # Stop the execute model loop in parallel workers until there are  # 注释：停止并行工作进程中的执行模型循环，直到有
                # more requests to process. This avoids waiting indefinitely in  # 注释：更多请求需要处理。这避免了在
                # torch.distributed ops which may otherwise timeout, and unblocks  # 注释：torch.distributed 操作中无限期等待（可能超时），并解除阻塞
                # the RPC thread in the workers so that they can process any other  # 注释：工作进程中的 RPC 线程，使它们可以处理任何其他
                # queued control plane messages, such as add/remove lora adapters.  # 注释：排队的控制平面消息，例如添加/删除 lora 适配器
                self.model_executor.stop_remote_worker_execution_loop()  # 停止远程工作进程的执行循环

            return ctx.request_outputs  # 返回上下文中的请求输出

        origin_method = LLMEngine.step  # 保存原始的 step 方法
        LLMEngine._old_step = origin_method  # 保存原始方法的备份
        LLMEngine.step = new_step  # 替换为新的 step 方法

    patch_vllm_abort_seq_group()  # 应用 Scheduler.abort_seq_group 补丁
    patch_vllm_engine()  # 应用 LLMEngine 补丁
