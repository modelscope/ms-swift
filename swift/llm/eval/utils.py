"""
模块功能概述：
本模块为评测后端（evalscope）适配的模型包装工具，提供：
- EvalModel: 适配 evalscope 的自定义模型接口，将 PyTorch 模型与模板封装为可预测对象；
- 统一的输入准备逻辑：将评测样本转换为对话消息格式（system/user），便于统一推理调用。

使用示例：
>>> model = ...  # 预训练/微调后的模型
>>> template = ...  # 模板对象，负责消息组装与生成配置
>>> eval_model = EvalModel(model, template, max_batch_size=8, model_name="my-llm")
>>> outputs = eval_model.predict(prompts=[{"data": ["What is AI?"], "system_prompt": "You are helpful."}],
...                             infer_cfg={"max_tokens": 256, "temperature": 0.0})
"""

# 从dataclasses导入asdict：将数据类对象转换为普通字典，便于序列化或统一返回格式
from dataclasses import asdict
# 从typing导入类型提示：Any、Dict、List、Union 用于函数签名与可读性
from typing import Any, Dict, List, Union

# 导入PyTorch的神经网络基类nn：用于类型标注和与transformers模型统一处理
import torch.nn as nn
# 从evalscope的自定义模型接口导入基类CustomModel：EvalModel将继承该基类对接评测框架
from evalscope.models.custom import CustomModel
# 从transformers导入预训练模型基类：用于与普通nn.Module做联合类型标注
from transformers import PreTrainedModel

# 从上级模块infer导入推理引擎与请求配置：PtEngine负责执行推理，RequestConfig承载生成参数
from ..infer import PtEngine, RequestConfig
# 从上级模块template导入推理请求数据结构：InferRequest封装多轮消息
from ..template import InferRequest


# 定义评测用模型适配器：承接evalscope调用并转发到内部PtEngine
class EvalModel(CustomModel):
    """
    类功能：
        将底层的PyTorch/Transformers模型与模板（Template）封装为evalscope可识别的自定义模型。
        负责：
        1) 按模板构建推理引擎（PtEngine）
        2) 适配evalscope的predict接口，执行批量推理并返回字典结果
        3) 准备输入，将多种评测样本格式转换为对话消息列表

    关键属性：
        model_name (str): 模型标识，用于evalscope侧显示与跟踪
        model (Union[PreTrainedModel, nn.Module]): 底层可调用模型
        template: 模板对象，负责消息规范与生成策略
        engine (PtEngine): 推理引擎，负责实际的批量推理
    """

    # 构造函数：初始化模型、模板与推理引擎
    def __init__(self, model: Union[PreTrainedModel, nn.Module], template, max_batch_size, model_name: str,
                 **kwargs) -> None:
        """
        函数功能：
            初始化EvalModel，保存模型与模板信息，基于模板构造PtEngine推理引擎。

        入参：
            model (Union[PreTrainedModel, nn.Module]): 底层模型（Transformers或任意nn.Module）。
            template: 模板对象，定义消息组织与encode/decode逻辑。
            max_batch_size: 推理阶段的最大批大小，用于限制一次推理的并发样本数。
            model_name (str): 模型标识，传递给evalscope以供报告标注。
            **kwargs: 透传给父类CustomModel的其他配置项。

        返回值：
            None

        示例：
            >>> EvalModel(model, template, max_batch_size=8, model_name="my-llm")
        """
        # 调用父类构造函数，设置evalscope侧的模型配置（此处仅传入model_id）
        super().__init__(config={'model_id': model_name}, **kwargs)
        # 保存模型标识字符串，便于日志与报告展示
        self.model_name = model_name
        # 保存底层模型对象，供推理引擎调用
        self.model = model
        # 保存模板对象，后续用于消息构建与引擎初始化
        self.template = template
        # 基于模型与模板创建推理引擎，限定最大批大小
        self.engine = PtEngine.from_model_template(model, template, max_batch_size=max_batch_size)

    # 预测接口：适配evalscope的CustomModel.predict签名
    def predict(self, prompts: List[dict], **kwargs) -> List[Dict[str, Any]]:
        """
        函数功能：
            执行批量推理。优先使用kwargs中的origin_inputs作为原始输入，否则使用prompts。
            将输入转换为InferRequest列表，构造RequestConfig并调用PtEngine进行推理，
            最终将返回的数据类对象转换为字典列表。

        入参：
            prompts (List[dict]): 评测框架传入的样本列表，通常包含data、system_prompt等字段。
            **kwargs: 额外参数：
                - origin_inputs: 原始输入（可覆盖prompts）
                - infer_cfg: 生成参数字典，将用于构造RequestConfig

        返回值：
            List[Dict[str, Any]]: 每个样本的推理结果，以字典形式返回，便于序列化。

        示例：
            >>> eval_model.predict(prompts=[{"data": ["hi"]}], infer_cfg={"max_tokens": 16})
        """
        # 选择原始输入来源：优先kwargs['origin_inputs']，否则使用prompts本身
        # use origin inputs
        infer_requests = self.prepare_inputs(kwargs.get('origin_inputs', prompts))

        # 从kwargs中取出推理配置副本（避免修改原对象）
        infer_cfg = kwargs['infer_cfg'].copy()
        # 将普通字典转为RequestConfig数据类，便于引擎统一处理生成参数
        generation_config = RequestConfig(**infer_cfg)

        # 调用推理引擎执行批量推理，关闭tqdm以避免评测时的多余输出
        response = self.engine.infer(infer_requests=infer_requests, request_config=generation_config, use_tqdm=False)
        # 将数据类对象列表转换为字典列表，评测框架通常以json友好格式消费结果
        dict_response = [asdict(item) for item in response]
        # 返回字典化的推理结果
        return dict_response

    # 输入准备：将评测样本转换为InferRequest列表
    def prepare_inputs(self, prompts: Union[List[dict], List[str]]) -> List[InferRequest]:
        """
        函数功能：
            将多样的输入格式（字符串或带data字段的字典）转换成统一的消息序列：
            - 若为字符串：作为用户query，system_prompt为空
            - 若为字典：优先从data字段读取文本；若data[0]为tuple（truthful_qa/hellaswag），拼接为多行

        入参：
            prompts (Union[List[dict], List[str]]): 输入样本列表，元素可为str或dict。

        返回值：
            List[InferRequest]: 统一格式的推理请求列表，每项包含messages（system/user）。

        示例：
            >>> self.prepare_inputs(["hello"])
        """
        # 初始化承载所有推理请求的列表
        infer_requests = []
        # 遍历每条输入样本，进行格式归一化
        for input_item in prompts:
            # 若样本为字符串，直接作为query使用，system_prompt为空
            if isinstance(input_item, str):
                query = input_item
                system_prompt = None
            else:
                # 否则样本为字典，读取其中的data字段（列表形式）
                data: list = input_item['data']
                # 如果data[0]是tuple，表示题目由多段组成（如truthful_qa/hellaswag），需要拼接
                if isinstance(data[0], tuple):  # for truthful_qa and hellaswag
                    # 先将tuple内的片段连接，再按条目换行拼接为完整query
                    query = '\n'.join(''.join(item) for item in data)
                    # 可选读取system_prompt（若未提供则为None）
                    system_prompt = input_item.get('system_prompt', None)
                else:
                    # 否则直接将第一条文本作为query
                    query = data[0]
                    # 可选读取system_prompt
                    system_prompt = input_item.get('system_prompt', None)
            #  准备对话消息列表，遵循system在前、user在后的顺序
            #  prepare messages
            messages = []
            # 若存在system提示词，则先加入一条system消息
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            # 再加入用户query为user消息
            messages.append({'role': 'user', 'content': query})
            # 将该条消息列表封装为InferRequest并加入结果列表
            infer_requests.append(InferRequest(messages=messages))
        # 返回统一化的推理请求列表
        return infer_requests
