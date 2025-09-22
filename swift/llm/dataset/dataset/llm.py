"""
模块功能
-------
本模块集中定义 LLM 训练/微调常用公开数据集的预处理器（Preprocessor）与数据集注册（register_dataset）逻辑，
涵盖通用对话、长文本、数学、检索重排、函数调用、分类、法务、编程等多类型场景。每个数据集通过
`DatasetMeta` 描述，结合对应 `Preprocessor` 将原始样本转换为统一训练所需的 `messages/query/response` 等字段。

典型用法
-------
1. 导入本模块后，调用方会在内部注册表中查询 `ms_dataset_id/hf_dataset_id/subsets/split` 对应的数据集；
2. Trainer 或数据管道读取 `DatasetMeta` 与 `Preprocessor`，据此构建标准化样本；
3. 预处理器可选择性地修补（repair）或裁剪消息历史，保证样本质量与格式一致性。

说明：为便于阅读与维护，本文件为每一行代码添加了中文注释（含模块/类/函数文档注释与行内注释）。
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，表明代码归属与授权信息
import ast  # 抽象语法树工具，用于安全地从字符串解析 Python 字面量
import re  # 正则表达式库，用于文本模式匹配与替换
from functools import partial  # 偏函数工具，用于为回调函数预先绑定参数
from typing import Any, Dict, List, Optional, Tuple, Union  # 类型注解：通用字典/列表/可选/元组/并集类型

import json  # JSON 序列化/反序列化，用于处理工具调用等结构化字段
import numpy as np  # 数值计算库，这里用于随机选择等轻量操作

from ...template import split_str_parts_by  # 字符串切分辅助函数，按关键标记分段
from ..preprocessor import (AlpacaPreprocessor, ClsGenerationPreprocessor, ClsPreprocessor, MessagesPreprocessor,  # 导入各类预处理器
                            ResponsePreprocessor, RowPreprocessor, TextGenerationPreprocessor)  # 统一将原始数据转为标准格式
from ..register import DatasetMeta, SubsetDataset, register_dataset  # 数据集元信息描述、子集封装与注册入口


class AlpacaZhPreprocessor(AlpacaPreprocessor):
    """
    类说明
    -----
    基于 `AlpacaPreprocessor` 的中文适配版本。针对中文样本中常见的前缀“输入：”进行裁剪，
    再沿用父类的拼接逻辑，确保 `instruction + input` 的合成结果符合训练模板。

    继承关系
    -------
    - AlpacaPreprocessor: 提供通用的 alpaca 风格样本拼接与字段归一化能力。
    """

    @classmethod
    def concat_inst_input(cls, instruction, input_):
        """
        将指令与输入拼接为统一的 `instruction + input` 字符串。

        参数
        ----
        - instruction: 指令文本。
        - input_: 输入文本，若以“输入：”开头则剥离此前缀。

        返回
        ----
        - str: 处理并拼接后的文本，由父类实现最终拼接细节。

        示例
        ----
        >>> AlpacaZhPreprocessor.concat_inst_input('请翻译', '输入：你好')
        '请翻译\n你好'
        """
        if input_ and input_.startswith('输入：'):  # 若输入以中文前缀“输入：”开头
            input_ = input_[3:]  # 去除前三个字符以剥离前缀
        return super().concat_inst_input(instruction, input_)  # 调用父类方法完成标准拼接


register_dataset(
    DatasetMeta(  # 创建数据集元信息，描述数据来源与预处理方式
        ms_dataset_id='AI-ModelScope/alpaca-gpt4-data-zh',  # ModelScope 平台的数据集标识
        hf_dataset_id='llm-wizard/alpaca-gpt4-data-zh',  # HuggingFace Hub 的数据集标识
        preprocess_func=AlpacaZhPreprocessor(),  # 使用中文适配版 Alpaca 预处理器
        tags=['chat', 'general', '🔥'],  # 标签：对话/通用/热门
    ))  # 立即注册到数据集注册表


class LongAlpacaPreprocessor(AlpacaPreprocessor):
    """
    类说明
    -----
    面向长文本 Alpaca 样本的预处理器。对部分样本中以“Answer: ”为前缀的响应字段进行规整，
    去除该前缀后回落到父类通用流程处理。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将长文本样本中的响应字段进行前缀修正后，交给父类做标准化处理。

        参数
        ----
        - row: 原始样本字典，预期包含 `response` 等字段。

        返回
        ----
        - Optional[Dict[str, Any]]: 规范化后的样本；若样本无效可返回 None。

        示例
        ----
        >>> pre = LongAlpacaPreprocessor()
        >>> pre.preprocess({'response': 'Answer: hello'})['output']
        'hello'
        """
        response = row['response']  # 取出响应字段
        prefix_prompt = 'Answer: '  # 需要剥离的前缀
        if response and response.startswith(prefix_prompt):  # 若响应以该前缀开头
            response = response[len(prefix_prompt):].strip()  # 去掉前缀并清理首尾空白
            row['output'] = response  # 写回标准输出字段供父类处理
        return super().preprocess(row)  # 交由父类进行统一样本格式化


register_dataset(
    DatasetMeta(  # 注册 LongAlpaca 长序列问答数据
        ms_dataset_id='AI-ModelScope/LongAlpaca-12k',  # MS 平台标识
        hf_dataset_id='Yukang/LongAlpaca-12k',  # HF 平台标识
        preprocess_func=LongAlpacaPreprocessor(),  # 绑定对应预处理器
        tags=['long-sequence', 'QA'],  # 标签：长序列/问答
    ))  # 完成注册


class RuozhibaPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    处理“若知吧”中文论坛数据：
    - 优先使用 `title` 字段，否则回退到 `content`；
    - 若存在摘要 `abs` 且不同于标题，则拼接到标题后；
    - 通过正则去除前缀序号等噪声，仅保留主要内容；
    - 以 assistant 单轮消息形式返回。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将原始条目规整为单轮 assistant 消息。

        参数
        ----
        - row: 原始样本，包含 `title/content/abs` 等字段。

        返回
        ----
        - Optional[Dict[str, Any]]: 包含 `messages` 的标准样本；无可用标题时返回 None。

        示例
        ----
        >>> pre = RuozhibaPreprocessor()
        >>> pre.preprocess({'title': '1. 测试标题', 'content': '正文', 'abs': '摘要'})['messages'][0]['content']
        '测试标题，摘要'
        """
        title = row['title'] if row.get('title', None) is not None else row['content']  # 选取标题，否则回退内容
        abs = row['abs'] if 'abs' in row else None  # 读取摘要（可选）
        if abs and abs != title:  # 若摘要存在且不同于标题
            title = title + '，' + abs  # 将摘要拼接在标题后，丰富信息

        pattern = r'\d+[\.,\s,\、](.+)'  # 匹配以序号开头的标题并捕获后续主体
        match = re.search(pattern, title)  # 执行正则匹配
        if match:  # 命中则裁剪出主体部分
            title = match.group(1)  # 提取第一个捕获组
        if title:  # 标题非空则构造消息
            return {'messages': [{'role': 'assistant', 'content': title}]}  # 返回仅含 assistant 的单轮消息


register_dataset(
    DatasetMeta(  # 注册若知吧数据集
        ms_dataset_id='AI-ModelScope/ruozhiba',  # MS 平台数据集 ID
        subsets=['post-annual', 'title-good', 'title-norm'],  # 可用子集列表
        preprocess_func=RuozhibaPreprocessor(),  # 绑定预处理器
        tags=['pretrain', '🔥']))  # 标签：预训练/热门


class MathTrnPreprocessor(ResponsePreprocessor):
    """
    数学训练数据的轻量规整：保持 `query/response` 字段命名一致后交由父类处理。
    """

    def preprocess(self, row):
        """
        读取原始 `query/response`，重组后调用父类标准流程。

        参数
        ----
        - row: 原始样本，含 `query/response`。

        返回
        ----
        - Dict[str, Any]: 规范化后的样本。

        示例
        ----
        >>> MathTrnPreprocessor().preprocess({'query': '1+1=?', 'response': '2'})['response']
        '2'
        """
        query = row['query']  # 读取题目
        output = row['response']  # 读取答案
        row = {
            'query': query,  # 写回查询
            'response': output,  # 写回答案
        }
        return super().preprocess(row)  # 调用父类统一处理


register_dataset(
    DatasetMeta(ms_dataset_id='AI-ModelScope/math-trn-format',  # 注册数学训练格式化数据
                preprocess_func=MathTrnPreprocessor(),  # 绑定数学预处理器
                tags=['math']))  # 标签：数学


def _repair_ms_bench(messages: str) -> Optional[List[Dict[str, str]]]:
    """
    修补 MS Bench 消息：
    - 字符串输入先用 `ast.literal_eval` 安全解析为列表；
    - 删除默认 system 提示；
    - 过滤包含“MOSS/role 提示”的样本，返回 None 以跳过。

    参数
    ----
    - messages: 消息列表或其字符串表示。

    返回
    ----
    - Optional[List[Dict[str, str]]]: 修补后的消息列表；若需跳过则返回 None。

    示例
    ----
    >>> _repair_ms_bench("[{'from':'user','value':'hi'}]")
    [{'from': 'user', 'value': 'hi'}]
    """
    if isinstance(messages, str):  # 若传入字符串
        messages = ast.literal_eval(messages)  # 安全解析为 Python 字面量
    default_system = 'You are a helpful assistant.'  # 默认的 system 模板
    messages: List[Dict[str, str]]  # 类型提示，标明消息列表元素结构
    if messages[0]['from'] == 'system' and messages[0]['value'] == default_system:  # 若第一条为默认 system
        messages.pop(0)  # 移除默认 system
    # skip MOSS  # 跳过包含 MOSS 或显式角色提示的样本
    for c in messages:  # 遍历每条消息
        value = c['value'].lower()  # 小写化便于匹配
        if 'moss' in value or 'human:' in value or 'assistant:' in value or 'user:' in value:  # 出现这些标记则跳过
            return  # 返回 None 表示丢弃
    return messages  # 返回修补后的消息列表


register_dataset(
    DatasetMeta(  # 注册 MS Bench 数据集
        ms_dataset_id='iic/ms_bench',  # 数据集 ID
        preprocess_func=MessagesPreprocessor(repair_messages=_repair_ms_bench),  # 使用消息修补函数
        tags=['chat', 'general', 'multi-round', '🔥']))  # 标签：对话/通用/多轮/热门


def _repair_agent_messages(messages: List[Dict[str, str]], use_mini: bool) -> Optional[List[Dict[str, str]]]:
    """
    修补 Agent 消息：
    - 当 use_mini=True 时，检查 system 中插件名称是否多样；若不足则跳过该样本。

    参数
    ----
    - messages: 消息列表。
    - use_mini: 是否采用 mini 子集的筛选逻辑。

    返回
    ----
    - Optional[List[Dict[str, str]]]: 通过筛选的消息列表，否则返回 None。
    """
    if use_mini:  # mini 子集需要较严格的多插件检验
        pattern = r'\d\. {"plugin_name": "(.+?)"'  # 匹配插件名称的模式
        if messages[0]['from'] != 'system':  # mini 子集要求首条为 system
            return  # 不满足直接跳过
        system = messages[0]['value']  # 读取 system 内容
        find_list = re.findall(pattern, system)  # 提取插件名称列表
        if len(set(find_list)) <= 1:  # 插件数量不足 2 种
            return  # 跳过
    return messages  # 返回原消息列表（通过筛选）


register_dataset(
    DatasetMeta(  # 注册 MSAgent-Bench 数据集
        ms_dataset_id='damo/MSAgent-Bench',  # 数据集 ID
        subsets=[  # 定义两个子集：默认与 mini
            SubsetDataset(  # 默认子集：不过滤插件多样性
                preprocess_func=MessagesPreprocessor(repair_messages=partial(_repair_agent_messages, use_mini=False))),
            SubsetDataset(  # mini 子集：要求插件多样性
                name='mini',
                preprocess_func=MessagesPreprocessor(repair_messages=partial(_repair_agent_messages, use_mini=True)),
                is_weak_subset=True)
        ],
        split=['train', 'validation'],  # 可用划分
        tags=['chat', 'agent', 'multi-round']))  # 标签：对话/智能体/多轮

advertise_gen_prompt = """Task: Generating advertisements based on keywords.
Keywords: {{QUERY}}
Advertisements:"""  # 文本生成提示模板：基于关键词生成广告文案

register_dataset(
    DatasetMeta(  # 注册 AdvertiseGen 文案生成数据
        ms_dataset_id='lvjianjin/AdvertiseGen',  # MS 数据集 ID
        hf_dataset_id='shibing624/AdvertiseGen',  # HF 数据集 ID
        preprocess_func=TextGenerationPreprocessor(  # 使用通用文本生成预处理器
            prompt=advertise_gen_prompt, columns={  # 指定提示模板与字段映射
                'content': 'query',  # 源字段 content -> 统一字段 query
                'summary': 'response'  # 源字段 summary -> 统一字段 response
            }),
        tags=['text-generation', '🔥'],  # 标签：文本生成/热门
        split=['train', 'validation'],  # 划分：训练/验证
    ))  # 完成注册


class FireflyPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    适配 Firefly 数据集的预处理器：仅保留 `kind` 属于白名单集合的样本，其余样本跳过。
    最终调用父类 `ResponsePreprocessor` 做标准化处理（产出 query/response/messages 等）。
    """
    _firefly_kind_list = {  # Firefly 数据集允许的任务种类白名单
        'ProseGeneration', 'MRC', 'JinYongGeneration', 'TextCorrection', 'ClassicalChinese', 'BELLE', 'StoryGeneration',
        'Couplet', 'Cot', 'Dictionary', 'Translation', 'Program', 'SentimentAnalyze', 'OpenQA', 'AncientPoem',
        'TextMatching', 'NLI', 'Summary', 'KeywordRecognition', 'ProductDesc', 'LyricGeneration', 'Composition',
        'MusicComment', 'NER'
    }

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        过滤不在白名单中的样本，并使用父类完成标准化。

        参数
        ----
        - row: 原始样本，包含 `kind` 字段。

        返回
        ----
        - Optional[Dict[str, Any]]: 规范化样本；当 `kind` 不符合时返回 None。

        示例
        ----
        >>> FireflyPreprocessor().preprocess({'kind': 'OpenQA', 'query': '问', 'response': '答'}) is not None
        True
        """
        if row['kind'] not in FireflyPreprocessor._firefly_kind_list:  # 若样本种类不在白名单
            return  # 跳过该样本
        return super().preprocess(row)  # 交由父类进行标准化处理


register_dataset(
    DatasetMeta(  # 注册 Firefly 训练数据集
        ms_dataset_id='AI-ModelScope/firefly-train-1.1M',  # MS 平台 ID
        hf_dataset_id='YeungNLP/firefly-train-1.1M',  # HF 平台 ID
        preprocess_func=FireflyPreprocessor(),  # 使用 Firefly 预处理器
        tags=['chat', 'general'],  # 标签：对话/通用
    ))  # 完成注册

register_dataset(
    DatasetMeta(  # 注册 CLUE cmnli 自然语言推断数据集
        ms_dataset_id='modelscope/clue',  # MS 数据集 ID
        hf_dataset_id='clue',  # HF 数据集 ID
        subsets=['cmnli'],  # 仅使用 cmnli 子集
        preprocess_func=ClsGenerationPreprocessor(['neutral', 'entailment', 'contradiction'],  # 生成式分类预处理器
                                                  task='Natural Language Inference',  # 任务名
                                                  is_pair_seq=True),  # 输入为句对
        tags=['text-generation', 'classification'],  # 标签：生成/分类
        split=['train', 'validation'],  # 划分：训练/验证
    ))  # 完成注册

register_dataset(
    DatasetMeta(  # 注册京东情感分类数据集
        ms_dataset_id='DAMO_NLP/jd',  # MS 数据集 ID
        subsets=[  # 两个子集：生成式分类与纯分类
            SubsetDataset(  # 生成式分类：输出为情感标签文本
                'default',  # 子集名
                'default',  # 源子集
                preprocess_func=ClsGenerationPreprocessor(['negative', 'positive'],  # 标签空间
                                                          task='Sentiment Classification',  # 任务名
                                                          is_pair_seq=False)),  # 单句分类
            SubsetDataset(  # 纯分类：仅构造 query/label
                'cls',
                'default',
                preprocess_func=ClsPreprocessor(columns={'sentence': 'query'}),  # 列映射：sentence -> query
            ),
        ],
        tags=['text-generation', 'classification', '🔥'],  # 标签：生成/分类/热门
        split=['train', 'validation'],  # 划分：训练/验证
    ))  # 完成注册


class SyntheticText2SqlPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    将合成的 NL2SQL 数据行拼接为带有表结构与提示语的 `query`，并将推理步骤+最终 SQL 组合为 `response`，
    再交给父类进行标准化处理。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        构造 NL2SQL 的 query/response 并执行标准化。

        参数
        ----
        - row: 包含 `sql_prompt/sql_context/sql/sql_explanation` 的原始样本。

        返回
        ----
        - Dict[str, Any]: 规范化后的样本。

        示例
        ----
        >>> pre = SyntheticText2SqlPreprocessor()
        >>> rec = pre.preprocess({'sql_prompt':'Q','sql_context':'T','sql':'S','sql_explanation':'E'})
        >>> 'Sql Table information' in rec['query'] and 'final sql' in rec['response']
        True
        """
        sql_prompt = row['sql_prompt']  # NL 查询提示
        sql_context = row['sql_context']  # 表结构/上下文
        sql = row['sql']  # 最终 SQL
        sql_explanation = row['sql_explanation']  # 逐步推理说明
        query = f'Sql Table information:\n{sql_context}\n{sql_prompt}'  # 构造包含表信息与提示的查询
        response = f'Let\'s think step by step:\n{sql_explanation}\nSo the final sql is:\n{sql}'  # 组合解释与最终 SQL
        return super().preprocess({'query': query, 'response': response})  # 交给父类标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/synthetic_text_to_sql',
        hf_dataset_id='gretelai/synthetic_text_to_sql',
        preprocess_func=SyntheticText2SqlPreprocessor(),
        tags=['nl2sql', 'en']))


def _repair_toolbench(conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    修补 ToolBench 对话角色：将第二条消息的角色从 caller/conclusion 规范为 assistant。

    参数
    ----
    - conversations: 两条消息的列表。

    返回
    ----
    - List[Dict[str, str]]: 角色修正后的消息列表。
    """
    assert len(conversations) == 2  # 预期恰有两条消息
    if conversations[1]['from'] in {'caller', 'conclusion'}:  # 若第二条为工具调用者/结论
        conversations[1]['from'] = 'assistant'  # 统一改为 assistant
    return conversations  # 返回修正结果


register_dataset(
    DatasetMeta(
        ms_dataset_id='shenweizhou/alpha-umi-toolbench-processed-v2',
        subsets=['backbone', 'caller', 'planner', 'summarizer'],
        preprocess_func=MessagesPreprocessor(repair_messages=_repair_toolbench),
        tags=['chat', 'agent', '🔥'],
        huge_dataset=True))


class BlossomMathPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    将 Blossom-Math 的输出与标准答案合并到 `response` 中，保持原始 `query` 不变。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        将输出与答案拼接为响应文本。

        参数
        ----
        - row: 包含 `query/output/answer` 的记录。

        返回
        ----
        - Dict[str, Any]: 标准化样本。
        """
        output, answer = row['output'], row['answer']  # 读取模型输出与答案
        return super().preprocess({'query': row['query'], 'response': f'{output}\n\nAnswer: {answer}'})  # 拼接并标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/blossom-math-v2',
        hf_dataset_id='Azure99/blossom-math-v2',
        preprocess_func=BlossomMathPreprocessor(),
        tags=['chat', 'math', '🔥']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/sql-create-context',
        hf_dataset_id='b-mc2/sql-create-context',
        preprocess_func=AlpacaPreprocessor(columns={
            'question': 'instruction',
            'context': 'input',
            'answer': 'output'
        }),
        tags=['chat', 'sql', '🔥']))


class TigerBotLawPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    将法务场景的多字段内容拼接为一个长响应：由 `type/title/chapter1-3/response` 组成，
    适合用于生成式/摘要式任务的训练。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        依次拼接类型、标题、章节与最终响应，构造长响应文本。

        参数
        ----
        - row: 包含 `type/title/chapter1-3/response` 的记录。

        返回
        ----
        - Dict[str, Any]: 只设置 `response` 的标准化样本。
        """
        prompt = """{type}
{title}
"""  # 顶部提示模板：类型与标题两行
        cur_prompt = prompt.format(type=row['type'], title=row['title'])  # 填充类型与标题
        for i in range(1, 4):  # 遍历三个可能的章节字段
            chapter = row[f'chapter{i}']  # 读取章节内容
            if chapter is not None:  # 若章节存在
                cur_prompt += f'{chapter}'  # 追加到响应正文
        cur_prompt += f'{row["response"]}'  # 末尾追加原响应内容
        return super().preprocess({'response': cur_prompt})  # 交给父类标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/tigerbot-law-plugin',
        hf_dataset_id='TigerResearch/tigerbot-law-plugin',
        preprocess_func=TigerBotLawPreprocessor(),
        tags=['text-generation', 'law', 'pretrained']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='codefuse-ai/CodeExercise-Python-27k',
        preprocess_func=MessagesPreprocessor(columns={'chat_rounds': 'messages'}),
        tags=['chat', 'coding', '🔥']))


class LeetcodePythonPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    解析 LeetCode Python 题解样本：
    - 从 `code_with_problem` 切分出题目与代码块；
    - 去掉题目前缀 `# `；
    - 将题目作为 `query`，代码与解释合并为 `response`。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析题目与代码，拼接解释文本后标准化。

        参数
        ----
        - row: 含 `code_with_problem/explanation_only` 的记录。

        返回
        ----
        - Dict[str, Any]: 标准化样本。
        """
        code_with_problem = row['code_with_problem']  # 包含题目与代码的整体文本
        idx = code_with_problem.find('```python')  # 定位代码块起始位置
        problem = code_with_problem[:idx]  # 提取题目部分
        if problem.startswith('# '):  # 若题目前有 Markdown 注释前缀
            problem = problem[2:]  # 去掉前缀
        code = code_with_problem[idx:].strip()  # 提取代码块并去掉首尾空白
        explanation = row['explanation_only']  # 读取文字解释
        return super().preprocess({'query': problem, 'response': f'{code}\n\n{explanation}'})  # 标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/leetcode-solutions-python',
        preprocess_func=LeetcodePythonPreprocessor(),
        tags=['chat', 'coding', '🔥']))


class StsbPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    STS-B 相似度数据预处理器：
    - 将 `(sentence1, sentence2, score)` 转换为 `(query, response, label)`；
    - 支持可选阈值 `sim_threshold`，低于阈值的样本可被丢弃。
    """

    def __init__(self, sim_threshold: Optional[float] = None):
        """
        初始化相似度阈值。

        参数
        ----
        - sim_threshold: 若设置，则只保留得分不低于该阈值的样本。
        """
        self.sim_threshold = sim_threshold  # 保存阈值配置
        super().__init__()  # 调用父类构造

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建标准字段并按需过滤低相似度样本。

        参数
        ----
        - row: 包含 `sentence1/sentence2/score` 的记录。

        返回
        ----
        - Dict[str, Any] 或 None: 通过筛选的样本；被过滤则返回 None。
        """
        row = {
            'query': row['sentence1'],  # 句子 1 作为查询
            'response': row['sentence2'],  # 句子 2 作为响应
            'label': row['score'],  # 相似度分数作为标签
        }
        if self.sim_threshold is None or float(row['label']) >= self.sim_threshold:  # 未设置阈值或分数达标
            return super().preprocess(row)  # 标准化
        else:
            return None  # 过滤掉低分样本


class StsbGeneratePreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    生成式 STS-B：构造带模板的 `query`，将浮点分数格式化为一位小数的字符串作为 `response`。
    """
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 1.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        构造问句与分数字符串，并交由父类标准化。
        """
        return super().preprocess({
            'query': self.prompt.format(text1=row['sentence1'], text2=row['sentence2']),  # 模板化查询
            'response': f"{row['score']:.1f}"  # 一位小数格式的分数
        })
        return super().preprocess({})


class StsbRegressionPreprocessor(StsbGeneratePreprocessor):
    """
    类说明
    -----
    回归式 STS-B：query 与 `StsbGeneratePreprocessor` 一致，但将数值分数作为 `label`。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        复用父类模板构造 query，并输出 label。
        """
        return super(StsbGeneratePreprocessor, self).preprocess({
            'query': self.prompt.format(text1=row['sentence1'], text2=row['sentence2']),  # 模板化查询
            'label': row['score']  # 使用回归标签
        })


register_dataset(
    DatasetMeta(
        ms_dataset_id='sentence-transformers/stsb',
        hf_dataset_id='sentence-transformers/stsb',
        subsets=[
            SubsetDataset('default', preprocess_func=StsbPreprocessor()),  # embedding
            SubsetDataset('positive', preprocess_func=StsbPreprocessor(sim_threshold=0.75)),  # infonce
            SubsetDataset('generate', preprocess_func=StsbGeneratePreprocessor()),
            SubsetDataset('reg', preprocess_func=StsbRegressionPreprocessor()),
        ],
        tags=['similarity', '🔥']))


class MTEBRerankPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    将 Reranking 任务中的一个查询与多个正/负样本展开为多条样本：
    - 每个 positive 变成一条样本，其 `rejected_response` 为全部 negatives；
    - 交由父类标准化后返回一个列表。
    """

    def preprocess(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        展开正负样本对，生成多条样本。
        """
        query = row['query']  # 查询文本
        positives = row['positive'] if isinstance(row['positive'], list) else [row['positive']]  # 规范正样本为列表
        negatives = row['negative'] if isinstance(row['negative'], list) else [row['negative']]  # 规范负样本为列表

        expanded_rows = []  # 存放展开后的样本
        for positive in positives:  # 遍历每个正样本
            expanded_row = {'query': query, 'response': positive, 'rejected_response': negatives}  # 构造一条样本
            expanded_rows.append(super().preprocess(expanded_row))  # 标准化并加入结果

        return expanded_rows  # 返回展开后的样本列表


register_dataset(
    DatasetMeta(
        ms_dataset_id='MTEB/scidocs-reranking',
        hf_dataset_id='mteb/scidocs-reranking',
        split=['validation', 'test'],
        preprocess_func=MTEBRerankPreprocessor(),
        tags=['rerank', '🔥']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='MTEB/stackoverflowdupquestions-reranking',
        hf_dataset_id='mteb/stackoverflowdupquestions-reranking',
        split=['train', 'test'],
        preprocess_func=MTEBRerankPreprocessor(),
        tags=['rerank', '🔥']))


def _repair_conversations_agent_instruct(s: str) -> List[Dict[str, Any]]:
    """
    修补 AgentInstruct 风格的会话字符串：
    - 统一在 `}{` 之间插入逗号，便于安全解析为列表；
    - 使用 `ast.literal_eval` 将字符串转换为 Python 对象。
    """
    s = s.replace('}\n {', '},\n {')  # 在分隔处插入逗号
    if isinstance(s, str):  # 若仍为字符串
        s = ast.literal_eval(s)  # 安全解析
    return s  # 返回解析后的对象


register_dataset(
    DatasetMeta(
        ms_dataset_id='huangjintao/AgentInstruct_copy',
        subsets=['alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'],
        preprocess_func=MessagesPreprocessor(repair_messages=_repair_conversations_agent_instruct),
        tags=['chat', 'agent', 'multi-round']))


class MultiRoleAgentPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    将多角色对话样本规整为标准三段式消息：system（规则+历史）、user（最后一轮用户输入）、
    assistant（最后一轮回复）。当无法抽取 user/assistant 时返回 None。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将多角色对话折叠为 system/user/assistant 三段结构。
        """
        conv = row['conversations']  # 原始多轮对话列表
        res_prompt = '\n\n【注意事项】\n1. 这是聊天室，不要发送私信给任何人\n2. 仅代表你个人说话,不要扮演其他人，只根据对话历史进行回复\n3. 长话短说，不要说太多话，不要超过50字 '  # 规则提示
        history_prompt = '\n\n【chat history】'  # 历史标题
        conv_prompt = '\n {name}:{content}'  # 历史项模板
        query, response = '', conv[-1]['value']  # 初始化当前用户提问与助手回复
        system = conv[0]['value'] if conv[0]['from'] == 'system' else ''  # 读取或置空 system
        if conv[0]['from'] == 'user':  # 首条即为用户提问
            query = conv[0]['value']  # 直接作为 query
        elif 'next_speakers:' not in system:  # 非用户开头且 system 中不含下一说话人提示
            if '【注意事项】' not in system and system:  # 若已有 system 但未包含注意事项
                system += res_prompt  # 追加注意事项
            system += history_prompt  # 追加历史标题
            system += ''.join([conv_prompt.format(name=c['from'], content=c['value']) for c in conv[1:-1]])  # 拼接历史摘要

        if not query or not response:  # 若缺少必要字段
            return  # 返回 None 跳过

        return {  # 返回标准三段式消息
            'messages': [{
                'role': 'system',
                'content': system
            }, {
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='iic/MSAgent-MultiRole',
        preprocess_func=MultiRoleAgentPreprocessor(),
        tags=['chat', 'agent', 'multi-round', 'role-play', 'multi-agent']))

register_dataset(DatasetMeta(ms_dataset_id='swift/ToolBench', tags=['chat', 'agent', 'multi-round']))  # 注册 ToolBench 数据占位元信息

register_dataset(
    DatasetMeta(
        ms_dataset_id='tastelikefeet/competition_math',
        subsets=[
            SubsetDataset(
                name='default',
                subset='default',
                split=['train', 'test'],
            ),
        ],
        tags=['qa', 'math']))

register_dataset(DatasetMeta(ms_dataset_id='modelscope/gsm8k', subsets=['main'], split=['train'], tags=['qa', 'math']))  # 注册 GSM8K 主子集

register_dataset(
    DatasetMeta(ms_dataset_id='modelscope/MathR', subsets=['default', 'clean'], split=['train'], tags=['qa', 'math']))

register_dataset(
    DatasetMeta(ms_dataset_id='modelscope/MathR-32B-Distill', subsets=['data'], split=['train'], tags=['qa', 'math']))


class CoundownTaskPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    倒计时算术任务数据预处理器：
    - 基于给定数字与目标值构造标准化 `query`；
    - 要求模型在 <think>/<answer> 标签中展示过程与答案；
    - 将 `target` 留存在样本中以供下游评估。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        构造带思维过程与答案标签的查询，并标准化样本。
        """
        numbers = row['nums']  # 可用数字列表
        target = row.pop('response', None)  # 目标值保存在 response 字段中，取出后移至 target
        query = (f'Using the numbers {numbers}, create an equation that equals {target}.\n'
                 'You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n'
                 'Show your work in <think> </think> tags. And return the final equation and answer '
                 'in <answer> </answer> tags, for example <answer> (1 + 2) / 3 * 4 = 4 </answer>.')  # 构造查询提示
        row.update({'target': target, 'query': query})  # 写回目标与查询
        return super().preprocess(row)  # 标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='zouxuhong/Countdown-Tasks-3to4',
        subsets=['default'],
        preprocess_func=CoundownTaskPreprocessor(),
        tags=['math']))


class HC3Preprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    HC3 生成式分类：
    - 对每条样本生成两条记录（Human 与 ChatGPT 各一）；
    - query 为模板化问题与一个候选回答，response 为对应类别名。
    """
    prompt = """Classification Task: Are the following responses from a human or from ChatGPT?
Question: {question}
Answer: {answer}
Category: Human, ChatGPT
Output:"""

    def preprocess(self, row):
        """
        生成两条样本用于分类训练。
        """
        rows = []  # 保存展开后的样本
        for response in ['Human', 'ChatGPT']:  # 遍历两个类别
            query = self.prompt.format(
                question=row['query'], answer=self.random_state.choice(row[f'{response.lower()}_answers']))  # 随机抽取一个该类答案
            rows.append(super().preprocess({'query': query, 'response': response}))  # 标注类别名为响应
        return rows  # 返回样本列表


class HC3ClsPreprocessor(HC3Preprocessor):
    """
    类说明
    -----
    HC3 纯分类：与 `HC3Preprocessor` 相似，但将类别以 `label` 数字形式给出（Human=0, ChatGPT=1）。
    """

    def preprocess(self, row):
        """
        生成两条样本并输出数值标签。
        """
        rows = []  # 保存样本
        for i, response in enumerate(['Human', 'ChatGPT']):  # Human->0, ChatGPT->1
            query = self.prompt.format(
                question=row['query'], answer=self.random_state.choice(row[f'{response.lower()}_answers']))  # 随机候选
            rows.append(ResponsePreprocessor.preprocess(self, {'query': query, 'label': i}))  # 标准化
        return rows  # 返回


hc3_subset_names = ['baike', 'open_qa', 'nlpcc_dbqa', 'finance', 'medicine', 'law', 'psychology']
hc3_subsets: List[SubsetDataset] = []
for hc3_subset_name in hc3_subset_names:
    hc3_subsets.append(
        SubsetDataset(
            name=hc3_subset_name,
            subset=hc3_subset_name,
            preprocess_func=HC3Preprocessor(),
        ))
    hc3_subsets.append(
        SubsetDataset(
            name=f'{hc3_subset_name}_cls',
            subset=hc3_subset_name,
            preprocess_func=HC3ClsPreprocessor(),
        ))

register_dataset(
    DatasetMeta(  # 注册 HC3 中文数据集
        ms_dataset_id='simpleai/HC3-Chinese',  # MS ID
        hf_dataset_id='Hello-SimpleAI/HC3-Chinese',  # HF ID
        subsets=hc3_subsets,  # 使用上文构造的子集列表
        tags=['text-generation', 'classification', '🔥']))  # 标签：生成/分类/热门

hc3_subset_names = ['finance', 'medicine']
hc3_subsets: List[SubsetDataset] = []
for hc3_subset_name in hc3_subset_names:
    hc3_subsets.append(
        SubsetDataset(
            name=hc3_subset_name,
            subset=hc3_subset_name,
            preprocess_func=HC3Preprocessor(),
        ))
    hc3_subsets.append(
        SubsetDataset(
            name=f'{hc3_subset_name}_cls',
            subset=hc3_subset_name,
            preprocess_func=HC3ClsPreprocessor(),
        ))

register_dataset(
    DatasetMeta(  # 注册 HC3 英文数据集
        ms_dataset_id='simpleai/HC3',  # MS ID
        hf_dataset_id='Hello-SimpleAI/HC3',  # HF ID
        subsets=hc3_subsets,  # 子集沿用
        preprocess_func=HC3Preprocessor(),  # 绑定生成式分类预处理器
        tags=['text-generation', 'classification', '🔥']))  # 标签


class DureaderPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    将 DuReader QG（问题生成）样本转换为两段式消息：
    - user: 基于 `context/answer` 构造的提问提示；
    - assistant: 对应的目标问题 `text2`。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 `text1` 中剥离答案与上下文，构造问题生成提示，并与 `text2` 组成问答对。
        """
        prompt = """Task: Question Generation
Context: {context}
Answer: {answer}
Question:"""  # 问题生成任务提示模板
        answer, context = row['text1'].split('[SEP]')  # 将 text1 按分隔符拆成答案与上下文
        return {
            'messages': [{
                'role': 'user',
                'content': prompt.format(context=context, answer=answer)  # 用户给出上下文与答案，请求生成问题
            }, {
                'role': 'assistant',
                'content': row['text2']  # 目标问题文本
            }]
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='modelscope/DuReader_robust-QG',
        preprocess_func=DureaderPreprocessor(),
        split=['train', 'validation', 'test'],
        tags=['text-generation', '🔥']))


class HHRLHFPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    处理 HH-RLHF 数据：
    - 将 `chosen/rejected` 中的对话片段按 `Human/Assistant` 标记拆分为轮次；
    - 产出 `messages` 与 `rejected_messages`，用于偏好建模（DPO/ORPO 等）。
    """

    @staticmethod
    def _to_messages(data):
        """
        将交替的用户/助手文本数组打包为消息列表。
        """
        messages = []  # 收集消息
        for query, response in zip(data[::2], data[1::2]):  # 以步长 2 成对遍历
            messages.append({'role': 'user', 'content': query})  # 用户消息
            messages.append({'role': 'assistant', 'content': response})  # 助手消息
        return messages  # 返回消息序列

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        按分隔符拆分 chosen/rejected，对齐首项并转为消息序列。
        """
        chosen = row['chosen'].strip()  # 选中答复文本
        rejected = row['rejected'].strip()  # 被拒绝答复文本
        parts_chosen = [s.strip() for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', chosen)]  # 拆分轮次
        parts_rejected = [s.strip() for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', rejected)]  # 拆分轮次
        if parts_chosen[0].startswith('Human:'):  # 若首项仍带有前缀
            assert parts_rejected[0].startswith('Human:')  # 两者应对齐
            parts_chosen[0] = parts_chosen[0][6:].strip()  # 去掉 'Human:'
            parts_rejected[0] = parts_rejected[0][6:].strip()  # 去掉 'Human:'
        row['messages'] = self._to_messages(parts_chosen)  # 构造正样本消息
        row['rejected_messages'] = self._to_messages(parts_rejected)  # 构造负样本消息
        return row  # 返回包含两套消息的记录


# TODO meta file broken
register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/hh-rlhf',
        subsets=['helpful-base', 'helpful-online', 'helpful-rejection-sampled'],
        preprocess_func=HHRLHFPreprocessor(),
        split=['train', 'test'],
        tags=['rlhf', 'dpo'],
        huge_dataset=True))


class XlamFunctionCallingPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    解析函数调用风格样本：
    - `query` 作为用户消息；
    - 将 `answers` 解析为 JSON 列表，逐条以 `tool_call` 角色追加；
    - 携带 `tools` 描述。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        构造用户消息并将解析后的工具调用作为 `tool_call` 追加。
        """
        messages = [{'role': 'user', 'content': row['query']}]  # 用户消息
        response = row['answers']  # 工具调用的 JSON 字符串
        response = json.loads(response)  # 解析为列表
        messages += [{'role': 'tool_call', 'content': json.dumps(content)} for content in response]  # 逐条加入消息
        return {'messages': messages, 'tools': row['tools']}  # 返回消息与工具列表


class XlamFunctionCallingGRPOPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    为 GRPO 训练准备数据：
    - 随机选择一个工具调用答案，格式化为 `Action/Action Input` 结构；
    - 同时保留 `solution` 与 `tools` 字段。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        随机采样一个函数调用答案，格式化为响应文本并标准化。
        """
        query = row['query']  # 用户查询
        answers = row['response']  # 候选工具调用答案（JSON 字符串或列表）
        if isinstance(answers, str):  # 若为字符串
            answers = json.loads(answers)  # 解析为列表
        answer = np.random.choice(answers)  # 随机选取一个答案
        name = answer['name']  # 工具名
        args = json.dumps(answer['arguments'])  # 参数序列化
        response = f'Action: {name}\nAction Input: {args}'  # 格式化响应
        row = {'query': query, 'response': response, 'solution': response, 'tools': row['tools']}  # 组装记录
        return super().preprocess(row)  # 标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='LLM-Research/xlam-function-calling-60k',
        hf_dataset_id='Salesforce/xlam-function-calling-60k',
        subsets=[
            SubsetDataset('default', 'dataset', preprocess_func=XlamFunctionCallingPreprocessor()),
            SubsetDataset('grpo', 'dataset', preprocess_func=XlamFunctionCallingGRPOPreprocessor())
        ],
        tags=['agent', 'grpo', '🔥']))


class HHRLHFCNPreprocessor(MessagesPreprocessor):
    """
    类说明
    -----
    处理中文 HH-RLHF 变体：
    - 将 `chosen` 追加到 `messages` 末尾；
    - 将 `rejected.text` 作为 `rejected_response`；
    - 交由父类进行字段映射与标准化。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        调整字段后调用父类预处理。
        """
        row['messages'].append(row.pop('chosen'))  # 将 chosen 附加到消息末尾
        row['rejected_response'] = row['rejected']['text']  # 提取被拒绝文本
        return super().preprocess(row)  # 标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/hh_rlhf_cn',
        subsets=['hh_rlhf', 'harmless_base_cn', 'harmless_base_en', 'helpful_base_cn', 'helpful_base_en'],
        preprocess_func=HHRLHFCNPreprocessor(columns={'context': 'messages'}, content_key='text'),
        split=['train', 'test'],
        tags=['rlhf', 'dpo', '🔥']))


def repair_conversations(s: Union[str, Any]) -> Any:
    """
    通用对话修补：将行间缺逗号的 JSON 片段规范化后解析为 Python 对象。

    参数
    ----
    - s: 原始字符串或已解析对象。

    返回
    ----
    - Any: 解析后的对象或原对象。
    """
    if isinstance(s, str):  # 仅处理字符串输入
        s = s.replace('}\n {', '},{')  # 各类缺逗号场景修补
        s = s.replace('}\n{', '},{')
        s = s.replace('}{', '},{')
        s = s.replace('}\n  {', '},{')
        return ast.literal_eval(s)  # 安全解析为 Python 对象
    return s  # 非字符串，直接返回


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/lmsys-chat-1m',
        hf_dataset_id='lmsys/lmsys-chat-1m',
        preprocess_func=MessagesPreprocessor(repair_messages=repair_conversations),
        tags=['chat', 'em']))


class EmojiPreprocessr(ResponsePreprocessor):
    """
    类说明
    -----
    清洗样本中常见的不可见 emoji 变体选择器字符（如 '️'），以降低训练时的噪声。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        去除 query/response/rejected_response 中的不可见字符后标准化。
        """
        # Remove dirty characters  # 清除不可见字符
        row['query'] = row['query'].replace('️', '')  # 清理 query
        row['response'] = row['response'].replace('️', '')  # 清理 response
        row['rejected_response'] = row['rejected_response'].replace('️', '')  # 清理 rejected_response
        return super().preprocess(row)  # 标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='hjh0119/shareAI-Llama3-DPO-zh-en-emoji',
        hf_dataset_id='shareAI/DPO-zh-en-emoji',
        preprocess_func=EmojiPreprocessr(columns={
            'answer_zh': 'response',
            'answer_en': 'rejected_response'
        }),
        tags=['rlhf', 'dpo']))

register_dataset(
    DatasetMeta(ms_dataset_id='AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto', tags=['rlhf', 'kto']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='OmniData/Zhihu-KOL-More-Than-100-Upvotes',
        hf_dataset_id='bzb2023/Zhihu-KOL-More-Than-100-Upvotes',
        tags=['zhihu', 'qa']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='OmniData/Zhihu-KOL',
        hf_dataset_id='wangrui6/Zhihu-KOL',
        huge_dataset=True,
        tags=['zhihu', 'qa'],
    ))


class GuanacoPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    解析 Guanaco 数据集：
    - 从 `instruction` 中按多种键名分割出历史轮次（User/Assistant 混杂大小写/中英文变体）；
    - 清洗 `input` 的 `User:` 前缀；
    - 构造多轮 `messages` 并在末尾追加当前 `input/output`。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将 instruction/input/output 转为多轮 messages。
        """
        instruction = row['instruction']  # 指令文本，其中可能包含历史多轮对话
        input = row['input']  # 当前用户输入
        output = row['output']  # 当前助手回复
        history = []  # 暂存历史轮次 [user, assistant]
        if instruction:  # 存在历史则解析
            parts = split_str_parts_by(
                instruction, ['User:', 'User：', 'Assistant：', 'Assistant:', 'Asssistent:', 'Assistent:', 'Assistenz:'])  # 按多语言键切分
            for idx, part in enumerate(parts):  # 枚举分段
                if idx % 2 == 0:  # 偶数段应为 user
                    if 'user' not in part['key'].lower():  # 防御性检查
                        return  # 结构异常，跳过
                    history.append([part['content'], None])  # 暂存 user 内容
                else:  # 奇数段应为 assistant
                    if 'assist' not in part['key'].lower() and 'asssist' not in part['key'].lower():  # 各种拼写
                        return  # 结构异常，跳过
                    history[-1][-1] = part['content']  # 填充 assistant 内容
        if input.startswith('User:'):  # 清理当前输入前缀
            input = input[len('User:'):].strip()  # 去除 'User:'
        if any([not h[0] or not h[1] for h in history]):  # 历史中若存在空轮次
            return  # 跳过该样本

        messages = []  # 构造标准消息序列
        for h in history:  # 逐轮添加历史
            messages.append({'role': 'user', 'content': h[0]})  # 用户发言
            messages.append({'role': 'assistant', 'content': h[1]})  # 助手回复
        messages.append({'role': 'user', 'content': input})  # 当前用户输入
        messages.append({'role': 'assistant', 'content': output})  # 当前助手回复
        return {
            'messages': messages,  # 返回统一格式
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/GuanacoDataset',
        hf_dataset_id='JosephusCheung/GuanacoDataset',
        preprocess_func=GuanacoPreprocessor(),
        tags=['chat', 'zh']))


class FunctionCallChatmlPreprocessor(MessagesPreprocessor):
    """
    类说明
    -----
    处理 Function-Calling ChatML 数据：
    - 若存在 `function_description`，拆分为 `tools`；
    - 若消息首条为 system，则移除；
    - 其余交由父类处理列映射与标准化。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        在父类标准化基础上，补齐 tools 并移除首条 system。
        """
        res = super().preprocess(row)  # 标准化字段

        if res['function_description']:  # 若存在函数描述
            res['tools'] = res['function_description'].split('\n\n')  # 拆分为工具列表
        messages = res['messages']  # 取出消息
        if messages[0]['role'] == 'system':  # 若首条为 system
            messages.pop(0)  # 移除之
        return res  # 返回处理结果


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/function-calling-chatml',
        hf_dataset_id='Locutusque/function-calling-chatml',
        preprocess_func=FunctionCallChatmlPreprocessor(),
        tags=['agent', 'en', 'sft', '🔥']))


class Dolly15kPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    适配 Databricks Dolly 15k 数据：
    - 将 `context`（可选）与 `instruction` 拼接为用户侧 `query`；
    - 将 `response` 用作助手回复；
    - 输出标准化的两段式消息。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        构造 query 并返回标准消息结构。

        参数
        ----
        - row: 输入记录，包含 `instruction/context/response`。

        返回
        ----
        - Optional[Dict[str, Any]]: 标准化样本。
        """
        instruction = row['instruction']  # 指令文本
        context = row['context']  # 上下文信息（可选）
        response = row['response']  # 参考答案/回复
        query = ''  # 初始化 query
        if context:  # 若存在上下文
            query = 'Here gives some useful information:\n'  # 前缀说明
            query += context  # 追加上下文
            query += '\n'  # 换行分隔
        query += instruction  # 最后追加指令
        return {
            'messages': [{
                'role': 'user',
                'content': query  # 用户侧合成的查询
            }, {
                'role': 'assistant',
                'content': response  # 助手侧目标回复
            }],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/databricks-dolly-15k',
        hf_dataset_id='databricks/databricks-dolly-15k',
        preprocess_func=Dolly15kPreprocessor(),
        tags=['multi-task', 'en', 'quality']))


class OrpoDPOMix40kPreprocessor(MessagesPreprocessor):
    """
    类说明
    -----
    针对 ORPO/DPO 混合数据：过滤来源为 `toxic-dpo-v0.2` 的样本，其余按父类逻辑标准化。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        过滤部分来源并委托父类处理。
        """
        if row['source'] == 'toxic-dpo-v0.2':  # 命中需过滤的数据来源
            return  # 丢弃该样本
        return super().preprocess(row)  # 其他样本标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/orpo-dpo-mix-40k',
        hf_dataset_id='mlabonne/orpo-dpo-mix-40k',
        preprocess_func=OrpoDPOMix40kPreprocessor(columns={
            'chosen': 'messages',
            'rejected': 'rejected_messages'
        }),
        tags=['dpo', 'orpo', 'en', 'quality']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/sharegpt',
        subsets=['common-zh', 'unknow-zh', 'common-en'],
        tags=['chat', 'general', 'multi-round']))


class SelfCognitionPreprocessor(ResponsePreprocessor):

    def __init__(self, *args, query_suffix: str = '', response_prefix: str = '', **kwargs):
        self.query_suffix = query_suffix
        self.response_prefix = response_prefix
        self.name: Optional[Tuple[str, str]] = None
        self.author: Optional[Tuple[str, str]] = None
        super().__init__(*args, **kwargs)

    def set_name_author(self, name, author):
        self.name = name
        self.author = author

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        for key in ['name', 'author']:
            val = getattr(self, key)
            if val is None:
                continue
            val = val[0] if row['tag'] == 'zh' else val[1]
            if val is None:
                continue
            placeholder = '{{' + key.upper() + '}}'
            row['query'] = row['query'].replace(placeholder, val)
            row['response'] = row['response'].replace(placeholder, val)

        row['query'] = row['query'] + self.query_suffix
        row['response'] = self.response_prefix + row['response']
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/self-cognition',
        hf_dataset_id='modelscope/self-cognition',
        subsets=[
            SubsetDataset(preprocess_func=SelfCognitionPreprocessor()),
            SubsetDataset(
                'qwen3',
                preprocess_func=SelfCognitionPreprocessor(
                    query_suffix=' /no_think', response_prefix='<think>\n\n</think>\n\n')),
            SubsetDataset(
                'empty_think', preprocess_func=SelfCognitionPreprocessor(response_prefix='<think>\n\n</think>\n\n')),
        ],
        dataset_name='self-cognition',
        tags=['chat', 'self-cognition', '🔥']))
