"""
模块功能
-------
提供若干附加预处理组件：
- GroundingMixin：为 grounding/caption 任务生成提示模版（中英双语），用于构造 query/response；
- TextGenerationPreprocessor：基于给定 prompt 模版，将 query 替换成模板化文本；
- ClsGenerationPreprocessor：将分类任务转化为生成式形式（输出类别名称），支持单句/句对模式。

典型用法
-------
>>> mixin = GroundingMixin(); mixin.task_type = 'grounding'; mixin.construct_grounding_prompt()
>>> pre = TextGenerationPreprocessor(prompt='Q: {{QUERY}}\nA: ')
>>> row = pre.preprocess({'query': 'hello', 'response': 'world'})
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明
from typing import Any, Dict, List, Optional  # 类型注解工具

import numpy as np  # 随机与数组工具

from .core import ResponsePreprocessor  # 引入响应式预处理基类


class GroundingMixin:
    """
    提供 grounding/caption 任务的提示模版（prompt）混入类：
    - 通过 `task_type` 指定任务（'grounding' 或 'caption'）；
    - `construct_grounding_prompt` 按语言概率随机生成 (query, response) 对。
    """
    task_type: Optional[str] = None  # 当前任务类型，影响模版选择

    _grounding_language_mixin = [0.8, 0.2]  # 语言选择概率：英文 80%，中文 20%
    _grounding_prompts = {  # grounding/caption 两类任务的中英提示模版库
        'grounding': {
            'en': [('<ref-object>', '<bbox>'), ('The positions of <ref-object> is', '<bbox>'),  # 位置询问
                   ('Find the positions of <ref-object>', '<bbox>'), ('Where is <ref-object>', '<bbox>'),
                   ('Find <ref-object>', '<bbox>'), ('Show me <ref-object>', '<bbox>'),
                   ('Detect <ref-object>', '<bbox>'), ('Locate <ref-object>', '<bbox>'),
                   ('Tell me the location of <ref-object>', '<bbox>'), ('Give the location of <ref-object>', '<bbox>'),
                   ('Provide the bounding box coordinate of <ref-object>', '<bbox>')],  # 直接请求坐标
            'zh': [('<ref-object>', '<bbox>'), ('<ref-object>的位置在图片中', '<bbox>'), ('<ref-object>在图片中', '<bbox>'),  # 中文版本
                   ('<ref-object>在', '<bbox>'), ('找到<ref-object>的位置', '<bbox>'), ('<ref-object>在哪里', '<bbox>'),
                   ('提供<ref-object>的坐标位置', '<bbox>')]
        },
        'caption': {
            'en': [
                ('<bbox>', '<ref-object>'),  # 描述指定区域
                ('The object at position <bbox>', '<ref-object>'),  # 询问区域内物体
                ('This <bbox> is', '<ref-object>'),  # 句式补全
                ('What is the object at <bbox>', '<ref-object>'),  # 询问
                ('Describe <bbox>', '<ref-object>'),  # 描述区域
                ('<bbox> is', '<ref-object>'),  # 句式补全
                ('The bounding box coordinate <bbox> contains', '<ref-object>'),  # 包含关系
            ],
            'zh': [
                ('<bbox>', '<ref-object>'),  # 直接描述区域
                ('<bbox>是什么', '<ref-object>'),  # 询问区域内对象
                ('<bbox>的位置包含', '<ref-object>'),  # 包含关系
                ('描述<bbox>', '<ref-object>'),  # 描述区域（注意示例含尖括号）
                ('<bbox>中是', '<ref-object>'),  # 句式补全
                ('坐标<bbox>描述了什么', '<ref-object>'),  # 询问
                ('描述<bbox>中的事物', '<ref-object>'),  # 描述
            ]
        },
    }

    def construct_grounding_prompt(self):
        """
        随机生成 grounding/caption 任务的 (query, response) 模版对。

        返回
        ----
        - Tuple[str, str]: (query 模版, response 模版)，例如 ('Find <ref-object>', '<bbox>')

        示例
        ----
        >>> mixin = GroundingMixin(); mixin.task_type = 'caption'
        >>> q, r = mixin.construct_grounding_prompt()
        >>> isinstance(q, str) and isinstance(r, str)
        True
        """
        # TODO Only support one bbox to one object  # 目前仅支持 1:1 对应
        lang = np.random.choice(['en', 'zh'], p=[0.8, 0.2])  # 按概率选择语言
        prompts = GroundingMixin._grounding_prompts[self.task_type][lang]  # 取对应任务与语言的模版列表
        query, response = prompts[np.random.choice(range(len(prompts)))]  # 随机抽取一组 (query, response)
        return query, response  # 返回模版对


class TextGenerationPreprocessor(ResponsePreprocessor):
    """
    文本生成预处理器：使用给定 `prompt` 模版，将原始 `query` 替换到 `{{QUERY}}` 占位符处，
    之后交由父类生成标准 messages。
    """

    def __init__(self,
                 *,
                 prompt: str,
                 query_tag: str = '{{QUERY}}',
                 columns: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        self.query_tag = query_tag  # 占位符标记（默认 {{QUERY}}）
        self.prompt = prompt  # 文本生成模板
        super().__init__(columns=columns, **kwargs)  # 初始化父类（列对齐等）

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        用模板替换占位符生成新的 `query`，再交由父类进行标准化。

        参数
        ----
        - row: 输入字典，包含原始 `query` 与 `response` 等。

        返回
        ----
        - Dict[str, Any]: 标准化后的样本。
        """
        row['query'] = self.prompt.replace(self.query_tag, row['query'])  # 替换占位符
        return super().preprocess(row)  # 父类生成 messages


class ClsGenerationPreprocessor(ResponsePreprocessor):
    """
    生成式分类预处理器：将分类任务格式化为指令式生成（输出类别名称），
    支持单句 `sentence` 与句对 `sentence1/sentence2` 两种输入形式。
    """

    def __init__(self,
                 labels: List[str],
                 *,
                 task: str,
                 is_pair_seq: bool = False,
                 columns: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        self.labels = labels  # 类别名称列表（按索引编码）
        self.task = task  # 任务名称（展示在模板中）
        self.is_pair_seq = is_pair_seq  # 是否为句对输入

        category = ', '.join(labels)  # 类别名称拼接成字符串
        self.sentence2_key = 'sentence2'  # 句对中的第二句键名
        self.label_key = 'label'  # 标签键名
        if is_pair_seq:  # 句对模式
            self.sentence_key = 'sentence1'  # 第一句键名
            inputs = 'Sentence1: {sentence1}\nSentence2: {sentence2}'  # 模板输入部分
        else:  # 单句模式
            self.sentence_key = 'sentence'  # 单句键名
            inputs = 'Sentence: {sentence}'  # 模板输入部分
        self.prompt = f"""Task: {task}
{inputs}
Category: {category}
Output:"""  # 最终模板文本
        super().__init__(columns=columns, **kwargs)  # 初始化父类

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        读取标签并将输入格式化到模板中，设置 `query/response` 后生成标准 messages。

        返回
        ----
        - Optional[Dict[str, Any]]: 若缺失 label 返回 None，否则返回标准化样本。
        """
        label = row.pop(self.label_key, None)  # 取出标签索引
        if label is None:  # 无标签样本跳过
            return

        if self.is_pair_seq:  # 句对模式：填充 sentence1/2
            query = self.prompt.format(sentence1=row.pop(self.sentence_key), sentence2=row.pop(self.sentence2_key))
        else:  # 单句模式：填充 sentence
            query = self.prompt.format(sentence=row.pop(self.sentence_key))
        row['query'] = query  # 写入模板化查询
        row['response'] = self.labels[int(label)]  # 将标签索引转为类别名称
        return super().preprocess(row)  # 父类生成 messages
