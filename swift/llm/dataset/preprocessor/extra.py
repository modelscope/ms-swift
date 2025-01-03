# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional

import numpy as np

from .core import ResponsePreprocessor


class GroundingMixin:
    """This class offers prompts to the grounding task"""
    task_type: Optional[str] = None

    _grounding_language_mixin = [0.8, 0.2]
    _grounding_prompts = {
        'grounding': {
            'en': [('<ref-object>', '<bbox>'), ('The positions of <ref-object> is', '<bbox>'),
                   ('Find the positions of <ref-object>', '<bbox>'), ('Where is <ref-object>', '<bbox>'),
                   ('Find <ref-object>', '<bbox>'), ('Show me <ref-object>', '<bbox>'),
                   ('Detect <ref-object>', '<bbox>'), ('Locate <ref-object>', '<bbox>'),
                   ('Tell me the location of <ref-object>', '<bbox>'), ('Give the location of <ref-object>', '<bbox>'),
                   ('Provide the bounding box coordinate of <ref-object>', '<bbox>')],
            'zh': [('<ref-object>', '<bbox>'), ('<ref-object>的位置在图片中', '<bbox>'), ('<ref-object>在图片中', '<bbox>'),
                   ('<ref-object>在', '<bbox>'), ('找到<ref-object>的位置', '<bbox>'), ('<ref-object>在哪里', '<bbox>'),
                   ('提供<ref-object>的坐标位置', '<bbox>')]
        },
        'caption': {
            'en': [
                ('<bbox>', '<ref-object>'),
                ('The object at position <bbox>', '<ref-object>'),
                ('This <bbox> is', '<ref-object>'),
                ('What is the object at <bbox>', '<ref-object>'),
                ('Describe <bbox>', '<ref-object>'),
                ('<bbox> is', '<ref-object>'),
                ('The bounding box coordinate <bbox> contains', '<ref-object>'),
            ],
            'zh': [
                ('<bbox>', '<ref-object>'),
                ('<bbox>是什么', '<ref-object>'),
                ('<bbox>的位置包含', '<ref-object>'),
                ('描述<bbox>', '<ref-object>'),
                ('<bbox>中是', '<ref-object>'),
                ('坐标<bbox>描述了什么', '<ref-object>'),
                ('描述<bbox>中的事物', '<ref-object>'),
            ]
        },
    }

    def construct_grounding_prompt(self):
        # TODO Only support one bbox to one object
        lang = np.random.choice(['en', 'zh'], p=[0.8, 0.2])
        prompts = GroundingMixin._grounding_prompts[self.task_type][lang]
        query, response = prompts[np.random.choice(range(len(prompts)))]
        return query, response


class TextGenerationPreprocessor(ResponsePreprocessor):

    def __init__(self,
                 *,
                 prompt: str,
                 query_tag: str = '{{QUERY}}',
                 columns_mapping: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        self.query_tag = query_tag
        self.prompt = prompt
        super().__init__(columns_mapping=columns_mapping, **kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = self.prompt.replace(self.query_tag, row['query'])
        return super().preprocess(row)


class ClsGenerationPreprocessor(ResponsePreprocessor):

    def __init__(self,
                 labels: List[str],
                 *,
                 task: str,
                 is_pair_seq: bool = False,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        self.labels = labels
        self.task = task
        self.is_pair_seq = is_pair_seq

        category = ', '.join(labels)
        self.sentence2_key = 'sentence2'
        self.label_key = 'label'
        if is_pair_seq:
            self.sentence_key = 'sentence1'
            inputs = 'Sentence1: {sentence1}\nSentence2: {sentence2}'
        else:
            self.sentence_key = 'sentence'
            inputs = 'Sentence: {sentence}'
        self.prompt = f"""Task: {task}
{inputs}
Category: {category}
Output:"""
        super().__init__(columns_mapping=columns_mapping, **kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        label = row.pop(self.label_key, None)
        if label is None:
            return

        if self.is_pair_seq:
            query = self.prompt.format(sentence1=row.pop(self.sentence_key), sentence2=row.pop(self.sentence2_key))
        else:
            query = self.prompt.format(sentence=row.pop(self.sentence_key))
        row['query'] = query
        row['response'] = self.labels[int(label)]
        return super().preprocess(row)
