# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from typing import Any, Dict, Optional

import torch

from swift.llm import (DatasetMeta, InferRequest, Model, ModelGroup, ModelMeta, PtEngine, RequestConfig,
                       ResponsePreprocessor, TemplateMeta, get_model_tokenizer_with_flash_attn, load_dataset,
                       register_dataset, register_model, register_template)


class CustomPreprocessor(ResponsePreprocessor):
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super().preprocess({
            'query': self.prompt.format(text1=row['text1'], text2=row['text2']),
            'response': f"{row['label']:.1f}"
        })


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/stsb',
        hf_dataset_id='SetFit/stsb',
        preprocess_func=CustomPreprocessor(),
    ))

register_template(
    TemplateMeta(
        template_type='custom',
        prefix=['<extra_id_0>System\n{{SYSTEM}}\n'],
        prompt=['<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n'],
        chat_sep=['\n']))

register_model(
    ModelMeta(
        model_type='custom',
        model_groups=[
            ModelGroup([Model('AI-ModelScope/Nemotron-Mini-4B-Instruct', 'nvidia/Nemotron-Mini-4B-Instruct')])
        ],
        template='custom',
        get_function=get_model_tokenizer_with_flash_attn,
        ignore_patterns=['nemo']))


class TestCustom(unittest.TestCase):

    def test_custom_model(self):
        infer_request = InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}])
        request_config = RequestConfig(max_tokens=512, temperature=0)
        engine = PtEngine('AI-ModelScope/Nemotron-Mini-4B-Instruct', torch.float16)
        response = engine.infer([infer_request], request_config)
        swift_response = response[0].choices[0].message.content

        engine.default_template.template_backend = 'jinja'
        response = engine.infer([infer_request], request_config)
        jinja_response = response[0].choices[0].message.content
        assert swift_response == jinja_response, (f'swift_response: {swift_response}\njinja_response: {jinja_response}')
        print(f'response: {swift_response}')

    def test_custom_dataset(self):
        dataset = load_dataset(['swift/stsb'])[0]
        assert len(dataset) == 5749
        assert list(dataset[0].keys()) == ['messages']
        print(f'dataset: {dataset}')
        print(f'dataset[0]: {dataset[0]}')


if __name__ == '__main__':
    unittest.main()
