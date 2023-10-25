import os
import unittest

from swift.llm import (ModelType, TemplateType, get_model_tokenizer,
                       get_template, inference, inference_stream)
from swift.utils import lower_bound, seed_everything


class TestLlmUtils(unittest.TestCase):

    def test_count_startswith(self):
        arr = [-100] * 1000 + list(range(1000))
        self.assertTrue(
            lower_bound(0, len(arr), lambda i: arr[i] != -100) == 1000)

    def test_count_endswith(self):
        arr = list(range(1000)) + [-100] * 1000
        self.assertTrue(
            lower_bound(0, len(arr), lambda i: arr[i] == -100) == 1000)

    def test_inference(self):
        model, tokenizer = get_model_tokenizer(ModelType.chatglm2_6b)
        template = get_template(TemplateType.chatml, tokenizer)
        inputs = template.encode({'query': '你好！'})

        seed_everything(42, True)
        print('stream=True')
        gen_text_stream = inference(inputs['input_ids'], model, tokenizer,
                                    True)
        print(f'[GEN]: {gen_text_stream}')
        #
        seed_everything(42, True)
        gen = inference_stream(inputs['input_ids'], model, tokenizer)
        for gen_text_stream2 in gen:
            pass
        print(f'[GEN]: {gen_text_stream2}')
        #
        seed_everything(42, True)
        print('stream=False')
        gen_text = inference(inputs['input_ids'], model, tokenizer, False)
        print(f'[GEN]: {gen_text}')
        self.assertTrue(gen_text_stream == gen_text_stream2 == gen_text)
        #
        inputs = template.encode({'query': 'hello!'})
        seed_everything(42, True)
        print('stream=True')
        gen_text_stream = inference(inputs['input_ids'], model, tokenizer,
                                    True)
        print(f'[GEN]: {gen_text_stream}')
        #
        seed_everything(42, True)
        gen = inference_stream(inputs['input_ids'], model, tokenizer)
        for gen_text_stream2 in gen:
            pass
        print(f'[GEN]: {gen_text_stream2}')
        #
        seed_everything(42, True)
        print('stream=False')
        gen_text = inference(inputs['input_ids'], model, tokenizer, False)
        print(f'[GEN]: {gen_text}')
        self.assertTrue(gen_text_stream == gen_text_stream2 == gen_text)


if __name__ == '__main__':
    unittest.main()
