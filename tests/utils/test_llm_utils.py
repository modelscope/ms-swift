import os
import unittest

from swift.llm import (MODEL_MAPPING, ModelType, get_model_tokenizer,
                       get_template, inference, inference_stream,
                       print_example)
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
        model_type = ModelType.chatglm2_6b
        model, tokenizer = get_model_tokenizer(model_type)
        template = get_template(MODEL_MAPPING[model_type]['template'],
                                tokenizer)
        model.generation_config.max_length = 128
        model.generation_config.do_sample = True
        for query in ['你好', 'hello']:
            seed_everything(42, True)
            print('stream=True')
            gen_text_stream, history = inference(
                model, template, query, stream=True)
            print(f'[GEN]: {gen_text_stream}')
            print(f'[HISTORY]: {history}')
            #
            seed_everything(42, True)
            gen = inference_stream(model, template, query)
            for gen_text_stream2, history2 in gen:
                pass
            print(f'[GEN]: {gen_text_stream2}')
            print(f'[HISTORY]: {history2}')
            #
            seed_everything(42, True)
            print('stream=False')
            gen_text, history3 = inference(
                model, template, query, stream=False)
            print(f'[GEN]: {gen_text}')
            print(f'[HISTORY]: {history3}')
            self.assertTrue(gen_text_stream == gen_text_stream2 == gen_text)
            self.assertTrue(history == history2 == history3)

    def test_print_example(self):
        input_ids = [1000, 2000, 3000, 4000, 5000, 6000]
        labels = [-100, -100, 1000, 2000, 3000, -100, -100, 4000, 5000, 6000]
        _, tokenizer = get_model_tokenizer(
            ModelType.chatglm3_6b, load_model=False)
        print_example({'input_ids': input_ids, 'labels': labels}, tokenizer)
        labels = [-100, -100, -100]
        print_example({'input_ids': input_ids, 'labels': labels}, tokenizer)
        labels = [1000, 2000, 3000, 4000, 5000, 6000]
        print_example({'input_ids': input_ids, 'labels': labels}, tokenizer)


if __name__ == '__main__':
    unittest.main()
