import os
import tempfile
import unittest

from swift.llm import (MODEL_MAPPING, ModelType, get_model_tokenizer,
                       get_template)


class TestTemplate(unittest.TestCase):

    def test_template(self):
        model_types = [ModelType.qwen_7b_chat_int4]
        for model_type in model_types:
            _, tokenizer = get_model_tokenizer(model_type, load_model=False)
            model_info = MODEL_MAPPING[model_type]
            template_type = model_info['template']
            template = get_template(template_type, tokenizer)
            history = [
                ('你好，你是谁？', '我是来自达摩院的大规模语言模型，我叫通义千问。'),
            ]
            data = {
                'system': 'you are a helpful assistant!',
                'query': '浙江的省会在哪？',
                'response': '浙江的省会是杭州。',
                'history': history
            }
            from swift.llm import print_example
            print_example(template.encode(data), tokenizer)
            input_ids = template.encode(data)['input_ids']
            print(model_type)
            text = tokenizer.decode(input_ids)
            result = """<|im_start|>system
you are a helpful assistant!<|im_end|>
<|im_start|>user
你好，你是谁？<|im_end|>
<|im_start|>assistant
我是来自达摩院的大规模语言模型，我叫通义千问。<|im_end|>
<|im_start|>user
浙江的省会在哪？<|im_end|>
<|im_start|>assistant
浙江的省会是杭州。<|im_end|>
<|endoftext|>"""
            self.assertTrue(result == text)


if __name__ == '__main__':
    unittest.main()
