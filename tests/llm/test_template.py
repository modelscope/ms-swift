import os
import tempfile
import torch
import unittest
from unittest.mock import patch

from PIL import Image
from swift.infer_engine import RequestConfig, TransformersEngine
from swift.model import get_processor
from swift.template import get_template
from swift.template.base import Template
from swift.utils import get_logger, seed_everything

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SWIFT_DEBUG'] = '1'

logger = get_logger()


def _infer_model(engine, system=None, messages=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    resp = engine.infer([{
        'messages': messages,
    }], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {engine.model_info.model_name}, messages: {messages}')
    return response


class TestTemplate(unittest.TestCase):

    @unittest.skipIf(not torch.cuda.is_available(), reason='GPTQ is only available on GPU')
    def test_template(self):
        engine = TransformersEngine('Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4')
        response = _infer_model(engine)
        engine.template.template_backend = 'jinja'
        response2 = _infer_model(engine)
        assert response == response2

    def test_tool_message_join(self):
        from copy import deepcopy

        from swift.agent_template import agent_template_map

        messages = [
            # first round
            {
                'role': 'user',
                'content': 'user1'
            },
            {
                'role': 'assistant',
                'content': 'assistant1'
            },
            {
                'role': 'assistant',
                'content': 'assistant2'
            },
            {
                'role': 'tool',
                'content': 'tool1'
            },
            # second round
            {
                'role': 'assistant',
                'content': 'assistant3'
            },
            {
                'role': 'tool',
                'content': 'tool2'
            },
            {
                'role': 'tool',
                'content': 'tool3'
            },
        ]

        # testing two template type.
        tokenizer = get_processor('Qwen/Qwen2.5-7B-Instruct')
        template = get_template(tokenizer)
        for agent_template_type in ('react_zh', 'qwen_zh'):
            template._agent_template = agent_template_type
            agent_template = template.agent_template
            observation = agent_template.keyword.observation
            test_messages = deepcopy(messages)
            test_messages[2]['content'] = 'assistant2' + observation
            test_messages[4]['content'] = (
                agent_template.keyword.action + agent_template.keyword.action_input + 'assistant3' + observation)
            encoded = template.encode({'messages': test_messages})
            res = template.safe_decode(encoded['input_ids'])

            ground_truth = (
                '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
                '<|im_start|>user\nuser1<|im_end|>\n'
                f'<|im_start|>assistant\nassistant1assistant2{observation}tool1'
                f'{agent_template.keyword.action}{agent_template.keyword.action_input}assistant3'
                f'{observation}tool2\n{observation}tool3\n')
            assert res == ground_truth

    def test_save_pil_image_uses_dimensions_in_cache_key(self):
        width_a, height_a = 120, 80
        width_b, height_b = 80, 120
        self.assertEqual(width_a * height_a, width_b * height_b)

        pixels = bytearray()
        for i in range(width_a * height_a):
            row = i // width_a
            pixels.extend((255, 60, 60) if row % 10 < 5 else (60, 60, 255))
        img_bytes = bytes(pixels)

        image_a = Image.frombytes('RGB', (width_a, height_a), img_bytes)
        image_b = Image.frombytes('RGB', (width_b, height_b), img_bytes)

        with tempfile.TemporaryDirectory() as cache_dir, patch('swift.template.base.get_cache_dir', return_value=cache_dir):
            path_a = Template._save_pil_image(image_a)
            path_b = Template._save_pil_image(image_b)

            self.assertNotEqual(path_a, path_b)
            self.assertEqual(Image.open(path_a).size, (width_a, height_a))
            self.assertEqual(Image.open(path_b).size, (width_b, height_b))


if __name__ == '__main__':
    unittest.main()
