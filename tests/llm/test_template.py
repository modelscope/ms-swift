import os
import unittest

import torch

from swift.infer_engine import RequestConfig, TransformersEngine
from swift.model import get_processor
from swift.template import get_template
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
            agent_template = agent_template_map[agent_template_type]()
            template.agent_template = agent_template
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


if __name__ == '__main__':
    unittest.main()
