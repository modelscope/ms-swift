import os
import unittest

from swift.llm import PtEngine, RequestConfig, get_model_tokenizer, get_template
from swift.utils import get_logger, seed_everything

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SWIFT_DEBUG'] = '1'

logger = get_logger()


def _infer_model(pt_engine, system=None, messages=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    resp = pt_engine.infer([{
        'messages': messages,
    }], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return response


class TestTemplate(unittest.TestCase):

    def test_template(self):
        pt_engine = PtEngine('Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4')
        response = _infer_model(pt_engine)
        pt_engine.default_template.template_backend = 'jinja'
        response2 = _infer_model(pt_engine)
        assert response == response2

    def test_tool_message_join(self):
        from copy import deepcopy

        from swift.plugin import agent_templates

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
        tokenizer = get_model_tokenizer('Qwen/Qwen2.5-7B-Instruct', load_model=False)[1]
        template = get_template(tokenizer.model_meta.template, tokenizer)
        for agent_template_type in ('react_zh', 'qwen_zh'):
            agent_template = agent_templates[agent_template_type]()
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
