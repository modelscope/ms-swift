import os
import unittest

from swift.llm import PtEngine, RequestConfig
from swift.utils import get_logger, seed_everything

logger = get_logger()
os.environ['SWIFT_DEBUG'] = '1'

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SKPT_TEST = True


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


if __name__ == '__main__':
    unittest.main()
