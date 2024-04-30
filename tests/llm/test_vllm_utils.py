import os
import unittest

import torch

from swift.llm.utils import *
from swift.utils import lower_bound, seed_everything

SKPT_TEST = True


class TestVllmUtils(unittest.TestCase):

    @unittest.skipIf(SKPT_TEST, 'To avoid citest error: OOM')
    def test_inference_vllm(self):
        model_type = ModelType.qwen_7b_chat
        llm_engine = get_vllm_engine(model_type, torch.float16)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, llm_engine.hf_tokenizer)
        request_list = [{'query': '浙江的省会在哪？'}, {'query': '你好!'}]
        # test inference_vllm
        response_list = inference_vllm(llm_engine, template, request_list, verbose=True)
        for response in response_list:
            print(response)

        # test inference_stream_vllm
        gen = inference_stream_vllm(llm_engine, template, request_list)
        for response_list in gen:
            print(response_list[0]['response'], response_list[0]['history'])
            print(response_list[1]['response'], response_list[1]['history'])


if __name__ == '__main__':
    unittest.main()
