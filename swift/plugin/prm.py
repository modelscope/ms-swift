import os
from typing import List

import json
import torch

from swift.llm import InferRequest
from swift.llm.infer.protocol import ChatCompletionResponse


class PRM:

    def __init__(self):
        # init here
        pass

    @torch.inference_mode()
    def infer(self, infer_requests: List[InferRequest], ground_truths: List[str],
              **kwargs) -> List[ChatCompletionResponse]:
        raise NotImplementedError


SYSTEM = """
You are a process reward model, give the reward value of the answer, you must follow the instructions below:

1. Output a float reward value between -1.0 and 1.0, -1.0 means the worst answer, 1.0 means the best answer, please think step by step to give your reasons and thoughts, but the reward must appare at the end with this format: **Reward: your-reward-value**.

2. The answer may be incomplete, you must give the reward by the existing part of the answer, taking into account semantic coherence, logical correctness, and clarity.

3. A ground truth answer will be given to you, it may be not the best one, consider it as a reference example.

Begin!
""" # noqa

QUERY = """
The original question or the previous conversation:

#query#

Here is the ground truth as the reference:

#ground_truth#

Given the upper information, give your reward(-1.0~1.0) of the following answer:

#response#
"""


class QwenMaxPRM(PRM):

    def infer(self, infer_requests: List[InferRequest], ground_truths: List[str],
              **kwargs) -> List[ChatCompletionResponse]:
        rewards = []
        for request, ground_truth in zip(infer_requests, ground_truths):
            from openai import OpenAI

            client = OpenAI(
                api_key=os.getenv('DASHSCOPE_API_KEY'),
                base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            )
            previous = request.messages[:-1]
            if previous[0]['role'] == 'system':
                previous = previous[1:]

            assert request.messages[-1]['role'] == 'assistant'
            query = QUERY.replace('#query#', json.dumps(previous))
            query = query.replace('#ground_truth#', ground_truth)
            query = query.replace('#response#', request.messages[-1]['content'])
            messages = [
                {
                    'role': 'system',
                    'content': SYSTEM
                },
                {
                    'role': 'user',
                    'content': query
                },
            ]
            completion = client.chat.completions.create(
                model='qwen-plus',
                messages=messages,
            )

            content = completion.choices[0].message.content
            if 'Reward:' not in content:
                rewards.append(None)
            try:
                reward = float(content.split('Reward:')[1].strip().replace('*', ''))
                rewards.append(reward)
            except Exception:
                rewards.append(None)

        return rewards


prms = {'qwen_max': QwenMaxPRM}
