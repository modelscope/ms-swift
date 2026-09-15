import os
from typing import Any, Dict, List, Union

import json

from swift.llm import InferRequest


class PRM:

    def __call__(self, **kwargs) -> List[Any]:
        raise NotImplementedError


SYSTEM = """
You are a process reward model, give the reward value of the answer, you must follow the instructions below:

1. Output a float reward value between -1.0 and 1.0, -1.0 means the worst answer, 1.0 means the best answer, please think step by step to give your reasons and thoughts, but the reward must appare at the end with this format: **Reward: your-reward-value**.

2. The answer may be incomplete, you must give the reward by the existing part of the answer, taking into account semantic coherence, logical correctness, and clarity.

3. A ground truth answer will be given to you, it may be not the best one, consider it as a reference example.
您是一个过程奖励模型，给出答案的奖励值，您必须遵循以下指示：

1. 输出一个介于 -1.0 和 1.0 之间的浮点奖励值，-1.0 表示最差的答案，1.0 表示最好的答案，请逐步思考并给出您的理由和想法，但奖励值必须以以下格式出现在最后：**奖励：您的奖励值**。

2. 答案可能不完整，您必须根据现有答案的部分给出奖励，考虑语义连贯性、逻辑正确性和清晰度。

3. 将给出一个真实答案供您参考，它可能不是最好的，请将其视为一个参考示例。
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
    '''
    生成式奖励模型，根据答案给出奖励值
    '''

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        # TODO: check request_config
        rewards = []

        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        for request, ground_truth in zip(infer_requests, ground_truths):
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
                model='qwen-max',
                messages=messages,
            )

            content = completion.choices[0].message.content
            if 'Reward:' not in content:
                rewards.append(0.)
            else:
                try:
                    reward = float(content.split('Reward:')[1].strip().replace('*', ''))
                    rewards.append(reward)
                except Exception:
                    rewards.append(0.)

        return rewards


class ClientPRM(PRM):

    def __init__(self, api_key=None, base_url=None, model=None):
        from swift.llm import InferClient
        import os
        if api_key is None:
            api_key = os.getenv('DASHSCOPE_API_KEY')
        if base_url is None:
            base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        if model is None:
            model = 'qwen-plus'
        self.infer_engine = InferClient(base_url=base_url, api_key=api_key)
        self.infer_engine.strict = False
        self.infer_kwargs = {
            'model': model,
        }

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        prm_infer_requests = []
        request_config = kwargs.get('request_config')
        for request, ground_truth in zip(infer_requests, ground_truths):
            previous = request['messages'][:-1]
            if previous[0]['role'] == 'system':
                previous = previous[1:]

            assert request['messages'][-1]['role'] == 'assistant'
            query = QUERY.replace('#query#', json.dumps(previous))
            query = query.replace('#ground_truth#', ground_truth)
            query = query.replace('#response#', request['messages'][-1]['content'])
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

            prm_infer_requests.append(InferRequest(messages=messages))

        responses = self.infer_engine.infer(prm_infer_requests, request_config=request_config, **self.infer_kwargs)
        rewards = []
        for response in responses:
            content = response.choices[0].message.content
            if 'Reward:' not in content:
                rewards.append(0.)
            else:
                try:
                    reward = float(content.split('Reward:')[1].strip().replace('*', ''))
                    rewards.append(reward)
                except Exception:
                    rewards.append(0.)
        return rewards


prms = {
    'qwen_max': QwenMaxPRM,
    'client': ClientPRM,
}
