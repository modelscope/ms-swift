import re
from typing import Dict, List, Union

import torch

from swift.llm import InferRequest
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage
from swift.plugin.orm import ORM, orms
from swift.utils import get_logger

logger = get_logger()


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        super().__init__()
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs):
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
            return rewards

    @torch.inference_mode()
    def infer(self, infer_requests: Union[List[InferRequest], List[Dict]], ground_truths: List[str],
              **kwargs) -> List[ChatCompletionResponse]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        rewards = self.__call__(predictions, solution=ground_truths)
        return [
            ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        message=ChatMessage(content=r if r else 0.0, role='assistant'), index=0, finish_reason='')
                ],
                model=None,
                usage=None) for r in rewards
        ]


class MathFormat(ORM):

    def __init__(self):
        super().__init__()

    def __call__(self, completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]

    @torch.inference_mode()
    def infer(self, infer_requests: Union[List[InferRequest], List[Dict]], ground_truths: List[str],
              **kwargs) -> List[ChatCompletionResponse]:
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        rewards = self.__call__(predictions)
        return [
            ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        message=ChatMessage(content=r if r else 0.0, role='assistant'), index=0, finish_reason='')
                ],
                model=None,
                usage=None) for r in rewards
        ]


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
