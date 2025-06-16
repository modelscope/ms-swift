from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

from swift.llm.infer.protocol import ChatCompletionResponseChoice, ChatCompletionResponseChoiceWithHistory
from swift.llm.template import RolloutInferRequest
"""
TODO
1. add reward_kwargs to InferRequest in training
2. refactor training multi turn compatible with multi turn scheduler
3. argument max_turns
4. 续写逻辑, 增加response 为 None
5. document about scheduler
6. retool implement and document
7. loss_mask
8. token increment in infer request
"""


class MultiTurnScheduler(ABC):

    def __init__(self, max_turns: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self.max_turns = max_turns

    @abstractmethod
    def step(self, infer_request: RolloutInferRequest, result: ChatCompletionResponseChoiceWithHistory,
             current_turn: int, **kwargs) -> RolloutInferRequest:
        pass

    def check_finished(self, infer_request: RolloutInferRequest, result: ChatCompletionResponseChoiceWithHistory,
                       current_turn: int) -> bool:
        # 默认逻辑： 长度截断或到达max_turns
        if result.finish_reason == 'length':
            return True

        if self.max_turns and current_turn >= self.max_turns:
            return True

        return False

    def set_max_turns(self, max_turns):
        self.max_turns = max_turns


class ReToolScheduler(MultiTurnScheduler):

    def __init__(self):
        super().__init__()
        self.sandbox = None

    def step(self, infer_request: RolloutInferRequest, result: ChatCompletionResponseChoiceWithHistory,
             current_turn: int, **kwargs) -> RolloutInferRequest:

        # if current_turn == 0 or not messages[-1]['content'] or messages[-1]['content'] == '<None>':
        #     messages = RolloutInferRequest.remove_response(infer_request.messages)

        pass  # TODO

    def extract_code(completions: Union[List[str], str]):

        pass  # TODO


class MathTipsScheduler(MultiTurnScheduler):
    from .orm import MathAccuracy
    tips_prompt = 'But wait... It seems I made a mistake,'
    acc_func = MathAccuracy()

    def check_finished(self, infer_request: RolloutInferRequest, result: ChatCompletionResponseChoiceWithHistory,
                       current_turn: int) -> bool:
        completion = result.message[-1]['content']
        # we only give tips once
        if self.tips_prompt in completion:
            return True
        solution = infer_request.data_dict['solution']

        acc = self.acc_func([completion], [solution])[0]
        if acc == 1:
            return True

        return super().check_finished(result, current_turn)

    def step(self, infer_request: RolloutInferRequest, result: ChatCompletionResponseChoiceWithHistory,
             current_turn: int, **kwargs) -> RolloutInferRequest:
        completion = result.message[-1]['content']
        if '<answer>' in completion:
            completion = completion[:completion.index('<answer>')]
        if '</think>' in completion:
            completion = completion[:completion.index('</think>')]
        completion += self.tips_prompt
        infer_request.messages[-1]['content'] = completion
        return infer_request


class MathTipsMultiTurnScheduler(MultiTurnScheduler):
    from .orm import MathAccuracy
    tips_prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'
    acc_func = MathAccuracy()

    def check_finished(self, infer_request: RolloutInferRequest, result: ChatCompletionResponseChoiceWithHistory,
                       current_turn: int) -> bool:
        query = result.message[-2]['content']
        # we only give tips once
        if self.tips_prompt in query:
            return True

        completion = result.message[-1]['content']
        solution = infer_request.data_dict['solution']
        acc = self.acc_func([completion], [solution])[0]
        if acc == 1:
            return True

        return super().check_finished(result, current_turn)

    def step(self, infer_request: RolloutInferRequest, result: ChatCompletionResponseChoiceWithHistory,
             **kwargs) -> RolloutInferRequest:
        completion = result.message[-1]['content']
        infer_request.messages.append(
            {
                'role': 'assistant',
                'content': completion
            },
            {
                'role': 'user',
                'content': self.tips_prompt
            },
        )
        return infer_request


multi_turns = {
    'math_tip_trick': MathTipsScheduler,
    'math_tip_trick_multi_turn': MathTipsMultiTurnScheduler,
}
