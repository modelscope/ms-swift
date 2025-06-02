from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union


class MultiTurnScheduler(ABC):

    def __init__(self, max_turns: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self.max_turns = max_turns

    @abstractmethod
    def step(*args, **kwargs):
        pass

    def check_finished(self, result, current_turn) -> bool:
        if result['finished'] or result['finish_reason'] == 'length':
            return True

        if self.max_turns:
            if current_turn >= self.max_turns:
                return True

        return False

    def set_max_turns(self, max_turns):
        self.max_turns = max_turns


class FunctionScheduler(MultiTurnScheduler):

    def __init__(self, callback: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback

    def step(self, *args, **kwargs):
        return self.callback(*args, **kwargs)


class ReToolScheduler(MultiTurnScheduler):

    def __init__(self):
        super().__init__()
        self.sandbox = None

    def step(completions: Union[List[str], str]):
        pass  # TODO

    def extract_code(completions: Union[List[str], str]):
        pass  # TODO


def check_math_result_and_give_tips(inputs):
    from .orm import MathAccuracy
    acc = MathAccuracy()
    # a trick
    prompt = 'But wait... It seems I made a mistake,'
    contents = [input['messages'][-1]['content'] for input in inputs]
    rewards = acc(contents, [input['solution'] for input in inputs])
    for reward, input in zip(rewards, inputs):
        content = input['messages'][-1]['content']
        if reward < 1 and prompt not in content:
            if '<answer>' in content:
                content = content[:content.index('<answer>')]
            if '</think>' in content:
                content = content[:content.index('</think>')]
            content += prompt
            input['messages'][-1]['content'] = content
            input['finished'] = False
        else:
            input['finished'] = True
    return inputs


def check_math_result_and_give_tips_multi_turn(inputs):
    from .orm import MathAccuracy
    acc = MathAccuracy()
    prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'
    contents = [input['messages'][-1]['content'] for input in inputs]
    rewards = acc(contents, [input['solution'] for input in inputs])
    for reward, input in zip(rewards, inputs):
        content = input['messages'][-2]['content']
        if reward < 1 and prompt not in content:
            input['messages'].append({'role': 'user', 'content': prompt})
            input['finished'] = False
        else:
            input['finished'] = True
    return inputs


multi_turns = {
    'math_tip_trick': FunctionScheduler(check_math_result_and_give_tips),
    'math_tip_trick_multi_turn': FunctionScheduler(check_math_result_and_give_tips_multi_turn),
}
