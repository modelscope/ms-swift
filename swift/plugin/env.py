# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from swift.plugin.orm import MathAccuracy

if TYPE_CHECKING:
    from swift.llm.template import RolloutInferRequest
    from swift.llm.utils import Messages


class Env(ABC):
    """Base environment interface for GRPO training."""

    def __init__(self, env_config):
        """Initialize environment."""
        self.env_config = env_config

    @abstractmethod
    async def reset(self, config: 'RolloutInferRequest') -> Tuple[str, Dict[str, Any], str]:
        """Reset environment to initial state.

        Args:
            config: Initial configuration containing dataset information

        Returns:
            Tuple of (observation, info, system_message):
            - observation: Initial query string for the agent
            - info: Environment debug information as dict
            - system_message: System prompt for this trajectory
        """
        pass

    @abstractmethod
    async def step(self, action: 'Messages') -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: LLM response choice containing the action to execute

        Returns:
            Tuple of (next_observation, reward, done, info):
            - next_observation: Next observation string
            - reward: Reward value for this step
            - done: Whether the episode is finished
            - info: Additional information as dict
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up environment resources."""
        pass


def count_qwen_tokens(messages: List[Dict[str, Any]], max_tokens: int = 2048) -> Tuple[int, bool]:
    """
    Calculate token count for Qwen messages and check if it exceeds the 16k limit

    Args:
        messages: List of messages in OpenAI format
        max_tokens: Maximum token limit, default 2k

    Returns:
        Tuple[int, bool]: (token count, whether within limit)
    """
    try:
        from modelscope import AutoTokenizer
        model_name = 'Qwen/Qwen2.5-3B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_count = len(tokenizer.encode(text))

        return token_count, token_count >= max_tokens

    except Exception as e:
        print(f'Token calculation failed: {e}')
        return 0, False


class SimpleMathEnv(Env):
    tips_prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'

    def __init__(self, env_config):
        super().__init__(env_config)
        self.acc_func = MathAccuracy()
        self.solution = ''

    async def reset(self, config: 'RolloutInferRequest') -> Tuple[str, Dict[str, Any], str]:
        obs = config.data_dict['problem']
        info = {}
        self.solution = config.data_dict['solution']
        system_prompt = """A conversation between User and Assistant.
        The user asks a question, and the Assistant solves it.
        The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags,
        respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
        """
        return obs, info, system_prompt

    async def step(self, action: 'Messages') -> Tuple[str, float, bool, Dict[str, Any]]:
        next_obs = self.tips_prompt

        reward = 0.0
        done = False
        info = {}
        acc = self.acc_func([action[-1]['content']], [self.solution])[0]
        if count_qwen_tokens(action)[1]:
            done = True
            info['stop_reason'] = 'Exceeded maximum length'

        if acc == 1:
            done = True
            reward = 1.0
            info['stop_reason'] = 'Correct'
        info['math_reward'] = reward
        return next_obs, reward, done, info

    async def close(self):
        pass


# Registry for environments
envs = {'math_env': SimpleMathEnv}
