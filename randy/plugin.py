import re
import math
import textwrap
from swift.utils import get_logger
from swift.plugin import ORM, orms, rm_plugins

logger = get_logger()


class AccReward(ORM):
    def _extract_answer(self, text):
        match = re.search(r'<answer>(.*?)</answer>', text)
        return match.group(1).strip() if match else text.strip()

    def __call__(self, completions, solution, **kwargs):
        rewards = []
        for output, answer in zip(completions, solution):
            try:
                correct = self._extract_answer(answer)
                generated = self._extract_answer(output)
                reward = 1.0 if generated == correct else 0.0
            except Exception:
                reward = 0.0
            rewards.append(reward)
        print_rewards = [f'{x:.2f}' for x in rewards]
        logger.info(f'acc_reward: {print_rewards}')
        return rewards


class GaussReward(ORM):
    def __init__(self, target=1024, sigma=256):
        self.target = target
        self.sigma = sigma

    def __call__(self, completions, **kwargs):
        rewards = []
        for ids in kwargs.get('response_token_ids'):
            exponent = -((len(ids) - self.target) ** 2) / (2 * self.sigma ** 2)
            reward = math.exp(exponent) - 1.0
            rewards.append(reward)
        print_rewards = [f'{x:.2f}' for x in rewards]
        logger.info(f'gauss_reward: {print_rewards}')
        return rewards


class CosineReward(ORM):
    def __init__(self,
        cosine_min_len_value_wrong=-0.5,
        cosine_max_len_value_wrong=0.0,
        cosine_min_len_value_correct=1.0,
        cosine_max_len_value_correct=0.5,
        cosine_max_len=1024
    ):
        self.min_wrong = cosine_min_len_value_wrong
        self.max_wrong = cosine_max_len_value_wrong
        self.min_correct = cosine_min_len_value_correct
        self.max_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.acc_orm = AccReward()

    def cos_interp(self, t, T, minv, maxv):
        return maxv - (maxv - minv) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs):
        acc_rewards = self.acc_orm(completions, solution, **kwargs)
        token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(token_ids, acc_rewards):
            is_correct = acc_reward >= 1.0
            if is_correct:
                min_value = self.max_correct
                max_value = self.min_correct
            else:
                min_value = self.max_wrong
                max_value = self.min_wrong
            reward = self.cos_interp(len(ids), self.max_len, min_value, max_value)
            rewards.append(reward)
        print_rewards = [f'{x:.2f}' for x in rewards]
        logger.info(f'cosine_reward: {print_rewards}')
        return rewards


orms['acc_reward'] = AccReward
orms['gauss_reward'] = GaussReward
orms['cosine_reward'] = CosineReward
