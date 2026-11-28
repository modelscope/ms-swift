import re
from typing import List

from swift.rewards import ORM, orms


class GSM8KAccuracy(ORM):

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract the last #### number from text."""
        text = text[-500:] if len(text) > 500 else text
        # Prefer \boxed{} format
        boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed[-1].replace(',', '').replace(' ', '').strip()
        # Fallback to #### format
        matches = re.findall(r'####\s*([\-\d,\.\s]+)', text)
        if matches:
            return matches[-1].replace(',', '').replace(' ', '').strip()
        return ''

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, gt_answer in zip(completions, solution):
            gt_num = self.extract_answer(gt_answer)
            pred_num = self.extract_answer(completion)
            correct = False
            if pred_num and gt_num:
                try:
                    correct = abs(float(pred_num) - float(gt_num)) < 1e-5
                except (ValueError, OverflowError):
                    correct = pred_num == gt_num
            rewards.append(1.0 if correct else 0.0)
        return rewards


class GSM8KFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            has_answer = bool(re.search(r'\\boxed\{[^}]+\}', completion) or re.search(r'####\s*[\-\d,\.]+', completion))
            rewards.append(1.0 if has_answer else 0.0)
        return rewards


orms['gsm8k_accuracy'] = GSM8KAccuracy
orms['gsm8k_format'] = GSM8KFormat
