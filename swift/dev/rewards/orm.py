# Copyright (c) ModelScope Contributors. All rights reserved.
# Outcome-reward (ORM) rule plugins over trajectory rows; comparators shared with twinkle/reward.

import json
import re
from typing import Any, Dict, List, Optional, Union

from swift.dev.utils.plugin import PluginRegistry

from .code_reward import CodeExecutionReward
from .reward import ORM
from twinkle.reward import FormatReward, MathReward
from twinkle.reward.gsm8k import MathVerifyAccuracyReward as _MathVerifyAccuracyReward


class MathAccuracy(_MathVerifyAccuracyReward):
    """twinkle's math-verify scorer with swift's 'solution' gold column as the default key."""

    def __init__(self, **kwargs):
        kwargs.setdefault('ground_truth_key', 'solution')
        super().__init__(**kwargs)


class ReActFormat(ORM):

    def __call__(self, trajectories, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, self.completion(t), re.DOTALL | re.MULTILINE) for t in trajectories]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, args: Optional[Any] = None, accuracy_orm=None):
        super().__init__(args)
        self.min_len_value_wrong = args.cosine_min_len_value_wrong
        self.max_len_value_wrong = args.cosine_max_len_value_wrong
        self.min_len_value_correct = args.cosine_min_len_value_correct
        self.max_len_value_correct = args.cosine_max_len_value_correct
        self.max_len = args.cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, trajectories, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(trajectories)
        rewards = []
        for trajectory, acc_reward in zip(trajectories, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(trajectory.get('response_token_ids') or [])
            rewards.append(self.cosfn(gen_len, self.max_len, min_value, max_value))
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, args: Optional[Any] = None, **kwargs):
        super().__init__(args)
        self.ngram_size = args.repetition_n_grams
        self.max_penalty = args.repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, trajectories, **kwargs) -> List[float]:
        """reward function the penalizes repetitions"""
        rewards = []
        for trajectory in trajectories:
            completion = self.completion(trajectory)
            if completion == '' or len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue
            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1
            rewards.append((1 - len(ngrams) / total) * self.max_penalty)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, args: Optional[Any] = None, **kwargs):
        super().__init__(args)
        assert args.soft_cache_length < args.soft_max_length
        self.soft_max_length = args.soft_max_length
        self.soft_cache_length = args.soft_cache_length

    def __call__(self, trajectories, **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion_length = len(trajectory.get('response_token_ids') or [])
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, trajectories: List[Dict], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            prediction = self.completion(trajectory)
            ground_truth = self.column(trajectory, 'ground_truth')
            if ground_truth is None:
                ground_truth = self.column(trajectory, 'solution')
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            reward = ReactORM.evaluate_action_reward(
                [pred_action or 'none'], [ref_action], [pred_input or '{}'], [ref_input])
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


orms = {
    'toolbench': ReactORM,
    # identical scorers are twinkle's own classes, used directly (no re-wrapper): resolve()
    # no-arg constructs them and the row protocol is already their native input.
    'math': MathReward,
    'format': FormatReward,
    'accuracy': MathAccuracy,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'code_execution': CodeExecutionReward,
}

#: The 'reward' extension point adopts ``orms`` *itself* as its registry, so
#: ``@PluginRegistry.register('reward', ...)`` and the historical ``orms['my'] = MyReward`` write to one
#: dict -- no consumer has to know which form a plugin arrived in, and there is no second roster to
#: keep in sync. The kind's base is the ``ORM`` rule itself (the twinkle ``Reward`` protocol plus the
#: plugin construction contract); an async reward is just an ``ORM`` with ``async def __call__``.
REWARD = PluginRegistry.register_kind(
    'reward', ORM, config_field='orm', entries=orms)
