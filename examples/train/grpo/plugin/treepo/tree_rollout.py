import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import List

import torch
from modelscope.preprocessors.templates.utils import Messages

from swift.llm.infer.protocol import ChatCompletionResponseChoice


class SampleStatus(Enum):
    INITIAL = 'initial'
    TO_INFER = 'to_infer'
    FINISH_NEXT_INFER = 'finish_next_infer'
    FINISHED = 'finished'
    ROLLBACK = 'rollback'


class FinishedReason(Enum):
    ANSWER = 'finished_with_answer'
    MAX_INFER_STEP = 'finished_with_max_infer_steps'
    UNFINISHED = 'unfinished'


@dataclass
class DataSampleTree:
    """
    Attributes:
        tree_idx (str):
            for example 0/1-2/2-3/4-0, root_node = 0, next node = 1-2 infer batch 1 and index 2 sample

        last_response (ChatCompletionResponseChoice):
            vllm previous round output
    """
    tree_idx: str
    request_id: str

    messages: Messages
    logprobs: List[List[float]] = field(default_factory=list)

    all_response_ids: List[List[int]] = field(default_factory=list)
    last_response: ChatCompletionResponseChoice = None

    token_count_per_step: List[int] = field(default_factory=list)

    status: SampleStatus = SampleStatus.INITIAL
    finished_reason: FinishedReason = FinishedReason.UNFINISHED

    @property
    def root_node(self):
        return int(self.tree_idx.split('/')[0])

    @property
    def depth(self):
        return len(self.tree_idx.split('/')) - 1

    @property
    def response_num(self):
        return len(self.all_response_ids)

    def response_truncate(self, truncate_len: int):
        """
        Before rollback, truncate the response.
        """

        if truncate_len < 1:
            return

        self.logprobs = self.logprobs[:-truncate_len]
        self.all_response_ids = self.all_response_ids[:-truncate_len]
        self.messages = self.messages[:-(truncate_len * 2 - 1)]
        self.last_response = None

    def extend_response(self, choice: ChatCompletionResponseChoice):
        self.extend_response_text(choice.message.content)
        self.extend_logprobs([item['logprob'] for item in choice.logprobs['content']])

        self.all_response_ids.append(choice.token_ids)
        self.token_count_per_step.append(len(choice.token_ids))

        choice.logprobs = None
        self.last_response = deepcopy(choice)

    def extend_response_text(self, response_text: str):
        self.messages.append({'role': 'assistant', 'content': response_text})

    def extend_logprobs(self, logprobs: List[float]):
        self.logprobs.append(logprobs)


def _repeat_list_interleave(any_list, repeat_times):
    # return [item for sublist in [[item] * repeat_times for item in any_list] for item in sublist]
    return [deepcopy(item) for sublist in [[item] * repeat_times for item in any_list] for item in sublist]


def _increment_tree_idx_depth(
    samples: list[DataSampleTree],
    next_infer_step: int,
) -> list[DataSampleTree]:
    for infer_batch_idx, sample in enumerate(samples):
        sample.tree_idx = sample.tree_idx + '/' + f'{next_infer_step}-{infer_batch_idx}'
    return samples


def extract_last_boxed(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'

    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(0)
    return None


class AbstractDivergence:

    @classmethod
    def calc_weights(cls, root_idx, samples_to_go_deeper, **kwargs) -> List[float]:
        pass

    @classmethod
    def allocate_with_weights(cls, weights, budget, max_divergence) -> List[int]:
        n = len(weights)
        alloc = [0] * n

        w = [float(wi) if wi is not None and wi > 0 else 0.0 for wi in weights]
        total_w = sum(w)
        if total_w == 0:
            return alloc

        # first round of allocation by weight ratio
        ideals = [(w[i] / total_w) * budget if w[i] > 0 else 0.0 for i in range(n)]
        for i in range(n):
            if w[i] <= 0:
                continue
            f = int(ideals[i])
            alloc[i] = min(f, max_divergence)

        # second round of allocation by greedy allocation
        remain = budget - sum(alloc)
        if remain <= 0:
            return alloc

        # weights desc, index asc
        remainders = [(ideals[i] - int(ideals[i]), i) for i in range(n) if w[i] > 0 and alloc[i] < max_divergence]
        remainders.sort(key=lambda x: (-x[0], x[1]))

        idx = 0
        while remain > 0 and remainders:
            frac, i = remainders[idx % len(remainders)]
            if alloc[i] < max_divergence:
                alloc[i] += 1
                remain -= 1

                if alloc[i] >= max_divergence:
                    remainders = [r for r in remainders if r[1] != i]
                    idx = 0
                    continue
            idx += 1

        return alloc

    @classmethod
    def apply(cls, root_idx, samples_to_go_deeper, divergence_budget, max_divergence, **kwargs) -> List[DataSampleTree]:
        """
        Args:
            root_idx: current root node idx
            samples_to_go_deeper: go deeper samples which root_node = root_idx
            divergence_budget: total divergence
            max_divergence: each sample max divergence
        """
        weights = cls.calc_weights(root_idx, samples_to_go_deeper, **kwargs)
        allocate_divergence = cls.allocate_with_weights(weights, divergence_budget, max_divergence)

        divergence_samples = []
        for sample, divergence in zip(samples_to_go_deeper, allocate_divergence):
            for _ in range(divergence):
                divergence_samples.append(deepcopy(sample))

        return divergence_samples


class LogProbDivergence(AbstractDivergence):

    @classmethod
    def calc_weights(cls, root_idx, samples_to_go_deeper, **kwargs) -> List[float]:
        """
        In this strategy, weight is proportional to entropy
        """
        entropies = []
        for sample in samples_to_go_deeper:
            log_probs = torch.tensor(sample.logprobs[-1])

            probs = torch.exp(log_probs)
            entropy = -torch.sum(probs * log_probs)
            entropies.append(entropy)

        entropies_tensor = torch.stack(entropies)
        weights = torch.softmax(entropies_tensor, dim=0)

        return weights.tolist()


class AvgDivergence(AbstractDivergence):

    @classmethod
    def calc_weights(cls, root_idx, samples_to_go_deeper, **kwargs) -> List[float]:
        avg = torch.ones(len(samples_to_go_deeper))
        weights = torch.softmax(avg, dim=0)

        return weights.tolist()


DivergenceStrategyMapping = {'logprobs': LogProbDivergence, 'average': AvgDivergence}
