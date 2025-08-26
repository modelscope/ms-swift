# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Literal

import numpy as np
import torch
import torch.distributed as dist
from transformers.trainer_utils import EvalPrediction

from swift.utils import Serializer, get_current_device, get_logger

logger = get_logger()


class Metric(ABC):

    def __init__(self):
        self._default = {}
        self._default_factory = {}

    def add_state(self, name: str, default=None, default_factory=None) -> None:
        if not hasattr(self, '_default'):
            raise AttributeError('Please call super().__init__() first.')
        if default is None:
            self._default_factory[name] = default_factory
            assert name not in self._default, f'self._default: {self._default}'
            default = default_factory()
        else:
            self._default[name] = default
            assert name not in self._default_factory, f'self._default_factory: {self._default_factory}'
        setattr(self, name, default)

    def reset(self):
        for k, v in self._default.items():
            setattr(self, k, v)
        for k, v in self._default_factory.items():
            setattr(self, k, v())

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass


class InferStats(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('start_runtime', default_factory=lambda: time.perf_counter())
        self.add_state('num_prompt_tokens', default_factory=dict)
        self.add_state('num_generated_tokens', default_factory=dict)

    def update(self, output):
        id_ = output.id
        self.num_prompt_tokens[id_] = output.usage.prompt_tokens
        self.num_generated_tokens[id_] = output.usage.completion_tokens

    def compute(self):
        runtime = time.perf_counter() - self.start_runtime
        num_samples = len(self.num_generated_tokens)
        num_generated_tokens = sum(self.num_generated_tokens.values())
        return {
            'num_prompt_tokens': sum(self.num_prompt_tokens.values()),
            'num_generated_tokens': num_generated_tokens,
            'num_samples': num_samples,
            'runtime': runtime,
            'samples/s': num_samples / runtime,
            'tokens/s': num_generated_tokens / runtime,
        }


class MeanMetric(Metric):

    def __init__(self, nan_value=0, device=None, group=None):
        super().__init__()
        self.nan_value = nan_value
        self.add_state('state', default=0.)
        self.add_state('count', default=0)
        if device is None:
            device = get_current_device()
        self.device = device
        self.group = group

    def update(self, state: torch.Tensor):
        if isinstance(state, (torch.Tensor, np.ndarray)):
            if state.ndim == 0:
                count = 1
                state = state.item()
            else:
                count = state.shape[0]
                state = state.sum().item()
        elif isinstance(state, (list, tuple)):
            count = len(state)
            state = sum(state)
        else:
            count = 1

        self.state += state
        self.count += count

    def compute(self):
        if dist.is_initialized():
            tensor = torch.tensor([self.state, self.count], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
            self.state, self.count = tensor[0].item(), int(tensor[1].item())
        if self.count == 0:
            value = self.nan_value
        else:
            value = self.state / self.count
        return {
            'value': value,
        }


def compute_rouge_bleu(preds: List[str], labels: List[str]):
    import jieba
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge.rouge import Rouge
    score_dict = {key: MeanMetric() for key in ['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']}

    for pred, label in zip(preds, labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        if not hypothesis or not reference:
            continue
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))[0]
        for k, v in scores.items():
            score_dict[k].update(v['f'])
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict['bleu-4'].update(bleu_score)

    return {k: round(v.compute()['value'] * 100, 6) for k, v in score_dict.items()}


def compute_nlg_metrics(prediction) -> Dict[str, float]:
    preds, labels = prediction[0], prediction[1]
    new_preds, new_labels = [], []
    for i in range(preds.shape[0]):
        new_preds.append(Serializer.from_tensor(preds[i]))
        new_labels.append(Serializer.from_tensor(labels[i]))
    return compute_rouge_bleu(new_preds, new_labels)


def compute_acc(preds,
                labels,
                *,
                acc_strategy: Literal['token', 'seq'] = 'token',
                is_encoder_decoder: bool = False) -> Dict[str, List[float]]:

    if isinstance(preds, torch.Tensor):
        if torch.is_floating_point(labels):
            return {}
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    if preds.ndim >= 2 and not is_encoder_decoder:
        labels = labels[..., 1:]
        preds = preds[..., :-1]
    if np.issubdtype(labels.dtype, np.floating) or preds.shape != labels.shape:
        return {}

    masks = labels != -100
    if acc_strategy == 'token' or preds.ndim == 1:
        acc_list = (preds[masks] == labels[masks]).tolist()
    else:
        acc_list = []
        for i, m in enumerate(masks):
            acc_list.append(np.all(preds[i, m] == labels[i, m]))
    return {f'{acc_strategy}_acc' if preds.ndim >= 2 else 'acc': acc_list}


def compute_acc_metrics(eval_prediction: EvalPrediction,
                        *,
                        acc_strategy: Literal['token', 'seq'] = 'token',
                        is_encoder_decoder: bool = False) -> Dict[str, float]:

    metric = compute_acc(
        eval_prediction.predictions,
        eval_prediction.label_ids,
        acc_strategy=acc_strategy,
        is_encoder_decoder=is_encoder_decoder)
    if len(metric) == 0:
        return {}
    return {k: sum(v) / len(v) for k, v in metric.items()}


def preprocess_logits_for_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = logits.argmax(dim=-1)
    return preds


# Add your own metric calculation method here, use --metric xxx to train
METRIC_MAPPING = {
    'acc': (compute_acc_metrics, preprocess_logits_for_acc),
    'nlg': (compute_nlg_metrics, None),
}


def get_metric(metric: str):
    return METRIC_MAPPING[metric]
