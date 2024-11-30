# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction

from swift.plugin import MeanMetric
from .logger import get_logger
from .torch_utils import Serializer

logger = get_logger()


def compute_nlg_metrics(prediction, tokenizer):
    import jieba
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge.rouge import Rouge
    preds, labels = prediction[0], prediction[1]
    score_dict = {key: MeanMetric() for key in ['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']}

    for pred, label in zip(preds, labels):
        pred = Serializer.from_tensor(pred)
        label = Serializer.from_tensor(label)
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


def compute_acc(preds,
                labels,
                *,
                acc_strategy: Literal['token', 'sentence'] = 'token',
                is_encoder_decoder: bool = False) -> Optional[List[float]]:

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

    if is_encoder_decoder:
        labels = labels[..., :]
        preds = preds[..., :]
    else:
        labels = labels[..., 1:]
        preds = preds[..., :-1]
    if preds.shape != labels.shape:
        return None

    masks = labels != -100
    if acc_strategy == 'sentence':
        acc_list = []
        for i, m in enumerate(masks):
            acc_list.append(np.all(preds[i, m] == labels[i, m]))
    else:
        acc_list = (preds[masks] == labels[masks]).tolist()
    return acc_list


def compute_acc_metrics(eval_prediction: EvalPrediction,
                        *,
                        acc_strategy: Literal['token', 'sentence'] = 'token',
                        is_encoder_decoder: bool = False) -> Dict[str, torch.Tensor]:

    acc_list = compute_acc(
        eval_prediction.predictions,
        eval_prediction.label_ids,
        acc_strategy=acc_strategy,
        is_encoder_decoder=is_encoder_decoder)
    if acc_list is None:
        return {}
    return {'acc': sum(acc_list) / len(acc_list)}


def preprocess_logits_for_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = logits.argmax(dim=-1)
    return preds
