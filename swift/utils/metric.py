# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, Literal

import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction

from .logger import get_logger

logger = get_logger()


def compute_nlg_metrics(prediction, tokenizer):
    import jieba
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge.rouge import Rouge
    preds, labels = prediction[0], prediction[1]

    score_dict = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}

    def _decode(tokens, ignore_pad_token_for_loss=False):
        if ignore_pad_token_for_loss:
            tokens = np.where(tokens != -100, tokens, tokenizer.pad_token_id)
        tokens = np.where(tokens < tokenizer.vocab_size, tokens, tokenizer.pad_token_id)
        return [t for t in tokenizer.batch_decode(tokens, skip_special_tokens=True)]

    for pred, label in zip(preds, labels):
        pred = ''.join(_decode(pred, False))
        label = ''.join(_decode(label, True))
        hypothesis = list(jieba.cut(pred))
        if len(hypothesis) == 0 or ''.join(hypothesis) == '.':
            hypothesis = [tokenizer.decode(tokenizer.eos_token_id)]
        reference = list(jieba.cut(label))
        try:
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v['f'] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict['bleu-4'].append(round(bleu_score * 100, 4))
        except Exception as e:
            logger.error(e)
            logger.error(f'eval error {hypothesis}, {reference}')

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict


def compute_acc_metrics(eval_prediction: EvalPrediction,
                        acc_strategy: Literal['token', 'sentence'] = 'token',
                        is_encoder_decoder: bool = False) -> Dict[str, torch.Tensor]:
    if is_encoder_decoder:
        labels = eval_prediction.label_ids[..., :]
        predictions = eval_prediction.predictions[..., :]
    else:
        labels = eval_prediction.label_ids[..., 1:]
        predictions = eval_prediction.predictions[..., :-1]
    if predictions.shape != labels.shape:
        return {}
    masks = labels != -100
    if acc_strategy == 'sentence':
        acc_list = []
        for i, m in enumerate(masks):
            acc_list.append(np.all(predictions[i, m] == labels[i, m]))
        acc = np.mean(np.array(acc_list))
    else:
        acc = np.mean((predictions[masks] == labels[masks]).astype(np.float64))
    return {'acc': acc}


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = logits.argmax(dim=-1)
    return preds
