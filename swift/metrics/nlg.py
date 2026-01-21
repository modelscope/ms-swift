# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, List

from transformers import EvalPrediction

from swift.utils import Serializer, get_logger
from .base import EvalMetrics
from .utils import MeanMetric

logger = get_logger()


def compute_rouge_bleu(preds: List[str], labels: List[str]):
    import jieba
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge.rouge import Rouge
    score_dict = {key: MeanMetric() for key in ['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']}

    for pred, label in zip(preds, labels):
        hypothesis = [w.strip(' ') for w in jieba.cut(pred) if w.strip(' ')]
        reference = [w.strip(' ') for w in jieba.cut(label) if w.strip(' ')]
        if not hypothesis or not reference:
            continue
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))[0]
        for k, v in scores.items():
            score_dict[k].update(v['f'])
        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
        score_dict['bleu-4'].update(bleu_score)

    return {k: round(v.compute()['value'] * 100, 6) for k, v in score_dict.items()}


class NlgMetrics(EvalMetrics):

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        # nlg: Natural Language Generation
        preds, labels = eval_prediction.predictions, eval_prediction.label_ids
        new_preds, new_labels = [], []
        for i in range(preds.shape[0]):
            new_preds.append(Serializer.from_tensor(preds[i]))
            new_labels.append(Serializer.from_tensor(labels[i]))
        return compute_rouge_bleu(new_preds, new_labels)
