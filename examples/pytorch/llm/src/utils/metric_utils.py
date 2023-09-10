# Copyright (c) Alibaba, Inc. and its affiliates.

import jieba
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge.rouge import Rouge

from swift import get_logger

logger = get_logger()


def compute_nlg_metrics(tokenizer, prediction):
    preds, labels = prediction[0], prediction[1]

    score_dict = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}

    def _decode(tokens, ignore_pad_token_for_loss=False):
        if ignore_pad_token_for_loss:
            tokens = np.where(tokens != -100, tokens, tokenizer.pad_token_id)
        tokens = np.where(tokens < tokenizer.vocab_size, tokens,
                          tokenizer.pad_token_id)
        return [
            t
            for t in tokenizer.batch_decode(tokens, skip_special_tokens=True)
        ]

    for pred, label in zip(preds, labels):
        pred = ''.join(_decode(pred, False))
        label = ''.join(_decode(label, True))
        hypothesis = list(jieba.cut(pred))
        if len(hypothesis) == 0 or ''.join(hypothesis) == '.':
            hypothesis = [tokenizer.decode(tokenizer.eos_token_id)]
        reference = list(jieba.cut(label))
        try:
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis),
                                      ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v['f'] * 100, 4))
            bleu_score = sentence_bleu(
                [list(label)],
                list(pred),
                smoothing_function=SmoothingFunction().method3)
            score_dict['bleu-4'].append(round(bleu_score * 100, 4))
        except Exception as e:
            logger.error(e)
            logger.error(f'eval error {hypothesis}, {reference}')

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict
