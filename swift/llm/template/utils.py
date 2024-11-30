# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
from transformers import (LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase, StoppingCriteria,
                          StoppingCriteriaList)

from swift.llm import History

Prompt = List[Union[str, List[int], List[str]]]
Word = Union[str, List[int]]
Context = Word


class ContextType:
    RESPONSE = 'response'
    SUFFIX = 'suffix'
    OTHER = 'other'


class StopWordsCriteria(StoppingCriteria):
    """Adding extra stop words in template to prevent unstoppable generation
        Like suffixes and chat seps in the template.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_words: List[Word], **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.tokenizer_kwargs = tokenizer_kwargs
        self.start_idx = -1
        self.is_done = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        if self.start_idx == -1:
            self.start_idx = len(input_ids[0]) - 1
            self.is_done = torch.full((input_ids.shape[0], ), False, device=input_ids.device, dtype=torch.bool)
        tokenizer = self.tokenizer
        stop_words = self.stop_words
        # [-20:]: Assuming the end tokens do not exceed 20 tokens,
        #   to avoid input_ids being too long and affecting efficiency.
        text_list = tokenizer.batch_decode(input_ids[:, self.start_idx:][-20:], **self.tokenizer_kwargs)
        for i, text in enumerate(text_list):
            if self.is_done[i]:
                continue
            is_finished = False
            for stop_word in stop_words:
                if isinstance(stop_word, str):
                    if stop_word in text:
                        is_finished = True
                else:  # list
                    assert len(stop_word) > 0
                    if input_ids[i][-len(stop_word):].tolist() == stop_word:
                        is_finished = True
            self.is_done[i] = is_finished
        return self.is_done


def fetch_one(element: Union[Tuple, List, Set, Dict, Any], item_type: Optional[Type] = None) -> Any:
    if isinstance(element, (tuple, set, list)):
        for ele in element:
            out = fetch_one(ele)
            if out and (item_type is None or isinstance(out, item_type)):
                return out
    elif isinstance(element, dict):
        return fetch_one(list(element.values()))
    else:
        return element


def findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
    """Find the index of a token in the token_list."""
    if isinstance(sub_token_list, int):
        sub_token_list = [sub_token_list]
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(sub_token_list[0], idx + 1)
            if len(sub_token_list) == 1 or sub_token_list == token_list[idx:idx + len(sub_token_list)]:
                res.append(idx)
    except ValueError:
        pass
    return res


def replace_img_tag(query: str,
                    response: Optional[str],
                    history: History,
                    replace_token: str,
                    pattern=r'<img>(.+?)</img>') -> Tuple[str, Optional[str], History, List[str]]:
    images_path = []
    new_history = []
    history = history.copy()
    history.append([query, response])
    for i, h in enumerate(history):
        new_h = []
        for content in h:
            if content is None:
                new_h.append(content)
            else:
                images_path += re.findall(pattern, content)
                new_h.append(re.sub(pattern, replace_token, content))
        new_history.append(new_h)
    return (*new_history[-1], new_history[:-1], images_path)


def align_image_inputs(input_ids: List[int], labels: List[int], new_input_ids,
                       image_token: int) -> Tuple[List[int], List[int]]:
    if isinstance(new_input_ids, torch.Tensor):
        new_input_ids = new_input_ids.tolist()

    # Find the tokens after the image_token in input_ids, and then align them.
    i, j = 0, 0
    while i < len(input_ids):
        x = input_ids[i]
        if x == image_token:
            assert i + 1 < len(input_ids), f'input_ids[-10:]: {input_ids[-10:]}'
            assert i - 1 >= 0, f'input_ids[:10]: {input_ids[:10]}'
            # [1, 2, 3(i-1), image_token(i), 4(i+1) ,5, 6]
            # [1, 2, 3(j_begin), a(j'), a, a, a, 4(j) ,5, 6]
            j_begin = j - 1
            for k in range(5):  # Increase robustness.
                if j_begin + k < len(new_input_ids) and new_input_ids[j_begin + k] == input_ids[i - 1]:
                    j_begin += k
                    break
                if j_begin - k >= 0 and new_input_ids[j_begin - k] == input_ids[i - 1]:
                    j_begin -= k
                    break
            else:
                raise ValueError(f'new_input_ids: {new_input_ids}, input_ids: {input_ids}')
            j_begin += 1
            while j < len(new_input_ids) and new_input_ids[j] != input_ids[i + 1]:
                j += 1
            input_ids = input_ids[:i] + new_input_ids[j_begin:j] + input_ids[i + 1:]
            if labels:
                labels = labels[:i] + [-100] * (j - j_begin) + labels[i + 1:]
            i += j - j_begin
        else:
            j += 1
        i += 1
    return input_ids, labels


def gather_list(batch: List[Dict[str, Any]], attr_name: str) -> Optional[List[Any]]:
    # List[Tensor] ->  List[Tensor]
    res = []
    for b in batch:
        if b.get(attr_name) is not None:
            res += b.pop(attr_name)
    return res
