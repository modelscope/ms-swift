# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from ..base import Template


class RLHFTemplateMixin:

    def encode(self: Template,
               example: Dict[str, Any],
               streaming: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        template_encode = self._old_encode
        inputs = {}
        tokenizer_kwargs = {}
        chosen_example, rejected_example = example, example.copy()
        rejected_example['response'] = example['rejected_response']
        if streaming:
            chosen_inputs, chosen_tokenizer_kwargs = template_encode(chosen_example), {}
            rejected_inputs, rejected_tokenizer_kwargs = template_encode(rejected_example), {}
        else:
            chosen_inputs, chosen_tokenizer_kwargs = template_encode(chosen_example)
            rejected_inputs, rejected_tokenizer_kwargs = template_encode(rejected_example)

        if len(chosen_inputs) == 0 or len(rejected_inputs) == 0:
            return {}, {}
        for suffix, res in zip(['inputs', 'tokenizer_kwargs'], [inputs, tokenizer_kwargs]):
            for prefix in ['chosen', 'rejected']:
                data = locals()[f'{prefix}_{suffix}']
                for k, v in data.items():
                    res[f'{prefix}_{k}'] = v
        return inputs, tokenizer_kwargs

    def data_collator(self: Template, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        _data_collator = self._old_data_collator
        new_batch = []
        for prefix in ['chosen_', 'rejected_']:
            for inputs in batch:
                new_inputs = {}
                for k, v in inputs.items():
                    if k.startswith(prefix):
                        new_k = k[len(prefix):]
                        new_inputs[new_k] = inputs[k]
                if len(new_inputs) > 0:
                    new_batch.append(new_inputs)
        assert len(new_batch) in {0, len(batch) * 2}, f'new_batch: {new_batch}'
        return _data_collator(new_batch or batch, padding_to)


class KTOTemplateMixin:

    def encode(self: Template,
               example: Dict[str, Any],
               streaming: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = self._old_encode(example, streaming)
        if len(inputs) > 0:
            inputs['label'] = example['label']
        return inputs, tokenizer_kwargs

    def data_collator(self: Template, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        for prefix in ['', 'KL_']:
            new_batch = []
            for b in batch:
                new_batch.append({'input_ids': b[f'{prefix}input_ids'], 'labels': b[f'{prefix}labels']})
            for k, v in self._old_data_collator(new_batch, padding_to).items():
                res[f'{prefix}completion_{k}'] = v
        res['label'] = [b['label'] for b in batch]
        return res
