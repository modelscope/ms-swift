"""Intern-Decision hard-label supervision for MS-SWIFT; no framework edits."""
import os
from collections.abc import Mapping
from copy import deepcopy

from swift.template import register_template
from swift.template.register import TEMPLATE_MAPPING
from swift.template.templates.qwen import Qwen3_5Template


class DecisionTemplate(Qwen3_5Template):

    def _data_collator(self, batch, *, padding_to=None):
        # Limit NPU compilation shapes without truncating any evidence or labels.
        multiple = int(os.environ.get('DECISION_PAD_MULTIPLE', '0'))
        return super()._data_collator(batch, padding_to=padding_to or multiple or None)

    def _encode(self, inputs):
        if inputs.images or inputs.videos:
            raise ValueError('This first reproduction is text-only')
        messages = deepcopy(inputs.messages)
        if inputs.system is not None:
            messages.insert(0, {'role': 'system', 'content': inputs.system})
        # Render with the checkpoint tokenizer, exactly as the reference inference
        # path does. Equivalent decoded text can still have different BPE boundaries.
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        if isinstance(ids, Mapping):
            ids = ids['input_ids']
        encoded = {'input_ids': ids, 'labels': None}
        targets = inputs.extra_kwargs.get('decision_targets')
        if targets is None:
            raise ValueError('Decision training requires explicit target symbols')
        marker = self.tokenizer.encode('<decision>', add_special_tokens=False)
        if len(marker) != 1:
            raise ValueError('Register <decision> with --new_special_tokens')
        positions = [i for i, token in enumerate(encoded['input_ids']) if token == marker[0]]
        if len(positions) != len(targets) or not positions:
            raise ValueError('Decision markers and targets do not match')
        labels = [-100] * len(encoded['input_ids'])
        for position, symbol in zip(positions, targets):
            token = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(token) != 1 or position == 0:
                raise ValueError('Invalid single-token decision target')
            labels[position] = token[0]
        # SWIFT/HF causal loss performs the shift. Do not shift here.
        encoded['labels'] = labels
        encoded.pop('loss_scale', None)
        return encoded

    def encode(self, *args, **kwargs):
        result = super().encode(*args, **kwargs)
        for item in result if isinstance(result, list) else [result]:
            item.pop('_extra_kwargs', None)
        return result


meta = deepcopy(TEMPLATE_MAPPING['qwen3_5'])
meta.template_type = 'intern_decision_training'
meta.template_cls = DecisionTemplate
register_template(meta)
