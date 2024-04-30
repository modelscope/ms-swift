# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from kmeng01/rome.
"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from typing import Any, Callable, List, Tuple, Union

import torch
from modelscope import AutoTokenizer

from .nethook import Trace


def get_reprs_at_word_tokens(
    model: torch.nn.Module,
    tokenizer: Any,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = 'in',
    batch_first: bool = True,
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tokenizer, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tokenizer,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
        batch_first,
    )


def get_words_idxs_in_templates(tokenizer: AutoTokenizer, context_templates: List[str], words: List[str],
                                subtoken: str) -> List:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(tmp.count('{}') == 1
               for tmp in context_templates), 'We currently do not support multiple fill-ins for context'

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index('{}') for tmp in context_templates]
    prefixes, suffixes = [tmp[:fill_idxs[i]] for i, tmp in enumerate(context_templates)
                          ], [tmp[fill_idxs[i] + 2:] for i, tmp in enumerate(context_templates)]

    lens = []
    for prefix, word, suffix in zip(prefixes, words, suffixes):
        prefix_token = tokenizer.encode(prefix)
        prefix_word_token = tokenizer.encode(prefix + word)
        prefix_word_suffix_token = tokenizer.encode(prefix + word + suffix)
        suffix_len = len(prefix_word_suffix_token) - len(prefix_word_token)

        # Compute indices of last tokens
        if subtoken == 'last' or subtoken == 'first_after_last':
            lens.append([
                len(prefix_word_token) -
                (1 if subtoken == 'last' or suffix_len == 0 else 0) - len(prefix_word_suffix_token)
            ])
        elif subtoken == 'first':
            lens.append([len(prefix_token) - len(prefix_word_suffix_token)])
        else:
            raise ValueError(f'Unknown subtoken type: {subtoken}')
    return lens


def get_reprs_at_idxs(
    model: torch.nn.Module,
    tokenizer: Callable,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = 'in',
    batch_first: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i:i + n], idxs[i:i + n]

    assert track in {'in', 'out', 'both'}
    both = track == 'both'
    tin, tout = (
        (track == 'in' or both),
        (track == 'out' or both),
    )
    module_name = module_template.format(layer)
    to_return = {'in': [], 'out': []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if isinstance(cur_repr, tuple) else cur_repr
        if not batch_first:
            cur_repr = cur_repr.transpose(0, 1)
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for batch_contexts, batch_idxs in _batch(n=512):
        contexts_tok = tokenizer(
            batch_contexts, padding=True, return_token_type_ids=False,
            return_tensors='pt').to(next(model.parameters()).device)

        with torch.no_grad():
            with Trace(
                    module=model,
                    layer=module_name,
                    retain_input=tin,
                    retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            _process(tr.input, batch_idxs, 'in')
        if tout:
            _process(tr.output, batch_idxs, 'out')

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return['in'] if tin else to_return['out']
    else:
        return to_return['in'], to_return['out']
