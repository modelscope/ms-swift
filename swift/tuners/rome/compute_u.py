# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from kmeng01/rome.
from typing import Dict, List

import torch
from modelscope import AutoTokenizer

from swift.utils.logger import get_logger
from .repr_tools import get_reprs_at_idxs, get_reprs_at_word_tokens
from .rome_hparams import ROMEHyperParams

logger = get_logger()


def compute_u(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
    batch_first=True,
) -> torch.Tensor:
    """
    Computes the left vector used in constructing the rank-1 update matrix.
    """

    logger.info('Computing left vector (u)...')

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track='in',
        batch_first=batch_first,
    )
    if 'subject_' in hparams.fact_token and hparams.fact_token.index('subject_') == 0:
        word = request['subject']
        logger.info(f'Selected u projection object {word}')
        cur_repr = get_reprs_at_word_tokens(
            context_templates=[templ.format(request['prompt']) for templ in context_templates],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len('subject_'):],
            **word_repr_args,
        ).mean(0)
    elif hparams.fact_token == 'last':
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = get_reprs_at_idxs(
            contexts=[templ.format(request['prompt'].format(request['subject'])) for templ in context_templates],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        logger.info('Selected u projection token with last token')
    else:
        raise ValueError(f'fact_token={hparams.fact_token} not recognized')

    # Apply inverse second moment adjustment
    u = cur_repr
    return u / u.norm()
