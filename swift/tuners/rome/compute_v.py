# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from kmeng01/rome.
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from modelscope import AutoTokenizer

from swift.utils.logger import get_logger
from .nethook import TraceDict, set_requires_grad
from .repr_tools import get_reprs_at_idxs, get_reprs_at_word_tokens, get_words_idxs_in_templates
from .rome_hparams import ROMEHyperParams

logger = get_logger()


def compute_v(model: torch.nn.Module,
              tokenizer: AutoTokenizer,
              request: Dict,
              hparams: ROMEHyperParams,
              layer: int,
              left_vector: torch.Tensor,
              context_templates: List[str],
              batch_first: bool = True) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    logger.info('Computing right vector (v)')

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request['prompt']) + request['target'] for context in context_templates
    ], ['{} is a', '{}是一个']
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tokenizer(
        [prompt.format(request['subject']) for prompt in all_prompts],
        return_tensors='pt',
        padding=True,
        return_token_type_ids=False,
    ).to(model.device)

    # Compute rewriting targets
    rewriting_targets = torch.tensor(
        -100, device=model.device).repeat(len(rewriting_prompts), *input_tok['input_ids'].shape[1:])

    prompt = context_templates[0].format(request['prompt'])
    prompt_full = prompt + request['target']
    target_len = len(tokenizer.tokenize(prompt_full)) - len(tokenizer.tokenize(prompt))
    for i in range(len(rewriting_prompts)):
        rewriting_targets[i, -target_len - 1:-1] = input_tok['input_ids'][i, -target_len:].clone()

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(prompt, request['subject'], tokenizer, hparams.fact_token, verbose=(i == 0))
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    logger.info(f'Rewrite layer is {layer}')

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    hidden_size = model.config.n_embd if hasattr(model.config, 'n_embed') else model.config.hidden_size
    delta = torch.zeros((hidden_size, ), requires_grad=True, device=model.device)
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        # Store initial value of the vector of interest
        if target_init is None:
            logger.info('Recording initial value of v*')
            # Initial value is recorded for the clean sentence
            target_init = cur_out[0, lookup_idxs[0]].detach().clone()

        for i, idx in enumerate(lookup_idxs):
            if batch_first:
                cur_out[i, idx, :] += delta
            else:
                cur_out[idx, i, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with TraceDict(
                module=model,
                layers=[
                    hparams.mlp_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
        ) as _:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [logits[i - len(kl_prompts), idx, :] for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_len
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction='batchmean')
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init)**2)
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        logger.info(f'loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + '
                    f'{np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} '
                    f"avg prob of [{request['target']}] "
                    f'{torch.exp(-nll_loss_each).mean().item()}')
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tokenizer,
        layer,
        context_template=request['prompt'],
        word=request['subject'],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
        batch_first=batch_first)

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    logger.info(f'Delta norm: {(target - cur_output).norm().item()}')
    logger.info(f'Change in target norm: {target_init.norm().item()} to {target.norm().item()} => '
                f'{(target.norm() - target_init.norm()).item()}')
    logger.info(f'Division Factor: {torch.dot(cur_input, left_vector).item()}')
    logger.info(f'Right vector norm: {right_vector.norm()}')

    return right_vector


def get_module_input_output_at_word(model: torch.nn.Module,
                                    tok: Any,
                                    layer: int,
                                    context_template: str,
                                    word: str,
                                    module_template: str,
                                    fact_token_strategy: str,
                                    batch_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model, tokenizer=tok, layer=layer, module_template=module_template, batch_first=batch_first)
    if 'subject_' in fact_token_strategy and fact_token_strategy.index('subject_') == 0:
        subtoken = fact_token_strategy[len('subject_'):]
        l_input, l_output = get_reprs_at_word_tokens(
            track='both',
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == 'last':
        l_input, l_output = get_reprs_at_idxs(
            track='both',
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f'fact_token={fact_token_strategy} not recognized')

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: Any,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    if fact_token_strategy == 'last':
        ret = -1
    elif ('subject_' in fact_token_strategy and fact_token_strategy.index('subject_') == 0):
        ret = get_words_idxs_in_templates(
            tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len('subject_'):],
        )[0][0]
    else:
        raise ValueError(f'fact_token={fact_token_strategy} not recognized')

    sentence = prompt.format(subject)
    if verbose:
        logger.info(
            f'Lookup index found: {ret} | Sentence: {sentence} | Token:'
            + tok.decode(tok(sentence)['input_ids'][ret]), )

    return ret
