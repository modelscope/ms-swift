# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from kmeng01/rome.
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn as nn

from swift import SwiftConfig
from swift.tuners.utils import SwiftAdapter, SwiftOutput
from swift.utils import get_logger
from .compute_u import compute_u
from .compute_v import compute_v
from .context_template import context_template
from .nethook import get_parameter
from .rome_hparams import ROMEHyperParams

CONTEXT_TEMPLATES_CACHE = None

logger = get_logger()


@dataclass
class RomeConfig(SwiftConfig):
    """
    The configuration class for the ROME module.
    This adapter can be used to inject/modify knowledge to models, without any training.
    ROME: [Rank-One Editing of Encoder-Decoder Models](https://arxiv.org/abs/2211.13317)
    Args:
        model_type(`str`): The model type, now support llama-7b/llama-13b
        tokenizer(`AutoTokenizer`): The tokenizer
        knowledge(`List[Dict]`): The knowledge to be injected to the model.
            format:
            >>> [
            >>>     {
            >>>         "prompt": "{} was the founder of",
            >>>         "subject": "Steve Jobs",
            >>>         "target": "Microsoft"
            >>>     }
            >>> ]
    """
    model_type: str = field(default=None, metadata={'help': 'The model type'})

    tokenizer: Any = field(default=None, metadata={'help': 'The tokenizer matching this model'})

    knowledge: List[Dict] = field(default=False, metadata={'help': 'The knowledge to be used'})

    batch_first: bool = field(default=True, metadata={'help': 'Batch at the first dimension or not'})

    def __post_init__(self):
        from swift.tuners.mapping import SwiftTuners
        self.swift_type = SwiftTuners.ROME

    @property
    def __dict__(self):
        _dict = super(RomeConfig, self).__dict__
        _dict.pop('tokenizer')
        return _dict


class Rome(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: RomeConfig, adapter_name: str):
        """
        Applies the selected model editing algorithm. Generates text both before and after
        for comparison of model behavior. Returns the updated model and the original values of
        weights that were changed.
        """
        modified_keys = set()
        if config.tokenizer is not None:
            for param in model.parameters():
                param.requires_grad = True

            hparams = ROMEHyperParams.from_name(config.model_type)
            modified_keys = apply_rome_to_model(model, config.tokenizer, config.knowledge, hparams, config.batch_first)

        def state_dict_callback(state_dict, adapter_name):
            return {key: value for key, value in state_dict.items() if key in modified_keys}

        def mark_trainable_callback(model):
            pass

        return SwiftOutput(
            config=config, state_dict_callback=state_dict_callback, mark_trainable_callback=mark_trainable_callback)

    @staticmethod
    def has_additional_modules():
        return False


def apply_rome_to_model(
    model: torch.nn.Module,
    tokenizer: Any,
    knowledge: List[Dict],
    hparams: ROMEHyperParams,
    batch_first: bool,
) -> Set:
    """Apply ROME to a model

    Args:
        model(`torch.nn.Module`): The model instance.
        tokenizer(`Any`): The tokenizer.
        knowledge(`List[Dict]`): The knowledge to be filled into the model.
        hparams(`ROMEHyperParams`): The hyperparameter of ROME
        batch_first(`bool`): Batch first of not.
    """
    modified_keys = set()
    for i, request in enumerate(knowledge):
        deltas = execute_rome(model, tokenizer, request, hparams, batch_first)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                w[...] += upd_matrix
        modified_keys.update(set(deltas.keys()))
    return modified_keys


def execute_rome(
    model: torch.nn.Module,
    tok: Any,
    knowledge: Dict,
    hparams: ROMEHyperParams,
    batch_first: bool,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(knowledge)
    logger.info(f'Executing ROME algorithm for the update: '
                f"[{request['prompt'].format(request['subject'])}] -> [{request['target']}]")

    # Retrieve weights that user desires to change
    weights = {
        f'{hparams.rewrite_module_tmp.format(layer)}.weight':
        get_parameter(model, f'{hparams.rewrite_module_tmp.format(layer)}.weight')
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            context_template,
            batch_first=batch_first,
        )
        logger.info(f'Left vector shape: {left_vector.shape}')
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            context_template,
            batch_first=batch_first,
        )
        logger.info(f'Right vector shape: {right_vector.shape}')
        right_vector = right_vector.to(left_vector.dtype)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f'{hparams.rewrite_module_tmp.format(layer)}.weight'
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    logger.info(f'Deltas successfully computed for {list(weights.keys())}')

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError('Update matrix computed by ROME does not match original weight shape. '
                         'Check for bugs in the code?')
