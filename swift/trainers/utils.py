# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import inspect
from types import FunctionType, MethodType
from typing import List, Union

from torch.nn import Module


def can_return_loss(model: Module) -> List[str]:
    """Check if a given model can return loss."""
    signature = inspect.signature(model.forward)
    for p in signature.parameters:
        if p == 'return_loss' and signature.parameters[p].default is True:
            return True
    return False


def find_labels(model: Module):
    """Find the labels used by a given model."""
    model_name = model.__class__.__name__
    signature = inspect.signature(model.forward)
    if 'QuestionAnswering' in model_name:
        return [
            p for p in signature.parameters
            if 'label' in p or p in ('start_positions', 'end_positions')
        ]
    else:
        return [p for p in signature.parameters if 'label' in p]


def get_function(
        method_or_function: Union[MethodType, FunctionType]) -> FunctionType:
    if isinstance(method_or_function, MethodType):
        method_or_function = method_or_function.__func__
    return method_or_function
