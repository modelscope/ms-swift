# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Optional, Tuple

import torch.nn as nn
from peft import PeftModel
from transformers import Trainer

from swift.utils import get_logger
from .base import OptimizerCallback

logger = get_logger()


def get_param_startswith(model,
                         chosen_prefix: List[str],
                         rejected_prefix: Optional[List[str]] = None) -> List[Tuple[str, nn.Parameter]]:
    chosen_prefix = chosen_prefix or []
    rejected_prefix = rejected_prefix or []
    res = []
    if not chosen_prefix:
        return res
    is_peft_model = isinstance(model, PeftModel)
    if is_peft_model:
        model = model.model
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_rejected = False
        for prefix in rejected_prefix:
            if n.startswith(prefix):
                is_rejected = True
                break
        if is_rejected:
            continue
        for prefix in chosen_prefix:
            if n.startswith(prefix):
                if is_peft_model:
                    n = f'base_model.model.{n}'
                res.append((n, p))
                break
    return res


class MultimodalOptimizerCallback(OptimizerCallback):

    def create_optimizer(self):
        args = self.args
        model = self.trainer.model
        """ViT/Aligner/LLM use different learning rates."""
        decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
        model_arch = model.model_meta.model_arch
        vit_parameters = get_param_startswith(model, model_arch.vision_tower, model_arch.aligner)
        aligner_parameters = get_param_startswith(model, model_arch.aligner)
        llm_parameters = get_param_startswith(model, model_arch.language_model)
        optimizer_grouped_parameters = []
        vit_lr = args.vit_lr if args.vit_lr is not None else args.learning_rate
        aligner_lr = args.aligner_lr if args.aligner_lr is not None else args.learning_rate
        logger.info(f'vit_lr: {vit_lr}, aligner_lr: {aligner_lr}, llm_lr: {args.learning_rate}')
        for lr, parameters in zip([vit_lr, aligner_lr, args.learning_rate],
                                  [vit_parameters, aligner_parameters, llm_parameters]):
            for use_wd, wd in zip([False, True], [0., args.weight_decay]):
                if use_wd:
                    params = [p for n, p in parameters if n in decay_parameters]
                else:
                    params = [p for n, p in parameters if n not in decay_parameters]
                if not params:
                    continue
                optimizer_grouped_parameters.append({
                    'params': params,
                    'weight_decay': wd,
                    'lr': lr,
                })
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
        return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
