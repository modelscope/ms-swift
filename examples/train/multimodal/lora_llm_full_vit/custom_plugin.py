import os
from typing import Optional

import safetensors.torch
import torch
from transformers import Trainer

from swift.llm import deep_getattr, get_model_arch, get_multimodal_target_regex
from swift.plugin import Tuner, extra_tuners, optimizers_map
from swift.tuners import LoraConfig, Swift
from swift.utils import get_logger

logger = get_logger()


def is_vit_param(model_arch, parameter_name: str) -> bool:
    for module_prefix in model_arch.vision_tower + model_arch.aligner:
        if f'.{module_prefix}.' in parameter_name:
            return True
    return False


class CustomTuner(Tuner):
    """Full-parameter training of ViT while LoRA training LLM"""

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        model = Swift.from_pretrained(model, model_id, **kwargs)
        state_dict = safetensors.torch.load_file(os.path.join(model_id, 'vit.safetensors'))
        model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        if state_dict is None:
            state_dict = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    state_dict[n] = p.detach().cpu()
        model.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization, **kwargs)
        # vit
        model_arch = get_model_arch(model.model_meta.model_arch)
        state_dict = {k: v for k, v in state_dict.items() if is_vit_param(model_arch, k)}
        safetensors.torch.save_file(
            state_dict, os.path.join(save_directory, 'vit.safetensors'), metadata={'format': 'pt'})

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        model_arch = get_model_arch(model.model_meta.model_arch)
        target_regex = get_multimodal_target_regex(model)
        logger.info(f'target_regex: {target_regex}')
        lora_config = LoraConfig(
            task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=target_regex)
        model = Swift.prepare_model(model, lora_config)
        for module_prefix in model_arch.vision_tower + model_arch.aligner:
            deep_getattr(model, module_prefix).requires_grad_(True)
        return model


def create_custom_optimizer(args, model, dataset):
    """ViT and LLM use different learning rates."""
    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    model_arch = get_model_arch(model.model_meta.model_arch)
    vit_parameters = [(n, p) for n, p in model.named_parameters() if is_vit_param(model_arch, n) and p.requires_grad]
    llm_parameters = [(n, p) for n, p in model.named_parameters()
                      if not is_vit_param(model_arch, n) and p.requires_grad]
    optimizer_grouped_parameters = [
        # vit & merger
        {
            'params': [p for n, p in vit_parameters if n in decay_parameters],
            'weight_decay': args.weight_decay,
            'lr': 0.1 * args.learning_rate,  # 1e-5
        },
        {
            'params': [p for n, p in vit_parameters if n not in decay_parameters],
            'weight_decay': 0.0,
            'lr': 0.1 * args.learning_rate,
        },
        # llm
        {
            'params': [p for n, p in llm_parameters if n in decay_parameters],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in llm_parameters if n not in decay_parameters],
            'weight_decay': 0.0,
        },
    ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


extra_tuners['custom'] = CustomTuner
optimizers_map['custom'] = create_custom_optimizer
