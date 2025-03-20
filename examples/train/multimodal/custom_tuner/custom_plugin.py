import deepspeed
import torch
from transformers import PreTrainedModel, Trainer

from swift.plugin import Tuner, extra_tuners, optimizers_map
from swift.tuners import LoraConfig, Swift
from transformers.pytorch_utils import id_tensor_storage

class CustomTuner(Tuner):

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ):
        with deepspeed.zero.GatheredParameters(model.parameters()):
            model.merge_adapter()
            state_dict = model.state_dict()
            # Remove base_model and base_layer prefixes
            state_dict = {
                k.removeprefix('base_model.model.').replace('.base_layer', ''): v
                for k, v in state_dict.items()
            }
            # Remove values with adapter prefix (example: "_lora")
            state_dict = {k: v for k, v in state_dict.items() if model.prefix not in k}
            # When module to save, remove its prefix and discard the original module
            state_dict = {
                k.replace('modules_to_save.default.', ''): v
                for k, v in state_dict.items() if 'original_module' not in k
            }
            if id_tensor_storage(state_dict['lm_head.weight']) == id_tensor_storage(state_dict['model.embed_tokens.weight']):
                # tie weight
                state_dict.pop('lm_head.weight')
            PreTrainedModel.save_pretrained(
                model, save_directory, state_dict=state_dict, safe_serialization=safe_serialization)
            model.unmerge_adapter()

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module):
        target_regex = r'^model.layers.*'
        lora_config = LoraConfig(
            task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=target_regex)
        model = Swift.prepare_model(model, lora_config)
        model.visual.requires_grad_(True)  # vit & merger
        return model


def create_custom_optimizer(args, model, dataset):
    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    optimizer_grouped_parameters = [
        # vit & merger
        {
            'params':
            [p for n, p in model.named_parameters() if ('.visual.' in n and n in decay_parameters and p.requires_grad)],
            'weight_decay':
            args.weight_decay,
            'lr':
            2e-6,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if ('.visual.' in n and n not in decay_parameters and p.requires_grad)
            ],
            'weight_decay':
            0.0,
            'lr':
            2e-6,
        },
        # llm
        {
            'params': [
                p for n, p in model.named_parameters()
                if ('.visual.' not in n and n in decay_parameters and p.requires_grad)
            ],
            'weight_decay':
            args.weight_decay,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if ('.visual.' not in n and n not in decay_parameters and p.requires_grad)
            ],
            'weight_decay':
            0.0,
        },
    ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


extra_tuners['custom'] = CustomTuner
optimizers_map['custom'] = create_custom_optimizer
