# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
import shutil
import time
from copy import copy
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import safetensors
import torch
import torch.nn as nn
import transformers
from datasets import Dataset as HfDataset
from modelscope import check_local_model_is_latest
from packaging import version
from peft import PeftModel
from torch.nn import Module
from transformers import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model
from transformers.trainer import Trainer, TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_torch_npu_available

from swift.hub import get_hub
from swift.llm import Template
from swift.plugin import extra_tuners
from swift.tuners import SwiftModel
from swift.utils import get_logger, is_mp_ddp
from .arguments import TrainingArguments
from .optimizers.galore import create_optimizer_and_scheduler
from .utils import can_return_loss, find_labels, get_function, is_instance_of_ms_model

try:
    from trl import AutoModelForCausalLMWithValueHead
except (ImportError, RuntimeError):
    AutoModelForCausalLMWithValueHead = None

logger = get_logger()


class SwiftMixin:

    def __init__(
            self,
            model: Union[PreTrainedModel, Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[HfDataset] = None,
            eval_dataset: Optional[Union[HfDataset, Dict[str, HfDataset]]] = None,
            template: Optional[Template] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_loss_func: Optional[Callable] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor],
                                                             torch.Tensor]] = None) -> None:
        if args.check_model and hasattr(model, 'model_dir'):
            check_local_model_is_latest(
                model.model_dir, user_agent={
                    'invoked_by': 'local_trainer',
                    'third_party': 'swift',
                })
        self._custom_metrics = {}
        self.template = template
        self.max_memory = 0
        self.hub = get_hub()
        if args.sequence_parallel_size > 1:
            from swift.trainers.xtuner import init_sequence_parallel_xtuner
            init_sequence_parallel_xtuner(args.sequence_parallel_size)

        with self.hub.patch_hub():
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=template.tokenizer,
                model_init=model_init,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                optimizers=optimizers,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics)

        self.compute_loss_func = compute_loss_func
        if get_function(model.__class__.forward) is not get_function(model.forward):
            self.label_names = find_labels(model) or ['labels']
            self.can_return_loss = can_return_loss(model)
        self.start_time = time.time()

    def _save_initial_model(self, output_dir):
        # pissa/olora/lora-ga
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            config = model.peft_config.get('default')
            init_lora_weights = getattr(config, 'init_lora_weights', None)
            if (isinstance(init_lora_weights, str)
                    and any(s in init_lora_weights for s in ('pissa', 'olora', 'lora-ga'))):
                config.init_lora_weights = True
                model.save_pretrained(os.path.join(output_dir, 'initial_model'))
                config.init_lora_weights = init_lora_weights

    def _save_converted_model(self, output_dir):
        # pissa/olora/lora-ga
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            config = model.peft_config.get('default')
            init_lora_weights = getattr(config, 'init_lora_weights', None)
            if isinstance(init_lora_weights, str):
                config = copy(config)
                os.makedirs(os.path.join(output_dir, 'converted'), exist_ok=True)
                if 'lora-ga' in init_lora_weights:
                    try:
                        from lora_ga.entrypoint import LoraGAContext
                        with LoraGAContext(model):
                            model.save_pretrained(
                                os.path.join(output_dir, 'converted', 'default'),
                                path_initial_model_for_weight_conversion=os.path.join(
                                    os.path.dirname(output_dir), 'initial_model'),
                            )
                            model.peft_config['default'] = config
                    except ImportError as e:
                        error_message = """
                        Since 'LoRA-GA' is not implemented by PEFT, you will need to install it directly from GitHub.
                        Command: 'pip install git+https://github.com/lxline/LoRA-GA.git'.
                        """
                        logger.info(error_message)
                        raise RuntimeError(error_message) from e
                elif 'pissa' in init_lora_weights or 'olora' in init_lora_weights:
                    model.save_pretrained(
                        os.path.join(output_dir, 'converted', 'default'),
                        path_initial_model_for_weight_conversion=os.path.join(
                            os.path.dirname(output_dir), 'initial_model'),
                    )
                    model.peft_config['default'] = config

    def _load_optimizer_and_scheduler(self, *args, **kwargs):
        super()._load_optimizer_and_scheduler(*args, **kwargs)
        if is_mp_ddp():
            # fix mp+ddp adamw
            for v in self.optimizer.state.values():
                if 'step' in v:
                    # not on the same device
                    device_set = set([t.device for t in v.values()]) - {v['step'].device, torch.device('cpu')}
                    if len(device_set) >= 1:
                        v['step'] = v['step'].to('cpu')

    def _save_model(self, output_dir: Optional[str] = None, state_dict=None):
        # model
        supported_classes = (SwiftModel, PreTrainedModel, PeftModel)
        if AutoModelForCausalLMWithValueHead is not None:
            supported_classes = supported_classes + (AutoModelForCausalLMWithValueHead, )
        save_safetensors = self.args.save_safetensors
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            _unwrap_model = unwrap_model(self.model)
            if isinstance(_unwrap_model, supported_classes):
                _unwrap_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
            else:
                logger.info('Trainer.model is not a `PreTrainedModel`, only saving its state dict.')
                if save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, 'model.safetensors'))
                else:
                    torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
        elif AutoModelForCausalLMWithValueHead and isinstance(self.model, AutoModelForCausalLMWithValueHead):
            # save reward model
            state_dict = self.model.state_dict()
            decoder_state_dict, v_head_state_dict = {}, {}
            for name, param in state_dict.items():
                if name.startswith('v_head.'):
                    v_head_state_dict[name] = param
                else:
                    decoder_state_dict[name.replace('pretrained_model.', '', 1)] = param
            self.model.pretrained_model.save_pretrained(
                output_dir, state_dict=decoder_state_dict or None, safe_serialization=save_safetensors)
            if save_safetensors:
                from safetensors.torch import save_file
                save_file(
                    v_head_state_dict, os.path.join(output_dir, 'value_head.safetensors'), metadata={'format': 'pt'})
            else:
                torch.save(v_head_state_dict, os.path.join(output_dir, 'value_head.bin'))
        elif is_instance_of_ms_model(self.model):
            PreTrainedModel.save_pretrained(
                self.model, output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
        elif self.args.train_type in extra_tuners:
            extra_tuners[self.args.train_type].save_pretrained(
                self.model, output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=save_safetensors)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Compatible with swift and peft"""
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._save_model(output_dir, state_dict)
        # training_args.bin
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self._save_converted_model(output_dir)
        # args.json
        args_path = os.path.join(os.path.dirname(output_dir), 'args.json')
        if os.path.exists(args_path):
            shutil.copy(args_path, os.path.join(output_dir, 'args.json'))
        # predict.jsonl
        predict_jsonl = os.path.join(os.path.dirname(output_dir), 'predict.jsonl')
        if os.path.exists(predict_jsonl):
            shutil.move(predict_jsonl, os.path.join(output_dir, 'predict.jsonl'))

        is_adapter = isinstance(self.model, (SwiftModel, PeftModel))
        # tokenizer
        if not is_adapter:
            from swift.llm import save_checkpoint
            additional_saved_files = self.model.model_meta.additional_saved_files
            save_checkpoint(None, self.template.processor, output_dir, additional_saved_files=additional_saved_files)

    def _fix_zero3_gather_all_parameters(self) -> None:
        if is_deepspeed_zero3_enabled() and not hasattr(self.deepspeed, '_zero3_consolidated_16bit_state_dict_origin'):
            parameters = inspect.signature(self.deepspeed._zero3_consolidated_16bit_state_dict).parameters
            if 'exclude_frozen_parameters' in parameters:

                def _zero3_consolidated_16bit_state_dict(model, exclude_frozen_parameters=False):
                    unwrapped = unwrap_model(model)
                    exclude_frozen_parameters = False
                    if isinstance(unwrapped, SwiftModel) and unwrapped.has_additional_modules:
                        exclude_frozen_parameters = True
                    if isinstance(unwrapped, PeftModel):
                        exclude_frozen_parameters = True
                    return model._zero3_consolidated_16bit_state_dict_origin(exclude_frozen_parameters)

                self.deepspeed._zero3_consolidated_16bit_state_dict_origin = (
                    self.deepspeed._zero3_consolidated_16bit_state_dict)
                self.deepspeed._zero3_consolidated_16bit_state_dict = MethodType(_zero3_consolidated_16bit_state_dict,
                                                                                 self.deepspeed)

    def _save_checkpoint(self, *args, **kwargs):
        self.state.last_model_checkpoint = os.path.join(self.args.output_dir, f'checkpoint-{self.state.global_step}')
        self._fix_zero3_gather_all_parameters()
        result = super()._save_checkpoint(*args, **kwargs)
        logger.info(f'Saving model checkpoint to {self.state.last_model_checkpoint}')
        return result

    def train(self, *args, **kwargs):
        if self.model.model_meta.is_multimodal:
            models = list(
                set([
                    v for k, v in self.__dict__.items()
                    if isinstance(v, nn.Module) and k in {'model', 'ref_model', 'reward_model', 'value_model'}
                ]))
            self.template.register_post_encode_hook(models)
            logger.info(f'Successfully registered post_encode hook: {[model.__class__.__name__ for model in models]}')
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        self._save_initial_model(self.args.output_dir)
        with self.hub.patch_hub():
            return super().train(*args, **kwargs)
        self.template.remove_post_encode_hook()

    def push_to_hub(self, *args, **kwargs):
        with self.hub.patch_hub():
            return super().push_to_hub(*args, **kwargs)

    def get_max_cuda_memory(self, device: Optional[Union[torch.device, int]] = None) -> float:
        if device is None:
            mems = [torch.cuda.max_memory_reserved(device=device) for device in range(torch.cuda.device_count())]
        else:
            mems = [torch.cuda.max_memory_reserved(device=device)]
        mem = sum(mems) / 1024**3
        self.max_memory = max(self.max_memory, mem)
        return mem

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            self.control.should_log = False

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            loss = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs: Dict[str, float] = {'loss': loss}  # loss first

            for k, metric in self._custom_metrics.items():
                value = metric.compute()
                if len(value) == 1:
                    val = list(value.values())[0]
                    logs[k] = val
                else:
                    for k_suffix, val in value.items():
                        new_k = f'{k}_{k_suffix}'
                        logs[new_k] = val
                metric.reset()

            if version.parse(transformers.__version__) >= version.parse('4.38'):
                grad_norm = args[0]
                if grad_norm is not None:
                    logs['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs['learning_rate'] = self._get_learning_rate()
            if not is_torch_npu_available():
                logs['memory(GiB)'] = round(self.get_max_cuda_memory(), 2)

            elapse_time = time.time() - self.start_time
            logs['train_speed(iter/s)'] = round(self.state.global_step / elapse_time, 6)
            for k in list(logs.keys()):
                if logs[k] is None:
                    logs.pop(k)
            tr_loss -= tr_loss
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)
        super()._maybe_log_save_evaluate(tr_loss, *args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if hasattr(self.args, 'galore_config'):
            optimizer, lr_scheduler = create_optimizer_and_scheduler(
                self.model,
                self.args,
                self.args.galore_config,
                num_training_steps,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)

    def create_optimizer(self):

        if self.optimizer is None and hasattr(self.model, 'create_optimizer_param_groups'):
            # Lora+ parameter groups
            optimizer_grouped_parameters = self.model.create_optimizer_param_groups(
                lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            if optimizer_grouped_parameters is not None:
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                return self.optimizer

        return super().create_optimizer()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.train_sampler_random:
            return super()._get_train_sampler()
        else:
            return self._get_eval_sampler(self.train_dataset)

    def get_train_dataloader(self):
        if self.args.sequence_parallel_size == 1:
            return super().get_train_dataloader()
        else:
            from swift.trainers.xtuner import get_xtuner_train_dataloader
            return get_xtuner_train_dataloader(self)
