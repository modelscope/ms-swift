# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
import re
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import safetensors
import torch
import torch.nn as nn
import transformers
from datasets import Dataset as HfDataset
from packaging import version
from peft import PeftModel
from torch.nn import Module
from transformers import PreTrainedModel, PreTrainedTokenizerBase, trainer
from transformers.data.data_collator import DataCollator
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME, Trainer, TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.utils import is_sagemaker_mp_enabled, is_torch_npu_available

from swift.hub.check_model import check_local_model_is_latest
from swift.torchacc_utils import (save_ta_ddp_checkpoint, save_ta_fsdp_checkpoint, ta_eval_dataloader,
                                  ta_load_optimizer_and_scheduler, ta_save_optimizer_and_scheduler, ta_test_dataloader,
                                  ta_train_dataloader, ta_trim_graph)
from swift.tuners import SwiftModel
from swift.utils import check_json_format, get_logger, use_torchacc
from swift.utils.constants import Invoke
from .callback import DefaultFlowCallbackNew, PrinterCallbackNew, ProgressCallbackNew
from .optimizers.galore import create_optimizer_and_scheduler
from .utils import can_return_loss, find_labels, get_function, is_instance_of_ms_model

try:
    from trl import AutoModelForCausalLMWithValueHead
except (ImportError, RuntimeError):
    AutoModelForCausalLMWithValueHead = None

logger = get_logger()


class SwiftMixin:

    def __init__(self,
                 model: Union[PreTrainedModel, Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[HfDataset] = None,
                 eval_dataset: Optional[Union[HfDataset, Dict[str, HfDataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 **kwargs) -> None:
        check_model = kwargs.pop('check_model', True)
        if check_model and hasattr(model, 'model_dir'):
            check_local_model_is_latest(
                model.model_dir,
                user_agent={
                    Invoke.KEY: Invoke.LOCAL_TRAINER,
                    Invoke.THIRD_PARTY: kwargs.pop(Invoke.THIRD_PARTY, Invoke.SWIFT),
                })

        # Compatible with transformers>=4.34
        from swift.tuners import SwiftModel
        is_quantized = getattr(model, 'is_quantized', False)
        _hf_peft_config_loaded = getattr(model, '_hf_peft_config_loaded', False)
        use_swift = isinstance(model, SwiftModel)
        if is_quantized and use_swift:
            model._hf_peft_config_loaded = True
        self.is_encoder_decoder = kwargs.pop('is_encoder_decoder', False)

        self.sequence_parallel_size = kwargs.pop('sequence_parallel_size', 1)
        if self.sequence_parallel_size > 1:
            from swift.trainers.xtuner import init_sequence_parallel_xtuner
            init_sequence_parallel_xtuner(self.sequence_parallel_size)
        if not hasattr(self, 'perf'):
            self.perf = {}
        # mro
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs)
        if not hasattr(self, 'label_names') or not self.label_names:
            self.label_names = ['labels']
        if is_quantized and use_swift:
            model._hf_peft_config_loaded = _hf_peft_config_loaded

        if get_function(model.__class__.forward) is not get_function(model.forward):
            self.label_names = find_labels(model)
            self.can_return_loss = can_return_loss(model)
        self.max_memory = 0.0
        self.start_time = time.time()
        self._resume_from_checkpoint = None
        self._resume_only_model = False
        # performance
        self.perf: Dict[str, Any] = {'memory': {}}
        if hasattr(self.model, 'get_trainable_parameters'):
            self.perf['model'] = self.model.get_trainable_parameters()

    @staticmethod
    def _create_configuration_file(model: Module, output_dir: str) -> None:
        cfg = getattr(model, 'cfg', None) or {}
        configuration_path = os.path.join(output_dir, 'configuration.json')
        new_cfg = {}
        if os.path.exists(configuration_path):
            with open(configuration_path, 'r', encoding='utf-8') as f:
                new_cfg = json.load(f)

        if 'framework' not in new_cfg:
            new_cfg['framework'] = cfg.get('framework', 'pytorch')
        if 'task' not in new_cfg:
            new_cfg['task'] = cfg.get('task', 'text-generation')
        with open(configuration_path, 'w', encoding='utf-8') as f:
            json.dump(new_cfg, f, ensure_ascii=False, indent=4)

    def _add_adapter_cfg(self, output_dir: str) -> None:
        if not hasattr(self, 'sft_args'):
            return
        sft_args = self.sft_args
        if sft_args.sft_type == 'full':
            return
        configuration_path = os.path.join(output_dir, 'configuration.json')
        new_cfg = {}
        if os.path.exists(configuration_path):
            with open(configuration_path, 'r', encoding='utf-8') as f:
                new_cfg = json.load(f)

        need_to_save = [
            'model_id_or_path', 'model_revision', 'sft_type', 'tuner_backend', 'template_type', 'dtype', 'system'
        ]
        quantization_bit = sft_args.quantization_bit
        if quantization_bit > 0:
            need_to_save += [
                'quant_method', 'quantization_bit', 'bnb_4bit_comp_dtype', 'bnb_4bit_quant_type',
                'bnb_4bit_use_double_quant'
            ]
        adapter_cfg = {}
        for k in need_to_save:
            adapter_cfg[k] = getattr(sft_args, k)
        new_cfg['adapter_cfg'] = adapter_cfg
        with open(configuration_path, 'w', encoding='utf-8') as f:
            json.dump(new_cfg, f, ensure_ascii=False, indent=4)

    def _save_sft_args(self, output_dir: str) -> None:
        sft_args = getattr(self, 'sft_args', None)
        if sft_args is None:
            return
        fpath = os.path.join(output_dir, 'sft_args.json')
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(check_json_format(self.sft_args.__dict__), f, ensure_ascii=False, indent=2)
        return

    def _save_optimizer_and_scheduler(self, output_dir):
        if not (use_torchacc() and self.sft_args.fsdp_num > 1):
            return super()._save_optimizer_and_scheduler(output_dir)

        ta_save_optimizer_and_scheduler(self.optimizer, self.lr_scheduler, output_dir)

    def _save_initial_model(self, output_dir):
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            config = model.peft_config.get('default', {})
            init_lora_weights = getattr(config, 'init_lora_weights', '')
            if isinstance(init_lora_weights, str) and ('pissa' in init_lora_weights or 'olora' in init_lora_weights):
                config.init_lora_weights = True
                model.save_pretrained(os.path.join(output_dir, 'initial_model'))
                config.init_lora_weights = init_lora_weights

    def _save_converted_model(self, output_dir):
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            config = model.peft_config.get('default', {})
            init_lora_weights = getattr(config, 'init_lora_weights', '')
            if isinstance(init_lora_weights, str) and ('pissa' in init_lora_weights or 'olora' in init_lora_weights):
                config = copy(config)
                os.makedirs(os.path.join(output_dir, 'converted'), exist_ok=True)
                model.save_pretrained(
                    os.path.join(output_dir, 'converted', 'default'),
                    path_initial_model_for_weight_conversion=os.path.join(os.path.dirname(output_dir), 'initial_model'),
                )
                model.peft_config['default'] = config

    def _load_optimizer_and_scheduler(self, checkpoint):
        if not (use_torchacc() and self.sft_args.fsdp_num > 1):
            if self._resume_only_model:
                checkpoint = self._resume_from_checkpoint
                if checkpoint is not None and (is_sagemaker_mp_enabled() or self.is_fsdp_enabled):
                    self._load_from_checkpoint(checkpoint, self.model_wrapped)
                return
            else:
                # Check if saved optimizer or scheduler states exist
                super()._load_optimizer_and_scheduler(checkpoint)
                try:
                    # fix mp+ddp adamw
                    for v in self.optimizer.state.values():
                        if 'step' in v:
                            # not on the same device
                            device_set = set([t.device for t in v.values()]) - {v['step'].device, torch.device('cpu')}
                            if len(device_set) >= 1:
                                v['step'] = v['step'].to('cpu')
                except Exception:
                    pass
                return

        if checkpoint is None or self.args.save_only_model:
            return

        self.optimizer, self.lr_scheduler = ta_load_optimizer_and_scheduler(self.optimizer, self.lr_scheduler,
                                                                            checkpoint, self.args.device)

    def _save_tpu(self, output_dir: Optional[str] = None):
        if not use_torchacc():
            return super()._save_tpu(output_dir)

        import torch_xla.core.xla_model as xm

        # Compatible with swift and peft
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        if xm.is_master_ordinal(local=False):
            os.makedirs(output_dir, exist_ok=True)
            # configuration.json
            model_dir = getattr(self.model, 'model_dir', None)
            if model_dir is not None:
                src_path = os.path.join(model_dir, 'configuration.json')
                dst_path = os.path.join(output_dir, 'configuration.json')
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
            else:
                self._create_configuration_file(self.model, output_dir)
            self._add_adapter_cfg(output_dir)
            self._save_sft_args(output_dir)
            # generation_config
            generation_config = getattr(self.args, 'generation_config', None)
            if generation_config is not None:
                generation_config.save_pretrained(output_dir)

        # model
        if self.sft_args.fsdp_num > 1:
            save_ta_fsdp_checkpoint(self.model, self.tokenizer, self.args, output_dir)
        else:
            save_ta_ddp_checkpoint(self.model, self.tokenizer, self.args, output_dir)
        sft_args = getattr(self, 'sft_args', None)

        # additional files
        if xm.is_master_ordinal(local=False):
            if sft_args is not None and sft_args.sft_type == 'full':
                additional_files = getattr(self.args, 'additional_saved_files',
                                           None) or [] + ['preprocessor_config.json']
                if model_dir is not None:
                    for file in additional_files:
                        src_path = os.path.join(model_dir, file)
                        dst_path = os.path.join(output_dir, file)
                        if os.path.isfile(src_path):
                            shutil.copy(src_path, dst_path)
                        elif os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Compatible with swift and peft"""
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # configuration.json
        model_dir = getattr(self.model, 'model_dir', None)
        if model_dir is not None:
            src_path = os.path.join(model_dir, 'configuration.json')
            dst_path = os.path.join(output_dir, 'configuration.json')
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
        else:
            self._create_configuration_file(self.model, output_dir)
        self._add_adapter_cfg(output_dir)
        self._save_sft_args(output_dir)
        # generation_config
        generation_config = getattr(self.args, 'generation_config', None)
        if generation_config is not None:
            generation_config.save_pretrained(output_dir)
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
        elif is_instance_of_ms_model(self.model):
            PreTrainedModel.save_pretrained(
                self.model, output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
        elif AutoModelForCausalLMWithValueHead is not None and isinstance(self.model,
                                                                          AutoModelForCausalLMWithValueHead):
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
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
        sft_args = getattr(self, 'sft_args', None)
        # tokenizer
        if self.tokenizer is not None and sft_args is not None and sft_args.sft_type == 'full':
            if hasattr(self.tokenizer, 'processor'):
                self.tokenizer.processor.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        # training_args.bin
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        # additional files
        if sft_args is not None and sft_args.sft_type == 'full':
            additional_files = getattr(self.args, 'additional_saved_files', None) or [] + ['preprocessor_config.json']
            if model_dir is not None:
                for file in additional_files:
                    src_path = os.path.join(model_dir, file)
                    dst_path = os.path.join(output_dir, file)
                    if os.path.isfile(src_path):
                        shutil.copy(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
        self._save_converted_model(output_dir)

    def _save_checkpoint(self, model, trial, metrics=None):
        self.state.last_model_checkpoint = os.path.join(self.args.output_dir, f'checkpoint-{self.state.global_step}')
        if is_deepspeed_zero3_enabled() and not hasattr(self.deepspeed, '_zero3_consolidated_16bit_state_dict_origin'):
            parameters = inspect.signature(self.deepspeed._zero3_consolidated_16bit_state_dict).parameters
            if 'exclude_frozen_parameters' in parameters:

                def _zero3_consolidated_16bit_state_dict(_model, exclude_frozen_parameters=False):
                    unwrapped = unwrap_model(_model)
                    exclude_frozen_parameters = False
                    if isinstance(unwrapped, SwiftModel) and unwrapped.has_additional_modules:
                        exclude_frozen_parameters = True
                    if isinstance(unwrapped, PeftModel):
                        exclude_frozen_parameters = True
                    return _model._zero3_consolidated_16bit_state_dict_origin(exclude_frozen_parameters)

                self.deepspeed._zero3_consolidated_16bit_state_dict_origin = (
                    self.deepspeed._zero3_consolidated_16bit_state_dict)
                self.deepspeed._zero3_consolidated_16bit_state_dict = MethodType(_zero3_consolidated_16bit_state_dict,
                                                                                 self.deepspeed)
        if version.parse(transformers.__version__) >= version.parse('4.36') or not self.args.save_only_model:
            result = super()._save_checkpoint(model, trial, metrics)
        else:
            result = self._save_only_model(model, trial, metrics)
        logger.info(f'Saving model checkpoint to {self.state.last_model_checkpoint}')
        return result

    def _save_only_model(self, model, trial, metrics=None):
        # Save model checkpoint
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith('eval_'):
                metric_to_check = f'eval_{metric_to_check}'
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # push to hub
        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        train_sampler_random = self.args.train_sampler_random
        if train_sampler_random:
            return super()._get_train_sampler()
        else:
            return self._get_eval_sampler(self.train_dataset)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        if model is None:
            model = self.model
        if use_torchacc():
            # Loading checkpoint of TorchAcc has been done in tuner.py when
            # sft_type is 'full'.
            if self.sft_args.fsdp_num > 1:
                model = model._get_underlay_model().module.module
            if isinstance(model, PreTrainedModel):
                return
        elif isinstance(model, SwiftModel) or is_deepspeed_zero3_enabled() and isinstance(model, PreTrainedModel):
            return
        else:
            # Avoid throwing exceptions
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _sorted_checkpoints(self,
                            output_dir=None,
                            checkpoint_prefix=PREFIX_CHECKPOINT_DIR,
                            use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f'{checkpoint_prefix}-*') if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f'.*{checkpoint_prefix}-([0-9]+)', path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if (self.state.best_model_checkpoint is not None
                and str(Path(self.state.best_model_checkpoint)) in checkpoints_sorted):
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, *args, **kwargs) -> torch.Tensor:
        sft_args = getattr(self, 'sft_args', None)
        self._resume_only_model = getattr(sft_args, 'resume_only_model', False)
        if self._resume_only_model:
            # Control the behavior of "resume_from_checkpoint" by swift.
            self._resume_from_checkpoint = resume_from_checkpoint
            resume_from_checkpoint = None
        if self._resume_from_checkpoint is not None and not is_sagemaker_mp_enabled() and not self.is_fsdp_enabled:
            self._load_from_checkpoint(self._resume_from_checkpoint)

        self._save_initial_model(self.args.output_dir)
        res = super().train(resume_from_checkpoint, *args, **kwargs)
        self._resume_from_checkpoint = None
        if self.max_memory != 0:
            self.perf['memory']['cuda'] = f'{self.max_memory:.2f}GiB'
        return res

    def _load_best_model(self):
        # Compatible with transformers>=4.35 (deepspeed)
        try:
            model = self.model
            if isinstance(model, SwiftModel):
                logger.info(
                    f'Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).')
                adapters = model.adapters
                for adapter_name in adapters.keys():
                    sub_folder = os.path.join(self.state.best_model_checkpoint, adapter_name)
                    state_dict = SwiftModel.load_state_file(sub_folder, device='cpu')
                    if state_dict is not None:
                        self.model.load_state_dict(state_dict, strict=False, adapter_name=adapter_name)
                state_dict = SwiftModel.load_state_file(self.state.best_model_checkpoint, device='cpu')
                if state_dict is not None:
                    self.model.load_state_dict(state_dict, strict=False, adapter_name='default')
            else:
                super()._load_best_model()
        except ValueError as e:
            logger.warning(e)

    def get_max_cuda_memory(self, device: Optional[Union[torch.device, int]] = None) -> float:
        if device is None:
            mems = [torch.cuda.max_memory_reserved(device=device) for device in range(torch.cuda.device_count())]
        else:
            mems = [torch.cuda.max_memory_reserved(device=device)]
        mem = sum([float(mem) / 1024 / 1024 / 1024 for mem in mems])
        if self.max_memory < mem:
            self.max_memory = mem
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return mem

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        if self.control.should_log:
            if use_torchacc():
                ta_trim_graph()
            self.control.should_log = False
            logs: Dict[str, float] = {}
            metrics_log = {'loss': tr_loss}  # loss first
            if hasattr(self, '_custom_metrics'):
                metrics_log.update(self._custom_metrics)
                self._custom_metrics = {}
            for k, v in metrics_log.items():
                # all_gather + mean() to get average loss over all processes
                v_scalar = self._nested_gather(v).mean().item()
                if k == 'loss':
                    self._total_loss_scalar += v_scalar
                logs[k] = round(v_scalar / (self.state.global_step - self._globalstep_last_logged), 8)
                if k == 'acc' and self._globalstep_last_logged > 0:
                    sft_args = getattr(self, 'sft_args', None)
                    acc_steps = 1 if sft_args is None else sft_args.acc_steps
                    logs[k] *= acc_steps
            if version.parse(transformers.__version__) >= version.parse('4.38'):
                grad_norm = args[0]
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()
                if grad_norm is not None:
                    logs['grad_norm'] = grad_norm
            logs['learning_rate'] = self._get_learning_rate()
            if not is_torch_npu_available():
                logs['memory(GiB)'] = round(self.get_max_cuda_memory(), 2)
            import time
            time_now = time.time()
            elapse_time = time_now - self.start_time
            logs['train_speed(iter/s)'] = round(self.state.global_step / elapse_time, 6)
            tr_loss -= tr_loss
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
            self.create_optimizer()
            self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            if version.parse(transformers.__version__) < version.parse('4.34.0'):
                logger.warning(f'If you are using lora+, please remember using transformers>=4.34.0, '
                               f'but now is {transformers.__version__}')
                return super().create_optimizer()

            optimizer_grouped_parameters = None
            if hasattr(self.model, 'create_optimizer_param_groups'):
                # Lora+ parameter groups
                optimizer_grouped_parameters = self.model.create_optimizer_param_groups(
                    lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

            if optimizer_grouped_parameters is None:
                # Default parameter groups
                decay_parameters = self.get_decay_parameter_names(opt_model)
                optimizer_grouped_parameters = [
                    {
                        'params':
                        [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        'weight_decay':
                        self.args.weight_decay,
                    },
                    {
                        'params':
                        [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        'weight_decay':
                        0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def get_train_dataloader(self):
        if self.sequence_parallel_size > 1:
            from swift.trainers.xtuner import get_xtuner_train_dataloader
            return get_xtuner_train_dataloader(self)
        elif use_torchacc():
            if trainer.is_datasets_available():
                import datasets

            if self.train_dataset is None:
                raise ValueError('Trainer: training requires a train_dataset.')

            train_dataset = self.train_dataset
            data_collator = self.data_collator

            if trainer.is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description='training')
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

            return ta_train_dataloader(train_dataset, data_collator, self._get_train_sampler(), self.args,
                                       self._train_batch_size)
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if not use_torchacc():
            return super().get_eval_dataloader(eval_dataset)
        else:
            if trainer.is_datasets_available():
                import datasets

            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError('Trainer: evaluation requires an eval_dataset.')
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            data_collator = self.data_collator

            if trainer.is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
                eval_dataset = self._remove_unused_columns(eval_dataset, description='evaluation')
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description='evaluation')

            return ta_eval_dataloader(eval_dataset, data_collator, self._get_eval_sampler(eval_dataset), self.args)

    def get_test_dataloader(self, test_dataset):
        if not use_torchacc():
            return super().get_test_dataloader(test_dataset)
        else:
            if trainer.is_datasets_available():
                import datasets

            data_collator = self.data_collator

            if trainer.is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
                test_dataset = self._remove_unused_columns(test_dataset, description='test')
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description='test')

            return ta_test_dataloader(test_dataset, data_collator, self._get_eval_sampler(test_dataset), self.args)


class ModelWrapper(nn.Module):
    # compat zero3 & rlhf
    def __init__(self, model: nn.Module, ref_model: nn.Module):
        super().__init__()
        self._model = model
        self._ref_model = ref_model

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._model, name)

    def load_state_dict(self, *args, **kwargs):
        return self._model.load_state_dict(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._model.parameters(*args, **kwargs)

    @contextmanager
    def _save_load_context(cls, trainer):
        # fix zero3 & save/load model
        deepspeed_model = trainer.deepspeed
        _new_model = deepspeed_model._model
        _old_model = deepspeed_model.__dict__['module']
        deepspeed_model.__dict__['module'] = _new_model
        deepspeed_model._modules['module'] = _new_model
        trainer.model = _new_model
        yield
        deepspeed_model.__dict__['module'] = _old_model
        deepspeed_model._modules['module'] = _old_model
        trainer.model = deepspeed_model


class RLHFTrainerMixin:

    @staticmethod
    def get_model_config_attr(config, key):
        for k in [None, 'language_config', 'llm_config', 'text_config']:
            if k is None:
                llm_config = config
            else:
                llm_config = getattr(config, k, None)
            if llm_config:
                val = getattr(llm_config, key)
                if val is not None:
                    return val

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import disable_dropout_in_model
        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        args = kwargs['args']
        self.beta = getattr(args, 'beta', 0.0)
        if getattr(args, 'disable_dropout', False):
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.is_encoder_decoder = kwargs['is_encoder_decoder']
        self.aux_loss_enabled = getattr(model.config, 'output_router_logits', False)
        self._peft_has_been_casted_to_bf16 = False
        self.generate_during_eval = getattr(args, 'generate_during_eval', False)
        self.is_multimodal = False
        if self.is_encoder_decoder:
            self.decoder_start_token_id = self.get_model_config_attr(model.config, 'decoder_start_token_id')
            self.pad_token_id = self.get_model_config_attr(model.config, 'pad_token_id')
        # not use
        self.is_vision_model = False
        tokenizer = kwargs['tokenizer']
        self.label_pad_token_id = -100
        self.padding_value = tokenizer.pad_token_id
        self.use_dpo_data_collator = True
        if is_deepspeed_zero3_enabled() and ref_model is not None:
            model = ModelWrapper(model, ref_model)
        super().__init__(model, *_args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        context = nullcontext()
        if hasattr(model, '_save_load_context'):
            context = model._save_load_context(self)
        with context:
            return super()._save_checkpoint(model, trial, metrics)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        model_kwargs = batch.copy()
        labels = model_kwargs.pop('labels', None)
        if self.is_encoder_decoder:
            model_kwargs['labels'] = labels

        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True
        outputs = model(**model_kwargs, use_cache=False)
        model_kwargs['labels'] = labels
        model_kwargs['chosen_labels'] = torch.zeros(model_kwargs['input_ids'].shape[0] // 2)  # just get shape
        if outputs.logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            outputs.logits = outputs.logits[:, -labels.shape[1]:]
        for key in ['input_ids', 'attention_mask', 'labels']:
            model_kwargs[f'concatenated_{key}'] = model_kwargs.pop(key)
        if self.__class__.__name__ == 'ORPOTrainer':  # Pass-through labels
            model_kwargs['concatenated_input_ids'] = model_kwargs['concatenated_labels']

        @contextmanager
        def _patch_concatenated_forward():
            _old_concatenated_inputs = self.concatenated_inputs
            _old_model_call = model.__class__.__call__
            self.concatenated_inputs = lambda *args, **kwargs: model_kwargs
            model.__class__.__call__ = lambda *args, **kwargs: outputs
            yield
            self.concatenated_inputs = _old_concatenated_inputs
            model.__class__.__call__ = _old_model_call

        with _patch_concatenated_forward():
            return super().concatenated_forward(model, model_kwargs)

    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor, *args, **kwargs):
        if self.is_encoder_decoder:
            labels = labels.clone()  # fix trl bug
        return super().get_batch_logps(logits, labels, *args, **kwargs)


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew
