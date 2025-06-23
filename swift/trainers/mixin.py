# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import logging
import os
import shutil
import time
from contextlib import contextmanager
from copy import copy
from functools import partial, wraps
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import safetensors
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from datasets import Dataset as HfDataset
from modelscope import check_local_model_is_latest
from packaging import version
from peft import PeftModel
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TrainerCallback
from transformers.trainer_utils import EvalPrediction, IntervalStrategy
from transformers.utils import is_torch_npu_available

from swift.hub import get_hub
from swift.llm import BatchSamplerShard, DataLoaderDispatcher, DataLoaderShard, Template
from swift.plugin import MeanMetric, compute_acc, extra_tuners
from swift.tuners import SwiftModel
from swift.utils import get_logger, is_dist, is_mp, is_mp_ddp, ms_logger_context, seed_worker, use_torchacc
from swift.utils.torchacc_utils import ta_trim_graph
from ..utils.torch_utils import get_device_count
from .arguments import TrainingArguments
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
                 template: Optional[Template] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_loss_func: Optional[Callable] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 **kwargs) -> None:
        if not hasattr(train_dataset, '__len__') and args.dataloader_num_workers > 1:
            args.dataloader_num_workers = 1
            logger.warning('Using IterableDataset, setting args.dataloader_num_workers to 1.')

        if args.check_model and hasattr(model, 'model_dir'):
            with ms_logger_context(logging.CRITICAL):
                check_local_model_is_latest(
                    model.model_dir, user_agent={
                        'invoked_by': 'local_trainer',
                        'third_party': 'swift',
                    })
        if eval_dataset is None and args:
            args.evaluation_strategy = IntervalStrategy.NO
            args.eval_strategy = IntervalStrategy.NO

        self._custom_metrics = {}
        self.template = template
        self.max_memory = 0
        self.hub = get_hub()

        self.model_meta = model.model_meta
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
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                **kwargs)

        self.compute_loss_func = compute_loss_func
        if get_function(model.__class__.forward) is not get_function(model.forward):
            self.label_names = find_labels(model)
            self.can_return_loss = can_return_loss(model)
        self.label_names = self.label_names or ['labels']
        self.start_time = time.time()
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.prepare_trainer(self)
        self._fix_gradient_checkpointing()

    def get_use_logits_to_keep(self, default_value: bool = True):
        use_logits_to_keep = self.args.use_logits_to_keep
        if use_logits_to_keep is None:
            base_model = self.template.get_base_model(self.model)
            use_logits_to_keep = (not self.model.model_meta.is_multimodal
                                  and 'logits_to_keep' in inspect.signature(base_model.forward).parameters
                                  and default_value)
        logger.info_once(f'use_logits_to_keep: {use_logits_to_keep}')
        return use_logits_to_keep

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
        supported_names = ('SentenceTransformer', )
        if AutoModelForCausalLMWithValueHead is not None:
            supported_classes = supported_classes + (AutoModelForCausalLMWithValueHead, )
        save_safetensors = self.args.save_safetensors
        if not isinstance(self.model, supported_classes) and self.model.__class__.__name__ not in supported_names:
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
            if self.model.__class__.__name__ != 'SentenceTransformer':
                self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
            else:

                @contextmanager
                def save_context():
                    save_pretrained = self.model[0].auto_model.save_pretrained
                    _state_dict = {
                        key[len('0.auto_model.'):] if 'auto_model' in key else key: value
                        for key, value in state_dict.items()
                    }
                    self.model[0].auto_model.save_pretrained = partial(
                        self.model[0].auto_model.save_pretrained, state_dict=_state_dict)
                    yield
                    self.model[0].auto_model.save_pretrained = save_pretrained

                with save_context():
                    self.model.save_pretrained(output_dir, safe_serialization=save_safetensors)
                    # copy sentencetransformers files
                    from swift.utils import copy_files_by_pattern
                    copy_files_by_pattern(self.model.model_dir, output_dir, '*.py')
                    copy_files_by_pattern(self.model.model_dir, output_dir, '*.json')

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
            additional_saved_files = self.model_meta.additional_saved_files
            save_checkpoint(
                None,
                self.template.processor,
                output_dir,
                model_dirs=[self.model.model_dir],
                additional_saved_files=additional_saved_files)
            if getattr(self.model, 'origin_generation_config', None):
                self.model.origin_generation_config.save_pretrained(output_dir)

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

    @staticmethod
    @contextmanager
    def _fix_grad_norm_nan():
        from accelerate import Accelerator
        origin_clip_grad_norm_ = Accelerator.clip_grad_norm_

        def clip_grad_norm_(self, parameters, *args, **kwargs):
            # If NaN occurs, ignore weight updates.
            parameters = list(parameters)
            grad_norm = origin_clip_grad_norm_(self, parameters, *args, **kwargs)
            if isinstance(grad_norm, torch.Tensor) and grad_norm.isnan().item():
                for p in parameters:
                    p.grad = None
            return grad_norm

        Accelerator.clip_grad_norm_ = clip_grad_norm_
        try:
            yield
        finally:
            Accelerator.clip_grad_norm_ = origin_clip_grad_norm_

    def _fix_gradient_checkpointing(self):
        # fix use_reentrant
        if hasattr(torch.utils.checkpoint, '_old_checkpoint'):  # avoid double patching
            return
        args = self.args
        if args.gradient_checkpointing_kwargs:
            use_reentrant_ = args.gradient_checkpointing_kwargs.get('use_reentrant')
        else:
            use_reentrant_ = None
        if use_reentrant_ is None:
            if is_dist() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                use_reentrant_ = False
            else:
                use_reentrant_ = True
        logger.info(f'use_reentrant: {use_reentrant_}')
        _old_checkpoint = torch.utils.checkpoint.checkpoint

        @wraps(_old_checkpoint)
        def _new_checkpoint(*args, use_reentrant=None, **kwargs):
            return _old_checkpoint(*args, use_reentrant=use_reentrant_, **kwargs)

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = _new_checkpoint
        try:
            # Fix the old version of transformers.
            import transformers.modeling_utils
            transformers.modeling_utils.checkpoint = _new_checkpoint
        except (ImportError, AttributeError):
            pass

    def _prepare_gradient_checkpointing(self, model) -> None:
        from swift.llm import HfConfigFactory, get_model_arch, deep_getattr, dynamic_gradient_checkpointing
        args = self.args
        HfConfigFactory.set_model_config_attr(model, 'use_cache', False)
        if args.gradient_checkpointing or args.vit_gradient_checkpointing:
            dynamic_gradient_checkpointing(model, args.vit_gradient_checkpointing)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
            model.enable_input_require_grads()

        model_meta = model.model_meta
        model_arch = get_model_arch(model_meta.model_arch)
        if model_meta.is_multimodal and model_arch:
            for vision_tower_name in model_arch.vision_tower:
                vision_tower = deep_getattr(model, vision_tower_name)
                if hasattr(vision_tower, 'enable_input_require_grads'):
                    try:
                        if args.vit_gradient_checkpointing:
                            vision_tower.gradient_checkpointing_enable(
                                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
                            vision_tower.enable_input_require_grads()
                        else:
                            vision_tower.gradient_checkpointing_disable()
                            vision_tower.disable_input_require_grads()
                    except (NotImplementedError, AttributeError):
                        pass
        # Avoid vit_gradient_checkpointing being overwritten by transformers.Trainer.gradient_checkpointing_enable.
        self.args.gradient_checkpointing = False

    def train(self, *args, **kwargs):
        if self.model_meta.is_multimodal:
            models = []
            for model_name in ['model', 'ref_model', 'value_model', 'teacher_model']:
                model = getattr(self, model_name, None)
                if isinstance(model, nn.Module):
                    models.append(model)

            reward_model = getattr(self, 'reward_model', None)
            if reward_model is not None:
                if isinstance(reward_model, list):
                    models.extend([m for m in reward_model if isinstance(m, nn.Module)])
                elif isinstance(reward_model, nn.Module):
                    models.append(reward_model)

            models = list(set(self.accelerator.unwrap_model(model) for model in models))  # Deduplicate
            self.template.register_post_encode_hook(models)
            logger.info(f'Successfully registered post_encode hook: {[model.__class__.__name__ for model in models]}.')
        self._save_initial_model(self.args.output_dir)

        # gradient_checkpointing
        gradient_checkpointing = self.args.gradient_checkpointing
        self._prepare_gradient_checkpointing(self.accelerator.unwrap_model(self.model))
        with self.hub.patch_hub(), self._fix_grad_norm_nan():
            res = super().train(*args, **kwargs)
        self.template.remove_post_encode_hook()
        self.args.gradient_checkpointing = gradient_checkpointing  # recover
        return res

    def push_to_hub(self, *args, **kwargs):
        with self.hub.patch_hub():
            return super().push_to_hub(*args, **kwargs)

    def get_max_cuda_memory(self, device: Optional[Union[torch.device, int]] = None) -> float:
        if device is None:
            mems = [torch.cuda.max_memory_reserved(device=device) for device in range(get_device_count())]
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

        if self.args.eval_use_evalscope and self.control.should_evaluate:
            self._evalscope_eval()
        super()._maybe_log_save_evaluate(tr_loss, *args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.optimizer is not None:
            from swift.plugin import optimizers_map
            optimizer_callback = optimizers_map[self.args.optimizer]
            self.optimizer, self.lr_scheduler = optimizer_callback(self.args, self.model, self.train_dataset)
            if self.optimizer is None:
                self.create_optimizer()
            if self.lr_scheduler is None:
                self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        else:
            super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)

    def _compute_acc(self, outputs, labels) -> None:
        args = self.args
        acc_steps = args.acc_steps
        preds = outputs.logits.argmax(dim=-1)
        if self.state.global_step % acc_steps == 0:
            if use_torchacc():
                ta_trim_graph()
                preds = preds.to('cpu')
                labels = labels.to('cpu')
            metrics = compute_acc(
                preds, labels, acc_strategy=args.acc_strategy, is_encoder_decoder=self.template.is_encoder_decoder)
            for k, v in metrics.items():
                if k not in self._custom_metrics:
                    self._custom_metrics[k] = MeanMetric(nan_value=None)
                self._custom_metrics[k].update(v)

    @torch.no_grad()
    def _evalscope_eval(self):
        from ..llm.eval.utils import EvalModel
        from evalscope import TaskConfig, run_task
        from evalscope.constants import EvalType

        self.model.eval()
        max_batch_size = self.args.per_device_eval_batch_size
        custom_model = EvalModel(
            self.model, self.template, max_batch_size=max_batch_size, model_name=f'model-step{self.state.global_step}')
        task_config = TaskConfig(
            model=custom_model,
            eval_type=EvalType.CUSTOM,
            datasets=self.args.eval_datasets,
            dataset_args=self.args.eval_datasets_args,
            limit=self.args.eval_limit,
            work_dir=os.path.join(self.args.output_dir, 'eval'),
            eval_batch_size=max_batch_size,
            generation_config=self.args.eval_generation_config or {'max_tokens': 512},
        )
        # start evaluation
        eval_report = run_task(task_config)
        # convert to dict
        eval_dict = {f'test_{k}': v.score for k, v in eval_report.items()}
        self.log(eval_dict)

        self.model.train()
        return eval_dict

    def get_logits_to_keep(self, labels):
        if labels.shape[0] == 1 and not is_mp():
            # device_map may encounter device mismatch issues.
            loss_mask = (labels != -100)[0]
            labels = labels[:, loss_mask]
            labels = nn.functional.pad(labels, (1, 0), value=-100)
            logits_to_keep = nn.functional.pad(loss_mask[1:], (0, 1), value=True)
        else:
            logits_to_keep = labels.shape[-1] - ((labels != -100).int().argmax(-1).min().item()) + 1
            assert logits_to_keep > 0
            labels = labels[:, -logits_to_keep:]
        return labels, logits_to_keep

    def get_cu_seqlens(self, position_ids, logits_to_keep) -> torch.Tensor:
        assert position_ids.shape[0] == 1
        position_ids = position_ids[0]
        indices = torch.arange(position_ids.shape[0], device=position_ids.device)
        cu_seqlens = torch.concat([
            indices[position_ids == 0],
            torch.tensor(position_ids.shape, device=position_ids.device),
        ])
        res_cu_seqlens = cu_seqlens.clone()
        if isinstance(logits_to_keep, torch.Tensor):
            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                res_cu_seqlens[i + 1:] -= (~logits_to_keep[start:end]).sum()
        elif isinstance(logits_to_keep, int):
            res_cu_seqlens[1:] -= position_ids.shape[0] + 1 - logits_to_keep
        return res_cu_seqlens

    def get_batch_samples(self, *args, **kwargs):
        res = super().get_batch_samples(*args, **kwargs)
        from swift.trainers.sequence_parallel import sequence_parallel
        if self.template.sequence_parallel_size == 1 or 'Ulysses' == sequence_parallel.__class__.__name__:
            # ulysses split inputs in the model hook, so no need to gather num_items_in_batch
            return res

        batch_samples, num_items_in_batch = res
        if num_items_in_batch is None:
            num_items_in_batch = torch.tensor(0).to(args[2])
        from swift.trainers.sequence_parallel import sequence_parallel
        dist.all_reduce(num_items_in_batch, dist.ReduceOp.SUM, sequence_parallel.sp_group)
        return batch_samples, num_items_in_batch


class DataLoaderMixin:

    def get_train_dataloader(self):
        dataloader = None
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            dataloader = sequence_parallel.get_dataloader(self, self.train_dataset, self._train_batch_size)
        if dataloader is None:
            # Higher efficiency
            if self.train_dataset is None:
                raise ValueError('Trainer: training requires a train_dataset.')
            args = self.args
            train_dataset = self.train_dataset

            dataloader_params = {
                'collate_fn': self.data_collator,
                'num_workers': args.dataloader_num_workers,
                'pin_memory': args.dataloader_pin_memory,
                'persistent_workers': args.dataloader_persistent_workers,
                'prefetch_factor': args.dataloader_prefetch_factor
            }
            batch_sampler_params = {
                'drop_last': args.dataloader_drop_last,
                'shuffle': args.train_dataloader_shuffle,
                'data_seed': args.data_seed,
            }

            if hasattr(train_dataset, '__len__'):
                batch_sampler = BatchSamplerShard(
                    len(train_dataset), batch_size=self._train_batch_size, **batch_sampler_params)
                dataloader_params['worker_init_fn'] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index)
                dataloader_params['batch_sampler'] = batch_sampler
                dataloader = DataLoaderShard(train_dataset, device=self.accelerator.device, **dataloader_params)
            else:
                # IterableDataset
                if dist.is_initialized() and dataloader_params['prefetch_factor']:
                    dataloader_params['prefetch_factor'] = dataloader_params['prefetch_factor'] * dist.get_world_size()
                dataloader = DataLoader(train_dataset, batch_size=self._train_batch_size, **dataloader_params)
                dataloader = DataLoaderDispatcher(dataloader, self.accelerator.device)

        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        dataloader = None
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError('Trainer: evaluation requires an eval_dataset.')
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            dataloader = sequence_parallel.get_dataloader(self, eval_dataset, self.args.eval_batch_size)
        if dataloader is None:
            return super().get_eval_dataloader(eval_dataset=eval_dataset)
        return dataloader
