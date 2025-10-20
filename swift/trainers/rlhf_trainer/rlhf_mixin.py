# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
from collections import OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import asdict, dataclass
from functools import partial
from math import ceil
from time import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from peft.utils.save_and_load import get_peft_model_state_dict
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from trl.models.utils import prepare_deepspeed
from trl.trainer.utils import selective_log_softmax

from swift.llm import (InferRequest, MultiModelKeys, RequestConfig, RolloutInferRequest, RowPreprocessor, Template,
                       to_device)
from swift.llm.argument import RLHFArguments
from swift.plugin import orms, rm_plugins
from swift.utils import get_logger, is_vllm_available
from swift.utils.torch_utils import get_current_device
from .utils import (FlattenedTensorBucket, TensorLoRARequest, _create_parameter_buckets, _ForwardRedirection,
                    _process_bucket_with_flattened_tensor, compute_chord_loss, get_gather_if_zero3_context,
                    identity_data_collator, load_pil_img, make_chord_sft_dataset, patch_lora_merge, patch_lora_unmerge,
                    patch_profiling_context, patch_profiling_decorator, patch_save_last_checkpoint,
                    patch_vllm_load_adapter, replace_assistant_response_with_ids, set_expandable_segments)
from .vllm_client import VLLMClient

logger = get_logger()


class RLHFTrainerMixin:

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import disable_dropout_in_model
        from swift.llm import HfConfigFactory
        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        args = kwargs['args']
        self.beta = getattr(args, 'beta', 0.0)
        if getattr(args, 'disable_dropout', False):
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.is_encoder_decoder = kwargs['template'].is_encoder_decoder
        self._peft_has_been_casted_to_bf16 = False
        self.generate_during_eval = getattr(args, 'generate_during_eval', False)
        if self.is_encoder_decoder:
            self.decoder_start_token_id = HfConfigFactory.get_config_attr(model.config, 'decoder_start_token_id')
            self.pad_token_id = HfConfigFactory.get_config_attr(model.config, 'pad_token_id')
        # not use
        self.is_vision_model = False
        self.label_pad_token_id = -100
        self.use_dpo_data_collator = True
        reward_funcs = kwargs.pop('reward_funcs', None)
        reward_model = kwargs.pop('reward_model', None)
        vllm_client = kwargs.pop('vllm_client', None)
        super().__init__(model, *_args, **kwargs)
        self.aux_loss_enabled = model.model_info.is_moe_model and args.router_aux_loss_coef > 0
        self.aux_loss_coef = args.router_aux_loss_coef
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        self.padding_value = self.tokenizer.pad_token_id
        if self.args.rlhf_type == 'grpo':
            self._prepare_rewards(reward_funcs, reward_model, **kwargs)

        if self.args.rlhf_type in ['grpo', 'gkd']:
            self._prepare_rollout_params()
            self._prepare_vllm(model, vllm_client)

    def create_loss_and_metric(self, args):
        return {}

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.prepare_inputs(inputs)
        return inputs

    def get_train_dataloader(self, *args, **kwargs):
        train_dataloader = super().get_train_dataloader(*args, **kwargs)
        base_dataloader = train_dataloader.base_dataloader if hasattr(
            train_dataloader, 'base_dataloader') and isinstance(train_dataloader.base_dataloader,
                                                                DataLoader) else train_dataloader
        if base_dataloader.worker_init_fn is not None and not isinstance(
                base_dataloader.worker_init_fn, partial) and 'num_workers' in inspect.signature(
                    base_dataloader.worker_init_fn).parameters:
            base_dataloader.worker_init_fn = partial(
                base_dataloader.worker_init_fn,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index)
        return train_dataloader

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
        model_kwargs['chosen_labels'] = torch.zeros(model_kwargs['labels'].shape[0] // 2)  # just get shape
        if outputs.logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            outputs.logits = outputs.logits[:, -labels.shape[1]:]
        for key in ['input_ids', 'attention_mask', 'labels']:
            model_kwargs[f'concatenated_{key}'] = model_kwargs.pop(key, None)
        if self.__class__.__name__ == 'ORPOTrainer':  # Pass-through labels
            model_kwargs['concatenated_input_ids'] = model_kwargs['concatenated_labels']

        @contextmanager
        def _patch_concatenated_forward():
            _old_concatenated_inputs = self.concatenated_inputs
            _old_model_call = model.__class__.__call__
            self.concatenated_inputs = lambda *args, **kwargs: model_kwargs
            model.__class__.__call__ = lambda *args, **kwargs: outputs
            try:
                yield
            finally:
                self.concatenated_inputs = _old_concatenated_inputs
                model.__class__.__call__ = _old_model_call

        with _patch_concatenated_forward():
            return super().concatenated_forward(model, model_kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        res = super().compute_loss(model, inputs, return_outputs=return_outputs)
        # compat transformers>=4.46.*
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = res[0] if return_outputs else res
            loss = loss / self.args.gradient_accumulation_steps
            return (loss, res[1:]) if return_outputs else loss
        return res

    def _get_train_sampler(self, train_dataset=None):
        get_train_sampler = super()._get_train_sampler
        parameters = inspect.signature(get_train_sampler).parameters
        kwargs = {'train_dataset': train_dataset} if 'train_dataset' in parameters else {}
        return get_train_sampler(**kwargs)

    def get_per_token_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id=-100,
        reduction='mean',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(f'Logits (batch and sequence length dim) {logits.shape[:-1]}'
                             'and labels must have the same shape {labels.shape}')
        loss_mask = labels != label_pad_token_id
        labels = labels.clone()
        labels[~loss_mask] = 0
        if reduction == 'mean':
            reduce_logits = logits.mean(-1)
        elif reduction == 'sum':
            reduce_logits = logits.sum(-1)
        else:
            raise ValueError(f'Invalid reduction: {reduction}')
        if self.template.sequence_parallel_size == 1:
            # https://github.com/huggingface/trl/pull/2799
            # Reduce peak vram consumption with efficient selective log_softmax
            per_token_logps = selective_log_softmax(logits, labels)
            per_token_logps[~loss_mask] = 0
            reduce_logits[~loss_mask] = 0
            return per_token_logps, reduce_logits, loss_mask
        else:
            labels = labels.to(logits.device)
            loss_mask = loss_mask.to(logits.device)
            mean_logits = reduce_logits
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            from swift.trainers.sequence_parallel.utils import GatherLoss
            from swift.trainers.sequence_parallel import sequence_parallel
            position_ids = sequence_parallel.real_position_ids
            total_per_token_logps, total_loss_mask = GatherLoss.apply(per_token_logps, loss_mask, 1, position_ids)
            total_mean_logits = sequence_parallel.gather(mean_logits, dim=1, position_ids=position_ids)
            if position_ids is not None and position_ids.min() == -1:
                _pos_mask = position_ids >= 0
                total_per_token_logps = total_per_token_logps[_pos_mask].contiguous()
                total_mean_logits = total_mean_logits[_pos_mask].contiguous()
                total_loss_mask = total_loss_mask[_pos_mask].contiguous()

            total_loss_mask = total_loss_mask.bool()
            total_per_token_logps = total_per_token_logps * (total_loss_mask)

            if total_per_token_logps.dim() == 1:
                total_per_token_logps = total_per_token_logps.unsqueeze(0)
                total_mean_logits = total_mean_logits.unsqueeze(0)
                total_loss_mask = total_loss_mask.unsqueeze(0)
            return total_per_token_logps, total_mean_logits, total_loss_mask

    def _prepare_rewards(self, reward_funcs, reward_model=None, **kwargs):
        args: RLHFArguments = self.args
        device = self.accelerator.device

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        if reward_funcs:
            for i, reward_func in enumerate(reward_funcs):
                if reward_func in orms:
                    reward_func_class = orms[reward_func]
                    reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                    reward_func_kwargs = {
                        key: getattr(args, key)
                        for key in reward_func_args if key not in ['self', 'args', 'kwargs'] and hasattr(args, key)
                    }
                    if 'tokenizer' in reward_func_args:
                        reward_func_kwargs['tokenizer'] = self.processing_class
                    reward_funcs[i] = reward_func_class(**reward_func_kwargs)
                elif not callable(reward_func):
                    raise ValueError(f'reward_function {reward_func} is not implemented in swift.plugin')

        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for reward_func in reward_funcs:
            if inspect.isfunction(reward_func):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = reward_func.__class__.__name__
            self.reward_func_names.append(reward_func_name)

        self.reward_model_plugins = [None] * len(self.reward_funcs)

        if reward_model is not None:
            reward_template = kwargs.pop('reward_template')
            reward_plugins = args.reward_model_plugin
            if reward_plugins is None:
                reward_plugins = ['default'] * len(reward_model)
            assert len(reward_plugins) == len(reward_model), (
                f"The number of 'reward_model_plugin' ({len(reward_plugins)}) does not match "
                f"the number of 'reward_model' ({len(reward_model)}). "
                "Please provide a corresponding 'reward_model_plugin' for each 'reward_model'.")
            for rm, rm_plugin, rm_template in zip(reward_model, reward_plugins, reward_template):
                # Set encoding mode train(see details in Template.encode).
                # Set max_length to None to disable truncation, as the input length has already been truncated earlier.
                rm_template.set_mode('train')
                rm_template.max_length = None
                if rm_plugin not in rm_plugins:
                    raise ValueError(f'rm_plugin {rm_plugin} is not implemented in swift.llm.plugin')
                self.reward_model_plugins.append(rm_plugins[rm_plugin](model=rm, template=rm_template))
                self.reward_funcs.append(rm)
                self.reward_func_names.append(rm.config._name_or_path.split('/')[-1])

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32).to(device)
        else:
            self.reward_weights = torch.ones(len(self.reward_func_names), dtype=torch.float32).to(device)

        # after init trainer
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True)

    def _prepare_rollout_params(self):
        args: RLHFArguments = self.args
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.max_completion_length = args.max_completion_length
        self.completion_length_limit_scope = args.completion_length_limit_scope  # grpo multi turn
        self.async_generate = args.async_generate

        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
            return_details=True)

    def _prepare_vllm(self, model, vllm_client=None):
        if not is_vllm_available():
            raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                              'Please install vLLM with `pip install vllm -U` to use it.')

        if not self.args.use_vllm:
            return

        assert self.args.rlhf_type in ['grpo', 'gkd']

        if self.vllm_mode == 'server':
            assert vllm_client is not None
            self.vllm_client: VLLMClient = vllm_client
            if self.accelerator.is_main_process:
                self.vllm_client.get_engine_type()
                vllm_use_async_engine = [self.vllm_client.use_async_engine]
                use_gym_env = [self.vllm_client.use_gym_env]
                enable_multi_turn = [self.vllm_client.enable_multi_turn]
                enable_lora = [self.vllm_client.enable_lora]
            else:
                vllm_use_async_engine = [False]
                use_gym_env = [False]
                enable_multi_turn = [self.enable_server_multi_turn]
                enable_lora = [False]
            self.vllm_use_async_engine = broadcast_object_list(vllm_use_async_engine, from_process=0)[0]
            self.use_gym_env = broadcast_object_list(use_gym_env, from_process=0)[0]
            self.enable_server_multi_turn = broadcast_object_list(enable_multi_turn, from_process=0)[0]
            self.rollout_enable_lora = broadcast_object_list(enable_lora, from_process=0)[0]
            if self.use_gym_env:
                self.reward_func_names = ['gym_reward']

        elif self.vllm_mode == 'colocate':
            if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                raise ValueError(f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                                 f'({self.accelerator.num_processes}) evenly.')

            if self.vllm_tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration([
                    list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                    for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                ])
            self.enable_offload = self.args.offload_model or self.args.offload_optimizer
            context = self.offload_context if self.enable_offload else nullcontext

            with context():
                self.engine = self._prepare_vllm_engine(model)
                if self.args.sleep_level > 0:
                    self.engine.engine.sleep(self.args.sleep_level)

        self.base_sync_done = False

    def _prepare_vllm_engine(self, model):
        from swift.tuners import Swift
        from swift.llm.infer.infer_engine import GRPOVllmEngine
        max_num_seqs = (
            self.args.per_device_train_batch_size * self.vllm_tensor_parallel_size * self.args.steps_per_generation)
        vllm_template = copy(self.template)
        vllm_template.padding_free = False
        lora_kwargs = {}
        is_moe = model.model_info.is_moe_model
        vllm_enable_lora = self.args.vllm_enable_lora
        if self.args.train_type == 'lora' and vllm_enable_lora:
            lora_kwargs = {
                'enable_lora': self.args.vllm_enable_lora,
                'max_loras': 1,
                'max_lora_rank': self.args.lora_rank,
            }
            self.rollout_enable_lora = True

            if is_moe:
                logger.warning(
                    'vLLM LoRA is enabled for an MoE model. This may cause errors when applying LoRA to expert layers, '
                    'as vLLM currently does not support LoRA in MoE configurations. If you encounter errors, '
                    'please set vllm_enable_lora to False.')

            if self.is_multimodal:
                logger.warning('vLLM LoRA is enabled for a multimodal model. This may lead to unexpected issues '
                               'when applying LoRA to the ViT component, as vLLM does not yet support this setup. '
                               'If errors occur, please disable LoRA by setting vllm_enable_lora to False.')

            patch_vllm_load_adapter()
        with Swift.grpo_context(model, self.template.processor):
            set_expandable_segments(False)
            engine = GRPOVllmEngine(
                model.model_dir,
                model.model_info.torch_dtype,
                model_type=model.model_meta.model_type,
                use_async_engine=False,  # TODO: async engine for colocate
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                max_num_seqs=max_num_seqs,
                enforce_eager=self.args.vllm_enforce_eager,
                limit_mm_per_prompt=self.args.vllm_limit_mm_per_prompt,
                enable_sleep_mode=self.args.sleep_level > 0,
                max_model_len=self.args.vllm_max_model_len,
                seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                disable_cascade_attn=self.args.vllm_disable_cascade_attn,
                load_format='dummy',
                template=vllm_template,
                distributed_executor_backend='external_launcher',
                **lora_kwargs,
            )
            set_expandable_segments(True)

        self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()

        return engine

    def split_batches(self):
        """Split model parameters into batches for synchronized weight transfer.

        This method divides model parameters into manageable batches to facilitate efficient
        weight synchronization between vLLM and DeepSpeed. The splitting strategy focuses on
        LLM layers while grouping other components separately.

        Batching Strategy:
            1. LLM layers: Split into N batches based on layer indices (configurable via args.move_model_batches)
            2. Embeddings and LM heads: Grouped into a single batch
            3. Multi-modal components (e.g., vision tower): Grouped into a single batch

        Returns:
            tuple[list[list[str]], list[list[str]]]: A tuple containing two lists:
                - First list: Batches of parameter names (with original prefixes and LoRA layers)
                - Second list: Batches of parameter names with LoRA layers removed and prefixes stripped
                  (for vLLM compatibility). Contains None if no batching is configured.

        Notes:
            - If args.move_model_batches is None, all parameters are returned in a single batch
            - Reference model parameters (containing 'ref_model') are excluded from batching
            - For multi-modal models, components are identified using model_arch metadata
            - LoRA layers are filtered based on the rollout_enable_lora flag

        Raises:
            AssertionError: If no ModuleList is found in the model to determine layer count
        """
        model = self.accelerator.unwrap_model(self.model)
        if self.args.move_model_batches is None:
            # All in one
            return [[n for n, p in model.named_parameters() if 'ref_model' not in n]], [None]

        model_arch = model.model_meta.model_arch
        non_llm_parameters = []
        llm_embeds = []
        parameters = []
        pattern = r'\.(\d+)\.'

        layer_count = None
        # Get the number of layers in LLM modules
        for name, module in model.named_modules():
            if isinstance(module, ModuleList):
                if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                    llm = model_arch.language_model
                    vision_tower = model_arch.vision_tower
                    if any(vt in name for vt in vision_tower):
                        continue
                    if isinstance(llm, list):
                        llm = llm[0]
                    if name.startswith('base_model'):
                        name = name.replace('base_model.', '')
                    if llm in name:
                        layer_count = len(module)
                else:
                    layer_count = len(module)
        assert layer_count is not None, 'Cannot find ModuleList to split modules.'

        n_layers = ceil(layer_count / self.args.move_model_batches)
        for _ in range(self.args.move_model_batches):
            parameters.append([])

        def replace_lora(name):
            if 'lora_' in name:
                return ''
            else:
                if not self.rollout_enable_lora:
                    return name.replace('.base_layer', '')
                else:
                    return name

        def remove_lora_and_prefix(names):
            names = set([re.sub(r'^_model\.', '', replace_lora(n)) for n in names])
            return [n for n in names if n]

        def split_llm(name):
            match = re.search(pattern, name)
            if match:
                number = match.group(1)
                group = int(number) // n_layers
                parameters[group].append(name)
            else:
                llm_embeds.append(name)

        for name, parameter in model.named_parameters():
            if 'ref_model' in name:
                continue
            if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                llm = model_arch.language_model
                vision_tower = model_arch.vision_tower
                if any(vt in name for vt in vision_tower):
                    non_llm_parameters.append(name)
                elif isinstance(llm, list):
                    llm = llm[0]
                    if llm in name:
                        split_llm(name)
                    else:
                        non_llm_parameters.append(name)
            else:
                split_llm(name)

        if llm_embeds:
            parameters.append(llm_embeds)
        if non_llm_parameters:
            parameters.append(non_llm_parameters)
        parameters = [p for p in parameters if p]
        parameters_no_lora = [remove_lora_and_prefix(p_list) for p_list in parameters]
        return parameters, parameters_no_lora

    @patch_profiling_decorator
    def _move_model_to_vllm(self, skip_async_check=False):
        if self.args.async_generate and not skip_async_check:
            # before sync weight, we should wait async generate finish
            self._wait_queue()

        train_type = self.args.train_type

        if train_type == 'full' or (train_type == 'lora' and not self.base_sync_done) or not self.rollout_enable_lora:
            self._move_full_model_to_vllm()
        else:
            self._move_adapter_to_vllm()

    def _move_adapter_to_vllm(self):
        lora_params = OrderedDict()
        for i, parameter_group in enumerate(self.parameter_groups):  # < this is the change
            parameters = [
                parameter for name, parameter in self.model.named_parameters()
                if not parameter_group or name in parameter_group
            ]
            gather_if_zero3 = get_gather_if_zero3_context(self)
            with gather_if_zero3(parameters), patch_lora_merge(self.model, parameter_group):
                assert len(parameters) == len(parameter_group)
                state_dict = {name: p for p, name in zip(parameters, parameter_group)}
                peft_config = self.model.peft_config.get('default', None)
                self.model.merge_adapter()
                cur_lora_params = get_peft_model_state_dict(self.model, state_dict)
                cur_lora_params = {
                    name: param.full_tensor().detach() if hasattr(param, 'full_tensor') else param.detach()
                    for name, param in cur_lora_params.items()
                }
                lora_params.update(cur_lora_params)
                with patch_lora_unmerge(self.model):
                    self.model.unmerge_adapter()
                del cur_lora_params

        if self.vllm_mode == 'server' and self.accelerator.is_main_process:
            bucked = FlattenedTensorBucket(named_tensors=list(lora_params.items()))
            metadatas = bucked.get_metadata()
            flattened_tensor = bucked.get_flattened_tensor()
            self.vllm_client.update_adapter_flattened_param(peft_config, metadatas, flattened_tensor)
        elif self.vllm_mode == 'colocate':
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f'{lora_int_id}',
                lora_int_id=lora_int_id,
                lora_path='dummy_lora_path',
                peft_config=asdict(peft_config),
                lora_tensors=lora_params,
            )
            self.engine.engine.add_lora(lora_reqest)
        del lora_params

    def _load_state_dict_to_vllm(self, state_dict):
        """Load state_dict to vLLM engine (server or colocate mode)"""
        if self.vllm_mode == 'server' and self.accelerator.is_main_process:
            bucket_size_mb = int(os.environ.get('SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE', 512))
            named_params = list(state_dict.items())
            parameter_buckets = _create_parameter_buckets(named_params, bucket_size_mb=bucket_size_mb)

            for bucket in parameter_buckets:
                _process_bucket_with_flattened_tensor(self, bucket)

            del named_params, parameter_buckets
        elif self.vllm_mode == 'colocate':
            llm_model = self.engine.inner_model
            llm_model.load_weights(state_dict.items())
        del state_dict

    def _move_full_model_to_vllm(self):
        gather_if_zero3 = get_gather_if_zero3_context(self)
        is_peft = is_peft_model(self.model)

        for i, parameter_group in enumerate(self.parameter_groups):
            parameter_group_no_lora = self.parameter_groups_no_lora[i]
            parameters = [
                parameter for name, parameter in self.model.named_parameters()
                if not parameter_group or name in parameter_group
            ]

            # Use patch_lora_merge for PEFT models, nullcontext otherwise
            context_manager = patch_lora_merge(self.model, parameter_group) if is_peft else nullcontext()

            with gather_if_zero3(parameters), context_manager:
                if is_peft and self.should_merge_adapter:
                    self.model.merge_adapter()

                state_dict = self.model.state_dict()

                # Process state_dict for PEFT models
                if is_peft:
                    prefix_removed = {k.removeprefix('base_model.model.'): v for k, v in state_dict.items()}
                    state_dict = prefix_removed if self.rollout_enable_lora else {
                        k.replace('.base_layer', ''): v
                        for k, v in prefix_removed.items()
                    }
                    state_dict = {k: v for k, v in state_dict.items() if self.model.prefix not in k}
                    state_dict = {
                        k.replace('modules_to_save.default.', ''): v
                        for k, v in state_dict.items() if 'original_module' not in k
                    }

                # Filter by parameter_group_no_lora
                if parameter_group_no_lora:
                    if is_peft:
                        parameter_group_no_lora = [n.replace('base_model.model.', '') for n in parameter_group_no_lora]
                    state_dict = {k: v for k, v in state_dict.items() if k in parameter_group_no_lora}

                if is_peft:
                    assert len(state_dict) > 0 and all(
                        [state.shape != torch.Size([0]) for state in state_dict.values()])

                # Load to vLLM
                self._load_state_dict_to_vllm(state_dict)

                if is_peft and self.should_merge_adapter:
                    with patch_lora_unmerge(self.model):
                        self.model.unmerge_adapter()

        if is_peft:
            self.base_sync_done = True

        # Reset prefix cache
        if self.vllm_mode == 'server' and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == 'colocate':
            self.engine.engine.reset_prefix_cache()
