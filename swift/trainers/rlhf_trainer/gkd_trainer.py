# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from trl import GKDTrainer as HFGKDTrainer
from trl import SFTTrainer as HFSFTTrainer

from swift.utils import get_logger, unwrap_model_for_generation
from ..mixin import SwiftMixin
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import identity_data_collator, prepare_deepspeed

del HFGKDTrainer.__init__
del HFSFTTrainer.__init__

logger = get_logger()


class GKDTrainer(RolloutTrainerMixin, SwiftMixin, HFGKDTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        teacher_model = kwargs.pop('teacher_model')
        teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
        self.vllm_client = kwargs.pop('vllm_client', None)
        rlhf_args = kwargs.pop('rlhf_args', None)
        kwargs['data_collator'] = identity_data_collator
        super().__init__(model, None, *_args, **kwargs)
        args = kwargs['args']
        self.lmbda = args.lmbda
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        self.generation_config = model.generation_config

        # Initialize teacher adapter for conditional distillation
        teacher_adapter_name = getattr(rlhf_args, 'teacher_adapter', None) if rlhf_args else getattr(args, 'teacher_adapter', None)
        self.teacher_adapter = None
        if teacher_adapter_name is not None:
            from swift.plugin import teacher_adapters
            adapter_cls = teacher_adapters.get(teacher_adapter_name)
            if adapter_cls is None:
                raise ValueError(f'Unknown teacher_adapter: {teacher_adapter_name}. '
                                 f'Available adapters: {list(teacher_adapters.keys())}')
            self.teacher_adapter = adapter_cls()
            logger.info(f"Loaded teacher_adapter: {type(self.teacher_adapter).__name__}")

        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize teacher model
        if self.is_deepspeed_enabled:
            if teacher_deepspeed_config is not None:
                self.teacher_model = prepare_deepspeed(
                    teacher_model, self.accelerator, deepspeed_config=teacher_deepspeed_config, training_args=args)
            else:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.teacher_model.eval()
        if self.args.offload_teacher_model:
            self.offload_model(self.accelerator.unwrap_model(self.teacher_model))

        # Initialize rollout infrastructure for vLLM support
        if args.use_vllm:
            self.prepare_rollout()
            logger.info('vLLM engine initialized for GKD training')

        # Initialize activation offloading context
        args.activation_offloading = False  # TODO: remove
        if args.activation_offloading:
            from trl.models import get_act_offloading_ctx_manager
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = nullcontext()

    # Code borrowed from huggingface/trl
    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        assert not self.template.padding_free, 'generate not support padding_free/packing.'
        # Generate output with respect to the prompt only
        model_inputs = {k: v for k, v in inputs.items()
                       if not k.startswith('prompt')
                       and not k.startswith('teacher_prompt')
                       and k not in ('labels', 'teacher_prompts')}
        model_inputs['input_ids'] = inputs['prompts']
        model_inputs.update({k[len('prompt_'):]: v for k, v in inputs.items() if k.startswith('prompt_')})
        model_inputs.pop('position_ids', None)
        kwargs = {}
        base_model = self.template.get_base_model(model)
        parameters = inspect.signature(base_model.generate).parameters
        if 'use_model_defaults' in parameters:
            kwargs['use_model_defaults'] = False
        with self.template.generate_context():
            if self.model.model_meta.is_multimodal:
                _, model_inputs = self.template.pre_forward_hook(model, None, model_inputs)
            generated_outputs = model.generate(
                **model_inputs, generation_config=generation_config, return_dict_in_generate=True, **kwargs)
        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        if not self.template.skip_prompt:
            generated_tokens = torch.concat([inputs['prompts'], generated_tokens], dim=1)
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()
        new_labels[:, :inputs['prompts'].shape[1]] = -100

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        new_position_ids = new_attention_mask.cumsum(dim=1) - 1
        new_position_ids[new_position_ids < 0] = 0
        inputs['position_ids'] = new_position_ids
        return generated_tokens, new_attention_mask, new_labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {k: v for k, v in inputs.items()
                       if k not in {'prompt', 'labels', 'teacher_prompts'}
                       and not k.startswith('prompt_')
                       and not k.startswith('teacher_prompt_')}
        # If generate is used, then use_logits_to_keep must be set to False.
        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            model_inputs['logits_to_keep'] = inputs['logits_to_keep']
        if self.args.sft_alpha > 0:
            model_inputs['labels'] = inputs['labels']
        # compute student output
        outputs_student = model(**model_inputs)

        model_inputs.pop('labels', None)
        model_inputs.pop('logits_to_keep', None)

        # Prepare teacher model inputs
        teacher_model_inputs = model_inputs.copy()
        if 'teacher_prompts' in inputs:
            # Conditional distillation: replace student prompts with teacher prompts
            # Use labels to identify response tokens (where labels != -100)
            batch_size = inputs['input_ids'].shape[0]

            # Get teacher prompt lengths
            if 'teacher_prompt_attention_mask' in inputs:
                teacher_prompts_len = inputs['teacher_prompt_attention_mask'].sum(dim=1)  # [batch_size]
            else:
                teacher_prompts_len = torch.full((inputs['teacher_prompts'].shape[0],),
                                                 inputs['teacher_prompts'].shape[1],
                                                 device=inputs['teacher_prompts'].device)

            # Build teacher_input_ids: teacher_prompt + response_tokens
            # Extract response tokens using labels (where labels != -100)
            teacher_input_ids_list = []
            teacher_attention_mask_list = []

            for i in range(batch_size):
                # When use_logits_to_keep=True, labels is truncated to the relevant portion
                # Extract the corresponding tokens from input_ids (last N tokens)
                # Reference: grpo_trainer.py line 1355, 1459
                labels_len = inputs['labels'][i].shape[0]
                relevant_input_ids = inputs['input_ids'][i, -labels_len:]

                # Find response tokens using labels (where labels != -100)
                response_mask = inputs['labels'][i] != -100
                response_tokens = relevant_input_ids[response_mask]

                # Get teacher prompt (without padding)
                teacher_prompt_len = teacher_prompts_len[i].item()
                teacher_prompt = inputs['teacher_prompts'][i, :teacher_prompt_len]

                # Concatenate: teacher_prompt + response_tokens
                teacher_seq = torch.cat([teacher_prompt, response_tokens])
                teacher_input_ids_list.append(teacher_seq)

                # Create attention mask (all 1s for valid tokens)
                teacher_attn = torch.ones(len(teacher_seq), dtype=torch.long, device=teacher_seq.device)
                teacher_attention_mask_list.append(teacher_attn)

            # Pad to same length
            max_len = max(len(seq) for seq in teacher_input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

            teacher_input_ids = torch.full((batch_size, max_len), pad_token_id,
                                          dtype=inputs['input_ids'].dtype,
                                          device=inputs['input_ids'].device)
            teacher_attention_mask = torch.zeros((batch_size, max_len),
                                                dtype=torch.long,
                                                device=inputs['input_ids'].device)

            for i, (seq, attn) in enumerate(zip(teacher_input_ids_list, teacher_attention_mask_list)):
                teacher_input_ids[i, :len(seq)] = seq
                teacher_attention_mask[i, :len(attn)] = attn

            teacher_model_inputs['input_ids'] = teacher_input_ids
            teacher_model_inputs['attention_mask'] = teacher_attention_mask

            # Recalculate position_ids if present
            if 'position_ids' in teacher_model_inputs:
                teacher_model_inputs['position_ids'] = teacher_attention_mask.cumsum(dim=1) - 1
                teacher_model_inputs['position_ids'][teacher_model_inputs['position_ids'] < 0] = 0

        load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
        with torch.no_grad(), load_context:
            outputs_teacher = self.teacher_model(**teacher_model_inputs)

        # Extract logits for distillation
        if 'teacher_prompts' in inputs:
            # Different prompt lengths: need to extract logits carefully
            # Student logits: from response part only
            shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
            student_mask = shifted_labels != -100
            shifted_student_logits = outputs_student.logits[student_mask][None]

            # Teacher logits: also from response part, but at different positions
            # Calculate where teacher's response starts
            if 'teacher_prompt_attention_mask' in inputs:
                teacher_prompts_len = inputs['teacher_prompt_attention_mask'].sum(dim=1)  # [batch_size]
            else:
                teacher_prompts_len = torch.full((inputs['teacher_prompts'].shape[0],),
                                                 inputs['teacher_prompts'].shape[1],
                                                 device=inputs['teacher_prompts'].device)

            # Create labels for teacher based on response length
            teacher_labels = torch.full_like(teacher_model_inputs['input_ids'], -100)
            response_len = student_mask.sum(dim=1)  # Length of response for each sample
            for i in range(batch_size):
                start_pos = teacher_prompts_len[i].item()
                teacher_labels[i, start_pos:start_pos + response_len[i].item()] = 1

            shifted_teacher_labels = torch.roll(teacher_labels, shifts=-1, dims=1)
            teacher_mask = shifted_teacher_labels != -100
            shifted_teacher_logits = outputs_teacher.logits[teacher_mask][None]
        else:
            # Standard case: same prompts for both
            shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
            mask = shifted_labels != -100
            shifted_student_logits = outputs_student.logits[mask][None]
            shifted_teacher_logits = outputs_teacher.logits[mask][None]

        # Fix the vocab_size mismatch between Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct.
        stu_dim = shifted_student_logits.shape[-1]
        tea_dim = shifted_teacher_logits.shape[-1]
        if stu_dim < tea_dim:
            shifted_student_logits = F.pad(shifted_student_logits, (0, tea_dim - stu_dim), 'constant', 0)
            shifted_student_logits[..., stu_dim:] = shifted_teacher_logits[..., stu_dim:]
        elif stu_dim > tea_dim:
            shifted_teacher_logits = F.pad(shifted_teacher_logits, (0, stu_dim - tea_dim), 'constant', 0)
            shifted_teacher_logits[..., tea_dim:] = shifted_student_logits[..., tea_dim:]

        # compute loss
        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            beta=self.beta,
        )
        if self.args.sft_alpha > 0:
            loss = loss + self.args.sft_alpha * outputs_student.loss

        # Return loss
        return (loss, outputs_student) if return_outputs else loss

    def _prepare_batch_inputs(self, inputs: list) -> Dict[str, torch.Tensor]:
        template = self.template
        batch_encoded_inputs = []

        for data in inputs:
            if 'response_token_ids' in data and data['response_token_ids']:
                from .utils import replace_assistant_response_with_ids
                data['messages'] = replace_assistant_response_with_ids(data['messages'], data['response_token_ids'])

            # Use teacher_adapter to generate teacher_history if available
            if self.teacher_adapter is not None:
                # Prepare student messages (without final assistant response if present)
                student_messages = data['messages'][:-1] if data['messages'][-1]['role'] == 'assistant' else data['messages']
                # Pass complete data dict to adapter (similar to GRPO's reward_model_plugin design)
                # Temporarily replace messages with student_messages for adapter
                original_messages = data['messages']
                data['messages'] = student_messages
                teacher_history = self.teacher_adapter.shape_context(data)
                data['messages'] = original_messages  # Restore
                data['teacher_history'] = teacher_history

            encoded = template.encode(data, return_length=True)
            batch_encoded_inputs.append(encoded)

        from swift.llm import to_device
        batch_encoded = to_device(template.data_collator(batch_encoded_inputs), self.model.device)

        return batch_encoded

    # Code borrowed from huggingface/trl
    def training_step(self,
                      model: nn.Module,
                      inputs: DataType,
                      num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper.
        With probability `self.lmbda`, it generates new responses using the student model,
        which are then used for training instead of the original inputs.

        When use_vllm is enabled, vLLM engine is used for faster generation.
        """
        args = self.args
        if self._get_random_num() <= self.lmbda:
            # On-policy: student model generates responses
            if args.use_vllm:
                processed_inputs = self._preprocess_inputs(inputs)
                inputs = self._fast_infer(processed_inputs)
                inputs = self._prepare_batch_inputs(inputs)
            else:
                inputs = self._prepare_batch_inputs(inputs)
                with unwrap_model_for_generation(
                        model, self.accelerator,
                        gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                    unwrapped_model.eval()
                    new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                        unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id)
                    unwrapped_model.train()
                inputs['input_ids'] = new_input_ids
                inputs['attention_mask'] = new_attention_mask
                inputs['labels'] = new_labels

        elif self.seq_kd:
            inputs = self._prepare_batch_inputs(inputs)
            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with load_context, unwrap_model_for_generation(
                    self.teacher_model, self.accelerator,
                    gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id)
            inputs['input_ids'] = new_input_ids
            inputs['attention_mask'] = new_attention_mask
            inputs['labels'] = new_labels

        else:
            inputs = self._prepare_batch_inputs(inputs)
        assert not isinstance(inputs, list)
        with self.template.forward_context(self.model, inputs):
            loss = HFSFTTrainer.training_step(self, model, inputs, num_items_in_batch)
        return loss

    def prediction_step(self, model, inputs, *args, **kwargs):
        inputs = self._prepare_batch_inputs(inputs)
        with self.template.forward_context(self.model, inputs):
            return super().prediction_step(model, inputs, *args, **kwargs)

    @contextmanager
    def offload_context(self):
        """Context manager for offloading model and optimizer during vLLM inference

        This offloads:
        - Student model (self.model)
        - Optimizer states

        to CPU to free up GPU memory for vLLM engine.
        """
        if self.args.offload_model:
            self.offload_model(self.accelerator.unwrap_model(self.model))
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            self.offload_optimizer()

        try:
            yield
        finally:
            # reload (load back) model when exiting context
            if self.args.offload_model:
                self.load_model(self.accelerator.unwrap_model(self.model))
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                self.load_optimizer()

    def _get_random_num(self) -> float:
        """
        Generate a deterministic random number.

        Uses an isolated Random instance to avoid interfering with the global
        random state, ensuring thread-safety and consistent behavior across processes.

        Returns:
            float: A random number in the range [0.0, 1.0).
        """
        seed = int(getattr(self.args, 'seed', 0))
        seed += int(self.state.global_step)
        rng = random.Random(seed)
        return rng.random()

    @contextmanager
    def load_teacher_model_context(self):
        """
        Context manager to load and offload the teacher model with memory and timing profiling.
        """
        if not self.args.offload_teacher_model:
            yield
            return

        self.load_model(self.accelerator.unwrap_model(self.teacher_model))
        yield
        self.offload_model(self.accelerator.unwrap_model(self.teacher_model))
