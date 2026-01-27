# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import random
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import trl
from accelerate.utils import gather_object, is_peft_model
from packaging import version
from transformers import PreTrainedModel
from trl import GKDTrainer as HFGKDTrainer
from trl import SFTTrainer as HFSFTTrainer

from swift.template import TemplateInputs
from swift.trainers import SwiftMixin, disable_gradient_checkpointing
from swift.utils import (JsonlWriter, get_logger, is_swanlab_available, is_wandb_available, remove_response, to_device,
                         unwrap_model_for_generation)
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import (get_gather_if_zero3_context, identity_data_collator, patch_profiling_context,
                    patch_profiling_decorator, prepare_deepspeed)

try:
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
    _liger_kernel_available = True
except ImportError:
    _liger_kernel_available = False

del HFGKDTrainer.__init__
del HFSFTTrainer.__init__

logger = get_logger()
if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab


class DataSource(str, Enum):
    STUDENT = 'student'  # On-policy: student model generates responses
    TEACHER = 'teacher'  # Sequential KD: teacher model generates responses
    DATASET = 'dataset'  # Off-policy: use dataset responses


class GKDTrainer(RolloutTrainerMixin, SwiftMixin, HFGKDTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        teacher_model = kwargs.pop('teacher_model', None)
        teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
        self.vllm_client = kwargs.pop('vllm_client', None)
        self.teacher_api_client = kwargs.pop('teacher_api_client', None)
        super().__init__(model, None, *_args, **kwargs)
        args = kwargs['args']
        self.lmbda = args.lmbda
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        self.generation_config = model.generation_config
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._total_train_tokens = 0

        # GKD top-k logits configuration
        self.gkd_logits_topk = getattr(args, 'gkd_logits_topk', None)
        self.use_teacher_api = self.teacher_api_client is not None

        # Initialize logging components
        self._prepare_logging()

        # Initialize liger loss (only when not using top-k mode)
        if self.gkd_logits_topk is None:
            self._prepare_liger_loss()
        else:
            self.use_liger_gkd_loss = False
            logger.info(f'Using top-k logits (k={self.gkd_logits_topk}) for KL computation, liger loss disabled.')

        self.teacher_ds3_gather_for_generation = args.ds3_gather_for_generation
        self.is_teacher_ds3 = None
        self.teacher_model = None

        # Initialize teacher model (skip if using API)
        if not self.use_teacher_api:
            if teacher_model is None:
                raise ValueError('teacher_model is required when not using teacher_model_server')
            if self.is_deepspeed_enabled:
                if teacher_deepspeed_config is not None:
                    self.is_teacher_ds3 = teacher_deepspeed_config.get('zero_optimization', {}).get('stage') == 3
                    if not self.is_teacher_ds3:
                        self.teacher_ds3_gather_for_generation = False
                    self.teacher_model = prepare_deepspeed(
                        teacher_model, self.accelerator, deepspeed_config=teacher_deepspeed_config, training_args=args)
                else:
                    self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            elif self.is_fsdp_enabled:
                from .utils import prepare_fsdp
                self.teacher_model = prepare_fsdp(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
            self.teacher_model.eval()
            if self.args.offload_teacher_model:
                self.offload_model(self.accelerator.unwrap_model(self.teacher_model))
        else:
            logger.info(f'Using teacher model API for logprobs, top_logprobs={self.gkd_logits_topk}')

        # Initialize rollout infrastructure for vLLM support
        self.prepare_rollout()

        # Initialize activation offloading context
        args.activation_offloading = False  # TODO: remove
        if args.activation_offloading:
            from trl.models import get_act_offloading_ctx_manager
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = nullcontext()
        self._trl_version_gte_0_24 = version.parse(trl.__version__) >= version.parse('0.24')

        # Initialize resample data iterator for truncation_strategy 'raise'('delete')
        if self.template.truncation_strategy == 'raise':
            self._prepare_resample_data_iterator()

    def _get_data_collator(self, args, template):
        return identity_data_collator

    # Code borrowed from huggingface/trl
    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs using the model.

        When encode_prompt_only=True, inputs['input_ids'] already contains only the prompt part.
        """
        assert not self.template.padding_free, 'generate not support padding_free/packing.'
        prompt_input_ids = inputs['input_ids']
        model_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        model_inputs.pop('position_ids', None)
        model_inputs.pop('text_position_ids', None)
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
            generated_tokens = torch.concat([prompt_input_ids, generated_tokens], dim=1)
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()
        new_labels[:, :prompt_input_ids.shape[1]] = -100

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        new_position_ids = new_attention_mask.cumsum(dim=1) - 1
        new_position_ids[new_position_ids < 0] = 0
        inputs['position_ids'] = new_position_ids
        return generated_tokens, new_attention_mask, new_labels

    @patch_profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get data source: DataSource.STUDENT, DataSource.TEACHER, or DataSource.DATASET
        data_source = inputs.pop('_data_source', DataSource.DATASET)
        # Get teacher logprobs from API if available (set in training_step)
        teacher_api_logprobs = inputs.pop('_teacher_api_logprobs', None)
        teacher_api_indices = inputs.pop('_teacher_api_indices', None)

        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}
        # If generate is used, then use_logits_to_keep must be set to False.
        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep and not self.use_liger_gkd_loss:
            self.prepare_logits_to_keep(inputs)
            model_inputs['logits_to_keep'] = inputs['logits_to_keep']

        if self.use_liger_gkd_loss:
            # Liger fused JSD loss for memory efficiency
            # Get base models (exclude lm_head to save memory)
            unwrapped_student = self.accelerator.unwrap_model(model)
            if is_peft_model(unwrapped_student):
                unwrapped_student = unwrapped_student.base_model.model
            base_student = getattr(unwrapped_student, getattr(unwrapped_student, 'base_model_prefix', 'model'),
                                   unwrapped_student)

            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            base_teacher = getattr(unwrapped_teacher, getattr(unwrapped_teacher, 'base_model_prefix', 'model'),
                                   unwrapped_teacher)

            # Forward through base models
            student_outputs = base_student(**model_inputs, use_cache=False)

            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with load_context:
                with torch.no_grad(), disable_gradient_checkpointing(self.teacher_model,
                                                                     self.args.gradient_checkpointing_kwargs):
                    teacher_outputs = base_teacher(**model_inputs, use_cache=False)

                # Get hidden states (shifted)
                student_hidden = student_outputs.last_hidden_state[:, :-1]
                teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

                # Release full outputs to free memory
                del student_outputs, teacher_outputs

                # Prepare labels (shifted)
                labels_mask = inputs['labels'] != -100
                masked_input_ids = torch.where(labels_mask, inputs['input_ids'],
                                               torch.full_like(inputs['input_ids'], -100))
                true_labels = masked_input_ids[:, 1:].contiguous()

                # Release intermediate tensors
                del labels_mask, masked_input_ids

                # Get output heads
                student_head = unwrapped_student.get_output_embeddings()
                teacher_head = unwrapped_teacher.get_output_embeddings()

                # Prepare context managers for gathering parameters in zero3
                teacher_context = get_gather_if_zero3_context(self, is_zero3=self.is_teacher_ds3)(teacher_head.weight)
                student_context = get_gather_if_zero3_context(self)(student_head.weight)

                with teacher_context, student_context:
                    # Compute liger fused JSD loss
                    loss = self.liger_jsd_loss(
                        student_input=student_hidden,
                        student_weight=student_head.weight,
                        teacher_input=teacher_hidden,
                        teacher_weight=teacher_head.weight,
                        true_labels=true_labels,
                        student_bias=getattr(student_head, 'bias', None),
                        teacher_bias=getattr(teacher_head, 'bias', None),
                    )
                    # loss / grad norm is unexpectedly large, normalize by sequence length
                    # https://github.com/linkedin/Liger-Kernel/blob/v0.6.3/src/liger_kernel/chunked_loss/jsd_loss.py#L9-L39
                    loss /= student_hidden.shape[1]
                # Release hidden states after loss computation
                del student_hidden, teacher_hidden, true_labels
            outputs_student = None
        elif self.use_teacher_api:
            assert teacher_api_logprobs is not None
            # API mode: use teacher logprobs from external service
            if self.args.sft_alpha > 0:
                model_inputs['labels'] = inputs['labels']
            outputs_student = model(**model_inputs)

            shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)

            # Compute top-k JSD loss with API logprobs
            loss = self.generalized_jsd_loss(
                student_logits=outputs_student.logits,
                teacher_logits=None,  # Not used in API mode
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
                teacher_topk_logprobs=teacher_api_logprobs,
                teacher_topk_indices=teacher_api_indices,
            )

            # Add SFT loss if enabled (skip for student-generated responses)
            if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
                loss = loss + self.args.sft_alpha * outputs_student.loss
        else:
            # Standard loss computation (local teacher model)
            if self.args.sft_alpha > 0:
                model_inputs['labels'] = inputs['labels']
            # compute student output
            outputs_student = model(**model_inputs)

            model_inputs.pop('labels', None)
            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with torch.no_grad(), load_context, disable_gradient_checkpointing(self.teacher_model,
                                                                               self.args.gradient_checkpointing_kwargs):
                outputs_teacher = self.teacher_model(**model_inputs)

            shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)

            if self.gkd_logits_topk is not None:
                # Top-k mode with local teacher
                loss = self.generalized_jsd_loss(
                    student_logits=outputs_student.logits,
                    teacher_logits=outputs_teacher.logits,
                    labels=shifted_labels,
                    beta=self.beta,
                    temperature=self.temperature,
                    topk=self.gkd_logits_topk,
                )
            else:
                # Full vocabulary mode
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
                    temperature=self.temperature,
                )

            # Add SFT loss if enabled (skip for student-generated responses)
            if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
                loss = loss + self.args.sft_alpha * outputs_student.loss

        # Return loss
        if return_outputs:
            return (loss, outputs_student)
        else:
            return loss

    def _prepare_batch_inputs(self, inputs: list, encode_prompt_only: bool = False) -> Dict[str, torch.Tensor]:
        """Prepare batch inputs for training.

        Args:
            inputs: List of input data dictionaries
            encode_prompt_only: If True, only encode the prompt part (for on-policy/seq_kd generation).
                               If False, encode the full messages including response (for offline dataset).
        """
        from .utils import replace_assistant_response_with_ids

        template = self.template
        batch_encoded_inputs = []

        # Use 'transformers' mode for prompt-only encoding, 'train' mode for full encoding
        mode = 'transformers' if encode_prompt_only else 'train'
        with self._template_context(template, mode=mode):
            for data in inputs:
                if 'response_token_ids' in data and data['response_token_ids']:
                    data['messages'] = replace_assistant_response_with_ids(data['messages'], data['response_token_ids'])

                if encode_prompt_only:
                    # Remove response content for prompt-only encoding
                    messages = data.get('messages', [])
                    if messages and messages[-1].get('role') == 'assistant':
                        messages[-1]['content'] = None

                encoded = template.encode(data, return_length=True)
                batch_encoded_inputs.append(encoded)

            batch_encoded = to_device(template.data_collator(batch_encoded_inputs), self.model.device)

        return batch_encoded

    # Code borrowed from huggingface/trl
    @patch_profiling_decorator
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
        with patch_profiling_context(self, 'get_completions'):
            if self._get_random_num() <= self.lmbda:
                # On-policy: student model generates responses
                data_source = DataSource.STUDENT
                # Resample inputs that fail encoding when truncation_strategy is 'raise'('delete')
                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)
                if args.use_vllm:
                    processed_inputs = self._preprocess_inputs(inputs)
                    generated_inputs = self._fast_infer(processed_inputs)
                    if self.log_completions:
                        messages = [inp['messages'][:-1] for inp in generated_inputs]
                        completions = [deepcopy(inp['messages'][-1]['content']) for inp in generated_inputs]
                        valid_messages = gather_object(messages)
                        valid_completions = gather_object(completions)
                        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
                        self._logs['completion'].extend(valid_completions)
                    with self._template_context(self.template):
                        # vLLM already generated response, encode full messages
                        encoded_inputs = self._prepare_batch_inputs(generated_inputs, encode_prompt_only=False)
                else:
                    # Need prompt-only encoding for on-policy generation
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                    with unwrap_model_for_generation(
                            model, self.accelerator,
                            gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                        unwrapped_model.eval()
                        new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                            unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)
                        unwrapped_model.train()
                    # override with generated inputs
                    encoded_inputs['input_ids'] = new_input_ids
                    encoded_inputs['attention_mask'] = new_attention_mask
                    encoded_inputs['labels'] = new_labels

            elif self.seq_kd:
                # Sequential KD: teacher model generates responses
                data_source = DataSource.TEACHER

                # Resample inputs that fail encoding when truncation_strategy is 'raise'('delete')
                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)
                # Need prompt-only encoding for teacher generation
                encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
                with load_context, unwrap_model_for_generation(
                        self.teacher_model,
                        self.accelerator,
                        gather_deepspeed3_params=self.teacher_ds3_gather_for_generation) as unwrapped_model:
                    unwrapped_model.eval()
                    new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                        unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)
                # override with generated inputs
                encoded_inputs['input_ids'] = new_input_ids
                encoded_inputs['attention_mask'] = new_attention_mask
                encoded_inputs['labels'] = new_labels

            else:
                # Off-policy: use dataset responses, encode full messages
                data_source = DataSource.DATASET
                total_length = self.template.max_length + self.max_completion_length
                with self._template_context(self.template, max_length=total_length):
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)

            # Mark data source for downstream processing (e.g., conditional SFT loss)
            encoded_inputs['_data_source'] = data_source

            # Fetch teacher logprobs from API if using external teacher service
            if self.use_teacher_api:
                teacher_logprobs, teacher_indices = self._fetch_teacher_logprobs_from_api(encoded_inputs)
                encoded_inputs['_teacher_api_logprobs'] = teacher_logprobs
                encoded_inputs['_teacher_api_indices'] = teacher_indices

        with self.template.forward_context(self.model, encoded_inputs):
            loss = HFSFTTrainer.training_step(self, model, encoded_inputs, num_items_in_batch)
        return loss

    def _fetch_teacher_logprobs_from_api(self, encoded_inputs: Dict[str, torch.Tensor]):
        """Fetch teacher logprobs from external API service.

        Args:
            encoded_inputs: Dictionary containing input_ids, attention_mask, labels, etc.

        Returns:
            Tuple of (teacher_logprobs, teacher_indices) tensors
        """
        import asyncio

        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        batch_size, seq_len = input_ids.shape
        topk = self.gkd_logits_topk

        # Prepare requests for API
        # We need logprobs for each position, so we send the full sequence
        # and request prompt_logprobs
        async def fetch_batch():
            results = await self.teacher_api_client.get_logprobs_batch(
                input_ids=input_ids.tolist(),
                attention_mask=attention_mask.tolist(),
                top_logprobs=topk,
            )
            return results

        # Run async function
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, fetch_batch())
                api_results = future.result()
        else:
            api_results = loop.run_until_complete(fetch_batch())

        # Parse API results into tensors
        # api_results should be list of dicts with 'logprobs' and 'indices' for each sample
        teacher_logprobs = torch.zeros(batch_size, seq_len, topk, device=input_ids.device, dtype=torch.float32)
        teacher_indices = torch.zeros(batch_size, seq_len, topk, device=input_ids.device, dtype=torch.long)

        for batch_idx, result in enumerate(api_results):
            for pos_idx, pos_logprobs in enumerate(result.get('logprobs', [])):
                if pos_idx >= seq_len:
                    break
                for k_idx, (token_id, logprob) in enumerate(pos_logprobs[:topk]):
                    teacher_logprobs[batch_idx, pos_idx, k_idx] = logprob
                    teacher_indices[batch_idx, pos_idx, k_idx] = token_id

        return teacher_logprobs, teacher_indices

    def prediction_step(self, model, inputs, *args, **kwargs):
        # Prediction uses full messages
        encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)
        with self.template.forward_context(self.model, encoded_inputs):
            return super().prediction_step(model, encoded_inputs, *args, **kwargs)

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

    def _prepare_liger_loss(self):
        """Initialize liger loss if enabled."""
        args = self.args
        self.use_liger_gkd_loss = False
        if getattr(args, 'use_liger_kernel', False):
            if not _liger_kernel_available:
                raise ImportError(
                    'Liger kernel is not installed. Please install liger-kernel by running: pip install liger-kernel')
            assert self.args.sft_alpha == 0, 'SFT loss is not supported with liger loss'

            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=self.beta,
                ignore_index=-100,
                temperature=self.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        chunk_size=512,
        topk=None,
        teacher_topk_logprobs=None,
        teacher_topk_indices=None,
    ):
        """Compute generalized JSD loss with optional top-k support.

        This method supports three modes:
        1. Full vocabulary mode (default): Uses complete logits from both models
        2. Top-k mode with local teacher: Extracts top-k from teacher_logits
        3. Top-k mode with API logprobs: Uses pre-computed teacher_topk_logprobs and indices

        For top-k mode, uses the teacher model's top-k tokens (following ROLL framework).
        This reduces memory usage while maintaining training effectiveness.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size] or [num_tokens, vocab_size]
            teacher_logits: Teacher model logits (same shape as student_logits), can be None for API mode
            labels: Token labels for masking, shape [batch, seq_len]
            beta: JSD interpolation coefficient (0=Forward KL, 0.5=JSD, 1=Reverse KL)
            temperature: Temperature for softmax scaling
            chunk_size: Chunk size for memory-efficient processing (full vocab mode only)
            topk: Number of top-k logits to use (teacher's top-k). None for full vocabulary mode.
            teacher_topk_logprobs: Pre-computed teacher log probs [batch, seq_len, topk] (API mode)
            teacher_topk_indices: Pre-computed teacher token indices [batch, seq_len, topk] (API mode)

        Returns:
            Scalar loss value
        """
        # Determine mode
        use_api_mode = teacher_topk_logprobs is not None and teacher_topk_indices is not None
        use_topk = topk is not None or use_api_mode

        # ============== Top-K Mode ==============
        if use_topk:
            # Apply temperature scaling to student logits
            student_logits_scaled = student_logits / temperature

            if use_api_mode:
                # API mode: teacher logprobs already computed (with temperature on server)
                teacher_topk_log_probs = teacher_topk_logprobs
                teacher_topk_probs = torch.exp(teacher_topk_logprobs)
                topk_indices = teacher_topk_indices
            else:
                # Local mode: extract top-k from teacher logits
                teacher_logits_scaled = teacher_logits / temperature
                teacher_topk_logits, topk_indices = torch.topk(teacher_logits_scaled, k=topk, dim=-1)
                teacher_topk_probs = F.softmax(teacher_topk_logits, dim=-1)
                teacher_topk_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)

            # Gather student logits at teacher's top-k indices and renormalize
            student_topk_logits = torch.gather(student_logits_scaled, dim=-1, index=topk_indices)
            student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)

            # Compute JSD on top-k distribution
            if beta == 0:
                # Forward KL: KL(teacher || student)
                jsd = (teacher_topk_probs * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1)
            elif beta == 1:
                # Reverse KL: KL(student || teacher)
                student_topk_probs = F.softmax(student_topk_logits, dim=-1)
                jsd = (student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)).sum(dim=-1)
            else:
                # Full JSD with mixture distribution
                student_topk_probs = F.softmax(student_topk_logits, dim=-1)
                mixture_probs = beta * teacher_topk_probs + (1 - beta) * student_topk_probs
                mixture_log_probs = torch.log(mixture_probs + 1e-10)
                kl_teacher = (teacher_topk_probs * (teacher_topk_log_probs - mixture_log_probs)).sum(dim=-1)
                kl_student = (student_topk_probs * (student_topk_log_probs - mixture_log_probs)).sum(dim=-1)
                jsd = beta * kl_teacher + (1 - beta) * kl_student

            # Apply mask and compute mean
            if labels is not None:
                mask = labels != -100
                jsd = jsd * mask.float()
                num_valid = mask.sum()
            else:
                num_valid = jsd.numel()

            if num_valid == 0:
                return student_logits.new_zeros(())
            return jsd.sum() / num_valid

        # ============== Full Vocabulary Mode ==============
        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Apply masking if labels provided
        if labels is not None:
            mask = labels != -100
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
            num_valid = mask.sum()
        else:
            # Flatten to [num_tokens, vocab_size]
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
            num_valid = student_logits.size(0)

        if num_valid == 0:
            return student_logits.new_zeros(())

        num_valid_int = num_valid if isinstance(num_valid, int) else num_valid.item()
        total_loss = student_logits.new_zeros(())

        # Precompute beta tensor once if needed
        if beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        # Process in chunks to reduce peak memory
        for start_idx in range(0, num_valid_int, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid_int)
            s_chunk = student_logits[start_idx:end_idx]
            t_chunk = teacher_logits[start_idx:end_idx]

            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            del s_chunk, t_chunk

            if beta == 0:
                jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
            elif beta == 1:
                jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
            else:
                mixture_log_probs = torch.logsumexp(
                    torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                    dim=0,
                )

                kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
                kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
                del mixture_log_probs

                jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
                del kl_teacher, kl_student

            total_loss = total_loss + jsd_chunk.sum()
            del jsd_chunk, s_log_probs, t_log_probs

        return total_loss / num_valid

    def _prepare_logging(self):
        """Initialize logging components for on-policy rollout tracking."""
        args = self.args
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = getattr(args, 'wandb_log_unique_prompts', False)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))

        # Initialize logs deque for storing rollout data (aligned with GRPO)
        self._logs = {
            'prompt': deque(),
            'completion': deque(),
        }

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        """Convert messages list to prompt text list using template (aligned with GRPO)."""
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log method to include completion table logging (aligned with GRPO)."""
        # Call parent log method
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:
            super().log(logs)

        # Log completions table if we have data (only for on-policy generations)
        if self.accelerator.is_main_process and self.log_completions and len(self._logs['prompt']) > 0:
            seen_nums = len(self._logs['prompt'])
            table = {
                'step': [str(self.state.global_step)] * seen_nums,
                'prompt': list(self._logs['prompt'])[:seen_nums],
                'completion': list(self._logs['completion'])[:seen_nums],
            }

            # Write to jsonl
            self.jsonl_writer.append(table)

            self._logs['prompt'].clear()
            self._logs['completion'].clear()
            # Log to wandb if enabled
            report_to_wandb = self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None
            if report_to_wandb:
                wandb_table = table.copy()
                import pandas as pd
                df = pd.DataFrame(wandb_table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

            # Log to swanlab if enabled
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab.get_run(
            ) is not None
            if report_to_swanlab:
                headers = list(table.keys())
                rows = []
                for i in range(len(table['step'])):
                    row = [table[header][i] for header in headers]
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})
