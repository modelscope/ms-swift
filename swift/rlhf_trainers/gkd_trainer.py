# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import trl
from accelerate.utils import gather_object, is_peft_model
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from packaging import version
from transformers import PreTrainedModel
from trl import SFTTrainer as HFSFTTrainer
from typing import Dict, Optional, Union

from swift.infer_engine import InferClient, InferRequest, RequestConfig
from swift.template import TemplateInputs
from swift.trainers import SwiftMixin, disable_gradient_checkpointing
from swift.utils import (JsonlWriter, get_logger, is_swanlab_available, is_wandb_available, remove_response, to_device,
                         unwrap_model_for_generation)
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import (get_gather_if_zero3_context, identity_data_collator, prepare_deepspeed, profiling_context,
                    profiling_decorator)

try:
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
    _liger_kernel_available = True
except ImportError:
    _liger_kernel_available = False

if version.parse(trl.__version__) >= version.parse('0.26.0'):
    from trl.experimental.gkd import GKDTrainer as HFGKDTrainer
else:
    from trl import GKDTrainer as HFGKDTrainer

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


@dataclass
class TeacherOutput:
    """Unified container for teacher model outputs from all three sources:
    local full-vocab, local top-k, and external API top-k.
    """
    full_logits: Optional[torch.Tensor] = None
    topk_logprobs: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None
    opsd_teacher_labels: Optional[torch.Tensor] = None

    @property
    def is_topk_mode(self) -> bool:
        return self.topk_logprobs is not None and self.topk_indices is not None

    def validate(self):
        if self.full_logits is None and not self.is_topk_mode:
            raise ValueError('TeacherOutput must provide either full_logits or '
                             '(topk_logprobs, topk_indices). Got neither.')


teacher_model_server_model_name = None


class GKDTrainer(RolloutTrainerMixin, SwiftMixin, HFGKDTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        teacher_model = kwargs.pop('teacher_model', None)
        teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
        self.vllm_client = kwargs.pop('vllm_client', None)
        self.gkd_logits_topk = kwargs.pop('gkd_logits_topk', None)
        teacher_model_server = kwargs.pop('teacher_model_server', None)

        # Self-distillation: reuse base model as teacher via disable_adapter().
        teacher_use_disable_adapter = kwargs.pop('teacher_use_disable_adapter', False)
        super().__init__(model, None, *_args, **kwargs)
        args = kwargs['args']
        self.lmbda = args.lmbda
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        self.generation_config = model.generation_config
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._total_train_tokens = 0

        self.teacher_model_server = teacher_model_server
        self.use_teacher_api = teacher_model_server is not None

        # Initialize logging components
        self._prepare_logging()

        # Initialize liger loss if enabled
        self._prepare_liger_loss()

        self.teacher_ds3_gather_for_generation = args.ds3_gather_for_generation
        self.is_teacher_ds3 = None
        self._teacher_use_disable_adapter = teacher_use_disable_adapter
        self._is_self_distillation = (teacher_model is None and teacher_model_server is None)

        # Initialize teacher model
        if teacher_model is not None:
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
            self.teacher_model = None

        # Initialize rollout infrastructure for vLLM support
        self.prepare_rollout()

        # Initialize resample data iterator for truncation_strategy 'raise'('delete')
        if self.template.truncation_strategy == 'raise':
            self._prepare_resample_data_iterator()

    def _get_data_collator(self, args, template):
        return identity_data_collator

    def _build_opsd_teacher_data(self, inputs):
        """Build teacher data for OPSD by replacing the last user message with teacher_prompt.

        Returns None if teacher_prompt is not available in all examples.
        """
        if not all('teacher_prompt' in data and data['teacher_prompt'] for data in inputs):
            return None

        assert not self.use_liger_gkd_loss, 'OPSD is only supported when use_liger_gkd_loss is False.'

        teacher_data = []
        for data in inputs:
            teacher_item = {k: v for k, v in data.items() if k != 'teacher_prompt'}
            messages = [dict(m) for m in data.get('messages', [])]
            if messages and messages[-1]['role'] == 'assistant':
                messages.pop()
            for msg in reversed(messages):
                if msg['role'] == 'user':
                    msg['content'] = data['teacher_prompt']
                    break
            teacher_item['messages'] = messages
            teacher_data.append(teacher_item)
        return teacher_data

    def _compute_jsd_loss(self, student_logits, teacher_output: TeacherOutput, labels):
        """Compute JSD loss using unified TeacherOutput.

        Args:
            student_logits: Student model logits [B, S, V].
            teacher_output: Unified teacher output container.
            labels: Student-side labels for masking.
        """
        teacher_output.validate()
        shifted_labels = torch.roll(labels, shifts=-1, dims=1)
        opsd_teacher_labels = teacher_output.opsd_teacher_labels

        # OPSD mode: student and teacher have different prompts, so apply separate masks
        if opsd_teacher_labels is not None:
            shifted_teacher_labels = torch.roll(opsd_teacher_labels, shifts=-1, dims=1)
            student_mask = shifted_labels != -100
            teacher_mask = shifted_teacher_labels != -100
            assert student_mask.sum() == teacher_mask.sum(), (
                f'OPSD label count mismatch: student={student_mask.sum().item()}, '
                f'teacher={teacher_mask.sum().item()}. '
                'Student and teacher must share the same response tokens.')
            s_logits = student_logits[student_mask][None]
            if teacher_output.is_topk_mode:
                t_logits = None
                topk_logprobs = teacher_output.topk_logprobs[teacher_mask][None]
                topk_indices = teacher_output.topk_indices[teacher_mask][None]
            else:
                t_logits = teacher_output.full_logits[teacher_mask][None]
                topk_logprobs = None
                topk_indices = None
            return self.generalized_jsd_loss(
                student_logits=s_logits,
                teacher_logits=t_logits,
                beta=self.beta,
                temperature=self.temperature,
                topk=self.gkd_logits_topk if t_logits is not None else None,
                teacher_topk_logprobs=topk_logprobs,
                teacher_topk_indices=topk_indices,
            )

        # Top-k mode: teacher logprobs from API
        if teacher_output.is_topk_mode:
            return self.generalized_jsd_loss(
                student_logits=student_logits,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
                teacher_topk_logprobs=teacher_output.topk_logprobs,
                teacher_topk_indices=teacher_output.topk_indices,
            )

        # Full-vocab teacher with top-k reduction (local teacher model)
        if self.gkd_logits_topk is not None:
            return self.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_output.full_logits,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
                topk=self.gkd_logits_topk,
            )

        # Full-vocab mode without top-k: vocab alignment handled inside generalized_jsd_loss
        return self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_output.full_logits,
            labels=shifted_labels,
            beta=self.beta,
            temperature=self.temperature,
        )

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

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get data source: DataSource.STUDENT, DataSource.TEACHER, or DataSource.DATASET
        data_source = inputs.pop('_data_source', DataSource.DATASET)
        # Get teacher logprobs from API if available (set in training_step)
        teacher_api_logprobs = inputs.pop('_teacher_api_logprobs', None)
        teacher_api_indices = inputs.pop('_teacher_api_indices', None)
        opsd_teacher_inputs = inputs.pop('_opsd_teacher_inputs', None)
        inputs.pop('_opsd_teacher_messages', None)

        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}

        if opsd_teacher_inputs is not None:
            teacher_fwd_inputs = {k: v for k, v in model_inputs.items()}
            teacher_fwd_inputs.update({k: v for k, v in opsd_teacher_inputs.items() if k != 'labels'})
        else:
            teacher_fwd_inputs = None
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
                # Release hidden states after loss computation
                del student_hidden, teacher_hidden, true_labels
            outputs_student = None
        # Teacher API mode: top-k logprobs fetched from external teacher server
        elif self.use_teacher_api:
            assert teacher_api_logprobs is not None
            if self.args.sft_alpha > 0:
                model_inputs['labels'] = inputs['labels']
            outputs_student = model(**model_inputs)

            # teacher_api shape: [batch, seq_len-1, topk]
            # Pad to [batch, seq_len, topk] so it aligns with student logits.
            teacher_api_logprobs = F.pad(teacher_api_logprobs, (0, 0, 0, 1), value=float('-inf'))
            teacher_api_indices = F.pad(teacher_api_indices, (0, 0, 0, 1), value=0)
            logits_to_keep = inputs.get('logits_to_keep')
            if logits_to_keep is not None:
                if isinstance(logits_to_keep, torch.Tensor) and logits_to_keep.dtype == torch.bool:
                    teacher_api_logprobs = teacher_api_logprobs[:, logits_to_keep]
                    teacher_api_indices = teacher_api_indices[:, logits_to_keep]
                else:
                    n = logits_to_keep.item() if isinstance(logits_to_keep, torch.Tensor) else int(logits_to_keep)
                    teacher_api_logprobs = teacher_api_logprobs[:, -n:]
                    teacher_api_indices = teacher_api_indices[:, -n:]

            opsd_labels = opsd_teacher_inputs.get('labels') if opsd_teacher_inputs is not None else None
            teacher_out = TeacherOutput(
                topk_logprobs=teacher_api_logprobs,
                topk_indices=teacher_api_indices,
                opsd_teacher_labels=opsd_labels,
            )
            loss = self._compute_jsd_loss(outputs_student.logits, teacher_out, inputs['labels'])

            if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
                loss = loss + self.args.sft_alpha * outputs_student.loss
        # Self-distillation mode: student model doubles as teacher
        elif self._is_self_distillation:
            if self.args.sft_alpha > 0:
                model_inputs['labels'] = inputs['labels']
            outputs_student = model(**model_inputs)

            t_fwd = teacher_fwd_inputs if teacher_fwd_inputs is not None else {
                k: v
                for k, v in model_inputs.items() if k != 'labels'
            }

            adapter_ctx = (
                self.accelerator.unwrap_model(model).disable_adapter()
                if self._teacher_use_disable_adapter else nullcontext())
            with torch.no_grad(), adapter_ctx, \
                    disable_gradient_checkpointing(model, self.args.gradient_checkpointing_kwargs):
                outputs_teacher = model(**t_fwd)

            opsd_labels = opsd_teacher_inputs.get('labels') if opsd_teacher_inputs is not None else None
            teacher_out = TeacherOutput(full_logits=outputs_teacher.logits, opsd_teacher_labels=opsd_labels)
            loss = self._compute_jsd_loss(outputs_student.logits, teacher_out, inputs['labels'])

            if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
                loss = loss + self.args.sft_alpha * outputs_student.loss
        # Separate teacher model provided
        else:
            assert self.teacher_model is not None
            if self.args.sft_alpha > 0:
                model_inputs['labels'] = inputs['labels']
            outputs_student = model(**model_inputs)

            t_fwd = teacher_fwd_inputs if teacher_fwd_inputs is not None else {
                k: v
                for k, v in model_inputs.items() if k != 'labels'
            }

            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with torch.no_grad(), load_context, disable_gradient_checkpointing(self.teacher_model,
                                                                               self.args.gradient_checkpointing_kwargs):
                outputs_teacher = self.teacher_model(**t_fwd)

            opsd_labels = opsd_teacher_inputs.get('labels') if opsd_teacher_inputs is not None else None
            teacher_out = TeacherOutput(full_logits=outputs_teacher.logits, opsd_teacher_labels=opsd_labels)
            loss = self._compute_jsd_loss(outputs_student.logits, teacher_out, inputs['labels'])

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
    @profiling_decorator
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

        # build data if dataset has teacher_prompt column
        teacher_data = self._build_opsd_teacher_data(inputs)

        with profiling_context(self, 'get_completions'):
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

                    # OPSD: encode teacher inputs with vLLM-generated response
                    if teacher_data is not None:
                        for i, gen_data in enumerate(generated_inputs):
                            teacher_data[i]['messages'].append(dict(gen_data['messages'][-1]))
                            if 'response_token_ids' in gen_data:
                                teacher_data[i]['response_token_ids'] = gen_data['response_token_ids']
                            teacher_data[i]['add_eos'] = False
                        encoded_inputs['_opsd_teacher_messages'] = [deepcopy(td['messages']) for td in teacher_data]
                        with self._template_context(self.template):
                            encoded_inputs['_opsd_teacher_inputs'] = self._prepare_batch_inputs(
                                teacher_data, encode_prompt_only=False)
                else:
                    # Need prompt-only encoding for on-policy generation
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                    student_prompt_len = encoded_inputs['input_ids'].shape[1]
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

                    if teacher_data is not None:
                        for i, td in enumerate(teacher_data):
                            mask = new_attention_mask[i, student_prompt_len:]
                            resp_token = new_input_ids[i, student_prompt_len:][mask == 1].tolist()
                            td['messages'].append({'role': 'assistant', 'content': resp_token})
                            td['add_eos'] = False
                        encoded_inputs['_opsd_teacher_messages'] = [
                            deepcopy(td['messages'][:-1])
                            + [{
                                'role': 'assistant',
                                'content': self.processing_class.decode(td['messages'][-1]['content'])
                            }] for td in teacher_data
                        ]
                        with self._template_context(self.template):
                            encoded_inputs['_opsd_teacher_inputs'] = self._prepare_batch_inputs(
                                teacher_data, encode_prompt_only=False)

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
                assert teacher_data is None, 'OPSD teacher data is not supported for off-policy mode'
                data_source = DataSource.DATASET
                total_length = self.template.max_length + self.max_completion_length
                with self._template_context(self.template, max_length=total_length):
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)

            # Mark data source for downstream processing (e.g., conditional SFT loss)
            encoded_inputs['_data_source'] = data_source

            # Fetch teacher logprobs from API if using external teacher service
            if self.use_teacher_api:
                teacher_logprobs, teacher_indices = self._fetch_teacher_logprobs_from_api(
                    encoded_inputs, raw_inputs=inputs)
                encoded_inputs['_teacher_api_logprobs'] = teacher_logprobs
                encoded_inputs['_teacher_api_indices'] = teacher_indices

        with self.template.forward_context(self.model, encoded_inputs):
            loss = HFSFTTrainer.training_step(self, model, encoded_inputs, num_items_in_batch)
        return loss

    def _fetch_teacher_logprobs_from_api(self,
                                         encoded_inputs: Dict[str, torch.Tensor],
                                         raw_inputs: Optional[list] = None):
        """Fetch teacher logprobs from external API service.

        For multimodal data, uses /v1/chat/completions and aligns response logprobs
        by resp_count (number of response tokens from local labels).

        Returns:
            Tuple of (teacher_logprobs, teacher_indices) tensors with shapes
            [batch, seq_len-1, topk], aligned to the local encode sequence grid.
        """
        opsd_teacher_inputs = encoded_inputs.get('_opsd_teacher_inputs')
        source = opsd_teacher_inputs if opsd_teacher_inputs is not None else encoded_inputs
        input_ids = source['input_ids']
        labels = source['labels']
        batch_size, tea_seq_len = input_ids.shape
        topk = self.gkd_logits_topk

        # Build multimodal inputs if applicable
        mm_raw_inputs = None
        if raw_inputs is not None and any(
                raw.get('images') or raw.get('audios') or raw.get('videos') for raw in raw_inputs):
            # When under multi-modal training,
            # we should use /v1/chat/completions when need messages inputs
            opsd_teacher_messages = encoded_inputs.get('_opsd_teacher_messages')
            mm_raw_inputs = []
            for i, raw in enumerate(raw_inputs):
                item = {k: raw[k] for k in ('images', 'audios', 'videos') if k in raw}
                if opsd_teacher_messages is not None and opsd_teacher_messages[i] is not None:
                    item['messages'] = opsd_teacher_messages[i]
                else:
                    item['messages'] = raw['messages']
                mm_raw_inputs.append(item)

        logprobs_raw, indices_raw, vllm_seq_lens = fetch_teacher_logprobs(
            self.teacher_model_server, input_ids.tolist(), topk=topk, mm_raw_inputs=mm_raw_inputs)

        out_len = tea_seq_len - 1
        if logprobs_raw.shape[1] == out_len:
            # Text-only or lengths match: direct use
            return logprobs_raw.to(input_ids.device), indices_raw.to(input_ids.device)

        # Multimodal: vLLM seq lengths may differ from local encode. Align by resp_count:
        # take the last resp_count logprobs from vLLM output and place them at the
        # corresponding response positions in the local sequence grid.
        device = input_ids.device
        logprobs_out = torch.full((batch_size, out_len, topk), float('-inf'), dtype=torch.float32, device=device)
        indices_out = torch.zeros((batch_size, out_len, topk), dtype=torch.long, device=device)
        resp_mask = labels != -100
        resp_counts = resp_mask.sum(dim=1).tolist()

        for idx in range(batch_size):
            vllm_out_len = vllm_seq_lens[idx] - 1
            resp_count = resp_counts[idx]
            n = min(resp_count, vllm_out_len, out_len)
            if n <= 0:
                continue
            # logprobs[i] predicts token[i+1], so response ending at labels position
            # resp_end aligns with logprobs position resp_end - 1.
            dest_end = min(resp_mask[idx].nonzero()[-1].item(), out_len)
            logprobs_out[idx, dest_end - n:dest_end] = logprobs_raw[idx, vllm_out_len - n:vllm_out_len].to(device)
            indices_out[idx, dest_end - n:dest_end] = indices_raw[idx, vllm_out_len - n:vllm_out_len].to(device)

        return logprobs_out, indices_out

    def prediction_step(self, model, inputs, *args, **kwargs):
        # Prediction uses full messages
        encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)

        # Fetch teacher logprobs from API if using external teacher service (for eval)
        if self.use_teacher_api:
            teacher_logprobs, teacher_indices = self._fetch_teacher_logprobs_from_api(encoded_inputs, raw_inputs=inputs)
            encoded_inputs['_teacher_api_logprobs'] = teacher_logprobs
            encoded_inputs['_teacher_api_indices'] = teacher_indices

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
            assert self.gkd_logits_topk is None, 'Top-k mode is not supported with liger loss'
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=self.beta,
                ignore_index=-100,
                temperature=self.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

    @staticmethod
    def _align_vocab_size(student_logits, teacher_logits):
        """Align vocab dimensions between student and teacher by padding the smaller one."""
        stu_vocab = student_logits.shape[-1]
        tea_vocab = teacher_logits.shape[-1]
        if stu_vocab == tea_vocab:
            return student_logits, teacher_logits
        if stu_vocab < tea_vocab:
            student_logits = F.pad(student_logits, (0, tea_vocab - stu_vocab), 'constant', 0)
            student_logits[..., stu_vocab:] = teacher_logits[..., stu_vocab:]
        else:
            teacher_logits = F.pad(teacher_logits, (0, stu_vocab - tea_vocab), 'constant', 0)
            teacher_logits[..., tea_vocab:] = student_logits[..., tea_vocab:]
        return student_logits, teacher_logits

    def generalized_jsd_loss(
        self,
        student_logits,
        teacher_logits=None,
        labels=None,
        beta=0.5,
        temperature=1.0,
        chunk_size=512,
        topk=None,
        teacher_topk_logprobs=None,
        teacher_topk_indices=None,
    ):
        # Align vocab sizes when student and teacher have different vocabulary dimensions
        if teacher_logits is not None:
            student_logits, teacher_logits = self._align_vocab_size(student_logits, teacher_logits)

        # Top-k mode: gather/topk first to get small [*, k] tensors, then scale in-place
        if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
            student_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)
            student_logits.div_(temperature)
            teacher_logits = teacher_topk_logprobs / temperature
            temperature = 1.0
        elif topk is not None and teacher_logits is not None:
            teacher_logits, topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)
            teacher_logits.div_(temperature)
            student_logits = torch.gather(student_logits, dim=-1, index=topk_idx)
            student_logits.div_(temperature)
            temperature = 1.0

        if labels is not None:
            mask = labels != -100
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
            num_valid = mask.sum()
        else:
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
            num_valid = student_logits.size(0)
        student_logits.div_(temperature)
        teacher_logits.div_(temperature)

        if num_valid == 0:
            return student_logits.new_zeros(())

        num_valid_int = num_valid if isinstance(num_valid, int) else num_valid.item()
        total_loss = student_logits.new_zeros(())

        if beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

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


def _build_teacher_session(max_retries=5):
    """Build a requests.Session with transport-level retry for teacher API calls."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=3,
        status_forcelist=[500, 502, 503],
        backoff_factor=2,
        allowed_methods=['POST', 'GET'],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


_teacher_session = None


def _parse_prompt_logprobs(prompt_logprobs_list, topk):
    """Parse vLLM prompt_logprobs into per-position (logprobs, indices) lists.

    Skips pos 0 (always None). Returns lists of length num_positions, where each
    element is a (logprobs_row, indices_row) tuple of length topk.
    """
    lps, ixs = [], []
    for raw_pos in range(1, len(prompt_logprobs_list)):
        pos_lp = prompt_logprobs_list[raw_pos]
        if pos_lp is None:
            lps.append([float('-inf')] * topk)
            ixs.append([0] * topk)
        else:
            sorted_items = sorted(pos_lp.items(), key=lambda x: -x[1]['logprob'])[:topk]
            lp_row = [info['logprob'] for _, info in sorted_items]
            ix_row = [int(tid_str) for tid_str, _ in sorted_items]
            pad = topk - len(lp_row)
            if pad > 0:
                lp_row += [float('-inf')] * pad
                ix_row += [0] * pad
            lps.append(lp_row)
            ixs.append(ix_row)
    return lps, ixs


def fetch_teacher_logprobs(base_url, input_ids, topk=20, timeout=300.0, mm_raw_inputs=None):
    """Fetch top-k prompt logprobs from a vLLM-compatible server.

    Text-only: uses /v1/completions with token IDs.
    Multimodal (mm_raw_inputs provided): uses /v1/chat/completions with messages + media.

    Args:
        base_url: vLLM server URL.
        input_ids: List of token ID sequences (used for text-only path).
        topk: Number of top log probabilities per token.
        timeout: Request timeout in seconds.
        mm_raw_inputs: Optional list of dicts with 'messages' and media keys
            for multimodal chat completions. When provided, text-only path is skipped.

    Returns:
        (logprobs, indices, vllm_seq_lens) tensors of shape [batch, max_seq_len - 1, topk].
        For multimodal, max_seq_len is the max vLLM tokenized length across the batch.
        vllm_seq_lens is the length of the vLLM tokenized sequence for each sample.
    """
    from concurrent.futures import ThreadPoolExecutor

    global _teacher_session
    if _teacher_session is None:
        _teacher_session = _build_teacher_session()
    session = _teacher_session

    base_url = base_url.rstrip('/')
    global teacher_model_server_model_name
    if teacher_model_server_model_name is None:
        try:
            resp = session.get(f'{base_url}/v1/models', timeout=10)
            teacher_model_server_model_name = resp.json()['data'][0]['id'] if resp.ok else 'default'
        except Exception:
            teacher_model_server_model_name = 'default'
    model = teacher_model_server_model_name

    batch_size = len(input_ids)
    # Per-sample results: (logprobs_list, indices_list, total_token_count)
    results = [None] * batch_size
    errors = {}

    def _fetch_one(idx):
        try:
            if mm_raw_inputs is not None and mm_raw_inputs[idx] is not None:
                raw = mm_raw_inputs[idx]
                infer_request = InferRequest(
                    **{k: raw[k]
                       for k in ('messages', 'images', 'audios', 'videos') if k in raw})
                payload = InferClient._prepare_request_data(model, infer_request,
                                                            RequestConfig(max_tokens=1, temperature=0))
                payload.update(prompt_logprobs=topk, add_generation_prompt=False, return_token_ids=True)
                resp = session.post(f'{base_url}/v1/chat/completions', json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                choice = data.get('choices', [{}])[0]
                raw_lps = data.get('prompt_logprobs') or choice.get('prompt_logprobs', [])
                token_count = len(data.get('prompt_token_ids') or choice.get('prompt_token_ids', raw_lps))
            else:
                payload = {
                    'model': model,
                    'prompt': input_ids[idx],
                    'max_tokens': 1,
                    'temperature': 0,
                    'prompt_logprobs': topk
                }
                resp = session.post(f'{base_url}/v1/completions', json=payload, timeout=timeout)
                resp.raise_for_status()
                raw_lps = resp.json()['choices'][0].get('prompt_logprobs', [])
                token_count = len(input_ids[idx])

            lps, ixs = _parse_prompt_logprobs(raw_lps, topk)
            results[idx] = (lps, ixs, token_count)
        except Exception as e:
            errors[idx] = e
            logger.error(f'Failed to get teacher logprobs for sequence {idx}: {e}')

    with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as pool:
        list(pool.map(_fetch_one, range(batch_size)))

    if errors:
        failed = sorted(errors.keys())
        raise RuntimeError(f'Failed to fetch teacher logprobs for {len(errors)} sequence(s). '
                           f'Failed indices: {failed}. Last errors: ' + '; '.join(f'seq {i}: {errors[i]}'
                                                                                  for i in failed[:3]))

    # Batch into tensors: shape [batch, max_out_len, topk]
    max_out_len = max(len(r[0]) for r in results)
    logprobs_out = torch.full((batch_size, max_out_len, topk), float('-inf'), dtype=torch.float32)
    indices_out = torch.zeros((batch_size, max_out_len, topk), dtype=torch.long)
    for idx, (lps, ixs, _) in enumerate(results):
        n = len(lps)
        logprobs_out[idx, :n] = torch.tensor(lps, dtype=torch.float32)
        indices_out[idx, :n] = torch.tensor(ixs, dtype=torch.long)

    vllm_seq_lens = [r[2] for r in results]
    return logprobs_out, indices_out, vllm_seq_lens
