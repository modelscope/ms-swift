# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import trl
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from packaging import version
from transformers import PreTrainedModel, Trainer
from trl import SFTTrainer as HFSFTTrainer
from trl.trainer.utils import RepeatSampler
from typing import Dict, Optional, Union

from swift.infer_engine.protocol import RequestConfig
from swift.rlhf_trainers.gkd_loss import DataSource, TeacherOutput, build_opsd_teacher_data, gkd_loss
from swift.rlhf_trainers.utils import (assemble_teacher_topk_logprobs, build_teacher_infer_request,
                                       get_non_thinking_prefix_ids, parse_prompt_logprobs, prepare_fsdp,
                                       replace_assistant_response_with_ids)
from swift.rlhf_trainers.vllm_client import VLLMInferClient
from swift.template import TemplateInputs
from swift.trainers import SwiftMixin, disable_gradient_checkpointing
from swift.utils import (JsonlWriter, get_cu_seqlens_from_position_ids, get_logger, is_swanlab_available,
                         is_wandb_available, remove_response, to_device, unwrap_model_for_generation)
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
        if self.use_teacher_api:
            if self.accelerator.is_main_process:
                self.teacher_client = VLLMInferClient(base_urls=[teacher_model_server])
            else:
                self.teacher_client = None
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

        self._step = 0
        self._buffered_inputs = None

    def _get_data_collator(self, args, template):
        return identity_data_collator

    def _get_train_sampler(self, train_dataset=None):
        return RepeatSampler(
            data_source=train_dataset or self.train_dataset,
            mini_repeat_count=1,
            batch_size=self.args.generation_batch_size,
            repeat_count=self.args.steps_per_generation,
            shuffle=getattr(self, 'shuffle_dataset', True),
            seed=self.args.seed,
        )

    def get_train_dataloader(self):
        return self._get_dataloader(
            dataset=self.train_dataset,
            description='Training',
            batch_size=self._train_batch_size * self.args.steps_per_generation,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def _build_opsd_teacher_data(self, inputs):
        assert not self.use_liger_gkd_loss, 'OPSD is only supported when use_liger_gkd_loss is False.'
        return build_opsd_teacher_data(inputs, strip_assistant=True)

    def _compute_jsd_loss(self, student_logits, teacher_output: TeacherOutput, labels):
        shifted_labels = torch.roll(labels, shifts=-1, dims=1)
        if teacher_output.opsd_teacher_labels is not None:
            teacher_output = TeacherOutput(
                full_logits=teacher_output.full_logits,
                topk_logprobs=teacher_output.topk_logprobs,
                topk_indices=teacher_output.topk_indices,
                opsd_teacher_labels=torch.roll(teacher_output.opsd_teacher_labels, shifts=-1, dims=1),
            )
        if self.gkd_logits_topk is not None:
            teacher_output = teacher_output.to_topk(self.gkd_logits_topk)
        total, num_valid = gkd_loss(student_logits, teacher_output, shifted_labels, self.beta, self.temperature)
        if num_valid == 0:
            return total * 0
        return total / num_valid

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
        with self.template.generate_context():
            if self.model.model_meta.is_multimodal:
                _, model_inputs = self.template.pre_forward_hook(model, None, model_inputs)
            generated_outputs = self.template.generate(
                model, **model_inputs, generation_config=generation_config, return_dict_in_generate=True)
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
        opsd_teacher_inputs = inputs.pop('_opsd_teacher_inputs', None)

        # If generate is used, then use_logits_to_keep must be set to False.
        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if self.use_teacher_api:
            use_logits_to_keep = False
        if use_logits_to_keep and not self.use_liger_gkd_loss:
            self.prepare_logits_to_keep(inputs)
            if opsd_teacher_inputs is not None:
                self.prepare_logits_to_keep(opsd_teacher_inputs)

        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}

        if opsd_teacher_inputs is not None:
            teacher_fwd_inputs = {k: v for k, v in model_inputs.items()}
            teacher_fwd_inputs.update({k: v for k, v in opsd_teacher_inputs.items() if k != 'labels'})
        else:
            teacher_fwd_inputs = None

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
        # Teacher API: topk logprobs already fetched and assembled in _prepare_inputs.
        elif self.use_teacher_api:
            if self.args.sft_alpha > 0:
                model_inputs['labels'] = inputs['labels']
            outputs_student = model(**model_inputs)

            teacher_topk_logprobs = inputs.pop('_teacher_topk_logprobs', None)
            teacher_topk_indices = inputs.pop('_teacher_topk_indices', None)
            assert teacher_topk_logprobs is not None and teacher_topk_indices is not None, (
                '_teacher_topk_logprobs/_teacher_topk_indices missing on inputs; '
                'ensure _prepare_inputs (or prediction_step) populated them.')
            opsd_labels = opsd_teacher_inputs.get('labels') if opsd_teacher_inputs is not None else None
            teacher_out = TeacherOutput(
                topk_logprobs=teacher_topk_logprobs,
                topk_indices=teacher_topk_indices,
                opsd_teacher_labels=opsd_labels,
            )
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
        template = self.template
        batch_encoded_inputs = []

        # Use 'transformers' mode for prompt-only encoding, 'train' mode for full encoding
        mode = 'transformers' if encode_prompt_only else 'train'
        original_mode = template.mode
        template.set_mode(mode)
        non_thinking_prefix_ids = get_non_thinking_prefix_ids(template)
        try:
            for data in inputs:
                if 'response_token_ids' in data and data['response_token_ids']:
                    data = {**data}
                    data['messages'] = replace_assistant_response_with_ids(
                        data['messages'], data['response_token_ids'], non_thinking_prefix_ids=non_thinking_prefix_ids)

                if encode_prompt_only:
                    # Remove response content for prompt-only encoding
                    messages = data.get('messages', [])
                    if messages and messages[-1].get('role') == 'assistant':
                        data = {**data, 'messages': messages[:-1] + [{**messages[-1], 'content': None}]}

                encoded = template.encode(data, return_length=True)
                batch_encoded_inputs.append(encoded)

            batch_encoded = to_device(template.data_collator(batch_encoded_inputs), self.model.device)
        finally:
            template.set_mode(original_mode)

        return batch_encoded

    def _log_student_completions(self, generated_inputs: DataType) -> None:
        if not self.log_completions:
            return
        messages = [inp['messages'][:-1] for inp in generated_inputs]
        completions = [deepcopy(inp['messages'][-1]['content']) for inp in generated_inputs]
        valid_messages = gather_object(messages)
        valid_completions = gather_object(completions)
        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
        self._logs['completion'].extend(valid_completions)

    def _build_encoded_inputs(self,
                              model: nn.Module,
                              inputs: DataType,
                              data_source: DataSource,
                              generated_inputs: Optional[DataType] = None) -> Dict[str, torch.Tensor]:
        """Build encoded training inputs for one micro-batch.

        Args:
            data_source: Pre-determined by _prepare_inputs (STUDENT / TEACHER / DATASET).
            generated_inputs: Pre-generated vLLM outputs for the STUDENT branch.
        """
        args = self.args

        # build data if dataset has teacher_prompt column
        teacher_data = self._build_opsd_teacher_data(inputs)

        with profiling_context(self, 'get_completions'):
            if data_source == DataSource.STUDENT:
                # On-policy: student model generates responses
                if generated_inputs is None and self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)
                    teacher_data = self._build_opsd_teacher_data(inputs)

                if args.use_vllm:
                    if generated_inputs is None:
                        processed_inputs = self._preprocess_inputs(inputs)
                        generated_inputs = self._fast_infer(processed_inputs)

                    # vLLM already generated complete responses; lift max_length to
                    # avoid truncating the completion during encoding.
                    with self._template_context(self.template):
                        encoded_inputs = self._prepare_batch_inputs(generated_inputs, encode_prompt_only=False)

                        # OPSD: encode teacher inputs with vLLM-generated response
                        if teacher_data is not None:
                            for i, gen_data in enumerate(generated_inputs):
                                teacher_data[i]['messages'].append(dict(gen_data['messages'][-1]))
                                if 'response_token_ids' in gen_data:
                                    teacher_data[i]['response_token_ids'] = gen_data['response_token_ids']
                                teacher_data[i]['add_eos'] = False
                            encoded_inputs['_opsd_teacher_inputs'] = self._prepare_batch_inputs(
                                teacher_data, encode_prompt_only=False)
                else:
                    # Need prompt-only encoding for on-policy generation
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                    student_prompt_len = encoded_inputs['input_ids'].shape[1]
                    with unwrap_model_for_generation(
                            model, self.accelerator,
                            gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                        was_training = unwrapped_model.training
                        unwrapped_model.eval()
                        new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                            unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)
                        if was_training:
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
                        with self._template_context(self.template):
                            encoded_inputs['_opsd_teacher_inputs'] = self._prepare_batch_inputs(
                                teacher_data, encode_prompt_only=False)

            elif data_source == DataSource.TEACHER:
                # Sequential KD: teacher model generates responses
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
                total_length = self.template.max_length + self.max_completion_length
                with self._template_context(self.template, max_length=total_length):
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)

            # Mark data source for downstream processing (e.g., conditional SFT loss)
            encoded_inputs['_data_source'] = data_source

            if self.use_teacher_api:
                if teacher_data is not None:
                    encoded_inputs['_teacher_raw'] = teacher_data
                elif data_source == DataSource.STUDENT:
                    if args.use_vllm and generated_inputs is not None:
                        encoded_inputs['_teacher_raw'] = list(generated_inputs)
                    else:
                        raise NotImplementedError(
                            'Teacher API + STUDENT mode requires use_vllm=True; HF generate path not supported.')
                elif data_source == DataSource.TEACHER:
                    raise NotImplementedError('Teacher API + sequential KD (TEACHER mode) is not supported yet.')
                else:
                    encoded_inputs['_teacher_raw'] = list(inputs)

        return encoded_inputs

    @profiling_decorator
    def _prepare_inputs(self, inputs: DataType) -> Dict[str, torch.Tensor]:
        model = self.model

        def _prepare_input(inputs):
            if self._get_random_num() <= self.lmbda:
                data_source = DataSource.STUDENT
            elif self.seq_kd:
                data_source = DataSource.TEACHER
            else:
                data_source = DataSource.DATASET

            if data_source == DataSource.STUDENT and self.args.use_vllm:
                rollout_inputs = inputs
                if self.template.truncation_strategy == 'raise':
                    rollout_inputs = self.resample_encode_failed_inputs(rollout_inputs)

                processed_inputs = self._preprocess_inputs(rollout_inputs)
                generated_inputs = self._fast_infer(processed_inputs)
                self._log_student_completions(generated_inputs)

                input_chunks = self.split_by_mini_batches(rollout_inputs)
                generated_chunks = self.split_by_mini_batches(generated_inputs)
                results = [
                    self._build_encoded_inputs(
                        model, chunk_inputs, data_source=data_source, generated_inputs=chunk_generated)
                    for chunk_inputs, chunk_generated in zip(input_chunks, generated_chunks)
                ]
            else:
                input_chunks = self.split_by_mini_batches(inputs)
                results = [
                    self._build_encoded_inputs(model, chunk_inputs, data_source=data_source)
                    for chunk_inputs in input_chunks
                ]

            if self.use_teacher_api:
                self._fetch_and_assemble_teacher_logprobs(results)
            return results

        mode = 'train' if model.training else 'eval'
        steps_per_generation = self.args.steps_per_generation
        if mode == 'train':
            if self._step % steps_per_generation == 0 or self._buffered_inputs is None:
                self._buffered_inputs = _prepare_input(inputs)
            inputs = self._buffered_inputs[self._step % steps_per_generation]
            self._step += 1
        else:
            inputs = _prepare_input(inputs)[0]
        return inputs

    def _assemble_topk_for_chunk(self, parsed_chunk, encoded_chunk: Dict[str, torch.Tensor]):
        """Assemble parsed teacher (logprobs, indices) into a chunk-aligned tensor.

        Derives cu_seqlens from parsed lengths directly (avoids mrope position_ids issues).
        """
        input_ids = encoded_chunk['input_ids']
        server_seq_lens = None
        offsets = None
        if self.template.padding_free:
            server_seq_lens = [0]
            for lps, ixs in parsed_chunk:
                server_seq_lens.append(server_seq_lens[-1] + len(lps) + 1)
            trainer_seq_lens = encoded_chunk.get('cu_seq_lens_q')
            if trainer_seq_lens is None:
                position_ids = encoded_chunk.get('text_position_ids')
                if position_ids is None:
                    position_ids = encoded_chunk.get('position_ids')
                if position_ids is not None:
                    trainer_seq_lens = get_cu_seqlens_from_position_ids(position_ids)
            if trainer_seq_lens is not None and server_seq_lens[-1] != int(trainer_seq_lens[-1]):
                logger.warning('The number of tokens returned by the teacher server differs from that of the trainer. '
                               'This may be caused by non-aligned processing.')
        else:
            attention_mask = encoded_chunk.get('attention_mask')
            if attention_mask is not None:
                offsets = []
                for i in range(attention_mask.shape[0]):
                    nz = attention_mask[i].nonzero().flatten()
                    offsets.append(int(nz[0]) if nz.numel() else 0)

        batch_size, seq_len = input_ids.shape
        return assemble_teacher_topk_logprobs(
            parsed_chunk,
            batch_size=batch_size,
            seq_len=seq_len,
            cu_seqlens=server_seq_lens,
            topk=self.gkd_logits_topk,
            device=input_ids.device,
            offsets=offsets,
        )

    def _fetch_and_assemble_teacher_logprobs(self, chunks):
        local_chunk_sizes = [len(c.get('_teacher_raw', [])) for c in chunks]
        local_raw = []
        for c in chunks:
            local_raw.extend(c.pop('_teacher_raw'))

        # gather across DP: every rank gets the full list (accelerate semantics)
        all_raw = gather_object(local_raw)

        if self.accelerator.is_main_process:
            non_thinking_prefix_ids = get_non_thinking_prefix_ids(self.template)
            requests = [
                build_teacher_infer_request(d, non_thinking_prefix_ids=non_thinking_prefix_ids) for d in all_raw
            ]
            request_config = RequestConfig(prompt_logprobs=self.gkd_logits_topk, max_tokens=1, temperature=0.0)
            responses = self.teacher_client.infer(requests, request_config=request_config, use_tqdm=False)
            parsed_global = [parse_prompt_logprobs(r, topk=self.gkd_logits_topk) for r in responses]
        else:
            parsed_global = None

        container = [parsed_global]
        broadcast_object_list(container, from_process=0)
        parsed_global = container[0]

        # Slice this rank's contiguous segment in gather order, then redistribute to chunks.
        n_local = sum(local_chunk_sizes)
        rank = self.accelerator.process_index
        parsed_local = parsed_global[rank * n_local:(rank + 1) * n_local]

        offset = 0
        for c, cs in zip(chunks, local_chunk_sizes):
            chunk_parsed = parsed_local[offset:offset + cs]
            offset += cs
            target = c.get('_opsd_teacher_inputs') or c
            topk_lp, topk_ix = self._assemble_topk_for_chunk(chunk_parsed, target)
            c['_teacher_topk_logprobs'] = topk_lp
            c['_teacher_topk_indices'] = topk_ix

    @profiling_decorator
    def training_step(self,
                      model: nn.Module,
                      inputs: DataType,
                      num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        return HFSFTTrainer.training_step(self, model, inputs, num_items_in_batch)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    @contextmanager
    def offload_context(self):
        """Offload student model and optimizer to CPU during vLLM on-policy generation."""
        if self.args.offload_model:
            self.offload_model(self.accelerator.unwrap_model(self.model))
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            self.offload_optimizer()

        try:
            yield
        finally:
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
