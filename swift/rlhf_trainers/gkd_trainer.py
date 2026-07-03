# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import random
import torch
import torch.nn as nn
import trl
from accelerate.utils import gather_object, is_peft_model
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from packaging import version
from transformers import PreTrainedModel, Trainer
from trl import SFTTrainer as HFSFTTrainer
from trl.trainer.utils import RepeatSampler
from typing import Any, Dict, List, Optional, Union

from swift.rl_core.data import GKDBatch, GKDSample
from swift.rlhf_trainers.gkd_helpers import (assemble_teacher_output, build_teacher_requests, encode_gkd_samples,
                                             fetch_teacher_parsed_by_routing)
from swift.rlhf_trainers.gkd_loss import DataSource, TeacherOutput, gkd_loss
from swift.template import TemplateInputs
from swift.trainers import SwiftMixin, disable_gradient_checkpointing
from swift.utils import (JsonlWriter, get_logger, is_swanlab_available, is_wandb_available, remove_response,
                         swanlab_get_run, to_device)
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import get_gather_if_zero3_context, identity_data_collator, profiling_decorator

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

    sample_cls = GKDSample

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        self.vllm_client = kwargs.pop('vllm_client', None)
        self.gkd_logits_topk = kwargs.pop('gkd_logits_topk', None)
        self._pop_teacher_kwargs(kwargs)  # teacher kwargs (shared with GRPO via RolloutTrainerMixin)
        super().__init__(model, None, *_args, **kwargs)
        args = kwargs['args']
        self.lmbda = args.lmbda
        self.temperature = args.temperature
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize logging components
        self._prepare_logging()

        # Initialize liger loss if enabled
        self._prepare_liger_loss()

        self._setup_teacher()

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

    def _compute_jsd_loss(self, student_logits, teacher_output: TeacherOutput, labels):
        """Compute JSD loss. teacher_output.labels is always set (equals student labels when non-OPSD)."""
        shifted_labels = torch.roll(labels, shifts=-1, dims=1)
        teacher_output.labels = torch.roll(teacher_output.labels, shifts=-1, dims=1)
        if self.gkd_logits_topk is not None:
            teacher_output = teacher_output.to_topk(self.gkd_logits_topk)
        total, num_valid = gkd_loss(student_logits, teacher_output, shifted_labels, self.beta, self.temperature)
        if num_valid == 0:
            return total * 0
        return total / num_valid

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = inputs['model_inputs']
        gkd_batch: GKDBatch = inputs['gkd_batch']
        data_source = gkd_batch.data_source
        teacher_model_inputs = inputs['teacher_model_inputs']

        # logits_to_keep trims each forward to its completion region. student and teacher
        # are masked independently in the loss via their own labels (extract_active), so the
        # two sides can be trimmed independently even when OPSD makes their lengths differ.
        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep and not self.use_liger_gkd_loss:
            self.prepare_logits_to_keep(model_inputs)
            # Trim the teacher only when it runs a forward. Teacher API does not (top-k is
            # pre-assembled on the full sequence). For a local teacher forward, non-OPSD reuses
            # the student encoding (separate dict) and OPSD has its own encoding — both are trimmed
            # here so the teacher forward also benefits, while extract_active aligns each side via
            # its own labels.
            if not self.use_teacher_api:
                self.prepare_logits_to_keep(teacher_model_inputs)

        # Teacher labels come from the teacher-side encoding (== student encoding when non-OPSD).
        # Must be read AFTER prepare_logits_to_keep which may truncate labels.
        teacher_labels = teacher_model_inputs['labels']

        # model_inputs is clean from template.encode; exclude 'labels' for model forward.
        forward_inputs = {k: v for k, v in model_inputs.items() if k != 'labels'}

        teacher_fwd_inputs = {k: v for k, v in teacher_model_inputs.items() if k != 'labels'}

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
            student_outputs = base_student(**forward_inputs, use_cache=False)

            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with load_context:
                with torch.no_grad(), disable_gradient_checkpointing(self.teacher_model,
                                                                     self.args.gradient_checkpointing_kwargs):
                    teacher_outputs = base_teacher(**forward_inputs, use_cache=False)

                # Get hidden states (shifted)
                student_hidden = student_outputs.last_hidden_state[:, :-1]
                teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

                # Release full outputs to free memory
                del student_outputs, teacher_outputs

                # Prepare labels (shifted)
                labels_mask = model_inputs['labels'] != -100
                masked_input_ids = torch.where(labels_mask, model_inputs['input_ids'],
                                               torch.full_like(model_inputs['input_ids'], -100))
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
        # Non-liger path: student forward + teacher output construction + JSD loss
        else:
            if self.args.sft_alpha > 0:
                forward_inputs['labels'] = model_inputs['labels']
            outputs_student = model(**forward_inputs)

            # Construct teacher_out — the only part that differs across modes
            if self.use_teacher_api:
                teacher_topk_logprobs = gkd_batch.teacher_topk_logprobs
                teacher_topk_indices = gkd_batch.teacher_topk_indices
                assert teacher_topk_logprobs is not None and teacher_topk_indices is not None, (
                    'teacher_topk_logprobs/teacher_topk_indices missing on gkd_batch; '
                    'ensure _prepare_inputs (or prediction_step) populated them.')
                teacher_out = TeacherOutput(
                    topk_logprobs=teacher_topk_logprobs,
                    topk_indices=teacher_topk_indices,
                    labels=teacher_labels,
                )
            else:
                t_fwd = teacher_fwd_inputs
                if self._is_self_distillation:
                    adapter_ctx = (
                        self.accelerator.unwrap_model(model).disable_adapter()
                        if self._teacher_use_disable_adapter else nullcontext())
                    with torch.no_grad(), adapter_ctx, \
                            disable_gradient_checkpointing(model, self.args.gradient_checkpointing_kwargs):
                        outputs_teacher = model(**t_fwd)
                else:
                    assert self.teacher_model is not None
                    load_context = (
                        self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext())
                    with torch.no_grad(), load_context, disable_gradient_checkpointing(
                            self.teacher_model, self.args.gradient_checkpointing_kwargs):
                        outputs_teacher = self.teacher_model(**t_fwd)
                teacher_out = TeacherOutput(full_logits=outputs_teacher.logits, labels=teacher_labels)

            loss = self._compute_jsd_loss(outputs_student.logits, teacher_out, model_inputs['labels'])

            if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
                loss = loss + self.args.sft_alpha * outputs_student.loss

        # Return loss
        if return_outputs:
            return (loss, outputs_student)
        else:
            return loss

    def _log_student_completions(self, generated_samples) -> None:
        """Log student completions from generated GKDSample list."""
        if not self.log_completions:
            return
        messages = [s.messages[:-1] for s in generated_samples]
        completions = [deepcopy(s.messages[-1]['content']) for s in generated_samples]
        valid_messages = gather_object(messages)
        valid_completions = gather_object(completions)
        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
        self._logs['completion'].extend(valid_completions)

    def _encode_samples(self, samples: List[GKDSample]):
        """Encode student + teacher (OPSD-aware). Mirrors Megatron GKD _encode_samples.

        Args:
            samples: Samples to encode (may be generated samples for STUDENT mode).

        Returns:
            (encoded_inputs, teacher_encoded_or_None) — ``teacher_encoded`` is non-None
            only for OPSD; non-OPSD returns None (teacher reuses the student encoding).
        """
        template = self.template

        student_encoded_list, teacher_encoded_list, has_opsd = encode_gkd_samples(samples, template)

        with self._template_context(template):
            encoded_inputs = to_device(template.data_collator(student_encoded_list), self.model.device)
        if has_opsd:
            with self._template_context(template):
                teacher_encoded = to_device(template.data_collator(teacher_encoded_list), self.model.device)
        else:
            teacher_encoded = None

        return encoded_inputs, teacher_encoded

    def _build_teacher_requests(self, samples: List[GKDSample]):
        """Build teacher API requests from samples (mirrors Megatron GKD)."""
        return build_teacher_requests(samples, self.template)

    def _rollout_samples(self, inputs: DataType) -> List[GKDSample]:
        """Pick the data source, convert rows to samples, and generate for the STUDENT branch."""
        if self._get_random_num() <= self.lmbda:
            self._data_source = DataSource.STUDENT
        else:
            self._data_source = DataSource.DATASET

        if self.template.truncation_strategy == 'raise':
            inputs = self.resample_encode_failed_inputs(inputs, strip_response=self._data_source != DataSource.DATASET)
        samples = self.to_samples(inputs)

        if self._data_source == DataSource.STUDENT:
            # Generation at full-batch level (uses _generate_completions from mixin)
            samples = self._generate_completions(samples)
            self._log_student_completions(samples)
        return samples

    def _score_completions(self, samples: List[GKDSample]) -> List[GKDSample]:
        """GKD has no reward scoring; teacher signals are filled in ``_postprocess_batch``."""
        return samples

    @profiling_decorator
    def _prepare_batch_inputs(self, samples: List[GKDSample]) -> List[Dict[str, Any]]:
        """Encode + collate samples into per-micro-batch ``{model_inputs, gkd_batch}`` dicts.

        Mirrors GRPO ``_prepare_batch_inputs``: takes all samples and splits internally.
        """
        sample_chunks = self.split_by_mini_batches(samples)
        ga_batch_encoded_inputs: List[Dict[str, Any]] = []
        for chunk_samples in sample_chunks:
            encoded_inputs, teacher_encoded = self._encode_samples(chunk_samples)
            # teacher_encoded is non-None only for OPSD. Non-OPSD reuses the student encoding
            # for the teacher, but as a separate dict so each side can run prepare_logits_to_keep
            # independently without aliasing.
            teacher_model_inputs = teacher_encoded or dict(encoded_inputs)
            result: Dict[str, Any] = {
                'model_inputs': encoded_inputs,
                'gkd_batch': GKDBatch(data_source=self._data_source),
                'teacher_model_inputs': teacher_model_inputs,
            }
            if self.use_teacher_api:
                result['_teacher_requests'] = self._build_teacher_requests(chunk_samples)
                result['_chunk_samples'] = chunk_samples
            ga_batch_encoded_inputs.append(result)
        return ga_batch_encoded_inputs

    def _postprocess_batch(self, samples: List[GKDSample], batch_encoded_inputs: List[Dict[str, Any]]) -> None:
        """Fetch teacher logprobs via API and fill them into each ``gkd_batch``."""
        if self.use_teacher_api:
            self._fetch_and_assemble_teacher_logprobs(batch_encoded_inputs)

    def _log_rollout(self, samples: List[GKDSample]) -> None:
        """Student completions are logged in ``_rollout_samples``; nothing extra here."""

    @profiling_decorator
    def _prepare_inputs(self, inputs: DataType) -> Dict[str, torch.Tensor]:
        mode = 'train' if self.model.training else 'eval'
        steps_per_generation = self.args.steps_per_generation
        if mode == 'train':
            if self._step % steps_per_generation == 0 or self._buffered_inputs is None:
                self._buffered_inputs = self._generate_and_score_completions(inputs)
            inputs = self._buffered_inputs[self._step % steps_per_generation]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(inputs)[0]
        return inputs

    def _fetch_and_assemble_teacher_logprobs(self, chunks):
        """Fetch teacher logprobs via API using sample-based _teacher_requests.

        Each sample routes to exactly one teacher by tag (single teacher = all samples). Routing
        runs once over the flattened chunks, so a single teacher is one DP gather → infer → slice.
        The parsed results are sliced back per chunk and fed to ``assemble_teacher_output``.
        """
        flat_samples, flat_requests, chunk_sizes = [], [], []
        for c in chunks:
            reqs = c.pop('_teacher_requests', [])
            flat_samples.extend(c.pop('_chunk_samples', []))
            flat_requests.extend(reqs)
            chunk_sizes.append(len(reqs))

        parsed = fetch_teacher_parsed_by_routing(
            flat_samples,
            flat_requests,
            self.teacher_configs,
            self.teacher_clients,
            gather_fn=self._gather_teacher_requests,
            infer_fn=lambda handle, client: self._infer_teacher_requests(
                handle, topk=self.gkd_logits_topk, teacher_client=client),
            scatter_fn=self._scatter_teacher_parsed,
            is_main_process=self.accelerator.is_main_process,
            tag_key=getattr(self.args, 'teacher_tag_key', 'dataset'))

        per_chunk_parsed, offset = [], 0
        for cs in chunk_sizes:
            per_chunk_parsed.append(parsed[offset:offset + cs])
            offset += cs

        for c, chunk_parsed in zip(chunks, per_chunk_parsed):
            gkd_batch: GKDBatch = c['gkd_batch']
            target = c['teacher_model_inputs']
            teacher_out = assemble_teacher_output(
                chunk_parsed,
                teacher_model_inputs=target,
                topk=self.gkd_logits_topk,
                template_padding_free=self.template.padding_free,
                device=target['input_ids'].device,
            )
            gkd_batch.teacher_topk_logprobs = teacher_out.topk_logprobs
            gkd_batch.teacher_topk_indices = teacher_out.topk_indices

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
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab_get_run(
            ) is not None
            if report_to_swanlab:
                headers = list(table.keys())
                rows = []
                for i in range(len(table['step'])):
                    row = [table[header][i] for header in headers]
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})
