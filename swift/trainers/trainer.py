# Copyright (c) ModelScope Contributors. All rights reserved.
# Part of the implementation is borrowed from huggingface/transformers.
import functools
import math
import os
import shutil
import sys
import time
import torch
import torch.distributed as dist
from contextlib import contextmanager, nullcontext
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import Trainer as HfTrainer
from transformers.trainer import (TRAINER_STATE_NAME, TrainerState, TrainOutput, _is_peft_model, deepspeed_init,
                                  deepspeed_load_checkpoint, get_model_param_count, has_length, is_sagemaker_mp_enabled,
                                  skip_first_batches, speed_metrics, unwrap_model)
from typing import Any

from swift.sequence_parallel import sequence_parallel
from swift.utils import get_logger
from .arguments import TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin

logger = get_logger()
try:
    from transformers.trainer import release_memory
except ImportError:
    from accelerate.utils import release_memory

try:
    from transformers.trainer import compare_trainer_and_checkpoint_args
except ImportError:
    compare_trainer_and_checkpoint_args = None

try:
    from transformers.trainer import ExportableState
except ImportError:
    ExportableState = None

try:
    from transformers.trainer import GreedyLR
except ImportError:
    GreedyLR = None


class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    args: TrainingArguments

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        # For tasks whose `labels` are per-sample (e.g. seq_cls/reranker/embedding), we must NOT let
        # SP code treat them as token labels. We detect that case by `labels.dim() == 1` and temporarily
        # remove labels during `prepare_inputs`.
        if self.template.sequence_parallel_size > 1:
            labels = inputs.get('labels', None)
            pop_labels = isinstance(labels, torch.Tensor) and labels.dim() == 1
            if pop_labels:
                labels = inputs.pop('labels', None)
            try:
                sequence_parallel.prepare_inputs(inputs)
            finally:
                if pop_labels and labels is not None:
                    inputs['labels'] = labels
        return inputs

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @functools.wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss

    # eval_on_start
    def _inner_training_loop(
        self,
        batch_size: int | None = None,
        args: TrainingArguments | None = None,
        resume_from_checkpoint: str | None = None,
        trial: dict[str, Any] | None = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """Run the actual training loop: forward, backward, optimizer step, logging, and checkpointing."""
        # reset everything
        self.accelerator.free_memory()
        if args.auto_find_batch_size:
            self._update_auto_batch_size(batch_size)
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            total_train_batch_size,
            steps_in_epoch,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader)

        epochs_trained, steps_trained_in_current_epoch = self._init_training_state(max_steps,
                                                                                   num_update_steps_per_epoch,
                                                                                   num_train_epochs,
                                                                                   resume_from_checkpoint, trial)
        model, train_dataloader = self._prepare_for_training(max_steps, train_dataloader, resume_from_checkpoint)

        # Train!
        logger.info('***** Running training *****')
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Num update steps per epoch = {num_update_steps_per_epoch:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        if resume_from_checkpoint is not None:
            logger.info(f"  Resuming training from checkpoint with epoch {epochs_trained} and "
                        f"global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(f"  Fast-forwarding the dataloader past {epochs_trained} epochs and"
                            f" {steps_trained_in_current_epoch} batches to resume from the exact training state.")

        start_time = time.time()
        # needed to calculate tokens/s
        self._initial_num_input_tokens_seen = self.state.num_input_tokens_seen
        # Logging state: _tr_loss accumulates on-device between logging steps (avoiding costly .item() syncs
        # on TPUs), then gets drained into _total_loss_scalar at each logging step.
        self._tr_loss = torch.tensor(0.0, device=args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            self._run_epoch(
                model=model,
                epoch=epoch,
                train_dataloader=train_dataloader,
                steps_in_epoch=steps_in_epoch,
                num_update_steps_per_epoch=num_update_steps_per_epoch,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                start_time=start_time,
                resume_from_checkpoint=resume_from_checkpoint,
                epochs_trained=epochs_trained,
                steps_trained_in_current_epoch=steps_trained_in_current_epoch,
            )
            if self.control.should_training_stop:
                break

        return self._finalize_training(trial, num_train_samples, start_time)

    def _update_auto_batch_size(self, batch_size):
        """Free memory, reset model wrapping, and update DeepSpeed config for the new batch size when using
        `auto_find_batch_size`"""
        # `_train_batch_size` value might have changed to `auto_find_batch_size`
        self._train_batch_size = batch_size
        # frees the wrapped model and resets it back to the unwrapped base model
        release_memory(self.model_wrapped)

        if self.is_fsdp_enabled:
            # Remove FSDP wrapping from sub-models because self.model points to the wrapped model in FSDP case
            self.model = unwrap_model(self.model, recursive=True)

        self.model_wrapped = self.model

        # Check for DeepSpeed *after* the initial pass and modify the config
        if self.is_deepspeed_enabled:
            # Temporarily unset `self.args.train_batch_size`
            original_bs = self.args.per_device_train_batch_size
            self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
            if hasattr(self, 'propagate_args_to_deepspeed'):
                self.propagate_args_to_deepspeed()
            else:
                from transformers.trainer import propagate_args_to_deepspeed
                propagate_args_to_deepspeed(self.accelerator, self.args, auto_find_batch_size=True)
            self.args.per_device_train_batch_size = original_bs

    # get_total_train_batch_size, get_sp_size
    def set_initial_training_values(self, args: TrainingArguments,
                                    dataloader: DataLoader) -> tuple[int, int, int, int, int, int | None, int]:
        """
        Calculates and returns the following values:
        - `num_train_epochs`
        - `num_update_steps_per_epoch`
        - `num_examples`
        - `num_train_samples`
        - `total_train_batch_size`
        - `steps_in_epoch` (total batches per epoch)
        - `max_steps`
        """
        # Case 1: we rely on `args.max_steps` first
        max_steps = args.max_steps
        # If max_steps is negative, we use the number of epochs to determine the number of total steps later
        epoch_based = max_steps < 0
        len_dataloader = len(dataloader) if has_length(dataloader) else None
        total_train_batch_size = self.get_total_train_batch_size(args)

        # Account for Sequence Parallelism (SP) dataloader adapter's effect
        sp_size = self.get_sp_size()
        if sp_size > 1 and len_dataloader is not None:
            len_dataloader = len_dataloader * sp_size

        # Case 2: We have a dataloader length and can extrapolate
        if len_dataloader is not None:
            num_update_steps_per_epoch = max(
                len_dataloader // args.gradient_accumulation_steps
                + int(len_dataloader % args.gradient_accumulation_steps > 0),
                1,
            )
            # Case 3: We have a length but are using epochs, we can extrapolate the number of steps
            if epoch_based:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
        if len_dataloader:
            num_examples = self.num_examples(dataloader)
            if args.max_steps > 0:
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(max_steps
                                                                                 % num_update_steps_per_epoch > 0)
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = max_steps * total_train_batch_size
            else:
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError('args.max_steps must be set to a positive value if dataloader does not have a length, was'
                             f" {args.max_steps}")
        steps_in_epoch = len_dataloader if len_dataloader is not None else max_steps * args.gradient_accumulation_steps
        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            total_train_batch_size,
            steps_in_epoch,
            max_steps,
        )

    # _load_callback_state, compute_steps
    def _init_training_state(self, max_steps, num_update_steps_per_epoch, num_train_epochs, resume_from_checkpoint,
                             trial) -> tuple[int, int]:
        """Initialize TrainerState, optionally restoring from checkpoint.
        Returns (epochs_trained, steps_trained_in_current_epoch)."""
        state_kwargs = {}
        if ExportableState is not None:
            stateful_callbacks = [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
            state_kwargs = {'stateful_callbacks': stateful_callbacks}

        self.state = TrainerState(**state_kwargs)
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size
        self.state.compute_steps(self.args, max_steps)

        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if compare_trainer_and_checkpoint_args is not None:
                compare_trainer_and_checkpoint_args(self.args, self.state)
            if hasattr(self, '_load_callback_state'):
                self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        return epochs_trained, steps_trained_in_current_epoch

    # _load_scaler
    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        """Wrap model, create optimizer and scheduler, and run accelerator.prepare.
        Returns (model, train_dataloader)."""
        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        self.create_optimizer()

        # Pass `self.model_wrapped` so that `_wrap_model` can detect if the model is already
        # wrapped (e.g. in DataParallel) on subsequent `train()` calls and avoid double wrapping.
        model = self._wrap_model(self.model_wrapped)

        # If the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases in accelerate such as FSDP-XLA, SageMaker MP/DP, DataParallel
        use_accelerator_prepare = model is self.model

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            if self.is_deepspeed_enabled and type(self.lr_scheduler).__name__ == 'DummyScheduler':
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer,
                                                                                    self.lr_scheduler)
            else:
                model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # Create scheduler now that the optimizer won't change anymore
        self.create_scheduler(num_training_steps=max_steps)

        # updating self.model_wrapped
        self.model_wrapped = model

        if self.is_fsdp_enabled or self.is_fsdp_xla_enabled:
            # breaking convention for FSDP model
            # TODO: check if this is really needed
            self.model = self.model_wrapped = model

        # backward compatibility
        # TODO: check if we really need this
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # Important: at this point:
        # self.model         is the Transformers Model except when we are using FSDP
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        if self.is_fsdp_enabled:
            # Fix `got mixed torch.Tensor and DTensor` error in model.generate() for FSDP2 with LoRA
            if hasattr(self.model, 'generate'):
                dist.fsdp.register_fsdp_forward_method(self.model, 'generate')

        # since DataLoader was Accelerate prepared w/o a model arg in the same call,
        # we now have to complete the DL wrapping for ALST/UlyssesSP, after model has been prepared
        pc = getattr(self.accelerator, 'parallelism_config', None)
        if pc is not None and pc.sp_backend == 'deepspeed' and pc.sp_enabled:
            train_dataloader = self.accelerator.deepspeed_ulysses_dl_adapter(train_dataloader, model)

        # load checkpoint
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model))
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

            self._load_optimizer_and_scheduler(resume_from_checkpoint)
            self._load_scaler(resume_from_checkpoint)

        # Update the references for the callback_handler
        for attr in ('model', 'optimizer', 'lr_scheduler'):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        return model, train_dataloader

    # get_batch_samples
    # _track_num_input_tokens, _clip_grad_norm, _get_grad_norm
    def _run_epoch(
        self,
        model,
        epoch,
        train_dataloader,
        steps_in_epoch,
        num_update_steps_per_epoch,
        trial,
        ignore_keys_for_eval,
        start_time,
        resume_from_checkpoint,
        epochs_trained,
        steps_trained_in_current_epoch,
    ):
        """Run one full pass over the dataloader."""

        step = -1
        grad_norm = None
        learning_rate = None
        rng_to_sync = False

        # Handle resumption from checkpoint: skip already-trained batches in the resumed epoch
        num_update_steps_trained = 0
        if epoch == epochs_trained and resume_from_checkpoint is not None:
            if steps_trained_in_current_epoch > 0 and not self.args.ignore_data_skip:
                train_dataloader = skip_first_batches(train_dataloader, steps_trained_in_current_epoch)
                step = steps_trained_in_current_epoch - 1
                num_update_steps_trained = steps_trained_in_current_epoch // self.args.gradient_accumulation_steps
                rng_to_sync = True
            elif steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

        if hasattr(train_dataloader, 'set_epoch'):
            train_dataloader.set_epoch(epoch)
        epoch_iterator = iter(train_dataloader)

        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % self.args.gradient_accumulation_steps
        if remainder == 0:
            remainder = self.args.gradient_accumulation_steps

        # Outer loop: one iteration per optimizer step. Each iteration prefetches
        # `gradient_accumulation_steps` batches (fewer for the last step if the epoch
        # doesn't divide evenly).
        for update_step in range(num_update_steps_trained, num_update_steps_per_epoch):
            num_batches = (
                self.args.gradient_accumulation_steps if update_step != (num_update_steps_per_epoch - 1) else remainder)
            batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, self.args.device)

            # This is used to correctly scale the loss when the last accumulation step has fewer batches.
            # Not used if `num_items_in_batch` is not None.
            self.current_gradient_accumulation_steps = len(batch_samples)

            # need to sync after if we skipped the batches in `get_batch_samples` for shuffle order reason
            if rng_to_sync:
                self._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False

            # Inner loop: forward + backward for each micro-batch. Gradients are
            # accumulated without syncing until the last micro-batch, then we clip,
            # step the optimizer, and log/save/evaluate.
            for i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if step % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # We sync the gradients in the following cases:
                # 1. sync_each_batch set to True
                # 2. Using deepspeed
                # 3. when we are at the last batch sample
                if (self.accelerator.gradient_state.plugin_kwargs.get('sync_each_batch', False)
                        or self.accelerator.distributed_type == 'DEEPSPEED' or i == len(batch_samples) - 1):
                    sync_context = nullcontext
                else:
                    sync_context = functools.partial(self.accelerator.no_sync, model=model)
                with sync_context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                if (self.args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
                    # if loss is nan or inf simply add the average of previous logged losses
                    self._tr_loss += self._tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if self._tr_loss.device != tr_loss_step.device:
                        raise ValueError(f"Calculated loss must be on the original device: {self._tr_loss.device} "
                                         f"but device in use is {tr_loss_step.device}")
                    self._tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                self._track_num_input_tokens(inputs)

                if do_sync_step:
                    grad_norm = None
                    if self.args.max_grad_norm > 0:
                        grad_norm = self._clip_grad_norm(model)
                    grad_norm = self._get_grad_norm(model, grad_norm=grad_norm)

                    self.control = self.callback_handler.on_pre_optimizer_step(self.args, self.state, self.control)
                    self.optimizer.step()
                    self.control = self.callback_handler.on_optimizer_step(self.args, self.state, self.control)

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        lr_scheduler_cls = [torch.optim.lr_scheduler.ReduceLROnPlateau]
                        if GreedyLR is not None:
                            lr_scheduler_cls.append(GreedyLR)
                        if not isinstance(self.lr_scheduler, tuple(lr_scheduler_cls)):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    self._maybe_log_save_evaluate(
                        self._tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(self.args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        if step < 0:
            logger.warning('There seems not to be a single sample in your epoch_iterator, stopping training at step'
                           f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                           f" num_steps ({self.state.max_steps}) higher than the number of available samples.")
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
        self._maybe_log_save_evaluate(
            self._tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )

    def _finalize_training(self, trial, num_train_samples, start_time):
        """Finalize training: metrics, best-model loading, cleanup. Returns TrainOutput."""
        logger.info('\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n')

        # add remaining tr_loss
        self._total_loss_scalar += self._tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            'train',
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics['total_flos'] = self.state.total_flos
        metrics['train_loss'] = train_loss

        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        run_dir = self._get_output_dir(trial)
        if hasattr(self, '_sorted_checkpoints'):
            checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
        else:
            from transformers.trainer import sort_checkpoints
            checkpoints_sorted = sort_checkpoints(
                output_dir=run_dir, best_model_checkpoint=self.state.best_model_checkpoint)

        # Delete the last checkpoint when save_total_limit=1 if it's different from
        # the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            if hasattr(self, '_deactivate_neftune'):
                self._deactivate_neftune(self.model)
            else:
                from transformers.trainer import deactivate_neftune
                deactivate_neftune(self.model, self.neftune_hook_handle, self.accelerator)
        self.is_in_train = False

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _evaluate(
        self,
        trial: 'optuna.Trial | dict[str, Any] | None',
        ignore_keys_for_eval: list[str] | None,
        skip_scheduler: bool = False,
    ) -> dict[str, float]:
        """Run evaluation, report to HP search, and step ReduceLROnPlateau/GreedyLR if needed."""
        metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        # Run delayed LR scheduler now that metrics are populated
        if (isinstance(self.lr_scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, GreedyLR))
                and not skip_scheduler):
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith('eval_'):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                self.lr_scheduler.step(metrics[metric_to_check])
            except KeyError as exc:
                raise KeyError(f"The `metric_for_best_model` training argument is set to '{metric_to_check}', "
                               f"which is not found in the evaluation metrics. "
                               f"The available evaluation metrics are: {list(metrics.keys())}. "
                               f"Please ensure that the `compute_metrics` function returns a dictionary that "
                               f"includes '{metric_to_check}' or consider changing the `metric_for_best_model` "
                               'via the TrainingArguments.') from exc
        return metrics
