# Copyright (c) Alibaba, Inc. and its affiliates.
import contextlib
import json
import math
import os
import shutil
import time
from copy import deepcopy
from functools import partial
from types import MethodType
from typing import List, Union, Tuple, Optional

import numpy as np
import torch
from accelerate import skip_first_batches, DistributedType
from datasets import Dataset
from modelscope import GenerationConfig
from transformers import PreTrainedModel, TrainerState
from transformers.integrations import deepspeed_init, deepspeed_load_checkpoint
from transformers.trainer import _is_peft_model
from transformers.trainer_callback import ExportableState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import speed_metrics, TrainOutput
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import is_accelerate_available
from trl.models.utils import unwrap_model_for_generation

from swift.llm import load_dataset
from swift.llm.template.template_inputs import StdTemplateInputs, InferRequest
from swift.utils import get_logger, get_dist_setting
from .rlhf import SwiftRLHF
from .. import PtEngine
from ..argument import RLFTArguments
from ...plugin.orm import orms
from ...plugin.prm import prms
from ...plugin.sampler import samplers

logger = get_logger()


def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None,
        num_rollout_iters=50, num_rollout_batches=300, rollout_func=None
):
    self.accelerator.free_memory()
    self._train_batch_size = batch_size
    # Data loader and number of training steps

    total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
    max_steps = num_rollout_iters * num_rollout_batches // total_train_batch_size

    # We need to reset the scheduler, as its parameters may be different on subsequent calls
    if self._created_lr_scheduler:
        self.lr_scheduler = None
        self._created_lr_scheduler = False

    if self.is_deepspeed_enabled:
        self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

    self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    self.state = TrainerState(
        stateful_callbacks=[
            cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
        ]
    )
    self.state.is_hyper_param_search = trial is not None
    self.state.train_batch_size = self._train_batch_size

    # Compute absolute values for logging, eval, and save if given as ratio
    if args.logging_steps is not None:
        if args.logging_steps < 1:
            self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
        else:
            self.state.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        if args.eval_steps < 1:
            self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
        else:
            self.state.eval_steps = args.eval_steps
    if args.save_steps is not None:
        if args.save_steps < 1:
            self.state.save_steps = math.ceil(max_steps * args.save_steps)
        else:
            self.state.save_steps = args.save_steps

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

    model = self._wrap_model(self.model_wrapped)

    use_accelerator_prepare = True if model is self.model else False

    # prepare using `accelerator` prepare
    if use_accelerator_prepare:
        self.model.train()
        if hasattr(self.lr_scheduler, "step"):
            model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
    elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        # In this case we are in DDP + LOMO, which should be supported
        self.optimizer = self.accelerator.prepare(self.optimizer)

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not self.model:
        self.model_wrapped = model

    # backward compatibility
    if self.is_deepspeed_enabled:
        self.deepspeed = self.model_wrapped

    # ckpt loading
    if resume_from_checkpoint is not None:
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
            )

    # Check if saved optimizer or scheduler states exist
    self._load_optimizer_and_scheduler(resume_from_checkpoint)

    # important: at this point:
    # self.model         is the Transformers Model
    # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num rollout iters = {num_rollout_iters:,}")
    logger.info(f"  Num Epochs = {args.num_train_epochs:,}")
    logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
    if self.args.per_device_train_batch_size != self._train_batch_size:
        logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    self.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None

    # Update the references
    self.callback_handler.model = self.model
    self.callback_handler.optimizer = self.optimizer
    self.callback_handler.lr_scheduler = self.lr_scheduler
    if self.hp_name is not None and self._trial is not None:
        # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
        # parameter to Train when using DDP.
        self.state.trial_name = self.hp_name(self._trial)
    self.state.trial_params = None
    # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    # to set this after the load.
    self.state.max_steps = max_steps
    self.state.num_train_epochs = args.num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0).to(args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()
    grad_norm: Optional[float] = None
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    if args.eval_on_start:
        self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    total_batched_samples = 0

    for rollout_iter in range(num_rollout_iters):
        rollout_func(rollout_iter)
        for epoch in range(epochs_trained, int(args.num_train_epochs)):
            epoch_iterator = self.get_train_dataloader()
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator)
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += (
                            torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(
                                        inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                    )
                                )
                            )
                            .cpu()
                            .item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        _grad_norm = self.accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )

                        if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, None)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, None)

            if self.control.should_training_stop:
                break

    if args.past_index and hasattr(self, "_past"):
        # Clean the state at the end of training
        delattr(self, "_past")

    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
        if args.parallel_mode == ParallelMode.DISTRIBUTED:
            import torch.distributed as dist
            dist.barrier()

        self._load_best_model()

    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
    train_loss = self._total_loss_scalar / effective_global_step

    metrics = speed_metrics(
        "train",
        start_time,
        num_samples=num_rollout_iters * num_rollout_batches,
        num_steps=self.state.max_steps,
    )
    self.store_flos()
    metrics["total_flos"] = self.state.total_flos
    metrics["train_loss"] = train_loss

    self.is_in_train = False

    self._memory_tracker.stop_and_update_metrics(metrics)

    self.log(metrics)

    run_dir = self._get_output_dir(trial)
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
        for checkpoint in checkpoints_sorted:
            if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                shutil.rmtree(checkpoint, ignore_errors=True)

    self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    # Wait for the checkpoint to be uploaded.
    self._finish_current_push()

    # After training we make sure to retrieve back the original forward pass method
    # for the embedding layer by removing the forward post hook.
    if self.neftune_noise_alpha is not None:
        self._deactivate_neftune(self.model)

    return TrainOutput(self.state.global_step, train_loss, metrics)


class SwiftRLFT(SwiftRLHF):
    args_class = RLFTArguments
    args: args_class

    splitter = [
        # '.', '。', '\n\n'
    ]

    def _prepare_rm(self):
        if self.args.prm_model is None:
            self.prm_model = None
            return
        if self.args.prm_model in prms:
            self.prm_model = prms[self.args.prm_model]()
        else:
            self.prm_model = PtEngine(self.args.prm_model, max_batch_size=64)

        if self.args.orm_model is None:
            self.orm_model = None
            return
        elif self.args.orm_model in orms:
            self.orm_model = orms[self.args.orm_model]()
        else:
            self.orm_model = PtEngine(self.args.orm_model, max_batch_size=64)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        self.template.set_mode('train')

    def _sample(self, model, batch, generation_config: GenerationConfig):
        queries = batch["input_ids"]
        generation_config.num_return_sequences = self.args.num_return_sequences
        generation_config.return_legacy_cache = False
        # generation_config.num_beam_groups = 5
        # generation_config.num_beams = 10
        # generation_config.do_sample = False
        # generation_config.diversity_penalty = 0.1

        with torch.no_grad():
            responses = batch_generation(model, queries,
                                         local_rollout_forward_batch_size=queries.shape[0],
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         generation_config=generation_config)
            return responses

    def _prepare_sampler(self):
        if self.args.sampler_type in samplers:
            self.sampler = samplers[self.args.sampler_type]()
        elif self.args.sampler_type == 'sample':
            self.sampler = self._sample
        elif self.args.sampler_type == 'mcts':
            pass
        else:
            raise NotImplementedError

    def _get_reward(self, model, infer_requests: List[InferRequest], request_config=None):
        resp_list = model.infer(infer_requests, request_config=request_config)
        arr = [float(resp_list[i].choices[0].message.content) for i in range(len(resp_list))]

        def normalize(arr):
            min_val = np.min(arr)
            max_val = np.max(arr)
            if min_val == max_val:
                if min_val == 0:
                    constant_value = 0.0
                else:
                    constant_value = 0.5
                return np.full_like(arr, fill_value=constant_value, dtype=np.float64)
            normalized = (arr - min_val) / (max_val - min_val + 1e-5)
            return normalized
        
        return normalize(arr)

    def rollout(self, data, trainer, step):
        with torch.no_grad():
            trainer.model_wrapped.eval()
            trainer.model.eval()
            if hasattr(trainer.optimizer, "eval") and callable(trainer.optimizer.eval):
                trainer.optimizer.eval()
            eos = [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]
            for s in self.splitter:
                eos.extend(self.tokenizer.encode(s, add_special_tokens=False))
            generation_config = GenerationConfig(
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.step_temperature(step),
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos,
            )

            res = []
            origin = []
            with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator) as unwrapped_model:
                generated = self.sampler(unwrapped_model, data, generation_config)
                for i, gen in enumerate(generated):
                    _data = deepcopy(data)
                    messages = _data['_messages'][i]
                    assert messages[-1]['content'] is None
                    batch_decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=True)
                    infer_requests = []
                    for decoded in batch_decoded:
                        _messages = deepcopy(messages)
                        _messages[-1]['content'] = decoded
                        infer_requests.append(InferRequest(messages=_messages,
                                                            ground_truths=_data['ground_truth'][i]))
                    _messages = deepcopy(messages)
                    _messages[-1]['content'] = _data['ground_truth'][i]
                    infer_requests.append(InferRequest(messages=_messages,
                                                            ground_truths=_data['ground_truth'][i]))
                    orm_score = self._get_reward(self.orm_model, infer_requests)
                    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
                    if is_deepspeed_zero3_enabled():
                        import deepspeed
                        context = deepspeed.zero.GatheredParameters(self.prm_model.engine.parameters())
                    else:
                        context = nullcontext()
                    with context:
                        prm_score = self._get_reward(self.prm_model, infer_requests)
                    
                    if not any([score > 0 for score in orm_score]):
                        raise
                    score = np.array(prm_score) + np.array(orm_score)
                    sorted_indices = np.argsort(score)
                    batch_decoded.append(_data['ground_truth'][i])
                    logger.info(f'orm:{orm_score}, positive index: {sorted_indices[-1]}, negative index: {sorted_indices[0]}')
                    positive = batch_decoded[sorted_indices[-1]]
                    negative = batch_decoded[sorted_indices[0]]
                    messages[-1]['content'] = positive
                    encoded = self.template.encode(
                        StdTemplateInputs.from_dict({'messages': messages, 'rejected_response': negative},
                                                    tools_prompt=self.args.tools_prompt))
                    encoded.pop('_messages', None)
                    res.append(encoded)
                    origin.append(json.dumps({'messages': messages, 'rejected_response': negative}) + '\n')
            return res, origin

    def step_temperature(self, step):
        # Linear
        step_wise = (self.args.end_temperature - self.args.temperature) / self.args.num_rollout_iters
        return self.args.temperature + step_wise * step

    def step_reward_threshold(self, step):
        # Linear
        step_wise = (self.args.end_threshold - self.args.start_threshold) / self.args.num_rollout_iters
        return self.args.start_threshold + step_wise * step

    @staticmethod
    @contextlib.contextmanager
    def switch_dataset(trainer, sampled_ds):
        origin_dataset: Dataset = trainer.train_dataset
        trainer.train_dataset = sampled_ds
        yield
        origin_dataset = origin_dataset.shuffle()
        trainer.train_dataset = origin_dataset

    def rollout_or_load(self, _iter, trainer):
        _, local_rank, world_size, _ = get_dist_setting()
        logger.info(f'Starting iter:{_iter}')
        if hasattr(trainer, 'origin_dataset'):
            trainer.train_dataset = trainer.origin_dataset.shuffle()
        iter_file = os.path.join(self.args.sampler_output, f'step_{_iter}.jsonl')
        if not os.path.exists(iter_file) or not self.args.use_cache_dataset:
            self.template.set_mode('train')
            train_dataloader = trainer.get_train_dataloader()
            dumped_ds = []

            for _index, batch in enumerate(train_dataloader):
                self.template.set_mode('rlhf' if self.args.rlft_type != 'causal_lm' else 'train')
                logger.info(f'Rolling out index:{_index}')
                _, origin = self.rollout(batch, trainer, _iter)
                self.template.set_mode('train')
                dumped_ds.extend(origin)
                if _index >= self.args.num_rollout_batches - 1:
                    break

            if world_size > 1:
                from accelerate.utils import gather_object
                dumped_ds = gather_object(dumped_ds)

            if local_rank <= 0:
                with open(iter_file, 'w') as f:
                    f.writelines(dumped_ds)
            
            if world_size > 1:
                import torch.distributed as dist
                dist.barrier()

        iter_file = [iter_file]
        if _iter >= 1:
            iter_file = [os.path.join(self.args.sampler_output, f'step_{_iter-1}.jsonl')] + iter_file
        self.template.set_mode('rlhf' if self.args.rlft_type != 'causal_lm' else 'train')
        local_dataset, _ = load_dataset(iter_file, split_dataset_ratio=0., **self.args.get_dataset_kwargs())
        new_dataset, _ = self._encode_dataset(local_dataset, None)
        trainer.origin_dataset = trainer.train_dataset
        trainer.train_dataset = new_dataset

    def train(self, trainer):
        trainer._inner_training_loop = MethodType(
            partial(_inner_training_loop, num_rollout_iters=self.args.num_rollout_iters,
                    num_rollout_batches=self.args.num_rollout_batches,
                    rollout_func=partial(self.rollout_or_load, trainer=trainer)), trainer)

        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        self._prepare_rm()
        self._prepare_sampler()
        os.makedirs(self.args.sampler_output, exist_ok=True)
        trainer.train()
        return self._save_trainer_state(trainer)


def generate(
        lm_backbone: PreTrainedModel, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Code borrowed from trl"""
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    # input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=queries,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
    )
    return output.sequences[:, context_length:]


@torch.no_grad()
def batch_generation(
        model: torch.nn.Module,
        queries: torch.Tensor,
        local_rollout_forward_batch_size: int,
        pad_token_id: int,
        generation_config: GenerationConfig,
):
    """Code borrowed from trl"""
    responses = []
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i: i + local_rollout_forward_batch_size]
        response = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        response = response.reshape(local_rollout_forward_batch_size, -1, response.shape[-1])
        responses.append(response)
    return torch.cat(responses, 0)


def rlft_main(args: Union[List[str], RLFTArguments, None] = None):
    return SwiftRLFT(args).main()
