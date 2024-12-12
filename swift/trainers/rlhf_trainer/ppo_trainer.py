# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import time
from typing import Optional, Union, Dict, Tuple, List

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, DataCollatorWithPadding, TrainerCallback
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback, TrainerControl, ExportableState
from trl import PPOv2Trainer as HFPPOTrainer, PPOv2Config
from trl.trainer.ppov2_trainer import PolicyAndValueWrapper
from trl.trainer.utils import exact_div, disable_dropout_in_model, OnlineTrainerState, prepare_deepspeed

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin


HFPPOTrainer.__init_origin__ = HFPPOTrainer.__init__


def init_v2(
        self,
        config: PPOv2Config,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
) -> None:
    if ref_policy is policy:
        raise ValueError(
            "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
            "same as `policy`, you must mass a copy of it, or `None` if you use peft."
        )

    self.args = config
    args = config
    self.tokenizer = tokenizer
    self.policy = policy

    self.policy.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

    self.ref_policy = ref_policy
    self.reward_model = reward_model
    self.train_dataset = train_dataset
    self.train_dataset_len = len(train_dataset)
    self.value_model = value_model
    self.data_collator = data_collator
    self.eval_dataset = eval_dataset
    self.optimizer, self.lr_scheduler = optimizers

    #########
    # calculate various batch sizes
    #########
    if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
        args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    self.accelerator = accelerator
    args.world_size = accelerator.num_processes
    args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
    )
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.mini_batch_size = exact_div(
        args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
    )
    args.local_mini_batch_size = exact_div(
        args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
    )
    if args.whiten_rewards:
        assert (
                args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    # `per_rank_rollout_batch_size` is our `args.local_batch_size`
    # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
    args.num_total_batches = math.ceil(
        args.total_episodes / args.batch_size
    )  # we may train for more than `total_episodes`
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
    if args.num_sample_generations > 0:
        self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
    self.local_dataloader_batch_size = args.local_batch_size

    #########
    # setup model, optimizer, and others
    #########
    for module in [policy, ref_policy, value_model, reward_model]:
        if module is not None:
            disable_dropout_in_model(module)
    if args.stop_token and args.stop_token == "eos":
        args.stop_token_id = tokenizer.eos_token_id
    self.model = PolicyAndValueWrapper(policy, value_model)
    self.model.config = policy.config  # needed for pushing to hub
    self.create_optimizer_and_scheduler(
        num_training_steps=args.num_total_batches
    )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

    #########
    ### trainer specifics
    #########
    default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
    self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
    self.callback_handler = CallbackHandler(
        self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
    )
    self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
    self.control = TrainerControl()
    self.state = OnlineTrainerState(
        is_local_process_zero=self.is_local_process_zero(),
        is_world_process_zero=self.is_world_process_zero(),
        stateful_callbacks=[
            cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
        ],
    )
    self.current_flos = 0
    self.hp_search_backend = None
    self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
    self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
    # Create distant repo and output directory if needed
    self.hub_model_id = None
    if self.args.push_to_hub:
        self.init_hf_repo()
    if self.args.should_save:
        os.makedirs(self.args.output_dir, exist_ok=True)

    #########
    ### setup dataloader
    #########
    self.dataloader = DataLoader(
        self.train_dataset,
        batch_size=self.local_dataloader_batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
        drop_last=True,  # needed; otherwise the last batch will be of ragged shape
    )
    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
    torch.manual_seed(self.local_seed)  # reset the local seed again

    self.eval_dataloader = DataLoader(
        self.eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=DataCollatorWithPadding(self.tokenizer),
        drop_last=True,
    )  # no need to shuffle eval dataset
    self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

    if self.is_deepspeed_enabled:
        self.reward_model = prepare_deepspeed(
            self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
        )
        self.ref_policy = prepare_deepspeed(
            self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
        )
    else:
        self.ref_policy = self.ref_policy.to(self.accelerator.device)
        self.reward_model = self.reward_model.to(self.accelerator.device)


HFPPOTrainer.__init__ = init_v2


class PPOTrainer(RLHFTrainerMixin, SwiftMixin, HFPPOTrainer):

    def __init__(self, model: PreTrainedModel, ref_model: PreTrainedModel, *_args, **kwargs):
        kwargs['policy'] = model
        kwargs['ref_policy'] = ref_model
        super().__init__(model, ref_model, *_args, **kwargs)
        # reset dataloader
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=kwargs['data_collator'],
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        self.accelerator.prepare(self.data_collator)
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=kwargs['data_collator'],
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

    def train(self, *args, **kwargs):
        # remove args that are not needed for the HFPPOTrainer
        HFPPOTrainer.train(self)


def patched_init(self, **kwargs):
    kwargs_to_pop = ['model', 'model_init', 'compute_metrics', 'preprocess_logits_for_metrics']
    for kwarg in kwargs_to_pop:
        kwargs.pop(kwarg, None)
    kwargs['config'] = kwargs.pop('args')
    original_init(self, **kwargs)


original_init = HFPPOTrainer.__init__
HFPPOTrainer.__init__ = patched_init
