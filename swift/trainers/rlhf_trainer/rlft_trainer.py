# Copyright (c) Alibaba, Inc. and its affiliates.
import gc
import math
import os
import time
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Optional, Union, Dict, Tuple, List
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import GenerationConfig, PreTrainedModel, StoppingCriteriaList
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, TrainerCallback
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback, TrainerControl, ExportableState
from trl import PPOv2Trainer as HFPPOTrainer, PPOv2Config
from trl.core import masked_mean, masked_whiten
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.ppov2_trainer import INVALID_LOGPROB
from trl.trainer.ppov2_trainer import PolicyAndValueWrapper
from trl.trainer.utils import (
    first_true_indices,
    forward,
    get_reward,
    truncate_response, print_rich_table,
)
from trl.trainer.utils import exact_div, disable_dropout_in_model, OnlineTrainerState, prepare_deepspeed

from .ppo_trainer import PPOTrainer

HFPPOTrainer.__init_origin__ = HFPPOTrainer.__init__

from ...llm.template.utils import StopWordsCriteria


def generate(
        lm_backbone: PreTrainedModel, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig,
        tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=StoppingCriteriaList([StopWordsCriteria(tokenizer, ['Observation:', '<|im_end|>', '<|endoftext|>', '\nObservation:'])]),
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
    tokenizer,
):
    query_responses = []
    logitss = []
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query = query.to('cuda:0')
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
            tokenizer
        )

        observation_tokens = tokenizer.encode('Observation:', add_special_tokens=False)
        for qr in query_response:
            r = qr[queries.shape[1]:]
            for i in range(len(r)):
                if len(r[i:i+len(observation_tokens)]) == len(observation_tokens) and all(r[i:i+len(observation_tokens)].cpu().numpy() == observation_tokens):
                    r[i:i+len(observation_tokens)] = torch.tensor([tokenizer.pad_token_id] * len(observation_tokens))
        
        query_responses.append(query_response)
        logitss.append(logits)
    return torch.cat(query_responses, 0), torch.cat(logitss, 0)


def init_v2(
        self,
        args: PPOv2Config,
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
        **kwargs
) -> None:
    if ref_policy is policy:
        raise ValueError(
            "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
            "same as `policy`, you must mass a copy of it, or `None` if you use peft."
        )

    self.args = args
    args = args
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
        if self.reward_model is not None:
            self.reward_model = self.reward_model.to(self.accelerator.device)


HFPPOTrainer.__init__ = init_v2


@dataclass
class RolloutState:
    advantages: List[torch.Tensor] = None
    scores: List[torch.Tensor] = None
    non_score_reward: List[torch.Tensor] = None
    kl: List[torch.Tensor] = None
    responses: List[torch.Tensor] = None
    query_responses: List[torch.Tensor] = None
    logprobs: List[torch.Tensor] = None
    ref_logprobs: List[torch.Tensor] = None
    returns: List[torch.Tensor] = None
    values: List[torch.Tensor] = None
    padding_mask: List[torch.Tensor] = None
    context_length: List[torch.Tensor] = None
    padding_mask_p1: List[torch.Tensor] = None
    postprocessed_responses: List[str] = None
    sequence_lengths: List[str] = None
    contain_eos_token: List[str] = None
    sequence_lengths_p1: List[str] = None
    response_idxs: List[str] = None
    rewards: List[str] = None
    actual_start: List[str] = None
    actual_end: List[str] = None


class RLFTTrainer(PPOTrainer):

    def __init__(self,
                 *args,
                 reward_func=None,
                 **kwargs):
        if 'reward_model' not in kwargs:
            kwargs['reward_model'] = None
        super().__init__(*args, **kwargs)
        self.reward_func = reward_func

        def repeat_generator():
            while True:
                yield from self.dataloader

        self.iter_dataloader = iter(repeat_generator())

    def train_reward_model(self, rollout_state: RolloutState, metrics):
        pass

    def train_policy_model(self, rollout_state: RolloutState, metrics, update):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        tokenizer = self.tokenizer
        device = accelerator.device
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(args.num_ppo_epochs):
            b_inds = np.random.permutation(args.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                mini_batch_end = mini_batch_start + args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                    with accelerator.accumulate(model):
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = rollout_state.advantages[micro_batch_inds]
                        mb_responses = rollout_state.responses[micro_batch_inds]
                        mb_query_responses = rollout_state.query_responses[micro_batch_inds]
                        mb_logprobs = rollout_state.logprobs[micro_batch_inds]
                        mb_return = rollout_state.returns[micro_batch_inds]
                        mb_values = rollout_state.values[micro_batch_inds]

                        output, vpred_temp = forward(model, mb_query_responses, tokenizer.pad_token_id)
                        if update < 0:
                            output.logits = output.logits.detach()
                        logits = output.logits[:, rollout_state.context_length - 1: -1]
                        logits /= args.temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(
                            new_logprobs, rollout_state.padding_mask[micro_batch_inds], INVALID_LOGPROB
                        )
                        vpred = vpred_temp[:, rollout_state.context_length - 1: -1].squeeze(-1)
                        vpred = torch.masked_fill(vpred, rollout_state.padding_mask_p1[micro_batch_inds], 0)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.cliprange_value,
                            mb_values + args.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(vf_loss_max, ~rollout_state.padding_mask_p1[micro_batch_inds])
                        vf_clipfrac = masked_mean(
                            (vf_losses2 > vf_losses1).float(), ~rollout_state.padding_mask_p1[micro_batch_inds]
                        )
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = masked_mean(pg_loss_max, ~rollout_state.padding_mask[micro_batch_inds])
                        loss = pg_loss + args.vf_coef * vf_loss
                        # print('==================>vf_loss:', vf_loss, 'loss', loss, 'scores', rollout_state.scores)
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        with torch.no_grad():
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~rollout_state.padding_mask[micro_batch_inds]
                            )
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff ** 2).mean()
                            approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                pg_clipfrac
                            )
                            pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                vf_clipfrac
                            )
                            entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                # del everything and empty cache
                # fmt: off
                del (
                    output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                    vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2,
                    pg_loss_max,
                    pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                    mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                )
                # fmt: on
                torch.cuda.empty_cache()
        metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
        metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
        metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
        metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
        metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
        metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
        metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
        metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()

    def rollout_mcts(self) -> RolloutState:
        pass

    def rollout(self) -> RolloutState:
        args = self.args
        accelerator = self.accelerator
        model = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        tokenizer = self.tokenizer
        device = accelerator.device

        data = next(self.iter_dataloader)
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=args.response_length,
                temperature=(args.temperature + 1e-7),
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.tokenizer.pad_token_id, self.tokenizer.eos_token_id],
            )
            queries = data["input_ids"].to(device)
            context_length = queries.shape[1]
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            scores = []
            sequence_lengths = []
            values = []
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generated_queries = []
                ground_truth_queries = []
                gt_index = []
                for i in range(queries.shape[0]):
                    if np.random.random() <= 1.0:
                        generated_queries.append(queries[i])
                    else:
                        gt_index.append(i)
                        ground_truth_queries.append(queries[i])
                print('gt_index', gt_index)
                query_responses = []
                logitss = []
                if generated_queries:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        torch.stack(generated_queries, dim=0),
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                        self.tokenizer,
                    )
                query_responses_gt = []
                logitss_gt = []
                if ground_truth_queries:
                    ground_truth_tokens = [self.tokenizer.encode(gt, add_special_tokens=False) for i, gt in enumerate(data["ground_truth"]) if i in gt_index]
                    ground_truth_qr = []
                    for query, ground_truth in zip(ground_truth_queries, ground_truth_tokens):
                        ground_truth_qr.append(torch.cat((query, torch.tensor(ground_truth).to(query.device)), dim=0))
                    query_responses_gt = pad_sequence(ground_truth_qr, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                    query_responses_gt = query_responses_gt.to(device)
                    gt_outputs = forward(unwrapped_model.policy, query_responses_gt, tokenizer.pad_token_id)
                    logitss_gt = gt_outputs.logits[:, context_length - 1: -1]
                    logitss_gt /= args.temperature + 1e-7

                qrs = []
                lts = []
                gti = 0
                gei = 0
                for i in range(queries.shape[0]):
                    if i in gt_index:
                        qrs.append(query_responses_gt[gti])
                        lts.append(logitss_gt[gti])
                        gti += 1
                    else:
                        qrs.append(query_responses[gei])
                        lts.append(logitss[gei])
                        gei += 1
                    
                query_responses = pad_sequence(qrs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                logitss = pad_sequence(lts, batch_first=True, padding_value=INVALID_LOGPROB)

            for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                query = queries[i: i + args.local_rollout_forward_batch_size]
                query_response = query_responses[i: i + args.local_rollout_forward_batch_size]
                response = query_response[:, context_length:]
                logits = logitss[i: i + args.local_rollout_forward_batch_size]
                all_logprob = F.log_softmax(logits, dim=-1)
                logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del logits, all_logprob
                torch.cuda.empty_cache()

                ref_output = forward(ref_policy, query_response, tokenizer.pad_token_id)
                ref_logits = ref_output.logits[:, context_length - 1: -1]
                ref_logits /= args.temperature + 1e-7
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del ref_output, ref_logits, ref_all_logprob
                torch.cuda.empty_cache()

                # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                postprocessed_response = response
                if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(
                        args.stop_token_id, tokenizer.pad_token_id, response
                    )

                # Response Processing 2. run reward model on the truncated responses
                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                unwrapped_value_model = accelerator.unwrap_model(model).value_model
                full_value, _, _ = get_reward(
                    unwrapped_value_model, query_response, tokenizer.pad_token_id, context_length
                )
                value = full_value[:, context_length - 1: -1].squeeze(-1)
                _, score, _ = self.get_reward(
                    reward_model, data, postprocessed_query_response, tokenizer.pad_token_id, context_length
                )

                responses.append(response)
                postprocessed_responses.append(postprocessed_response)
                logprobs.append(logprob)
                ref_logprobs.append(ref_logprob)
                sequence_lengths.append(sequence_length)
                scores.append(score)
                values.append(value)
            responses = torch.cat(responses, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            scores = torch.cat(scores, 0)
            values = torch.cat(values, 0)
            del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
            torch.cuda.empty_cache()
            gc.collect()

            # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
            # Completions not passing that filter will receive a lower score.
            contain_eos_token = torch.any(postprocessed_responses == self.tokenizer.eos_token_id, dim=-1)
            if self.args.missing_eos_penalty is not None:
                scores[~contain_eos_token] -= self.args.missing_eos_penalty
            # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
            sequence_lengths_p1 = sequence_lengths + 1
            padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
            values = torch.masked_fill(values, padding_mask_p1, 0)

            # 4. compute rewards
            kl = logprobs - ref_logprobs
            non_score_reward = -args.kl_coef * kl
            rewards = non_score_reward.clone()
            actual_start = torch.arange(rewards.size(0), device=rewards.device)
            actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
            rewards[[actual_start, actual_end]] += scores

            # 5. whiten rewards
            if args.whiten_rewards:
                rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

            # 6. compute advantages and returns
            lastgaelam = 0
            advantages_reversed = []
            gen_length = responses.shape[1]
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values
            advantages = masked_whiten(advantages, ~padding_mask)
            advantages = torch.masked_fill(advantages, padding_mask, 0)
            torch.cuda.empty_cache()

        return RolloutState(
            advantages=advantages,
            scores=scores,
            kl=kl,
            responses=responses,
            postprocessed_responses=postprocessed_responses,
            non_score_reward=non_score_reward,
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            returns=returns,
            values=values,
            padding_mask=padding_mask,
            context_length=context_length,
            padding_mask_p1=padding_mask_p1,
            sequence_lengths=sequence_lengths,
            contain_eos_token=contain_eos_token,
            sequence_lengths_p1=sequence_lengths_p1,
            response_idxs=response_idxs,
            rewards=rewards,
            actual_start=actual_start,
            actual_end=actual_end,
        )

    def get_reward(
            self, model: torch.nn.Module, batch, query_responses: torch.Tensor, pad_token_id: int, context_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reward_func:
            return self.reward_func(model, batch, query_responses, pad_token_id, context_length)
        else:
            return get_reward(model, query_responses, pad_token_id, context_length)

    def train(self, *args, **kwargs):
        args = self.args
        accelerator = self.accelerator
        model = self.model
        tokenizer = self.tokenizer
        accelerator.print("===training policy===")
        start_time = time.time()
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            rollout_state = self.rollout()
            metrics = {}
            self.train_reward_model(rollout_state, metrics)
            self.train_policy_model(rollout_state, metrics, update)
            with torch.no_grad():
                mean_kl = rollout_state.kl.sum(1).mean()
                mean_entropy = (-rollout_state.logprobs).sum(1).mean()
                mean_non_score_reward = rollout_state.non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + rollout_state.scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(rollout_state.scores.mean()).mean().item()
                metrics["val/num_eos_tokens"] = (rollout_state.responses == tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del rollout_state.kl, mean_kl, mean_entropy, mean_non_score_reward, rollout_state.scores, metrics, rollout_state.non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                rollout_state.query_responses,
                rollout_state.responses,
                rollout_state.postprocessed_responses,
                rollout_state.logprobs,
                rollout_state.ref_logprobs,
                rollout_state.values,
                rollout_state.sequence_lengths,
                rollout_state.contain_eos_token,
                rollout_state.sequence_lengths_p1,
                rollout_state.response_idxs,
                rollout_state.padding_mask,
                rollout_state.padding_mask_p1,
                rollout_state.rewards,
                rollout_state.actual_start,
                rollout_state.actual_end,
                rollout_state.advantages,
                rollout_state.returns,
            )
            del rollout_state
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        tokenizer = self.tokenizer
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.pad_token_id, self.tokenizer.eos_token_id],
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for batch in self.dataloader:
                query = batch["input_ids"].to('cuda:0')
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        tokenizer.pad_token_id,
                        generation_config,
                        self.tokenizer,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )
                    table["query"].extend(gather_object(tokenizer.batch_decode(query, skip_special_tokens=True)))
                    table["model response"].extend(gather_object(tokenizer.batch_decode(postprocessed_response)))
                    if 'ground_truth' in batch:
                        table["grounding truth"].extend(batch['ground_truth'])

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    _, score, _ = self.get_reward(
                        self.reward_model, batch, postprocessed_query_response, tokenizer.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0: 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})
