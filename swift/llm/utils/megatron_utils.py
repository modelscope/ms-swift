import os
import sys


os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
import torch
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Dict, Tuple
from functools import wraps
from swift.utils import subprocess_run, get_dist_setting


import torch
import sys
from functools import wraps
from megatron.training import get_args
from .dataset import get_dataset
from .template import get_template
from .utils import LazyLLMDataset, to_device
from megatron_patch.model.qwen2.model import GPTModel
from megatron_patch.model.qwen2.transformer_config import Qwen2TransformerConfig
from swift.utils import get_logger
from megatron.training.arguments import core_transformer_config_from_args
from megatron_patch.model.qwen2.layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_layer_local_spec
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.training import get_timers
from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.training import get_tokenizer
from megatron_patch.data.utils import get_batch_on_this_tp_rank_original
from megatron_patch.data import build_pretrain_dataset_from_original
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from functools import partial

logger = get_logger()


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    tokenizer = get_tokenizer()

    config = core_transformer_config_from_args(args, Qwen2TransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"

    if use_te:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm,
                                                                            args.qk_layernorm)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
    )

    return model


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    inputs = next(data_iterator)
    input_ids = inputs['input_ids']
    inputs['position_ids'] = torch.arange(input_ids.shape[1], device=input_ids.device)[None].expand_as(input_ids)
    timers('batch-generator').stop()
    inputs = to_device(inputs, torch.cuda.current_device())

    output_tensor = model(**inputs)

    return output_tensor, partial(loss_func, inputs['attention_mask'])


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    tokenizer = get_tokenizer()
    logger.info("> building train, validation, and test datasets for GPT ...")
    # Loading Dataset
    train_dataset, val_dataset = get_dataset(args.dataset, 0.01)
    template = get_template(
        args.template_type,
        tokenizer)
    train_dataset = LazyLLMDataset(train_dataset, template)
    if val_dataset is not None:
        val_dataset = LazyLLMDataset(val_dataset, template)

    from megatron.training import training
    assert not hasattr(training, '_old_build_pretraining_data_loader')
    _old_build_pretraining_data_loader = training.build_pretraining_data_loader

    @wraps(_old_build_pretraining_data_loader)
    def build_pretraining_data_loader(*args, **kwargs):
        res = _old_build_pretraining_data_loader(*args, **kwargs)
        if res is not None:
            res.collate_fn = template.data_collator
        return res

    training.build_pretraining_data_loader = build_pretraining_data_loader
    training._old_build_pretraining_data_loader = _old_build_pretraining_data_loader
    return train_dataset, val_dataset, None
