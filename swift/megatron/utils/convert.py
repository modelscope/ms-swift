# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import torch
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint as mg_save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import get_ltor_masks_and_position_ids

from swift.llm import ExportArguments, get_model_tokenizer, get_template, save_checkpoint
from swift.utils import get_logger, get_n_params_grads
from ..argument import MegatronArguments
from ..model import get_megatron_model_meta
from .patcher import patch_megatron_tokenizer, patch_torch_dist_shard

logger = get_logger()


def test_convert_precision(hf_model, mg_model, processor):
    torch_dtype = hf_model.dtype
    template = get_template(hf_model.model_meta.template, processor)
    input_ids = template.encode({'messages': [{'role': 'user', 'content': 'who are you?'}]})['input_ids']
    input_ids = torch.tensor(input_ids)[None].to('cuda')
    hf_model.to('cuda')
    hf_model.to(torch.float32)
    with torch.inference_mode():
        hf_logits = hf_model(input_ids).logits
    hf_model.to(torch_dtype)
    hf_model.to('cpu')

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    mg_model.to('cuda')
    mg_model.to(torch.float32)
    with torch.inference_mode():
        mg_logits = mg_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    mg_model.to(torch_dtype)
    mg_model.to('cpu')

    mean_diff = (mg_logits - hf_logits).abs().mean().item()
    max_diff = (mg_logits - hf_logits).abs().max().item()
    print(f'mean_diff: {mean_diff}, max_diff: {max_diff}')
    hf_tokens = hf_logits.argmax(-1)
    mg_tokens = mg_logits.argmax(-1)
    print(f'hf_tokens: {hf_tokens[0].tolist()}\nmg_tokens: {mg_tokens[0].tolist()}')
    assert mean_diff < 0.1
    assert (hf_tokens == mg_tokens).all()


convert_kwargs = {
    'use_cpu_initialization': True,
    'no_save_optim': True,
    'no_save_rng': True,
    'no_load_optim': True,
    'no_load_rng': True,
    'no_masked_softmax_fusion': True,
    'no_bias_dropout_fusion': True,
    'no_bias_swiglu_fusion': True,
    'no_rope_fusion': True
}


def convert_hf2mcore(args: ExportArguments) -> None:
    kwargs = args.get_model_kwargs()
    hf_model, processor = get_model_tokenizer(**kwargs)
    if args.thread_count is None:
        checkpoint_size = sum(get_n_params_grads(hf_model)[0]) * torch.finfo(args.torch_dtype).bits // 8e9
        args.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB
    patch_torch_dist_shard(args.thread_count)

    megatron_model_meta = get_megatron_model_meta(args.model_type)
    assert megatron_model_meta is not None, f'Model: {args.model} is not supported.'
    kwargs = megatron_model_meta.convert_hf_config(processor.model_info.config)
    megatron_args = MegatronArguments(**kwargs, **convert_kwargs, save=args.output_dir, torch_dtype=args.torch_dtype)
    patch_megatron_tokenizer(processor)
    extra_args = megatron_args.parse_to_megatron()
    initialize_megatron(args_defaults=extra_args)

    mg_model = megatron_model_meta.model_provider()
    logger.info('Megatron model created successfully.')
    megatron_model_meta.convert_hf2mcore(hf_model, mg_model)
    if args.test_convert_precision:
        test_convert_precision(hf_model, mg_model, processor)
    logger.info('Successfully transferred HF model weights to MG model.')
    mg_save_checkpoint(1, [mg_model], None, None, 0)
    args.save_args()
    logger.info(f'Successfully saved Megatron model weights in `{args.output_dir}`.')


def convert_mcore2hf(args: ExportArguments) -> None:
    kwargs = args.get_model_kwargs()
    hf_model, processor = get_model_tokenizer(**kwargs)
    if args.thread_count is None:
        checkpoint_size = sum(get_n_params_grads(hf_model)[0]) * torch.finfo(args.torch_dtype).bits // 8e9
        args.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB
    patch_torch_dist_shard(args.thread_count)

    megatron_model_meta = get_megatron_model_meta(args.model_type)
    assert megatron_model_meta is not None, f'Model: {args.model} is not supported.'
    kwargs = megatron_model_meta.convert_hf_config(processor.model_info.config)
    megatron_args = MegatronArguments(**kwargs, **convert_kwargs, load=args.mcore_model, torch_dtype=args.torch_dtype)
    patch_megatron_tokenizer(processor)
    extra_args = megatron_args.parse_to_megatron()
    initialize_megatron(args_defaults=extra_args)

    mg_model = megatron_model_meta.model_provider()
    load_checkpoint([mg_model], None, None, strict=True)
    logger.info('Megatron model created successfully.')
    megatron_model_meta.convert_mcore2hf(hf_model, mg_model)
    if args.test_convert_precision:
        test_convert_precision(hf_model, mg_model, processor)
    logger.info('Successfully transferred MG model weights to HF model.')
    save_checkpoint(
        hf_model,
        processor,
        args.output_dir,
        safe_serialization=args.safe_serialization,
        model_dirs=[args.mcore_model, args.model_dir],
        max_shard_size=args.max_shard_size,
        additional_saved_files=hf_model.model_meta.additional_saved_files)
    args.save_args()
    logger.info(f'Successfully saved HF model weights in `{args.output_dir}`.')
