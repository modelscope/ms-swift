# Copyright (c) Alibaba, Inc. and its affiliates.

import math
from contextlib import contextmanager
from dataclasses import fields

import torch
import torch.nn as nn
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint as mg_save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import get_ltor_masks_and_position_ids

from swift.llm import ExportArguments, HfConfigFactory, get_model_tokenizer, get_template, save_checkpoint
from swift.utils import get_logger, get_n_params_grads
from ..argument import MegatronArguments
from ..model import get_megatron_model_meta
from .patcher import patch_megatron_tokenizer, patch_torch_dist_shard

logger = get_logger()


def _test_params_sum(model):
    total_sum = 0
    zero_count = 0
    n_parameter = 0
    for n, p in model.named_parameters():
        n_parameter += 1
        sum_ = p.cuda().float().abs().sum().cpu().item()
        if sum_ == 0:
            zero_count += 1
            logger.warning(f'n: {n}, sum: {sum_}')
        elif math.isnan(sum_) or math.isinf(sum_) or sum_ > 1e10:
            logger.warning(f'n: {n}, sum: {sum_}')
        else:
            total_sum += sum_
    logger.info(f'n_parameter: {n_parameter}')
    logger.info(f'total_sum: {total_sum}')
    logger.info(f'zero_count: {zero_count}')


def _find_modules(model, recurse: bool = True):
    modules = []
    children = list(model.children())
    for module in children:
        if module.__class__ is nn.ModuleList:
            modules += _find_modules(module, False)
        elif recurse:
            modules += _find_modules(module)
        else:
            modules.append(module)
    if not children:
        modules.append(model)
    return modules


@contextmanager
def _model_cpu_forward_context(modules, torch_dtype=None, device=None, share_embedding: bool = False):
    origin_torch_dtype = next(modules[0].parameters()).dtype

    def _to_cuda_hook(module, args):
        if device is not None:
            module.to(device)
        if torch_dtype is not None:
            module.to(torch_dtype)

    def _to_cpu_hook(module, args, output):
        if share_embedding and module is modules[0]:
            return
        module.to('cpu')
        if torch_dtype is not None:
            module.to(origin_torch_dtype)

    hooks = []
    for module in modules:
        hooks.append(module.register_forward_pre_hook(_to_cuda_hook))
        hooks.append(module.register_forward_hook(_to_cpu_hook))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def test_convert_precision(hf_model, mg_model, processor, torch_dtype=torch.float32):
    _test_params_sum(hf_model)
    _test_params_sum(mg_model)

    template = get_template(hf_model.model_meta.template, processor)
    input_ids = template.encode({'messages': [{'role': 'user', 'content': 'who are you?'}]})['input_ids']
    input_ids = torch.tensor(input_ids)[None].to('cuda')

    HfConfigFactory.set_model_config_attr(hf_model, 'use_cache', False)
    share_embedding = mg_model.share_embeddings_and_output_weights
    hf_modules = _find_modules(hf_model)
    with torch.inference_mode(), _model_cpu_forward_context(hf_modules, torch_dtype, share_embedding=share_embedding):
        hf_logits = hf_model(input_ids).logits
    hf_model = hf_model.to('cpu')

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    packed_seq_params = None
    mg_torch_dtype = torch_dtype
    # thd
    # from ..train.utils import get_packed_seq_params
    # mg_torch_dtype = None
    # packed_seq_params = get_packed_seq_params(position_ids)
    # attention_mask = None
    mg_modules = _find_modules(mg_model)
    with torch.inference_mode(), _model_cpu_forward_context(
            mg_modules, mg_torch_dtype, 'cuda', share_embedding=share_embedding):
        mg_logits = mg_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params)

    mean_diff = (mg_logits - hf_logits).abs().mean().item()
    max_diff = (mg_logits - hf_logits).abs().max().item()
    print(f'mean_diff: {mean_diff}, max_diff: {max_diff} (Please check that mean_diff is less than 0.1).')
    hf_tokens = hf_logits.argmax(-1)
    mg_tokens = mg_logits.argmax(-1)
    print(f'hf_tokens: {hf_tokens[0].tolist()}\nmg_tokens: {mg_tokens[0].tolist()}')
    print(f'token_diff: {(hf_tokens != mg_tokens).sum().item()}')


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


def _check_megatron_kwargs(kwargs):
    # Make sure that the keys in kwargs have default values of None in MegatronArguments.
    default_mapping = {field.name: field.default for field in fields(MegatronArguments)}
    for k in kwargs.keys():
        assert default_mapping[k] is None


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
    logger.info(f'megatron_config: {kwargs}')
    _check_megatron_kwargs(kwargs)
    megatron_args = MegatronArguments(**kwargs, **convert_kwargs, save=args.output_dir, torch_dtype=args.torch_dtype)
    patch_megatron_tokenizer(processor)
    extra_args = megatron_args.parse_to_megatron()
    extra_args_provider = megatron_model_meta.extra_args_provider
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=extra_args)

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
    logger.info(f'megatron_config: {kwargs}')
    _check_megatron_kwargs(kwargs)
    megatron_args = MegatronArguments(**kwargs, **convert_kwargs, load=args.mcore_model, torch_dtype=args.torch_dtype)
    patch_megatron_tokenizer(processor)
    extra_args = megatron_args.parse_to_megatron()
    extra_args_provider = megatron_model_meta.extra_args_provider
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=extra_args)

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
