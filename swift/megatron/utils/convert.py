# Copyright (c) Alibaba, Inc. and its affiliates.

import math
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict

import torch
import torch.nn as nn
from megatron.training import get_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint as mg_save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import get_ltor_masks_and_position_ids

from swift.llm import ExportArguments, HfConfigFactory, prepare_model_template, save_checkpoint, to_device
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


def _find_modules(model, recurse: bool = True, prefix='', ignore_modules=None):
    ignore_modules = ignore_modules or []
    for k in ignore_modules:
        if prefix.startswith(k):
            return []
    else:
        named_children = list(model.named_children())

    modules = []
    for n, module in named_children:
        if module.__class__ is nn.ModuleList:
            modules += _find_modules(module, False, prefix=f'{prefix}{n}.', ignore_modules=ignore_modules)
        elif recurse:
            modules += _find_modules(module, prefix=f'{prefix}{n}.', ignore_modules=ignore_modules)
        else:
            modules.append(module)
    if not named_children:
        modules.append(model)
    return modules


@contextmanager
def _model_cpu_forward_context(modules, torch_dtype=None, device=None, share_embedding: bool = False):
    origin_torch_dtype = next(modules[0].parameters()).dtype

    def _to_cuda_hook(module, args):
        if device is not None or torch_dtype is not None:
            module.to(device=device, dtype=torch_dtype)

    def _to_cpu_hook(module, args, output):
        if share_embedding and module is modules[0]:
            return
        module.to(device='cpu', dtype=origin_torch_dtype)

    hooks = []
    for module in modules:
        hooks.append(module.register_forward_pre_hook(_to_cuda_hook))
        hooks.append(module.register_forward_hook(_to_cpu_hook))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def get_examples(is_multimodal: bool) -> Dict[str, Any]:
    mm_type = 'image'
    if is_multimodal:
        if mm_type == 'image':
            data = {
                'messages': [{
                    'role': 'user',
                    'content': '<image>describe the image.'
                }, {
                    'role':
                    'assistant',
                    'content':
                    'The image depicts a close-up of a kitten with striking features. '
                    'The kitten has a white and gray coat with distinct black stripes, '
                    'particularly noticeable on its face and ears. Its eyes are large '
                    'and expressive, with a captivating blue hue that stands out against '
                    "the darker fur around them. The kitten's nose is small and pink, "
                    'and it has long, delicate whiskers extending from either side of its mouth. '
                    "The background is blurred, drawing attention to the kitten's face and "
                    'making it the focal point of the image. The overall impression is '
                    'one of cuteness and charm.'
                }],
                'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
            }
        elif mm_type == 'audio':
            data = {
                'messages': [{
                    'role': 'user',
                    'content': '<audio>Caption the audio.'
                }, {
                    'role': 'assistant',
                    'content': "The audio contains a male voice speaking the phrase '今天天气真好呀' in Mandarin."
                }],
                'audios': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav']
            }
    else:
        data = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'Introduction to ms-swift.'
                },
                {
                    'role':
                    'assistant',
                    'content':
                    'ms-swift is an official framework provided by the ModelScope community for fine-tuning '
                    'and deploying large language models and multi-modal large models.'
                },
            ]
        }
    return data


def test_convert_precision(hf_model, mg_model, template, torch_dtype=torch.float32):
    hf_model.eval()
    mg_model.eval()
    _test_params_sum(hf_model)
    _test_params_sum(mg_model)

    template.set_mode('train')
    template.register_post_encode_hook([hf_model])
    is_multimodal = template.model_meta.is_multimodal
    inputs = get_examples(is_multimodal)
    inputs = template.encode(inputs)
    inputs = to_device(template.data_collator([inputs]), 'cuda')

    HfConfigFactory.set_model_config_attr(hf_model, 'use_cache', False)
    mg_language_model = mg_model.language_model if is_multimodal else mg_model
    share_embedding = mg_language_model.share_embeddings_and_output_weights
    model_arch = hf_model.model_meta.model_arch
    ignore_modules = (model_arch.vision_tower + model_arch.aligner) if is_multimodal else []

    hf_modules = _find_modules(hf_model, ignore_modules=ignore_modules)
    with torch.inference_mode(), _model_cpu_forward_context(hf_modules, torch_dtype, share_embedding=share_embedding):
        inputs.pop('text_position_ids', None)
        hf_logits = hf_model(**inputs).logits
    hf_model.to('cpu')

    input_ids = inputs['input_ids']
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    packed_seq_params = None
    mg_torch_dtype = torch_dtype
    # thd
    # from ..trainers.utils import get_packed_seq_params
    # mg_torch_dtype = None
    # packed_seq_params = get_packed_seq_params(position_ids)
    # attention_mask = None
    mg_language_model.config.fp8 = None  # compat fp8
    mg_modules = _find_modules(mg_language_model, ignore_modules=['visual'])
    kwargs = {k: v for k, v in inputs.items() if k not in ['input_ids', 'attention_mask', 'labels']}
    if 'position_ids' not in kwargs:
        kwargs['position_ids'] = position_ids
    with torch.inference_mode(), _model_cpu_forward_context(
            mg_modules, mg_torch_dtype, 'cuda', share_embedding=share_embedding):
        mg_logits = mg_model(
            input_ids=input_ids, attention_mask=attention_mask, packed_seq_params=packed_seq_params, **kwargs)
    args = get_args()
    if args.task_type == 'seq_cls':
        mg_logits = mg_logits[:, -1]
        mean_diff = (mg_logits - hf_logits).abs().mean().item()
        max_diff = (mg_logits - hf_logits).abs().max().item()
        print(f'mean_diff: {mean_diff}, max_diff: {max_diff}')
    else:
        token_mean_diff = (mg_logits - hf_logits).abs().mean(dim=-1)
        mean_diff = token_mean_diff.mean().item()
        max_diff = (mg_logits - hf_logits).abs().max().item()
        loss_mask = (torch.roll(inputs['labels'], -1) != -100)
        mean_diff_with_loss = token_mean_diff[loss_mask].mean().item()
        max_diff_with_loss = (mg_logits - hf_logits)[loss_mask].abs().max().item()
        print(f'token_mean_diff: {token_mean_diff}')
        print(f'mean_diff: {mean_diff}, max_diff: {max_diff}')
        print(f'mean_diff (with loss): {mean_diff_with_loss}, max_diff (with loss): {max_diff_with_loss} '
              '(Please check that mean_diff is less than 0.1).')
        hf_tokens = hf_logits.argmax(-1)
        mg_tokens = mg_logits.argmax(-1)
        print(f'hf_tokens: {hf_tokens[0].tolist()}\nmg_tokens: {mg_tokens[0].tolist()}')
        print(f'token_diff: {(hf_tokens != mg_tokens).sum().item()}')
        print(f'token_diff (with loss): {(hf_tokens[loss_mask] != mg_tokens[loss_mask]).sum().item()}')


convert_kwargs = {
    'use_cpu_initialization': True,
    'no_save_optim': True,
    'no_save_rng': True,
    'no_load_optim': True,
    'no_load_rng': True,
    'finetune': True,
    'attention_backend': 'unfused',
}


def _check_megatron_kwargs(kwargs):
    # Make sure that the keys in kwargs have default values of None in MegatronArguments.
    default_mapping = {field.name: field.default for field in fields(MegatronArguments)}
    for k in kwargs.keys():
        assert default_mapping[k] is None


def convert_hf2mcore(args: ExportArguments) -> None:
    hf_model, template = prepare_model_template(args, patch_offload=not args.test_convert_precision)
    processor = template.processor
    if args.thread_count is None:
        checkpoint_size = sum(get_n_params_grads(hf_model)[0]) * torch.finfo(args.torch_dtype).bits // 8e9
        args.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB
    patch_torch_dist_shard(args.thread_count)

    megatron_model_meta = get_megatron_model_meta(args.model_type)
    assert megatron_model_meta is not None, f'Model: {args.model} is not supported.'
    kwargs = megatron_model_meta.convert_hf_config(processor.model_info.config)
    logger.info(f'megatron_config: {kwargs}')
    _check_megatron_kwargs(kwargs)
    current_convert_kwargs = convert_kwargs.copy()
    if args.model_info.is_moe_model:
        current_convert_kwargs['moe_grouped_gemm'] = True
    megatron_args = MegatronArguments(
        **kwargs, **current_convert_kwargs, save=args.output_dir, torch_dtype=args.torch_dtype)
    patch_megatron_tokenizer(processor)
    extra_args = megatron_args.parse_to_megatron()
    extra_args['model_info'] = args.model_info
    extra_args['model_meta'] = args.model_meta
    extra_args['megatron_model_meta'] = megatron_model_meta
    extra_args_provider = megatron_model_meta.extra_args_provider
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=extra_args)

    mg_model = megatron_model_meta.model_provider()
    logger.info('Megatron model created successfully.')
    megatron_model_meta.convert_hf2mcore(hf_model, mg_model)
    if args.test_convert_precision:
        test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)
    del hf_model
    logger.info('Successfully transferred HF model weights to MG model.')
    args.save_args()
    logger.info('Saving the model...')
    mg_save_checkpoint(1, [mg_model], None, None, 0)
    logger.info(f'Successfully saved Megatron model weights in `{args.output_dir}`.')


def convert_mcore2hf(args: ExportArguments) -> None:
    from swift.megatron import prepare_mcore_model, adapter_state_dict_context
    _, template = prepare_model_template(args, load_model=False)
    processor = template.processor

    megatron_model_meta = get_megatron_model_meta(args.model_type)
    assert megatron_model_meta is not None, f'Model: {args.model} is not supported.'
    kwargs = megatron_model_meta.convert_hf_config(processor.model_info.config)
    logger.info(f'megatron_config: {kwargs}')
    _check_megatron_kwargs(kwargs)
    current_convert_kwargs = convert_kwargs.copy()
    if args.model_info.is_moe_model:
        current_convert_kwargs['moe_grouped_gemm'] = True
    adapter_load = args.mcore_adapters[0] if args.mcore_adapters else None
    extra_config = MegatronArguments.load_args_config(adapter_load or args.mcore_model)
    extra_config['adapter_load'] = adapter_load
    if args.mcore_model is not None:
        extra_config['load'] = args.mcore_model
    kwargs.update(extra_config)
    megatron_args = MegatronArguments(
        **kwargs,
        **current_convert_kwargs,
        save=args.output_dir if args.to_mcore else None,
        torch_dtype=args.torch_dtype)
    patch_megatron_tokenizer(processor)
    extra_args = megatron_args.parse_to_megatron()
    extra_args['model_info'] = args.model_info
    extra_args['model_meta'] = args.model_meta
    extra_args['megatron_model_meta'] = megatron_model_meta
    extra_args_provider = megatron_model_meta.extra_args_provider
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=extra_args)

    mg_model = megatron_model_meta.model_provider()
    if megatron_args.load is None:
        raise ValueError('Please specify `--mcore_model`.')
    load_checkpoint([mg_model], None, None, strict=True)
    if megatron_args.adapter_load is not None:
        peft_model = prepare_mcore_model(mg_model)
        with adapter_state_dict_context():
            load_checkpoint([mg_model], None, None, load_arg='adapter_load', strict=False)
        logger.info('Merge LoRA...')
        mg_model = peft_model.merge_and_unload()
    logger.info('Megatron model created successfully.')
    if args.to_hf:
        hf_model, template = prepare_model_template(args, patch_offload=not args.test_convert_precision)
        megatron_model_meta.convert_mcore2hf(hf_model, mg_model)
        if args.test_convert_precision:
            test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)
        del mg_model
        logger.info('Successfully transferred MG model weights to HF model.')
        ckpt_dir = megatron_args.load if megatron_args.adapter_load is None else megatron_args.adapter_load
        logger.info('Saving the model...')
        save_checkpoint(
            hf_model,
            processor,
            args.output_dir,
            safe_serialization=args.safe_serialization,
            model_dirs=[ckpt_dir, args.model_dir],
            max_shard_size=args.max_shard_size,
            additional_saved_files=hf_model.model_meta.additional_saved_files)
        logger.info(f'Successfully saved HF model weights in `{args.output_dir}`.')
    elif args.to_mcore:
        if args.thread_count is None:
            checkpoint_size = sum(get_n_params_grads(mg_model)[0]) * torch.finfo(args.torch_dtype).bits // 8e9
            args.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB
        patch_torch_dist_shard(args.thread_count)

        args.save_args()
        logger.info('Saving the model...')
        mg_save_checkpoint(1, [mg_model], None, None, 0)
        logger.info(f'Successfully saved Megatron model weights in `{args.output_dir}`.')
