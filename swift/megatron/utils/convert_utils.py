# Copyright (c) ModelScope Contributors. All rights reserved.

import math
from contextlib import contextmanager
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import mpu
from megatron.training import get_args

from swift.utils import HfConfigFactory, get_logger, to_device, to_float_dtype
from .utils import forward_step_helper, get_padding_to

logger = get_logger()


def _test_params_sum(model):
    total_sum = 0
    zero_count = 0
    n_parameter = 0
    for n, p in model.named_parameters():
        n_parameter += 1
        sum_ = p.to(device='cuda', dtype=torch.float32).abs().sum().cpu().item()
        if sum_ == 0 and '.lora_B.' not in n:
            zero_count += 1
            logger.warning(f'n: {n}, sum: {sum_}')
        elif math.isnan(sum_) or math.isinf(sum_) or sum_ > 1e10:
            logger.warning(f'n: {n}, sum: {sum_}')
        else:
            total_sum += sum_
    cond = mpu.get_data_parallel_rank() == 0
    logger.info_if(f'n_parameter: {n_parameter}', cond=cond)
    logger.info_if(f'total_sum: {total_sum}', cond=cond)
    logger.info_if(f'zero_count: {zero_count}', cond=cond)


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
def _model_cpu_forward_context(modules,
                               torch_dtype=None,
                               compute_device=None,
                               share_embedding: bool = False,
                               target_device='cpu'):
    try:
        origin_torch_dtype = next(modules[0].parameters()).dtype
    except StopIteration:
        origin_torch_dtype = next(modules[-1].parameters()).dtype

    def _to_cuda_hook(module, args):
        if compute_device is not None or torch_dtype is not None:
            module.to(device=compute_device, dtype=torch_dtype)
            args = to_float_dtype(args, dtype=torch_dtype)
        return args

    def _to_cpu_hook(module, args, output):
        if share_embedding and module is modules[0]:
            return
        module.to(device=target_device, dtype=origin_torch_dtype)

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
    template.set_mode('train')
    _test_params_sum(mg_model)

    is_multimodal = template.model_meta.is_multimodal
    mg_language_model = mg_model.language_model if is_multimodal else mg_model
    if mg_language_model.config.fp8 is not None:
        raise ValueError('fp8 models currently do not support testing convert_precision. '
                         'Please set `--test_convert_precision false`.')
    share_embedding = mg_language_model.share_embeddings_and_output_weights
    if hf_model is not None:
        hf_model.eval()
        if dist.get_world_size() == 1:
            _test_params_sum(hf_model)
        inputs = template.encode(get_examples(is_multimodal))
        hf_inputs = to_device(template.data_collator([inputs]), 'cuda')
        template.register_post_encode_hook([hf_model])
        HfConfigFactory.set_model_config_attr(hf_model, 'use_cache', False)
        model_arch = hf_model.model_meta.model_arch
        ignore_modules = (model_arch.vision_tower + model_arch.aligner) if is_multimodal else []
        hf_modules = _find_modules(hf_model, ignore_modules=ignore_modules)
        with torch.inference_mode(), _model_cpu_forward_context(
                hf_modules, torch_dtype, share_embedding=share_embedding):
            hf_inputs.pop('text_position_ids', None)
            hf_logits = hf_model(**hf_inputs).logits
            hf_logits = hf_logits.to('cuda')
        hf_model.to('cpu')

    args = get_args()
    template.use_megatron = True
    inputs = template.encode(get_examples(is_multimodal))
    mg_inputs = to_device(template.data_collator([inputs], padding_to=get_padding_to(args)), 'cuda')
    packed_seq_params = None
    mg_model.eval()
    # thd
    # from ..trainers.utils import get_packed_seq_params
    # packed_seq_params = get_packed_seq_params(position_ids)
    # attention_mask = None
    mg_language_model.config.fp8 = None  # compat fp8
    mg_modules = _find_modules(mg_language_model, ignore_modules=['visual'])
    for key in ['labels', 'num_samples', 'attention_mask_2d', 'text_position_ids']:
        mg_inputs.pop(key, None)
    mg_inputs.update({'packed_seq_params': packed_seq_params})
    _param = next(mg_language_model.parameters())
    mg_dtype = _param.dtype
    mg_device = _param.device
    if args.hf_model_type == 'minimax_m2':
        # router to bfloat16
        for n, m in mg_language_model.named_modules():
            if n.endswith('router'):
                m.to(mg_dtype)
    with torch.inference_mode(), _model_cpu_forward_context(
            mg_modules, torch_dtype, 'cuda', share_embedding=share_embedding, target_device=mg_device):
        mg_logits = forward_step_helper(mg_model, mg_inputs, dtype=torch_dtype)
        if args.tensor_model_parallel_size > 1 and args.task_type != 'seq_cls':
            from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
            if mg_logits is not None:
                mg_logits = gather_from_tensor_model_parallel_region(mg_logits)
    if hf_model is None:
        return
    if args.task_type == 'seq_cls':
        mg_logits = mg_logits[:, -1]
        mean_diff = (mg_logits - hf_logits).abs().mean().item()
        max_diff = (mg_logits - hf_logits).abs().max().item()
        print(f'mean_diff: {mean_diff}, max_diff: {max_diff}')
    else:
        mg_logits = mg_logits[:, :hf_logits.shape[1]]
        token_mean_diff = (mg_logits - hf_logits).abs().mean(dim=-1)
        mean_diff = token_mean_diff.mean().item()
        max_diff = (mg_logits - hf_logits).abs().max().item()
        loss_mask = (torch.roll(hf_inputs['labels'], -1) != -100)
        mean_diff_with_loss = token_mean_diff[loss_mask].mean().item()
        max_diff_with_loss = (mg_logits - hf_logits)[loss_mask].abs().max().item()
        print(f'token_mean_diff: {token_mean_diff}')
        print(f'mean_diff: {mean_diff}, max_diff: {max_diff}')
        print(f'mean_diff (with loss): {mean_diff_with_loss}, max_diff (with loss): {max_diff_with_loss} '
              '(Please check that mean_diff (with loss) is less than 0.1).')
        hf_tokens = hf_logits.argmax(-1)
        mg_tokens = mg_logits.argmax(-1)
        print(f'hf_tokens: {hf_tokens[0].tolist()}\nmg_tokens: {mg_tokens[0].tolist()}')
        print(f'token_diff: {(hf_tokens != mg_tokens).sum().item()}')
        print(f'token_diff (with loss): {(hf_tokens[loss_mask] != mg_tokens[loss_mask]).sum().item()}')
