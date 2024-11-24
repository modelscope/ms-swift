# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import json
import torch

from swift.llm import TrainArguments
from swift.plugin import Tuner, extra_tuners
from swift.tuners import Swift
from swift.utils import get_logger, use_torchacc

logger = get_logger()


def apply_liger(model_type: str):
    from liger_kernel.transformers import (apply_liger_kernel_to_llama, apply_liger_kernel_to_mistral,
                                           apply_liger_kernel_to_mixtral, apply_liger_kernel_to_gemma,
                                           apply_liger_kernel_to_qwen2)
    if 'llama3' in model_type:
        apply_liger_kernel_to_llama()
    elif 'mistral' in model_type:
        apply_liger_kernel_to_mistral()
    elif 'mixtral' in model_type:
        apply_liger_kernel_to_mixtral()
    elif 'gemma' in model_type:
        apply_liger_kernel_to_gemma()
    elif 'qwen2' in model_type:
        apply_liger_kernel_to_qwen2()
    else:
        raise ValueError(f'Unsupported liger model_type: {model_type}')


def torchacc_resume_from_checkpoint(args, model):
    import safetensors
    weights_file = os.path.join(args.resume_from_checkpoint, 'pytorch_model.bin')
    safe_weights_file = os.path.join(args.resume_from_checkpoint, 'model.safetensors')
    if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
        if args.save_safetensors and os.path.isfile(safe_weights_file):
            state_dict = safetensors.torch.load_file(safe_weights_file, device='cpu')
        else:
            state_dict = torch.load(weights_file, map_location='cpu')
        model.load_state_dict(state_dict, False)
        del state_dict
    else:
        from transformers.modeling_utils import load_sharded_checkpoint
        # We load the sharded checkpoint
        load_result = load_sharded_checkpoint(
            model, args.resume_from_checkpoint, strict=False, prefer_safe=args.save_safetensors)
        if len(load_result.missing_keys) != 0:
            if model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                    model._keys_to_ignore_on_save):
                model.tie_weights()
            else:
                logger.warning(f'There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.')
        if len(load_result.unexpected_keys) != 0:
            logger.warning(f'There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.')


def prepare_model(model, args: TrainArguments):
    if args.use_liger:
        # Apply liger
        apply_liger(args.model_type)

    if args.is_adapter:
        model.requires_grad_(False)
        if args.resume_from_checkpoint is None:
            model = args.prepare_adapter(model)
        else:
            if getattr(model, 'is_tuner_plugin', False):
                with open(os.path.join(args.resume_from_checkpoint, 'args.json'), 'r') as f:
                    content = json.load(f)

                tuner: Tuner = extra_tuners[content['sft_type']]
                model = tuner.from_pretrained(model, args.resume_from_checkpoint)
            elif use_torchacc():
                model = Swift.from_pretrained(
                    model, args.resume_from_checkpoint, adapter_name='default', is_trainable=True)
            else:
                model = Swift.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        # fix bug: Attempting to unscale FP16 gradients.
        #   peft: https://github.com/huggingface/peft/issues/1249
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                logger.info_once('Convert trainable parameters from fp16 to fp32.')
                p.data = p.data.to(dtype=torch.float32)
    elif args.train_type == 'full':
        model.requires_grad_(True)

        if use_torchacc() and args.resume_from_checkpoint is not None:
            torchacc_resume_from_checkpoint(args, model)
    elif args.train_type in extra_tuners:
        tuner: Tuner = extra_tuners[args.train_type]
        model = tuner.prepare_model(model, args)
        model.is_tuner_plugin = True
    else:
        raise ValueError(f'args.train_type: {args.train_type}')

    if args.sequence_parallel_size > 1:
        from swift.trainers.xtuner import dispatch_module_xtuner
        dispatch_module_xtuner(model)

    return model
