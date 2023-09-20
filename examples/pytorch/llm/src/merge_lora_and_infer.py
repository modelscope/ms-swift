# Copyright (c) Alibaba, Inc. and its affiliates.
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from llm_infer import llm_infer
from transformers import BitsAndBytesConfig, GenerationConfig, TextStreamer
from utils import (InferArguments, get_dataset, get_model_tokenizer,
                   get_preprocess)

from swift import Swift, get_logger
from swift.tuners import LoRA
from swift.utils import inference, parse_args, seed_everything

logger = get_logger()


def merge_lora(args: InferArguments) -> None:
    assert args.sft_type == 'lora'
    args.init_argument()
    logger.info(f'device_count: {torch.cuda.device_count()}')

    # ### Loading Model and Tokenizer
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, device_map='cpu')

    # ### Preparing LoRA
    model = Swift.from_pretrained(model, args.ckpt_dir, inference_mode=True)
    if not hasattr(model, 'peft_type'):
        LoRA.unpatch_lora(model, model.adapters['default'].config, 'default')
    else:
        model.merge_and_unload()

    new_ckpt_dir = os.path.abspath(
        os.path.join(args.ckpt_dir, '..', 'output_ckpt'))
    logger.info(f'new_ckpt_dir: `{new_ckpt_dir}`')
    logger.info("Setting args.sft_type: 'full'")
    logger.info(f'Setting args.ckpt_dir: {new_ckpt_dir}')
    args.ckpt_dir = new_ckpt_dir
    args.sft_type = 'full'
    if not os.path.exists(args.ckpt_dir):
        model.model.save_pretrained(args.ckpt_dir)
        tokenizer.save_pretrained(args.ckpt_dir)


if __name__ == '__main__':
    args, remaining_argv = parse_args(InferArguments)
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    merge_lora(args)
    llm_infer(args)
