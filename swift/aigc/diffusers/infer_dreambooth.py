# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os

import torch
from diffusers import StableDiffusionPipeline
from modelscope import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description='Simple example of a dreambooth inference.')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        required=True,
        help='Path to trained model.',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        required=True,
        help='The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`',
    )
    parser.add_argument(
        '--image_save_path',
        type=str,
        default=None,
        required=True,
        help='The path to save generated image',
    )
    parser.add_argument(
        '--torch_dtype',
        type=str,
        default=None,
        choices=['no', 'fp16', 'bf16'],
        help=('Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >='
              ' 1.10.and an Nvidia Ampere GPU.  Default to the value of the'
              ' mixed_precision passed with the `accelerate.launch` command in training script.'),
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=50,
        help=('The number of denoising steps. More denoising steps usually lead to a higher quality image at the \
                expense of slower inference.'),
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=7.5,
        help=('A higher guidance scale value encourages the model to generate images closely linked to the text \
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.'),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.torch_dtype == 'fp16':
        torch_dtype = torch.float16
    elif args.torch_dtype == 'bf16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to('cuda')

    image = pipe(
        args.prompt, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale).images[0]

    image.save(args.image_save_path)
