# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os

import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from modelscope import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description='Simple example of a text to image inference.')
    parser.add_argument(
        '--pretrained_model_name_or_path',
        type=str,
        default='AI-ModelScope/stable-diffusion-v1-5',
        required=True,
        help='Path to pretrained model or model identifier from modelscope.cn/models.',
    )
    parser.add_argument(
        '--revision',
        type=str,
        default=None,
        required=False,
        help='Revision of pretrained model identifier from modelscope.cn/models.',
    )
    parser.add_argument(
        '--unet_model_path',
        type=str,
        default=None,
        required=False,
        help='The path to trained unet model.',
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
        default=30,
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

    if os.path.exists(args.pretrained_model_name_or_path):
        model_path = args.pretrained_model_name_or_path
    else:
        model_path = snapshot_download(args.pretrained_model_name_or_path, revision=args.revision)

    if args.torch_dtype == 'fp16':
        torch_dtype = torch.float16
    elif args.torch_dtype == 'bf16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    if args.unet_model_path is not None:
        pipe.unet = UNet2DConditionModel.from_pretrained(args.unet_model_path, torch_dtype=torch_dtype)
    pipe.to('cuda')
    image = pipe(
        prompt=args.prompt, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale).images[0]
    image.save(args.image_save_path)
