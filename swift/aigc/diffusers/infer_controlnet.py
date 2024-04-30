# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
from modelscope import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description='Simple example of a ControlNet inference.')
    parser.add_argument(
        '--base_model_path',
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
        '--controlnet_path',
        type=str,
        default=None,
        required=False,
        help='The path to trained controlnet model.',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        required=True,
        help='The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`',
    )
    parser.add_argument(
        '--control_image_path',
        type=str,
        default=None,
        required=True,
        help='The path to conditioning image.',
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
    parser.add_argument('--seed', type=int, default=None, help='A seed for inference.')
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=20,
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

    if os.path.exists(args.base_model_path):
        base_model_path = args.base_model_path
    else:
        base_model_path = snapshot_download(args.base_model_path, revision=args.revision)

    if args.torch_dtype == 'fp16':
        torch_dtype = torch.float16
    elif args.torch_dtype == 'bf16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch_dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch_dtype)

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # memory optimization.
    pipe.enable_model_cpu_offload()

    control_image = load_image(args.control_image_path)

    # generate image
    generator = torch.manual_seed(args.seed)
    image = pipe(
        args.prompt, num_inference_steps=args.num_inference_steps, generator=generator, image=control_image).images[0]
    image.save(args.image_save_path)
