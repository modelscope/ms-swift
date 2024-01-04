# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import torch
from diffusers import DDIMScheduler, MotionAdapter
from diffusers.pipelines import AnimateDiffPipeline
from diffusers.utils import export_to_gif

from swift import Swift, snapshot_download
from swift.aigc.utils import AnimateDiffInferArguments
from swift.utils import get_logger, get_main

logger = get_logger()


def animatediff_infer(args: AnimateDiffInferArguments) -> None:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(args.seed)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        steps_offset=args.steps_offset,
        clip_sample=args.clip_sample,
    )

    if not os.path.exists(args.model_id_or_path):
        pretrained_model_path = snapshot_download(
            args.model_id_or_path, revision=args.model_revision)
    else:
        pretrained_model_path = args.model_id_or_path

    motion_adapter = None
    if args.motion_adapter_id_or_path is not None:
        if not os.path.exists(args.motion_adapter_id_or_path):
            args.motion_adapter_id_or_path = snapshot_download(
                args.motion_adapter_id_or_path,
                revision=args.motion_adapter_revision)
        motion_adapter = MotionAdapter.from_pretrained(
            args.motion_adapter_id_or_path)
    if args.sft_type == 'full':
        motion_adapter = MotionAdapter.from_pretrained(args.ckpt_dir)

    validation_pipeline = AnimateDiffPipeline.from_pretrained(
        pretrained_model_path,
        motion_adapter=motion_adapter,
    ).to('cuda')
    validation_pipeline.scheduler = noise_scheduler

    if not args.sft_type == 'full':
        Swift.from_pretrained(validation_pipeline.unet, args.ckpt_dir)

    validation_pipeline.enable_vae_slicing()
    validation_pipeline.enable_model_cpu_offload()

    if args.eval_human:
        idx = 0
        while True:
            prompt = input('<<< ')
            sample = validation_pipeline(
                prompt,
                negative_prompt='bad quality, worse quality',
                generator=generator,
                num_frames=args.sample_n_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).frames[0]
            os.makedirs(args.output_path, exist_ok=True)
            logger.info(
                f'Output saved to: {f"{args.output_path}/output-{idx}.gif"}')
            export_to_gif(sample, f'{args.output_path}/output-{idx}.gif')
            idx += 1
    else:
        with open(args.validation_prompts_path, 'r') as f:
            validation_data = f.readlines()

        for idx, prompt in enumerate(validation_data):
            sample = validation_pipeline(
                prompt,
                negative_prompt='bad quality, worse quality',
                generator=generator,
                num_frames=args.sample_n_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).frames[0]
            os.makedirs(args.output_path, exist_ok=True)
            logger.info(
                f'Output saved to: {f"{args.output_path}/output-{idx}.gif"}')
            export_to_gif(sample, f'{args.output_path}/output-{idx}.gif')


animatediff_infer_main = get_main(AnimateDiffInferArguments, animatediff_infer)
