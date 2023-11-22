# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import torch
from diffusers import (AutoencoderKL, DDIMScheduler, MotionAdapter,
                       UNetMotionModel)
from diffusers.pipelines import AnimateDiffPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from swift import snapshot_download
from swift.utils import get_logger
from .utils import AnimateDiffInferArguments

logger = get_logger()


def animatediff_infer(args: AnimateDiffInferArguments) -> None:
    generator = torch.Generator(device='cuda:0')
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

    pretrained_model_path = snapshot_download(args.base_model_id_or_path)
    unet = UNetMotionModel.from_pretrained(
        pretrained_model_path,
        subfolder='unet',
        _class_name=UNetMotionModel.__name__,
        down_block_types=[
            'CrossAttnDownBlockMotion', 'CrossAttnDownBlockMotion',
            'CrossAttnDownBlockMotion', 'DownBlockMotion'
        ],
        up_block_types=[
            'UpBlockMotion', 'CrossAttnUpBlockMotion',
            'CrossAttnUpBlockMotion', 'CrossAttnUpBlockMotion'
        ],
        low_cpu_mem_usage=False,
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder='vae')
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder='text_encoder')
    unet.load_state_dict(torch.load(os.path.join(args.model_id_or_path, 'checkpoint.ckpt'))['state_dict'])

    height = args.sample_size
    width = args.sample_size
    motion_adapter = MotionAdapter(
        motion_num_attention_heads=args.motion_num_attention_heads,
        motion_max_seq_length=args.motion_max_seq_length)
    motion_adapter.mid_block.motion_modules = unet.mid_block.motion_modules
    for db1, db2 in zip(motion_adapter.down_blocks, unet.down_blocks):
        db1.motion_modules = db2.motion_modules
    for db1, db2 in zip(motion_adapter.up_blocks, unet.up_blocks):
        db1.motion_modules = db2.motion_modules
    validation_pipeline = AnimateDiffPipeline(
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,
        motion_adapter=motion_adapter,
        text_encoder=text_encoder,
        scheduler=noise_scheduler,
    ).to('cuda:0')
    validation_pipeline.enable_vae_slicing()

    with open(args.validation_prompts_path, 'r') as f:
        validation_data = f.readlines()

    for idx, prompt in enumerate(validation_data):
        sample = validation_pipeline(
            prompt,
            generator=generator,
            num_frames=args.sample_n_frames,
            height=height,
            width=width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).frames[0]
        os.makedirs(args.output_path, exist_ok=True)
        sample[0].save(
            f'{args.output_path}/{idx}.gif',
            save_all=True,
            append_images=sample[1:],
            loop=0,
            duration=500)
