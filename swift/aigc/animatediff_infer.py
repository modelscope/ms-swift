# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Optional

import torch
from diffusers import DDIMScheduler, MotionAdapter
from diffusers.pipelines import AnimateDiffPipeline
from diffusers.utils import export_to_gif

from swift import Swift, snapshot_download
from swift.aigc.utils import AnimateDiffInferArguments
from swift.llm import merge_lora
from swift.utils import get_logger, get_main

logger = get_logger()


def merge_lora(args: AnimateDiffInferArguments,
               replace_if_exists=False,
               device_map: str = 'auto',
               **kwargs) -> Optional[str]:
    logger.info(f'replace_if_exists: {replace_if_exists}')
    assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
    assert args.sft_type == 'lora', "Only supports sft_type == 'lora'"
    old_ckpt_dir = args.ckpt_dir
    ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
    merged_lora_path = os.path.join(ckpt_dir, f'{ckpt_name}-merged')
    logger.info(f'merged_lora_path: `{merged_lora_path}`')
    logger.info("Setting args.sft_type: 'full'")
    logger.info(f'Setting args.ckpt_dir: {merged_lora_path}')
    args.sft_type = 'full'
    args.ckpt_dir = merged_lora_path
    if os.path.exists(args.ckpt_dir) and not replace_if_exists:
        logger.info(
            f'The weight directory for the merged LoRA already exists in {args.ckpt_dir}, '
            'skipping the saving process. '
            'you can pass `replace_if_exists=True` to overwrite it.')
        return
    # Loading Model and Tokenizer
    kwargs = {}

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

    if not args.sft_type == 'full':
        Swift.from_pretrained(validation_pipeline.unet, args.ckpt_dir)

    # Preparing LoRA
    model = Swift.from_pretrained(model, old_ckpt_dir, inference_mode=True)
    Swift.merge_and_unload(model)
    model = model.model
    logger.info('Saving merged weights...')
    model.save_pretrained(
        merged_lora_path, safe_serialization=args.save_safetensors)
    for add_file in get_additional_saved_files(args.model_type):
        shutil.copy(
            os.path.join(model.model_dir, add_file),
            os.path.join(merged_lora_path, add_file))
    tokenizer.save_pretrained(merged_lora_path)
    for fname in os.listdir(old_ckpt_dir):
        if fname in {'generation_config.json'}:
            src_path = os.path.join(old_ckpt_dir, fname)
            tgt_path = os.path.join(merged_lora_path, fname)
            shutil.copy(src_path, tgt_path)
    # configuration.json
    configuration_fname = 'configuration.json'
    old_configuration_path = os.path.join(old_ckpt_dir, configuration_fname)
    new_configuration_path = os.path.join(merged_lora_path,
                                          configuration_fname)
    if os.path.exists(old_configuration_path):
        with open(old_configuration_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        res.pop('adapter_cfg', None)
        with open(new_configuration_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
    # sft_args.json
    sft_args_fname = 'sft_args.json'
    old_sft_args_path = os.path.join(old_ckpt_dir, sft_args_fname)
    new_sft_args_path = os.path.join(merged_lora_path, sft_args_fname)
    if os.path.exists(old_sft_args_path):
        with open(old_sft_args_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        res['sft_type'] = 'full'
        with open(new_sft_args_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    logger.info(f'Successfully merged LoRA and saved in {merged_lora_path}.')
    return merged_lora_path


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
        motion_adapter_dir = args.ckpt_dir if args.ckpt_dir is not None else os.path.join(pretrained_model_path, 'motion_adapter')
        motion_adapter = MotionAdapter.from_pretrained(motion_adapter_dir)

    validation_pipeline = AnimateDiffPipeline.from_pretrained(
        pretrained_model_path,
        motion_adapter=motion_adapter,
    ).to('cuda')
    validation_pipeline.scheduler = noise_scheduler

    if not args.sft_type == 'full':
        model = Swift.from_pretrained(validation_pipeline.unet, args.ckpt_dir)
        if args.merge_lora_and_save:
            ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
            merged_lora_path = os.path.join(ckpt_dir, f'{ckpt_name}-merged')
            logger.info(f'merged_lora_path: `{merged_lora_path}`')
            logger.info("Setting args.sft_type: 'full'")
            logger.info(f'Setting args.ckpt_dir: {merged_lora_path}')
            args.sft_type = 'full'
            args.ckpt_dir = merged_lora_path
            if os.path.exists(args.ckpt_dir) and not args.replace_if_exists:
                logger.warn(
                    f'The weight directory for the merged LoRA already exists in {args.ckpt_dir}, '
                    'skipping the saving process. '
                    'you can pass `replace_if_exists=True` to overwrite it.')
                return

            Swift.merge_and_unload(model)
            validation_pipeline.unet = model.model
            validation_pipeline.save_pretrained(args.ckpt_dir)

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
