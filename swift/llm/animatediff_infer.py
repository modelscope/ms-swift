# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Tuple
import csv
import datetime
import inspect
import logging
import math
import os
import random
from typing import Dict
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from decord import VideoReader
from diffusers import (AutoencoderKL, DDIMScheduler, MotionAdapter,
                       UNet2DConditionModel, UNetMotionModel)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import AnimateDiffPipeline
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from modelscope import read_config, snapshot_download
from torch.utils.data import RandomSampler
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import requests
from swift import LoRAConfig, Swift, get_logger
from .utils import AnimateDiffArguments
import json
import torch
from transformers import PreTrainedModel
from swift.utils import (get_logger, print_model_info, seed_everything,
                         show_layers)
from .utils import (AnimateDiffInferArguments)

logger = get_logger()

def llm_infer(args: AnimateDiffInferArguments) -> None:
    samples = []

    generator = torch.Generator(device=latents.device)
    generator.manual_seed(seed)
    unet = UNetMotionModel.from_pretrained(
            args.model_id_or_path,
        )

    height = args.sample_size
    width = args.sample_size
    validation_pipeline = AnimateDiffPipeline(
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,
        motion_adapter=motion_adapter,
        text_encoder=text_encoder,
        scheduler=noise_scheduler,
    ).to('cuda')
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
        os.makedirs(
            f'{output_dir}/samples/sample-{global_step}',
            exist_ok=True)
        sample[0].save(
            f'{output_dir}/samples/sample-{global_step}/{idx}.gif',
            save_all=True,
            append_images=sample[1:],
            loop=0,
            duration=500)
        samples.append(sample)