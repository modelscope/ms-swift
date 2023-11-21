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
from diffusers import AutoencoderKL, DDIMScheduler, UNetMotionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines import AnimateDiffPipeline
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from modelscope import read_config, snapshot_download
from torch.utils.data import RandomSampler
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from swift import LoRAConfig, Swift, get_logger
from .utils import AnimateDiffArguments

logger = get_logger()


class AnimateDiffDataset(Dataset):

    VIDEO_ID = 'videoid'
    NAME = 'name'
    CONTENT_URL = 'contentUrl'

    def __init__(
        self,
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
    ):
        print(f'loading annotations from {csv_path} ...')
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        dataset = []
        for d in self.dataset:
            file_name = d[self.CONTENT_URL]
            if os.path.isfile(os.path.join(video_folder, file_name)):
                dataset.append(d)
        self.dataset = dataset
        self.length = len(self.dataset)
        print(f'data scale: {self.length}')

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        sample_size = tuple(sample_size) if not isinstance(
            sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def get_batch(self, idx):
        video_dict: Dict[str, str] = self.dataset[idx]
        name = video_dict[self.NAME]

        file_name = video_dict[self.CONTENT_URL]
        video_dir = os.path.join(self.video_folder, file_name)
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        clip_length = min(video_length,
                          (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx,
            start_idx + clip_length - 1,
            self.sample_n_frames,
            dtype=int)

        pixel_values = torch.from_numpy(
            video_reader.get_batch(batch_index).asnumpy()).permute(
                0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                logger.error(f'Error loading dataset batch: {e}')
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample


def save_videos_grid(videos: torch.Tensor,
                     path: str,
                     rescale=False,
                     n_rows=6,
                     fps=8):
    videos = rearrange(videos, 'b c t h w -> t b c h w')
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def animatediff_sft(args: AnimateDiffArguments) -> None:
    # Initialize distributed training
    local_rank = 0  # init_dist(launcher=launcher)
    global_rank = 0  # dist.get_rank()
    num_processes = 1  # dist.get_world_size()
    is_main_process = global_rank == 0

    seed = args.seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = datetime.datetime.now().strftime('-%Y-%m-%dT%H-%M-%S')
    output_dir = os.path.join(args.output_dir, folder_name)

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/samples', exist_ok=True)
        os.makedirs(f'{output_dir}/sanity_check', exist_ok=True)
        os.makedirs(f'{output_dir}/checkpoints', exist_ok=True)

    with open(args.validation_prompts_path, 'r') as f:
        validation_data = f.readlines()

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(
        **read_config(args.noise_scheduler_additional_config_path))
    pretrained_model_path = snapshot_download(args.model_id_or_path)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder='vae')
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder='text_encoder')
    unet = UNetMotionModel.from_pretrained(
        pretrained_model_path,
        subfolder='unet',
        unet_additional_kwargs=read_config(args.unet_additional_config_path),
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

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Set unet trainable parameters
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # Preparing LoRA
    if args.sft_type == 'lora':
        if args.resume_from_checkpoint is None:
            if args.sft_type == 'lora':
                lora_config = LoRAConfig(
                    r=args.lora_rank,
                    target_modules=args.lora_target_modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout_p)
                unet = Swift.prepare_model(unet, lora_config)
                logger.info(f'lora_config: {lora_config}')
        else:
            unet = Swift.from_pretrained(
                unet, args.resume_from_checkpoint, is_trainable=True)

    trainable_params = list(
        filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
    )

    if is_main_process:
        print(f'trainable params number: {len(trainable_params)}')
        print(
            f'trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M'
        )

    # Enable xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                'xformers is not available. Make sure it is installed correctly'
            )

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)

    # Get the training dataset
    train_dataset = AnimateDiffDataset(
        csv_path=args.csv_path,
        video_folder=args.video_folder,
        sample_size=args.sample_size,
        sample_stride=args.sample_stride,
        sample_n_frames=args.sample_n_frames,
    )
    sampler = RandomSampler(train_dataset)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    max_train_steps = args.num_train_epochs * len(train_dataloader)
    checkpointing_steps = args.save_steps

    # Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * max_train_steps
                             * args.gradient_accumulation_steps),
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    unet.to(local_rank)
    # unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.batch_size * num_processes * args.gradient_accumulation_steps

    if is_main_process:
        logging.info('***** Running training *****')
        logging.info(f'  Num examples = {len(train_dataset)}')
        logging.info(f'  Num Epochs = {num_train_epochs}')
        logging.info(
            f'  Instantaneous batch size per device = {args.batch_size}')
        logging.info(
            f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}'
        )
        logging.info(
            f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}'
        )
        logging.info(f'  Total optimization steps = {max_train_steps}')
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description('Steps')

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    for epoch in range(first_epoch, num_train_epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        for step, batch in enumerate(train_dataloader):
            if args.text_dropout_rate > 0:
                batch['text'] = [
                    name if random.random() > args.text_dropout_rate else ''
                    for name in batch['text']
                ]

            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(
                ), batch['text']
                pixel_values = rearrange(pixel_values,
                                         'b f c h w -> b c f h w')
                for idx, (pixel_value,
                          text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    file_name = '-'.join(text.replace('/', '').split(
                    )[:10]) if not text == '' else f'{global_rank}-{idx}'
                    save_videos_grid(
                        pixel_value,
                        f'{output_dir}/sanity_check/{file_name}.gif',
                        rescale=True)

            # Convert videos to latent space
            pixel_values = batch['pixel_values'].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values,
                                         'b f c h w -> (b f) c h w')
                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()
                latents = rearrange(
                    latents, '(b f) c h w -> b c f h w', f=video_length)
                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps, (bsz, ),
                device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise,
                                                      timesteps)

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'],
                    max_length=tokenizer.model_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt').input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == 'epsilon':
                target = noise
            elif noise_scheduler.config.prediction_type == 'v_prediction':
                raise NotImplementedError
            else:
                raise ValueError(
                    f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                )

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction='mean')

            optimizer.zero_grad()

            # Backpropagate
            if args.mixed_precision:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(),
                                               args.max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(),
                                               args.max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            # Save checkpoint
            if is_main_process and (global_step % args.save_steps == 0
                                    or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, 'checkpoints')
                state_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'state_dict': unet.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    torch.save(
                        state_dict,
                        os.path.join(save_path,
                                     f'checkpoint-epoch-{epoch + 1}.ckpt'))
                else:
                    torch.save(state_dict,
                               os.path.join(save_path, 'checkpoint.ckpt'))
                logging.info(
                    f'Saved state to {save_path} (global_step: {global_step})')

            # Periodically validation
            if is_main_process and global_step % args.eval_steps == 0:
                samples = []

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(seed)

                height = args.sample_size
                width = args.sample_size
                motion_adapter = MotionAdapter()
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
                ).to('cuda')
                validation_pipeline.enable_vae_slicing()

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

            logs = {
                'step_loss': loss.detach().item(),
                'lr': lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # dist.destroy_process_group()
