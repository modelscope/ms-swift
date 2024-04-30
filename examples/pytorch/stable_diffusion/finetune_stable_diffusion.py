import os
from dataclasses import dataclass, field

import cv2
import torch
from modelscope import get_logger, snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.trainers.training_args import TrainingArgs
from modelscope.utils.constant import DownloadMode, Tasks

from swift import LoRAConfig, Swift

logger = get_logger()


# Load configuration file and dataset
@dataclass(init=False)
class StableDiffusionLoraArguments(TrainingArgs):
    prompt: str = field(
        default='dog', metadata={
            'help': 'The pipeline prompt.',
        })

    lora_rank: int = field(
        default=4, metadata={
            'help': 'The rank size of lora intermediate linear.',
        })

    lora_alpha: int = field(
        default=32, metadata={
            'help': 'The factor to add the lora weights',
        })

    lora_dropout: float = field(
        default=0.0, metadata={
            'help': 'The dropout rate of the lora module',
        })

    bias: str = field(
        default='none', metadata={
            'help': 'Bias type. Values ca be "none", "all" or "lora_only"',
        })

    sample_nums: int = field(
        default=10, metadata={
            'help': 'The numbers of sample outputs',
        })

    num_inference_steps: int = field(
        default=50, metadata={
            'help': 'The number of denoising steps.',
        })


training_args = StableDiffusionLoraArguments(task='text-to-image-synthesis').parse_cli()
config, args = training_args.to_config()

if os.path.exists(args.train_dataset_name):
    # Load local dataset
    train_dataset = MsDataset.load(args.train_dataset_name)
    validation_dataset = MsDataset.load(args.train_dataset_name)
else:
    # Load online dataset
    train_dataset = MsDataset.load(args.train_dataset_name, split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD)
    validation_dataset = MsDataset.load(
        args.train_dataset_name, split='validation', download_mode=DownloadMode.FORCE_REDOWNLOAD)


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    cfg.train.lr_scheduler = {'type': 'LambdaLR', 'lr_lambda': lambda _: 1, 'last_epoch': -1}
    return cfg


# build models
model = Model.from_pretrained(training_args.model, revision=args.model_revision)
model_dir = snapshot_download(args.model)
lora_config = LoRAConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias=args.bias,
    target_modules=['to_q', 'to_k', 'to_v', 'query', 'key', 'value', 'to_out.0'])
model.unet = Swift.prepare_model(model.unet, lora_config)

# build trainer and training
kwargs = dict(
    model=model,
    cfg_file=os.path.join(model_dir, 'configuration.json'),
    work_dir=training_args.work_dir,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    torch_type=torch.float32,
    use_swift=True,
    cfg_modify_fn=cfg_modify_fn)

trainer = build_trainer(name=Trainers.stable_diffusion, default_args=kwargs)
trainer.train()

# save models
model.unet.save_pretrained(os.path.join(training_args.work_dir, 'unet'))
logger.info(f'model save pretrained {training_args.work_dir}')

# pipeline after training and save result
pipe = pipeline(
    task=Tasks.text_to_image_synthesis,
    model=training_args.model,
    model_revision=args.model_revision,
    lora_dir=os.path.join(training_args.work_dir, 'unet'),
    use_swift=True)

for index in range(args.sample_nums):
    image = pipe({'text': args.prompt, 'num_inference_steps': args.num_inference_steps})
    cv2.imwrite(f'./lora_result_{index}.png', image['output_imgs'][0])
