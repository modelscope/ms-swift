import os

import cv2
from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile

from swift import LoRAConfig, SCETuningConfig, Swift, snapshot_download

# load dataset
train_dataset = MsDataset.load(
    'style_custom_dataset', namespace='damo', subset_name='3D',
    split='train_short').remap_columns({'Image:FILE': 'Target:FILE'})

# load pretrained model
model_id = 'damo/multi-modal_efficient-diffusion-tuning-swift-base'
task = 'efficient-diffusion-tuning'
revision = 'v1.0.1'
model_dir = snapshot_download(model_id)
cfg_dict = Config.from_file(os.path.join(model_dir, ModelFile.CONFIGURATION))
cfg_dict.model.inference = False
model = Model.from_pretrained(model_id, cfg_dict=cfg_dict, revision=revision)

# init tuner
tuner_type = 'scetuning'  # "lora"

if tuner_type == 'lora':
    work_dir = 'tmp/multimodal_swift_lora_style'
    tuner_config = LoRAConfig(r=64, target_modules='.*unet.*.(to_q|to_k|to_v|to_out.0|net.0.proj|net.2)$')
    model = Swift.prepare_model(model, tuner_config)
elif tuner_type == 'scetuning':
    work_dir = 'tmp/multimodal_swift_scetuning_style'
    tuner_config = SCETuningConfig(
        dims=[1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320, 320],
        target_modules=r'^unet\.up_blocks\.\d+\.resnets\.\d+$')
    model = Swift.prepare_model(model, tuner_config)
else:
    raise Exception(f'Error tuner type: {tuner_type}')


# training
def cfg_modify_fn(cfg):
    cfg.preprocessor.resolution = 512
    cfg.train.lr_scheduler = {'type': 'LambdaLR', 'lr_lambda': lambda _: 1, 'last_epoch': -1}
    cfg.train.max_epochs = 100
    cfg.train.optimizer.lr = 1e-4
    cfg.train.dataloader.batch_size_per_gpu = 10
    cfg.model.inference = False
    cfg.model.pretrained_tuner = None
    trainer_hook = cfg.train.hooks
    trainer_hook.append({'type': 'SwiftHook'})
    trainer_hook.append({'type': 'CheckpointHook', 'interval': 50})
    cfg.train.hooks = trainer_hook
    return cfg


kwargs = dict(
    model=model,
    cfg_file=os.path.join(model_dir, 'configuration.json'),
    work_dir=work_dir,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(name='efficient-diffusion-tuning', default_args=kwargs)
trainer.train()

# inference
work_dir = os.path.join(work_dir, 'output_swift')
model_dir = snapshot_download(model_id)
cfg_dict = Config.from_file(os.path.join(model_dir, ModelFile.CONFIGURATION))
cfg_dict.model.inference = True
model = Model.from_pretrained(model_id, cfg_dict=cfg_dict, revision=revision)
model = Swift.from_pretrained(model, work_dir)
pipe = pipeline(task='efficient-diffusion-tuning', model=model)
test_prompt = 'A boy in a camouflage jacket with a scarf'
img_out = pipe({'prompt': test_prompt}, num_inference_steps=50, generator_seed=123)['output_imgs'][0]
cv2.imwrite(os.path.join(work_dir, 'inference.png'), img_out)
