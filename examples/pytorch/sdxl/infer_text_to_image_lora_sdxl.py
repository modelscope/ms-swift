from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import torch
from swift import Swift
import os
from modelscope import snapshot_download

model_path = snapshot_download("AI-ModelScope/stable-diffusion-v1-5")
lora_model_path = "/mnt/workspace/swift_trans_test/examples/pytorch/sdxl/train_text_to_image_lora_sdxl"

pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.unet = Swift.from_pretrained(pipe.unet, os.path.join(lora_model_path, 'unet'))
prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("sw_sdxl_lora_pokemon.png")
