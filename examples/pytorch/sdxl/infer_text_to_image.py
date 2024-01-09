import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from modelscope import snapshot_download

model_path = snapshot_download("AI-ModelScope/stable-diffusion-v1-5")

unet_model_path = "/mnt/workspace/swift/examples/pytorch/sdxl/train_text_to_image/unet"
unet = UNet2DConditionModel.from_pretrained(unet_model_path, torch_dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained(model_path, unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="yoda").images[0]
image.save("sw_yoda-pokemon.png")
