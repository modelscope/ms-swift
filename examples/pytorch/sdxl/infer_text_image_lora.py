from diffusers import StableDiffusionPipeline
import torch
from swift import Swift
from modelscope import snapshot_download


model_path = snapshot_download("AI-ModelScope/stable-diffusion-v1-5")
lora_model_path = "/mnt/workspace/swift/examples/pytorch/sdxl/train_text_to_image_lora"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.unet = Swift.from_pretrained(pipe.unet, lora_model_path)
pipe.to("cuda")

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("sw_sd_lora_pokemon.png")
