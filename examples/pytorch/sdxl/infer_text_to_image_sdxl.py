from diffusers import DiffusionPipeline
import torch

model_path = "/mnt/workspace/swift/examples/pytorch/sdxl/sdxl-pokemon-model"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("sdxl_pokemon.png")
