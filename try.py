from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import transformers
import torch
import copy

print(transformers.__version__)

from safetensors.torch import load_file as safe_load_file
import safetensors.torch

for i in [1, 2, 3, 4, ]:
    tensors = safe_load_file(f'/mnt/workspace/.cache/modelscope/hub/swift/llava-v1___6-llama-3.1-my/model-0000{i}-of-00005.safetensors.old')

    new_tensors = copy.deepcopy(tensors)

    for key, value in tensors.items():
        # new_key = "vision_tower." + key
        new_key = "language_model." + key    

        del new_tensors[key]
        new_tensors[new_key] = value

    safetensors.torch.save_file(new_tensors, f'/mnt/workspace/.cache/modelscope/hub/swift/llava-v1___6-llama-3.1-my/model-0000{i}-of-00005.safetensors',
        metadata={'format': 'pt'})




# model = LlavaNextForConditionalGeneration.from_pretrained("/mnt/workspace/.cache/modelscope/hub/swift/llava-v1___6-llama-3.1-my") 
# print(model)
