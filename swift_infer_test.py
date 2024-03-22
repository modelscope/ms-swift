import os
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
tokenizer = AutoTokenizer.from_pretrained('TeleAI/telechat-7B')
model = AutoModelForCausalLM.from_pretrained('TeleAI/telechat-7B', trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
generate_config = GenerationConfig.from_pretrained('TeleAI/telechat-7B')
question="生抽与老抽的区别？"
answer, history = model.chat(tokenizer = tokenizer, question=question, history=[], generation_config=generate_config, stream=False)
print(answer)