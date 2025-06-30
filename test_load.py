from swift.llm import get_model_tokenizer
from swift.utils import get_current_device
import torch
from vllm import LLM
import os
from contextlib import contextmanager
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_USE_MODELSCOPE'] = 'true'

model = None
engine = None

def main():
    global model, engine
    engine = LLM(model='Qwen/Qwen2.5-7B-Instruct')
    model, _ = get_model_tokenizer('Qwen/Qwen2.5-7B-Instruct')

def offload_model(model):
    for param in model.parameters():
        param.data = param.data.to(torch.device('cpu'), non_blocking=True)

@torch.no_grad()
def load_model_to_device(model):
    device = get_current_device()
    for param in model.parameters():
        param.data = param.data.to(device, non_blocking=True)

@contextmanager
def time_context(name=None):
    start_time = time.time()
    if name:
        print(f"{name} started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        if name:
            print(f"{name} ended at {time.strftime('%H:%M:%S', time.localtime(end_time))}, duration: {duration:.2f} seconds")

def load_model(model):
    for name, param in model.named_parameters():
        llm_model = engine.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights([(name, param.data)])

if __name__ == '__main__':
    main()
    offload_model(model)

    with time_context("cpu_load"):
        load_model(model)
        
    load_model_to_device()
    with time_context("gpu_load"):
        load_model(model)
