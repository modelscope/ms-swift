import torch
from transformers.utils import is_torch_npu_available


def get_npu_or_cpu_device():
    if is_torch_npu_available():
        return torch.device('npu')
    return torch.device('cpu')


def get_diffusers_unet_input(device):
    return {
        'sample': torch.ones((1, 4, 64, 64), device=device),
        'timestep': torch.tensor(10, device=device),
        'encoder_hidden_states': torch.ones((1, 77, 768), device=device)
    }
