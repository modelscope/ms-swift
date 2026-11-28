"""Run with torchrun to verify rank-0 loading and exact FSDP/TP/EP layouts."""
import os
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from fsdp_turbo.distributed.parallel_state import reset_parallel_state
from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import DistributedConfig, EPPlanConfig, FSDPPlanConfig, FSDPTurboConfig, TPPlanConfig
from torch import nn
from torch.distributed.tensor import DTensor
from types import SimpleNamespace

from swift.fsdpturbo.loading import load_rank0_model, materialize_sharded_model


class Experts(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.empty(4, 8, 8, dtype=torch.bfloat16))
        self.down_proj = nn.Parameter(torch.empty(4, 8, 4, dtype=torch.bfloat16))


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        self.experts = Experts()
        self.stable = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.register_buffer('indices', torch.arange(8, dtype=torch.int64))

    def forward(self, x):
        return self.proj(x)


def main():
    device = torch.accelerator.current_accelerator()
    torch.accelerator.set_device_index(int(os.environ['LOCAL_RANK']))
    backend = dist.get_default_backend_for_device(device)
    dist.init_process_group(backend=f'cpu:gloo,{device.type}:{backend}')
    calls = 0

    def load(**_kwargs):
        nonlocal calls
        calls += 1
        model = Model()
        with torch.no_grad():
            for param in model.parameters():
                param.copy_(torch.arange(param.numel()).reshape(param.shape).to(param.dtype))
        return model, None

    args = SimpleNamespace(
        model_info=SimpleNamespace(quant_method=None), get_model_processor=load, resume_from_checkpoint=None)
    try:
        source, dtypes = load_rank0_model(args)
        assert calls == int(dist.get_rank() == 0)
        with init_empty_weights(include_buffers=False):
            model = Model()
        for name, param in model.named_parameters():
            param.data = param.data.to(dtypes[name])
        config = FSDPTurboConfig(
            distributed=DistributedConfig(
                fully_shard_parallel_size=2,
                tensor_parallel_size=2,
                expert_parallel_size=2,
                expert_fully_shard_parallel_size=2,
                fsdp_plan=FSDPPlanConfig(apply_modules={'proj': {}}, ignored_params=['stable']),
                tp_plan=TPPlanConfig(colwise_parallel=['proj'], rowwise_parallel=[], sequence_parallel=[]),
                ep_plan=EPPlanConfig(
                    apply_modules=['experts'],
                    apply_efsdp_modules=['experts'],
                    dispatcher='all2all',
                    use_fused_ops=False),
            ))
        wrapped = FSDPTurbo(config, model)
        materialize_sharded_model(wrapped.model, source)
        for name, param in wrapped.model.named_parameters():
            actual = param.full_tensor() if isinstance(param, DTensor) else param
            expected = torch.arange(actual.numel()).reshape(actual.shape).to(dtype=actual.dtype, device=actual.device)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
        torch.testing.assert_close(wrapped.model.indices.cpu(), torch.arange(8))
        if dist.get_rank() == 0:
            print('PASS: rank0-only load and exact FSDP/TP/EP/EFSDP weights', flush=True)
    finally:
        dist.destroy_process_group()
        reset_parallel_state()


if __name__ == '__main__':
    main()
