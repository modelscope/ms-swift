# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the recipe's checkpoint format with real two-rank CPU FSDP2."""
import importlib.util
import os
import socket
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
from pathlib import Path
from types import SimpleNamespace


def checkpoint_worker(rank, directory):
    from accelerate.state import PartialState
    from accelerate.utils.fsdp_utils import load_fsdp_model, save_fsdp_model
    from peft import LoraConfig, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import (FullOptimStateDictConfig, FullStateDictConfig, ShardedOptimStateDictConfig,
                                        ShardedStateDictConfig, StateDictType, fully_shard)

    torch.set_num_threads(1)
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2', MASTER_ADDR='127.0.0.1')
    loopback = next((name for _, name in socket.if_nameindex() if name in ('lo', 'lo0')), None)
    if loopback:
        os.environ['GLOO_SOCKET_IFNAME'] = loopback
    dist.init_process_group('gloo', init_method='file://' + directory + '/rendezvous', rank=rank, world_size=2)
    try:
        PartialState(cpu=True)
        source = Path(__file__).resolve().parents[2] / 'examples/train/startlux_decision/decision_training.py'
        spec = importlib.util.spec_from_file_location('decision_contract', source)
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        state_type = StateDictType[contract.FSDP_STATE_DICT_TYPE]
        full = state_type == StateDictType.FULL_STATE_DICT
        plugin = SimpleNamespace(
            fsdp_version=2,
            state_dict_type=state_type,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            if full else ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            if full else ShardedOptimStateDictConfig(offload_to_cpu=True),
            sync_module_states=True)
        accelerator = SimpleNamespace(
            num_processes=2,
            process_index=rank,
            is_fsdp2=True,
            is_main_process=rank == 0,
            wait_for_everyone=dist.barrier)
        torch.manual_seed(42)
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(8, 8)
        model = get_peft_model(model, LoraConfig(r=2, target_modules=['proj']))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.copy_(torch.arange(parameter.numel()).view_as(parameter) / 100)
        expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if 'lora_' in name}
        mesh = init_device_mesh('cpu', (2, ))
        fully_shard(model.base_model.model.proj, mesh=mesh)
        fully_shard(model, mesh=mesh)
        save_fsdp_model(plugin, accelerator, model, directory, adapter_only=True)
        dist.barrier()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_' in name:
                    parameter.zero_()
        load_fsdp_model(plugin, accelerator, model, directory, adapter_only=True)
        for name, parameter in model.named_parameters():
            if name in expected:
                torch.testing.assert_close(parameter.full_tensor(), expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class DecisionCheckpoint(unittest.TestCase):

    def test_all_rank_adapter_weights_survive_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(checkpoint_worker, args=(directory, ), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
