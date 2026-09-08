"""torchrun entry for test_hf_sp_config.py::test_hf_sp_mesh_and_validate_over_gloo.

Runs on CPU under ``torchrun --nproc_per_node=2`` with a gloo process group (no accelerator):
  1. validate_configs ACCEPTS sp=world on the HF backend when every guard is satisfied
     (padding_free + flash_attention_2, right padding, local mode, dist initialized);
  2. build_hf_device_mesh returns a mesh with ulysses_size=sp and data_world_size=world/sp;
  3. validate_configs REJECTS sp=world+1 on divisibility.

Each rank writes ``{out}.rank{RANK}.json``.

Usage:
    torchrun --nproc_per_node=2 swift/dev/tests/_runners/hf_sp_mesh.py --out RESULT
"""

import argparse

import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch.distributed as dist

    dist.init_process_group("gloo")

    from swift.dev.builders import build_hf_device_mesh
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        validate_configs,
    )

    world = dist.get_world_size()
    distributed_config = DistributedConfig()
    result = {"rank": dist.get_rank(), "world": world}

    def _validate(sp):
        validate_configs(
            ModelConfig(model="dummy", attn_impl="flash_attention_2"),
            TemplateConfig(template="qwen2_5", sequence_parallel_size=sp, padding_free=True),
            DatasetConfig(dataset=["dummy"]),
            TrainConfig(),
            distributed_config,
            CheckpointConfig(),
        )

    # 1+2. accepted: pure-SP layout (sp == world, so dp=1) with all guards satisfied.
    _validate(world)
    result["validate_ok"] = True
    mesh = build_hf_device_mesh(distributed_config, world)
    result["ulysses_size"] = mesh.ulysses_size
    result["data_world_size"] = mesh.data_world_size

    # 3. rejected: world % sp != 0.
    try:
        _validate(world + 1)
        result["divisibility_raised"] = False
    except ValueError as e:
        result["divisibility_raised"] = "not divisible" in str(e)

    with open(f"{args.out}.rank{result['rank']}.json", "w") as f:
        json.dump(result, f)
    print(f"RUNNER_DONE {result}", flush=True)


if __name__ == "__main__":
    main()
