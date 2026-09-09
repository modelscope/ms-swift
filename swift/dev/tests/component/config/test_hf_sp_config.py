"""Ulysses sequence parallelism (TemplateConfig.sequence_parallel_size) on the HF backend.

Why these guards exist. The knob was accepted by the config and forwarded to the template
(builders/template.py) long before any training-path consumer existed: an sp>1 run SILENTLY trained
with SP=1. Now that run_sft wires a real mesh (build_hf_device_mesh), validate_configs must reject
every combination where the mesh would be meaningless or the twinkle SP strategy would crash at the
first forward -- failing at validation in milliseconds instead of after the weights load.

Nothing here touches an accelerator. The guard matrix is a pure function of the Configs and needs no
process group (the rejections all fire before the divisibility check); the accepted case and the mesh
math need world_size>=2, so they run under torchrun with a gloo (CPU) group via the hf_sp_mesh runner.
"""

from __future__ import annotations
import os
import sys

import json
import pytest

from ..._runners import Runners
from ...feature.sft.test_e2e import _master_port, _run_torchrun


def _configs(*, rlhf_type=None, **overrides):
    """HF-backend config dict for validate_configs; overrides key on 'field=value' via setattrs."""
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
    )

    configs = {
        "model_config": ModelConfig(model="dummy"),
        "template_config": TemplateConfig(template="qwen2_5"),
        "dataset_config": DatasetConfig(dataset=["dummy"]),
        "train_config": TrainConfig(),
        "distributed_config": DistributedConfig(),
        "checkpoint_config": CheckpointConfig(),
    }
    if rlhf_type is not None:
        from swift.dev.config import RLHFConfig

        configs["rlhf_config"] = RLHFConfig(rlhf_type=rlhf_type)
    for dotted, value in overrides.items():
        holder_name, _, attr = dotted.partition(".")
        setattr(configs[holder_name], attr, value)
    return configs


def _validate(**overrides):
    from swift.dev.config import validate_configs

    validate_configs(**_configs(**overrides))


def test_sp_off_by_default_is_accepted():
    """sp=1 must pass with no process group at all -- the guard returns before touching dist."""
    _validate()


def test_sp_on_megatron_backend_is_rejected():
    """Megatron SP is DistributedConfig.sequence_parallel (TP-SP), a different feature."""
    with pytest.raises(ValueError, match="only applies to the transformers backend"):
        _validate(
            **{
                "template_config.sequence_parallel_size": 2,
                "distributed_config.backend": "megatron",
                "distributed_config.nproc_per_node": 2,
            }
        )


@pytest.mark.parametrize("rlhf_type", ["dpo", "grpo", "kto"])
def test_sp_with_any_rlhf_type_is_rejected(rlhf_type):
    """dev wires no RLHF SP mesh; legacy's grpo/dpo allowance would silently train without SP."""
    with pytest.raises(ValueError, match="does not support sequence_parallel_size"):
        _validate(rlhf_type=rlhf_type, **{"template_config.sequence_parallel_size": 2})


def test_sp_in_ray_mode_is_rejected():
    """Ray passes a pure data-parallel mesh (_apply_ray_placement); SP would silently not apply."""
    with pytest.raises(NotImplementedError, match='mode="local"'):
        _validate(
            **{
                "template_config.sequence_parallel_size": 2,
                "distributed_config.mode": "ray",
            }
        )


def test_sp_with_fsdp_is_rejected():
    """The FSDP x ulysses composition in twinkle is unvalidated."""
    with pytest.raises(NotImplementedError, match="fsdp"):
        _validate(
            **{
                "template_config.sequence_parallel_size": 2,
                "distributed_config.fsdp": "full_shard",
            }
        )


def test_sp_with_left_padding_is_rejected():
    """The SP collator's per-row position_ids assume right padding (legacy asserts at collate time)."""
    with pytest.raises(ValueError, match="padding_side"):
        _validate(
            **{
                "template_config.sequence_parallel_size": 2,
                "template_config.padding_side": "left",
            }
        )


def test_sp_padding_free_without_flash_attn_is_rejected():
    """twinkle's SP strategy rejects the variable-length layout without FA2/3 at first forward."""
    with pytest.raises(ValueError, match="flash attention"):
        _validate(
            **{
                "template_config.sequence_parallel_size": 2,
                "template_config.padding_free": True,
                "model_config.attn_impl": "sdpa",
            }
        )


def test_sp_padding_free_with_unset_attn_impl_is_rejected():
    """attn_impl=None lands on the default (non-flash) kernel, so it is rejected too."""
    with pytest.raises(ValueError, match="flash attention"):
        _validate(
            **{
                "template_config.sequence_parallel_size": 2,
                "template_config.padding_free": True,
            }
        )


def test_sp_packing_inherits_the_flash_attn_requirement():
    """_check_packing forces padding_free=True before this guard runs, so packing+SP+sdpa must fail."""
    with pytest.raises(ValueError, match="flash attention"):
        _validate(
            **{
                "template_config.sequence_parallel_size": 2,
                "dataset_config.packing": True,
                "model_config.attn_impl": "sdpa",
            }
        )


def test_sp_without_torchrun_is_rejected():
    """Single process: no ranks to split a sequence across (WORLD_SIZE is unset here)."""
    with pytest.raises(ValueError, match="torchrun"):
        _validate(**{"template_config.sequence_parallel_size": 2})


def test_mesh_builder_returns_none_when_sp_is_off():
    """sp=1 must reach TransformersModel with NO mesh -- the deliberate default-mesh local path."""
    from swift.dev.builders import build_hf_device_mesh
    from swift.dev.config import DistributedConfig

    assert build_hf_device_mesh(DistributedConfig(), 1) is None
    # Non-local mode never builds this mesh either (validate rejects ray+sp>1 before this runs).
    assert build_hf_device_mesh(DistributedConfig(mode="ray"), 2) is None


def test_mesh_builder_without_torchrun_raises():
    """Plain pytest process: WORLD_SIZE unset -> world=1 -> the torchrun requirement fires."""
    from swift.dev.builders import build_hf_device_mesh
    from swift.dev.config import DistributedConfig

    with pytest.raises(ValueError, match="torchrun"):
        build_hf_device_mesh(DistributedConfig(), 2)


@pytest.mark.slow
def test_hf_sp_mesh_and_validate_over_gloo(tmp_path):
    """2-proc gloo: sp=2 is ACCEPTED with all guards satisfied, and the mesh math holds.

    The one combination the matrix above cannot cover: passing the divisibility guard needs a real
    process group with world_size >= sp. gloo keeps it CPU-only. Asserts, per rank:
      - validate_configs accepts sp=world with padding_free + flash_attention_2;
      - the mesh reports ulysses_size=world and data_world_size=1 (world/ulysses);
      - sp=world+1 raises on divisibility.
    """
    runner = Runners.path("hf_sp_mesh")
    result_prefix = str(tmp_path / "gloo")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        f"--master_port={_master_port(10)}",
        runner,
        "--out",
        result_prefix,
    ]
    out, err = _run_torchrun(cmd)

    for rank in range(2):
        path = f"{result_prefix}.rank{rank}.json"
        if not os.path.exists(path):
            raise AssertionError(
                f"gloo runner produced no result for rank {rank}. stdout tail:\n{out[-2000:]}\n"
                f"stderr tail:\n{err[-3000:]}"
            )
        with open(path) as f:
            r = json.load(f)
        assert r["world"] == 2
        assert r["validate_ok"], f"rank {rank}: sp=world with all guards satisfied must validate"
        assert r["ulysses_size"] == 2, f"rank {rank}: mesh ulysses_size={r['ulysses_size']}, not 2"
        assert r["data_world_size"] == 1, f"rank {rank}: data_world_size={r['data_world_size']}, not world/ulysses=1"
        assert r["divisibility_raised"], f"rank {rank}: sp=world+1 must raise on divisibility"
