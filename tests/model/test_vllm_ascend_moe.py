import torch

from swift.model.npu_patch.vllm_ascend_moe import (
    configure_vllm_ascend_moe_preprocessed_weight_sync,
    configure_vllm_ascend_moe_weight_sync,
    patch_vllm_ascend_moe_expert_weight_loader,
    should_keep_fused_moe_expert_for_vllm_ascend,
    should_skip_vllm_ascend_moe_post_load,
    use_vllm_ascend_moe_preprocessed_weight,
)


class _AscendQuantMethod:
    pass


_AscendQuantMethod.__module__ = 'vllm_ascend.test'


class _Config:

    def __init__(self, model_type):
        self.model_type = model_type


class _Model:

    def __init__(self, model_type=None):
        if model_type is not None:
            self.config = _Config(model_type)


class _Experts:

    def __init__(self, tp_rank=0):
        self.tp_rank = tp_rank
        self.quant_method = _AscendQuantMethod()

    def _map_global_expert_id_to_local_expert_id(self, global_expert_id):
        return global_expert_id if global_expert_id == 0 else -1

    @staticmethod
    def weight_loader(*args, **kwargs):
        raise AssertionError('origin weight_loader should not handle vLLM-Ascend runtime sync')


class _Param:

    def __init__(self, data):
        self.data = data


def test_qwen_moe_fsdp2_colocate_sync_uses_processed_layout_and_skips_post_load():
    for model_type in ('qwen3_moe', 'qwen3_5_moe'):
        vllm_model = _Model()
        train_model = _Model(model_type)

        configure_vllm_ascend_moe_weight_sync(vllm_model, train_model, is_fsdp2=True)

        assert not should_keep_fused_moe_expert_for_vllm_ascend(train_model)
        assert not use_vllm_ascend_moe_preprocessed_weight(vllm_model)
        assert should_skip_vllm_ascend_moe_post_load(vllm_model)


def test_server_preprocessed_sync_keeps_post_load_enabled():
    vllm_model = _Model()

    configure_vllm_ascend_moe_preprocessed_weight_sync(vllm_model)

    assert use_vllm_ascend_moe_preprocessed_weight(vllm_model)
    assert not should_skip_vllm_ascend_moe_post_load(vllm_model)


def test_vllm_ascend_loader_copies_fsdp2_fused_weights_to_processed_layout():
    hidden_size = 4
    intermediate_per_tp = 3
    experts = _Experts(tp_rank=0)

    w13_param = _Param(torch.zeros(1, hidden_size, 2 * intermediate_per_tp))
    patch_vllm_ascend_moe_expert_weight_loader(experts, 'w13_weight', w13_param)
    gate_weight = torch.arange(1 * intermediate_per_tp * hidden_size).reshape(1, intermediate_per_tp, hidden_size)
    up_weight = gate_weight + 1000

    assert w13_param.weight_loader(
        w13_param,
        gate_weight,
        'model.layers.0.mlp.experts.w13_weight',
        shard_id='w1',
        expert_id=0,
        return_success=True,
    )
    assert torch.equal(w13_param.data[0, :, :intermediate_per_tp], gate_weight[0].T)

    assert w13_param.weight_loader(
        w13_param,
        up_weight,
        'model.layers.0.mlp.experts.w13_weight',
        shard_id='w3',
        expert_id=0,
        return_success=True,
    )
    assert torch.equal(w13_param.data[0, :, intermediate_per_tp:], up_weight[0].T)

    w2_param = _Param(torch.zeros(1, intermediate_per_tp, hidden_size))
    patch_vllm_ascend_moe_expert_weight_loader(experts, 'w2_weight', w2_param)
    down_weight = torch.arange(1 * hidden_size * intermediate_per_tp).reshape(1, hidden_size, intermediate_per_tp)

    assert w2_param.weight_loader(
        w2_param,
        down_weight,
        'model.layers.0.mlp.experts.w2_weight',
        shard_id='w2',
        expert_id=0,
        return_success=True,
    )
    assert torch.equal(w2_param.data[0], down_weight[0].T)
