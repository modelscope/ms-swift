# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
import sys
import torch

from swift.model.models.xing4_0 import Xing4_0Loader, transformers_5

MODEL_ID = 'XingChen-AGI/Xing4.0-29B-A4B'
# Only the remote code and the config are needed: every module below is built from scratch, so the
# 29B checkpoint itself is never downloaded.
CODE_FILES = ['config.json', 'generation_config.json', 'configuration_xing4_0.py', 'modeling_xing4_0.py']

pytestmark = pytest.mark.skipif(not transformers_5, reason='the stacked experts need transformers>=5')


@pytest.fixture(scope='module')
def model_dir():
    from modelscope.hub.snapshot_download import snapshot_download
    return snapshot_download(MODEL_ID, allow_patterns=CODE_FILES)


@pytest.fixture(scope='module')
def modeling_module(model_dir):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    model_cls = get_class_from_dynamic_module('modeling_xing4_0.Xing4_0ForCausalLM', model_dir)
    return sys.modules[model_cls.__module__]


@pytest.fixture(scope='module')
def config(model_dir):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # The MLA head dims have to stay self-consistent, so only the MoE / depth / vocab are shrunk.
    for key, value in dict(
            intermediate_size=48,
            moe_intermediate_size=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=0,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
            vocab_size=64,
            max_position_embeddings=64,
    ).items():
        setattr(config, key, value)
    return config


@pytest.fixture(scope='module')
def official_moe_impl(modeling_module):
    """The unpatched `Xing4_0MoE.__init__` / `Xing4_0MoE.moe`, captured before any test patches them."""
    return modeling_module.Xing4_0MoE.__init__, modeling_module.Xing4_0MoE.moe


def _make_tiny_model(modeling_module, config):
    torch.manual_seed(0)
    with torch.device('meta'):
        model = modeling_module.Xing4_0ForCausalLM(config)
    model = model.to_empty(device='cpu').to(torch.float32)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.02)
    return model.eval()


def test_stacked_experts_match_the_official_loop(modeling_module, config, official_moe_impl):
    """`Xing4_0Experts` must compute exactly what the official 64-iteration python loop computes."""
    moe_cls = modeling_module.Xing4_0MoE
    # Put the unpatched implementation back first, so that the comparison does not depend on the order
    # in which the tests of this module run.
    moe_cls.__init__, moe_cls.moe = official_moe_impl
    torch.manual_seed(0)
    official = modeling_module.Xing4_0MoE(config).to(torch.float32)
    with torch.no_grad():
        for expert in official.experts:
            for param in expert.parameters():
                param.mul_(0.05)
        official.gate.weight.mul_(0.05)

    hidden_states = torch.randn(7, config.hidden_size)
    with torch.no_grad():
        _, topk_weights, topk_indices = official.gate(hidden_states)
        official_moe = official.moe(hidden_states, topk_indices, topk_weights)
        official_output = official(hidden_states)

    Xing4_0Loader._stack_experts(modeling_module)
    stacked = modeling_module.Xing4_0MoE(config).to(torch.float32)
    num_experts, intermediate_size = config.n_routed_experts, config.moe_intermediate_size
    with torch.no_grad():
        for i in range(num_experts):
            stacked.experts.gate_up_proj[i].copy_(
                torch.cat([official.experts[i].gate_proj.weight, official.experts[i].up_proj.weight], dim=0))
            stacked.experts.down_proj[i].copy_(official.experts[i].down_proj.weight)
        stacked.gate.load_state_dict(official.gate.state_dict())
        stacked.shared_experts.load_state_dict(official.shared_experts.state_dict())
        stacked_moe = stacked.moe(hidden_states, topk_indices, topk_weights)
        stacked_output = stacked(hidden_states)

    assert stacked.experts.gate_up_proj.shape == (num_experts, 2 * intermediate_size, config.hidden_size)
    assert stacked.experts.down_proj.shape == (num_experts, config.hidden_size, intermediate_size)
    # Fusing gate_proj/up_proj into one GEMM changes the rounding, not the math.
    assert torch.allclose(official_moe, stacked_moe, rtol=1e-5, atol=1e-6)
    assert torch.allclose(official_output, stacked_output, rtol=1e-5, atol=1e-6)


def test_checkpoint_weight_names_round_trip(modeling_module, config, tmp_path):
    """An exported checkpoint must keep the official `mlp.experts.{i}.{gate,up,down}_proj.weight` names.

    `save_pretrained` reverts the conversion mapping, so a model trained with the stacked experts still
    loads with the unpatched remote code (vLLM / SGLang), and reloading it here gives the same logits.
    """
    from safetensors import safe_open
    Xing4_0Loader._stack_experts(modeling_module)
    model = _make_tiny_model(modeling_module, config)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        expected = model(input_ids).logits

    export_dir = str(tmp_path / 'export')
    # `AutoModelForCausalLM.from_pretrained` does this for us; the model above was built directly.
    modeling_module.Xing4_0ForCausalLM.register_for_auto_class('AutoModelForCausalLM')
    model.save_pretrained(export_dir)
    with safe_open(os.path.join(export_dir, 'model.safetensors'), framework='pt') as f:
        exported = {key: f.get_tensor(key) for key in f.keys()}
    assert not any('gate_up_proj' in key for key in exported)

    experts = model.model.layers[0].mlp.experts
    intermediate_size = config.moe_intermediate_size
    for i in range(config.n_routed_experts):
        prefix = f'model.layers.0.mlp.experts.{i}.'
        assert torch.equal(exported[prefix + 'gate_proj.weight'], experts.gate_up_proj[i, :intermediate_size])
        assert torch.equal(exported[prefix + 'up_proj.weight'], experts.gate_up_proj[i, intermediate_size:])
        assert torch.equal(exported[prefix + 'down_proj.weight'], experts.down_proj[i])

    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True, dtype=torch.float32).eval()
    with torch.no_grad():
        assert torch.allclose(reloaded(input_ids).logits, expected, rtol=1e-4, atol=1e-5)
