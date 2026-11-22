"""defect124: the three task_forward SFT recipes thread the Megatron Configs through to assembly
and the loop (they used to pin None, so --backend megatron could never reach the worker build).

CPU-only: build_supervised_objects / run_supervised are captured, never executed -- the contract
under test is the FORWARDING, not the training. Reverse-verified: re-pinning None in any recipe
turns its threading case RED.
"""
from importlib import import_module
from types import SimpleNamespace

import pytest

from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    MegatronConfig,
    ModelConfig,
    MoEConfig,
    TemplateConfig,
    TrainConfig,
)

# The package __init__ re-exports the run_sft FUNCTION, shadowing the same-named submodule attr;
# import_module resolves the real module from sys.modules.
mod = import_module('swift.dev.recipe.sft.run_sft')


def _configs(task_type, **model_kw):
    return {
        'model_config': ModelConfig(model='m', task_type=task_type, **model_kw),
        'template_config': TemplateConfig(),
        'dataset_config': DatasetConfig(dataset=['d']),
        'train_config': TrainConfig(),
        'distributed_config': DistributedConfig(backend='megatron'),
        'checkpoint_config': CheckpointConfig(),
    }


@pytest.mark.parametrize('recipe,task_type,model_kw', [
    ('run_embedding', 'embedding', {}),
    ('run_reranker', 'reranker', {}),
    ('run_seq_cls', 'seq_cls', {'problem_type': 'single_label_classification', 'num_labels': 3}),
])
def test_recipe_threads_megatron_configs(monkeypatch, recipe, task_type, model_kw):
    mc, mo = MegatronConfig(), MoEConfig()
    seen = {}

    def fake_build(**kwargs):
        seen['build'] = kwargs
        return SimpleNamespace(forward_task=task_type)

    def fake_loop(*args, **kwargs):
        seen['loop_args'] = args
        return []

    monkeypatch.setattr(mod, 'build_supervised_objects', fake_build)
    monkeypatch.setattr(mod, 'run_supervised', fake_loop)
    getattr(mod, recipe)(**_configs(task_type, **model_kw), megatron_config=mc, moe_config=mo)
    assert seen['build']['megatron_config'] is mc
    assert seen['build']['moe_config'] is mo
    # run_supervised's 11th positional is megatron_config (manual_gc and friends read it there).
    assert seen['loop_args'][10] is mc


@pytest.mark.parametrize('recipe,task_type,model_kw', [
    ('run_embedding', 'embedding', {}),
    ('run_reranker', 'reranker', {}),
    ('run_seq_cls', 'seq_cls', {'problem_type': 'single_label_classification', 'num_labels': 3}),
])
def test_recipe_defaults_keep_the_hf_path_none(monkeypatch, recipe, task_type, model_kw):
    seen = {}

    def fake_build(**kwargs):
        seen['build'] = kwargs
        return SimpleNamespace(forward_task=task_type)

    def fake_loop(*args, **kwargs):
        seen['loop_args'] = args
        return []

    monkeypatch.setattr(mod, 'build_supervised_objects', fake_build)
    monkeypatch.setattr(mod, 'run_supervised', fake_loop)
    getattr(mod, recipe)(**_configs(task_type, **model_kw))
    assert seen['build']['megatron_config'] is None
    assert seen['build']['moe_config'] is None
    assert seen['loop_args'][10] is None
