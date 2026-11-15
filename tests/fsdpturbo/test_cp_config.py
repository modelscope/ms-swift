import importlib.util
import pytest
import torch
from types import SimpleNamespace

from swift.fsdpturbo.model_specs import get_model_spec

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('fsdp_turbo') is None, reason='requires the optional FSDPTurbo package')


def make_args(cp_size):
    train = SimpleNamespace(
        bf16=True,
        fp16=False,
        torch_dtype=torch.bfloat16,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        learning_rate=1e-5,
        warmup_ratio=0,
        lr_scheduler_type='cosine',
        max_grad_norm=1,
        seed=42,
        max_steps=20,
        num_train_epochs=3,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        train_dataloader_shuffle=False,
        logging_steps=1,
        save_only_model=False,
        save_strategy='no',
        save_steps=20,
    )
    return SimpleNamespace(
        training_args=train,
        model='/local/model',
        tuner_type='full',
        torch_dtype=torch.bfloat16,
        dataset=['/local/data.jsonl'],
        cached_dataset=[],
        max_length=256,
        output_dir='/local/output',
        resume_from_checkpoint=None,
        resume_only_model=False,
        fsdp_size=8,
        tp_size=1,
        cp_size=cp_size,
        ep_size=8,
        efsdp_size=1,
        forward_prefetch=0,
        backward_prefetch=0,
        offload_params=False,
        pin_memory=True,
        host_pin_memory=None,
        fsdp_implementation='native',
        ep_dispatcher='eager',
        gradient_checkpointing=True,
    )


def test_native_npu_full_unpinned_offload_fails_before_model_loading():
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    args = make_args(1)
    args.model_type, args.router_aux_loss_coef = 'deepseek_v41', 0
    args.offload_params, args.pin_memory = True, False
    args.training_args.device = SimpleNamespace(type='npu')
    with pytest.raises(ValueError, match='requires pin_memory=true'):
        FSDPTurboTrainer(args, None, None, None)


def test_native_npu_lora_keeps_unpinned_parameter_offload_available():
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    args = make_args(1)
    args.model_type, args.router_aux_loss_coef = 'deepseek_v41', 0
    args.tuner_type, args.offload_params, args.pin_memory = 'lora', True, False
    args.training_args.device = SimpleNamespace(type='npu')
    trainer = FSDPTurboTrainer(args, None, None, None)
    assert trainer.config.distributed.fsdp_plan.pin_memory is False


def test_cp1_does_not_enable_model_or_loss_patches():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    config = build_fsdpturbo_config(make_args(1), get_model_spec('qwen3_5_moe'))
    assert config.distributed.ulysses_parallel_size == 1
    assert config.module_patches == []
    assert config.distributed.cp_plan.ulysses_function_patches == []
    assert config.distributed.cp_plan.loss_function_patches == []


def test_checkpoint_identity_includes_raw_and_cached_training_inputs():
    from fsdp_turbo.training.checkpoint import dataset_fingerprint

    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    args = make_args(1)
    args.cached_dataset = ['/local/cache_A']
    first = build_fsdpturbo_config(args, get_model_spec('qwen3_5_moe'))
    args.cached_dataset = ['/local/cache_B']
    second = build_fsdpturbo_config(args, get_model_spec('qwen3_5_moe'))
    assert '/local/data.jsonl' in first.data.dataset_path
    assert '/local/cache_A' in first.data.dataset_path
    assert dataset_fingerprint(first.data) != dataset_fingerprint(second.data)


def test_qwen_cp2_uses_backend_gdn_and_postfusion_interfaces():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    config = build_fsdpturbo_config(make_args(2), get_model_spec('qwen3_5_moe'))
    assert config.distributed.ulysses_parallel_size == 2
    assert {item['type']
            for item in config.distributed.cp_plan.ulysses_function_patches} == {'full_attention', 'gated_delta_net'}
    assert config.distributed.cp_plan.loss_function_patches[0]['type'] == 'causal_lm_loss'
    assert config.module_patches[0]['replacement'].endswith('qwen3_5_moe_model_forward')


def test_native_cp_rejects_legacy_ulysses_selection():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    args = make_args(2)
    args.cp_implementation = 'ulysses'
    with pytest.raises(ValueError, match='supports kv_allgather CP'):
        build_fsdpturbo_config(args, get_model_spec('deepseek_v41'))


def test_native_v41_uses_intrinsic_reference_order_dispatcher():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    args = make_args(1)
    args.gradient_checkpointing = False
    spec = get_model_spec('deepseek_v41')
    config = build_fsdpturbo_config(args, spec)
    assert spec.native_model_factory.endswith('adapter.build_deepseek_v41_model')
    assert config.distributed.ep_plan.dispatcher == 'custom_native_eager_forward'
    assert not config.memory.recompute
    assert spec.tp_colwise_modules and not spec.cp_function_patches


@pytest.mark.parametrize('mode,fused', [('eager', False), ('fused', True)])
def test_cli_ep_modes_build_a_dispatcher_supported_by_current_backend(mode, fused):
    from fsdp_turbo.distributed.expert_parallel.expert_parallel import get_experts_forward_fn

    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    args = make_args(1)
    args.ep_dispatcher = mode
    config = build_fsdpturbo_config(args, get_model_spec('qwen3_5_moe'))
    plan = config.distributed.ep_plan
    assert plan.use_fused_ops is fused
    assert callable(get_experts_forward_fn(None, plan, plan.dispatcher))


def test_native_v41_entrypoint_resolves_after_backend_directory_reorganization():
    import importlib

    path = get_model_spec('deepseek_v41').native_model_factory
    module, _, name = path.rpartition('.')
    assert callable(getattr(importlib.import_module(module), name))


@pytest.mark.parametrize('field,value,message', [
    ('ep_size', 1, 'requires EP > 1'),
    ('ep_dispatcher', 'fused', 'requires EP > 1'),
    ('resume_from_checkpoint', '/checkpoint', 'no checkpoint resume'),
    ('router_aux_loss_coef', 0.01, 'auxiliary loss is not implemented'),
])
def test_native_v41_rejects_unvalidated_modes_before_loading(field, value, message):
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    args = make_args(1)
    args.model_type = 'deepseek_v41'
    args.gradient_checkpointing = False
    args.router_aux_loss_coef = 0
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        FSDPTurboTrainer(args, None, None, None)


def test_native_v41_trainer_rejects_model_only_save_before_loading():
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    args = make_args(1)
    args.tuner_type = 'full'
    args.model_type = 'deepseek_v41'
    args.gradient_checkpointing = False
    args.router_aux_loss_coef = 0
    args.training_args.save_only_model = True
    with pytest.raises(ValueError, match='save_only_model'):
        FSDPTurboTrainer(args, None, None, None)
