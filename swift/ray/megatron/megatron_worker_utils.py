# Copyright (c) ModelScope Contributors. All rights reserved.
"""Utility helpers for MegatronWorker co-locate offloading."""


def offload_megatron_for_vllm(worker):
    """Offload Megatron model/optimizer to CPU to free GPU for vLLM."""
    from swift.megatron.trainers.utils import offload_megatron_model_to_cpu, offload_megatron_optimizer

    trainer = worker.trainer
    args = trainer.args
    if getattr(args, 'offload_model', False):
        offload_megatron_model_to_cpu(trainer.wrapped_models)
        if hasattr(trainer, 'ref_models') and trainer.ref_models:
            offload_megatron_model_to_cpu(trainer.ref_models)
    if getattr(trainer, 'optimizer', None) and getattr(args, 'offload_optimizer', False):
        offload_megatron_optimizer(trainer.optimizer)


def reload_megatron_after_vllm(worker):
    """Reload Megatron model/optimizer back to GPU after vLLM is done."""
    from swift.megatron.trainers.utils import load_megatron_model_to_gpu, load_megatron_optimizer

    trainer = worker.trainer
    args = trainer.args
    if getattr(args, 'offload_model', False):
        load_megatron_model_to_gpu(trainer.wrapped_models)
        if hasattr(trainer, 'ref_models') and trainer.ref_models:
            load_megatron_model_to_gpu(trainer.ref_models)
    if getattr(trainer, 'optimizer', None) and getattr(args, 'offload_optimizer', False):
        load_megatron_optimizer(trainer.optimizer)
