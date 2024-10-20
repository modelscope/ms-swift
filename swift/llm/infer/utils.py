import inspect

from swift.llm import InferArguments
from .infer_engine import InferEngine


def get_infer_engine(args: InferArguments) -> InferEngine:
    kwargs = {
        'model_id_or_path': args.model,
        'model_type': args.model_type,
        'revision': args.model_revision,
        'torch_dtype': args.torch_dtype,
        'use_hf': args.use_hf,
    }
    if args.infer_backend == 'pt':
        from .infer_engine import PtEngine
        infer_engine_cls = PtEngine
        kwargs.update({
            'attn_impl': args.attn_impl,
            'device_map': args.device_map_config,
            'quantization_config': args.quantization_config,
        })
    elif args.infer_backend == 'vllm':
        from .infer_engine import VllmEngine
        infer_engine_cls = VllmEngine
        kwargs.update({
            'gpu_memory_utilization': args.gpu_memory_utilization,
            'tensor_parallel_size': args.tensor_parallel_size,
            'max_num_seqs': args.max_num_seqs,
            'max_model_len': args.max_model_len,
            'disable_custom_all_reduce': args.disable_custom_all_reduce,
            'enforce_eager': args.enforce_eager,
            'limit_mm_per_prompt': args.limit_mm_per_prompt,
            'enable_lora': args.enable_lora,
            'max_loras': args.max_loras,
            'max_lora_rank': args.max_lora_rank
        })
    else:
        from .infer_engine import LmdeployEngine
        infer_engine_cls = LmdeployEngine
        kwargs.update({
            'tp': args.tp,
            'cache_max_entry_count': args.cache_max_entry_count,
            'quant_policy': args.quant_policy,
            'vision_batch_size': args.vision_batch_size
        })

    return infer_engine_cls(**kwargs)
