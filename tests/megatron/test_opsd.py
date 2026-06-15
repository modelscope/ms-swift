import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if __name__ == '__main__':
    from swift.megatron import MegatronRLHFArguments, megatron_rlhf_main
    megatron_rlhf_main(
        MegatronRLHFArguments(
            rlhf_type='gkd',
            model='Qwen/Qwen3-4B',
            teacher_model='Qwen/Qwen3-4B',
            external_plugins=['examples/train/rlhf/opsd/opsd_plugin.py'],
            dataset=['open-r1/OpenThoughts-114k-math'],
            use_vllm=True,
            vllm_mode='colocate',
            vllm_gpu_memory_utilization=0.6,
            vllm_max_model_len=10240,
            tuner_type='lora',
            lora_rank=64,
            lora_alpha=128,
            sleep_level=1,
            lmbda=1.0,
            beta=0.5,
            temperature=1.2,
            sft_alpha=0,
            torch_dtype='bfloat16',
            micro_batch_size=2,
            global_batch_size=32,
            train_iters=1000,
            lr=2e-5,
            save_steps=100,
            save_total_limit=10,
            logging_steps=1,
            max_length=8192,
            max_completion_length=2048,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            attention_backend='flash',
            recompute_granularity='selective',
            finetune=True,
            no_save_optim=True,
            no_save_rng=True,
        ))
