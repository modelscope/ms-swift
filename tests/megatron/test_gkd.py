import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':
    from swift.megatron import megatron_rlhf_main, MegatronRLHFArguments
    megatron_rlhf_main(
        MegatronRLHFArguments(
            rlhf_type='gkd',
            model='Qwen/Qwen3-4B-Base',
            teacher_model='Qwen/Qwen3-8B',
            tuner_type='lora',
            dataset=['AI-ModelScope/alpaca-gpt4-data-en#2000', 'AI-ModelScope/alpaca-gpt4-data-zh#2000'],
            tensor_model_parallel_size=2,
            seq_kd=False,
            lmbda=1,
            beta=1,
            micro_batch_size=2,
            global_batch_size=16,
            num_train_epochs=1,
            lr=5e-6,
            log_interval=1,
            max_length=2048,
            max_completion_length=1024,
            attention_backend='flash',
            use_vllm=True,
            vllm_mode='colocate',
            vllm_gpu_memory_utilization=0.5,
            vllm_tensor_parallel_size=1,
            vllm_max_model_len=16384,
            sleep_level=1,
            offload_teacher_model=True,
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
            finetune=True,
            no_save_optim=True,
            no_save_rng=True,
            temperature=1,
            padding_free=True,
            sequence_parallel=True,
        ))
