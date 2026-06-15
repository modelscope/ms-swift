import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['NPROC_PER_NODE'] = '2'


def train():
    from swift import RLHFArguments, rlhf_main
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='gkd',
            model='Qwen/Qwen3.5-4B',
            teacher_model='Qwen/Qwen3.5-4B',
            tuner_type='lora',
            lora_rank=64,
            lora_alpha=128,
            target_modules=['all-linear'],
            use_vllm=True,
            vllm_mode='colocate',
            vllm_gpu_memory_utilization=0.7,
            vllm_max_model_len=10240,
            sleep_level=1,
            external_plugins=['examples/train/rlhf/opsd/opsd_plugin.py'],
            dataset=['open-r1/OpenThoughts-114k-math'],
            lmbda=1.0,
            beta=0.5,
            temperature=1.2,
            sft_alpha=0,
            torch_dtype='bfloat16',
            max_steps=1000,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            save_steps=100,
            save_total_limit=10,
            logging_steps=1,
            max_length=8192,
            max_completion_length=2048,
            save_only_model=True,
            gradient_checkpointing=True,
            deepspeed='zero0',
            attn_impl='flash_attn',
        ))
    return result


if __name__ == '__main__':
    train()
