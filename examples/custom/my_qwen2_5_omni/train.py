import os
import sys

from swift.llm import TrainArguments, sft_main

sys.path.append('examples/custom/my_qwen2_5_omni')

if __name__ == '__main__':
    import my_register
    os.environ['MAX_PIXELS'] = '1003520'
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-Omni-7B',
            dataset='AI-ModelScope/LaTeX_OCR#5000',
            model_type='my_qwen2_5_omni',
            template='my_qwen2_5_omni',
            load_from_cache_file=True,
            split_dataset_ratio=0.01,
            train_type='lora',
            torch_dtype='bfloat16',
            attn_impl='flash_attn',
            padding_free=True,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=1e-4,
            lora_rank=8,
            lora_alpha=32,
            target_modules='all-linear',
            freeze_vit=True,
            freeze_aligner=True,
            gradient_accumulation_steps=1,
            eval_steps=50,
            save_steps=50,
            save_total_limit=2,
            logging_steps=5,
            max_length=2048,
            output_dir='output',
            warmup_ratio=0.05,
            dataloader_num_workers=4,
            dataset_num_proc=1,
        ))
