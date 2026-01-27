import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 4,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_embedding():
    from swift import sft_main, SftArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen3-Embedding-0.6B',
            task_type='embedding',
            dataset=['sentence-transformers/stsb:positive'],
            split_dataset_ratio=0.01,
            load_from_cache_file=False,
            loss_type='infonce',
            attn_impl='flash_attn',
            max_length=2048,
            **kwargs,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    print(f'last_model_checkpoint: {last_model_checkpoint}')


def test_reranker():
    from swift import sft_main, SftArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen3-Reranker-4B',
            tuner_type='lora',
            load_from_cache_file=True,
            task_type='generative_reranker',
            dataset=['MTEB/scidocs-reranking#10000'],
            split_dataset_ratio=0.05,
            loss_type='pointwise_reranker',
            dataloader_drop_last=True,
            eval_strategy='steps',
            eval_steps=10,
            max_length=4096,
            attn_impl='flash_attn',
            num_train_epochs=1,
            save_steps=200,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            dataset_num_proc=2,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    print(f'last_model_checkpoint: {last_model_checkpoint}')


def test_reranker2():
    from swift import sft_main, SftArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2.5-VL-3B-Instruct',
            tuner_type='lora',
            load_from_cache_file=True,
            task_type='reranker',
            dataset=['MTEB/scidocs-reranking'],
            split_dataset_ratio=0.05,
            loss_type='listwise_reranker',
            dataloader_drop_last=True,
            eval_strategy='steps',
            eval_steps=10,
            max_length=4096,
            attn_impl='flash_attn',
            padding_side='right',
            num_train_epochs=1,
            save_steps=200,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            dataset_num_proc=1,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    print(f'last_model_checkpoint: {last_model_checkpoint}')


if __name__ == '__main__':
    # test_embedding()
    test_reranker()
