import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def test_embedding():
    from swift.megatron import megatron_sft_main, MegatronSftArguments
    megatron_sft_main(
        MegatronSftArguments(
            model='Qwen/Qwen3-Embedding-0.6B',
            task_type='embedding',
            dataset=['sentence-transformers/stsb:positive'],
            split_dataset_ratio=0.01,
            micro_batch_size=4,
            tensor_model_parallel_size=2,
            tuner_type='lora',
            max_epochs=1,
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
            loss_type='infonce',
            attn_impl='flash_attn',
            max_length=2048,
            eval_iters=5,
            save_interval=5,
            no_save_optim=True,
            no_save_rng=True,
            sequence_parallel=True,
            finetune=True))


def test_reranker():
    from swift.megatron import megatron_sft_main, MegatronSftArguments
    megatron_sft_main(
        MegatronSftArguments(
            model='Qwen/Qwen3-Reranker-4B',
            tuner_type='lora',
            load_from_cache_file=True,
            max_epochs=1,
            task_type='generative_reranker',
            dataset=['MTEB/scidocs-reranking#2000'],
            loss_type='pointwise_reranker',
            split_dataset_ratio=0.01,
            tensor_model_parallel_size=2,
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
            train_iters=100,
            eval_iters=5,
            save_interval=5,
            no_save_optim=True,
            no_save_rng=True,
            sequence_parallel=True,
            finetune=True))


if __name__ == '__main__':
    test_embedding()
    # test_reranker()
