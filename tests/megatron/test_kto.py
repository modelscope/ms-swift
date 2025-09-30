import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def test_kto():
    from swift.megatron import megatron_rlhf_main, MegatronRLHFArguments
    megatron_rlhf_main(
        MegatronRLHFArguments(
            load='Qwen2.5-7B-Instruct-mcore',
            rlhf_type='kto',
            train_type='lora',
            load_from_cache_file=True,
            dataset=['AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto#10000'],
            target_modules=['all-linear'],
            tensor_model_parallel_size=2,
            split_dataset_ratio=0.01,
            micro_batch_size=4,
            global_batch_size=16,
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
            eval_interval=10,
            save_interval=10,
            log_interval=1,
            finetune=True,
            max_epochs=1,
            max_length=2048,
            packing=True,
            dataset_num_proc=8,
            cross_entropy_loss_fusion=True,
            sequence_parallel=True,
        ))


if __name__ == '__main__':
    test_kto()
