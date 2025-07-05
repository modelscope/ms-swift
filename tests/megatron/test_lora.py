import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def test_sft():
    from swift.megatron import megatron_sft_main, MegatronTrainArguments
    megatron_sft_main(
        MegatronTrainArguments(
            load='Qwen2.5-3B-Instruct-mcore',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#5000'],
            split_dataset_ratio=0.01,
            tensor_model_parallel_size=2,
            train_type='lora',
            # pipeline_model_parallel_size=2,
            # freeze_parameters_ratio=0.5,
            train_iters=100,
            eval_iters=5,
            save_interval=5,
            no_save_optim=True,
            no_save_rng=True,
            sequence_parallel=True,
            finetune=True))


def test_sft2():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.megatron import megatron_sft_main, MegatronTrainArguments
    megatron_sft_main(
        MegatronTrainArguments(
            load='Qwen2.5-3B-Instruct-mcore',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#5000'],
            split_dataset_ratio=0.01,
            train_iters=100,
            eval_iters=5,
            no_save_optim=True,
            no_save_rng=True,
            finetune=True))


if __name__ == '__main__':
    test_sft()
    # test_sft2()
