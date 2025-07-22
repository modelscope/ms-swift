import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def test_sft():
    from swift.megatron import megatron_sft_main, MegatronTrainArguments
    megatron_sft_main(
        MegatronTrainArguments(
            load='Qwen2.5-3B-Instruct-mcore',
            dataset=['AI-ModelScope/function-calling-chatml#10000'],
            loss_scale='hermes',
            split_dataset_ratio=0.01,
            tensor_model_parallel_size=2,
            train_type='lora',
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
            # pipeline_model_parallel_size=2,
            # freeze_parameters_ratio=0.5,
            train_iters=100,
            modules_to_save=['word_embeddings', 'output_layer'],
            eval_iters=5,
            save_interval=5,
            no_save_optim=True,
            no_save_rng=True,
            sequence_parallel=True,
            finetune=True))


def test_moe():
    from swift.megatron import megatron_sft_main, MegatronTrainArguments
    megatron_sft_main(
        MegatronTrainArguments(
            load='Qwen1.5-MoE-A2.7B-mcore',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#5000'],
            split_dataset_ratio=0.01,
            moe_shared_expert_overlap=True,
            moe_grouped_gemm=True,
            tensor_model_parallel_size=2,
            # expert_model_parallel_size=2,
            train_type='lora',
            recompute_granularity='full',
            modules_to_save=['word_embeddings', 'output_layer'],
            recompute_method='uniform',
            recompute_num_layers=1,
            # pipeline_model_parallel_size=2,
            # freeze_parameters_ratio=0.5,
            train_iters=100,
            eval_iters=5,
            save_interval=5,
            no_save_optim=True,
            no_save_rng=True,
            sequence_parallel=True,
            finetune=True))


def test_convert():
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(
        mcore_adapters=['megatron_output/vx-xxx'],
        to_hf=True,
        test_convert_precision=True,
    ))


def test_embedding():
    pass


def test_resume():
    pass


if __name__ == '__main__':
    test_sft()
    # test_moe()
    # test_convert()
