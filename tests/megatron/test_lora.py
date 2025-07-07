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


def test_moe():
    pass


def test_convert():
    from swift.llm import export_main, ExportArguments
    # Qwen2.5-3B-Instruct-mcore
    export_main(
        ExportArguments(
            mcore_adapter=
            '/mnt/nas2/huangjintao.hjt/work/llmscope/megatron_output/Qwen2.5-3B-Instruct/v388-20250705-184538',
            to_hf=True,
        ))


def test_parallel():
    pass


def test_embedding():
    pass


def test_modules_to_save():
    pass


def test_resume():
    pass


if __name__ == '__main__':
    # test_sft()
    test_convert()
