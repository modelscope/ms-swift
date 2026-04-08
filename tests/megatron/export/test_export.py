import os

from swift.megatron import MegatronExportArguments, megatron_export_main

os.environ['SWIFT_TEST_CONVERT_PRECISION'] = '1'


def test_to_mcore():
    megatron_export_main(
        MegatronExportArguments(
            model='Qwen/Qwen2.5-7B-Instruct',
            output_dir='Qwen2.5-7B-Instruct-mcore',
            to_mcore=True,
            exist_ok=True,
            tensor_model_parallel_size=2,
            test_convert_precision=True))


def test_to_hf():
    megatron_export_main(
        MegatronExportArguments(
            mcore_model='Qwen3-30B-A3B-mcore',
            to_hf=True,
            exist_ok=True,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=2,
            test_convert_precision=True))


def test_peft_to_mcore():
    megatron_export_main(
        MegatronExportArguments(
            model='Qwen/Qwen3-30B-A3B',
            adapters=['megatron_output/Qwen3-30B-A3B/vx-xxx/checkpoint-xxx-hf'],
            merge_lora=False,
            to_mcore=True,
            exist_ok=True,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=4,
            test_convert_precision=True))


def test_peft_to_hf():
    megatron_export_main(
        MegatronExportArguments(
            mcore_model='Qwen3-30B-A3B-mcore',
            mcore_adapter='megatron_output/Qwen3-30B-A3B/vx-xxx/checkpoint-xxx',
            merge_lora=False,
            to_hf=True,
            exist_ok=True,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            test_convert_precision=True))


if __name__ == '__main__':
    # test_to_mcore()
    test_to_hf()
    # test_peft_to_mcore()
    # test_peft_to_hf()
