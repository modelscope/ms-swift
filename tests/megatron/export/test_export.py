from swift.megatron import MegatronExportArguments, megatron_export_main


def test_to_mcore():
    megatron_export_main(
        MegatronExportArguments(
            model='Qwen/Qwen2.5-7B-Instruct',
            save='Qwen2.5-7B-Instruct-mcore',
            to_mcore=True,
            exist_ok=True,
            tensor_model_parallel_size=2,
            test_convert_precision=True))


def test_to_hf():
    megatron_export_main(
        MegatronExportArguments(
            load='Qwen3-30B-A3B-mcore',
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
            adapters=['megatron_output/Qwen3-30B-A3B/vx-xxx-hf'],
            merge_lora=False,
            to_mcore=True,
            exist_ok=True,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=4,
            test_convert_precision=True))


def test_peft_to_hf():
    megatron_export_main(
        MegatronExportArguments(
            load='Qwen3-30B-A3B-mcore',
            adapter_load='megatron_output/Qwen3-30B-A3B/vx-xxx',
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
