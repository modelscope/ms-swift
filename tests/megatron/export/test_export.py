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
            load='Qwen2.5-7B-Instruct-mcore',
            to_hf=True,
            exist_ok=True,
            tensor_model_parallel_size=2,
            test_convert_precision=True))


if __name__ == '__main__':
    # test_to_mcore()
    test_to_hf()
