
def test_export_bin_dataset():
    from swift.llm import export_main, ExportArguments
    export_main(
        ExportArguments(
            model='Qwen/Qwen2.5-7B-Instruct',
            dataset='liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT',
            to_bin_dataset=True,
            dataset_num_proc=32,
            packing=True,
            load_from_cache_file=False,
    ))
    print()

def test_sft():
    from swift.llm import sft_main, TrainArguments
    sft_main(TrainArguments(
        model='Qwen/Qwen2.5-7B-Instruct',
        dataset='liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT#1000',
        dataset_num_proc=2,
        packing=True,
        attn_impl='flash_attn',
        # load_from_cache_file=False,
    ))

if __name__ == '__main__':
    # test_export_bin_dataset()
    test_sft()
# 186426.75
# 13896.07
