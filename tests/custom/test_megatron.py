def test_to_megatron():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(model_type='qwen2-0_5b', to_megatron=True))


def test_to_hf():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import export_main, ExportArguments
    export_main(
        ExportArguments(
            model_type='qwen2-0_5b', to_hf=True, ckpt_dir='/mnt/nas2/huangjintao.hjt/work/swift/qwen2-0_5b-tp1-pp1'))


def test_pretrain():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import sft_main, SftArguments, export_main, ExportArguments
    sft_main(SftArguments(model_type='qwen2-0_5b', dataset=['alpaca-zh'], template_type='qwen',
                          train_backend='megatron'))


if __name__ == '__main__':
    test_pretrain()
