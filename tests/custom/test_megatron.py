def test_to_megatron():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(model_type='qwen2-0_5b', to_megatron=True))


def test_to_hf():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(ckpt_dir='qwen2-0_5b-tp1-pp1', to_hf=True))
    # export_main(ExportArguments(ckpt_dir='/mnt/nas2/huangjintao.hjt/work/swift/output/qwen2-0_5b/v24-20240722-114451', to_hf=True, ))


# test_to_megatron()
# test_to_hf()


def test_pretrain():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import sft_main, SftArguments, export_main, ExportArguments
    sft_main(
        SftArguments(
            resume_from_checkpoint='qwen2-0_5b-tp1-pp1',
            dataset=['alpaca-zh'],
            template_type='qwen',
            train_backend='megatron'))


if __name__ == '__main__':
    test_pretrain()
