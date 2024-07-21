def test_to_megatron():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(model_type='qwen2-0_5b', to_megatron=True, check_model_forward=True))


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
    from threading import Thread
    thread = Thread(
        target=export_main,
        args=((ExportArguments(model_type='qwen2-0_5b', to_megatron=True, check_model_forward=True), )))
    thread.start()
    thread.join()
    print()
    # sft_main(SftArguments(model_type='qwen2-7b-instruct', ))


# test_pretrain()


def test_pretrain_raw():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    from swift.llm import get_model_tokenizer
    from swift.llm.megatron import (load_megatron_config, MegatronArguments, convert_megatron_to_hf, get_model_seires,
                                    patch_megatron, model_provider)
    model_type = 'qwen2-0_5b'
    _, tokenizer = get_model_tokenizer(model_type, load_model=False)
    res = load_megatron_config(tokenizer.model_dir)
    res['model_series'] = get_model_seires(model_type)
    res.update({
        'train_iters': 1000,
        'eval_iters': 100,
        'lr_warmup_iters': 100,
        'save': 'output/megatron',
        'tensorboard_dir': 'output/megatron/runs',
        'bf16': True,
        'load': '/mnt/nas2/huangjintao.hjt/work/swift/qwen2-0_5b-tp1-pp1',
    })
    megatron_args = MegatronArguments(**res)
    extra_args = megatron_args.parse_to_megatron()
    extra_args['dataset'] = 'alpaca-zh#10000'
    extra_args['template_type'] = 'default-generation'
    from swift.llm.utils.megatron_utils import forward_step, train_valid_test_datasets_provider
    from megatron.core.enums import ModelType
    from megatron.training import pretrain

    train_valid_test_datasets_provider.is_distributed = True
    patch_megatron(tokenizer)
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults=extra_args)


# test_pretrain()
