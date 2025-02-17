import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_cli(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(model='Qwen/Qwen2-VL-7B-Instruct', infer_backend=infer_backend)
    infer_main(args)


def test_cli_jinja(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(model='Qwen/Qwen2-VL-7B-Instruct', infer_backend=infer_backend, template_backend='jinja')
    infer_main(args)


def test_dataset(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(
        model='Qwen/Qwen2-7B-Instruct',
        infer_backend=infer_backend,
        val_dataset=['AI-ModelScope/alpaca-gpt4-data-zh#10'],
        stream=True)
    infer_main(args)


def test_mllm_dataset(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(
        model='Qwen/Qwen2-VL-7B-Instruct',
        infer_backend=infer_backend,
        val_dataset=['modelscope/coco_2014_caption:validation#1000'],
        stream=True)
    infer_main(args)


def test_dataset_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from swift.llm import infer_main, InferArguments
    args = InferArguments(
        model='Qwen/Qwen2-7B-Instruct', max_batch_size=64, val_dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000'])
    infer_main(args)


def test_dataset_mp_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from swift.llm import infer_main, InferArguments
    args = InferArguments(
        model='Qwen/Qwen2-7B-Instruct', max_batch_size=64, val_dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000'])
    infer_main(args)


def test_emu3_gen(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(
        model='BAAI/Emu3-Gen',
        infer_backend=infer_backend,
        stream=False,
        use_chat_template=False,
        top_k=2048,
        max_new_tokens=40960)
    infer_main(args)


if __name__ == '__main__':
    # test_cli('pt')
    # test_cli_jinja('pt')
    # test_dataset('pt')
    # test_mllm_dataset('pt')
    # test_dataset_ddp()
    # test_dataset_mp_ddp()
    test_emu3_gen('pt')
