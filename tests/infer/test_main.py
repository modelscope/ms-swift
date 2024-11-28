import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_cli(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(model='qwen/Qwen2-VL-7B-Instruct', infer_backend=infer_backend)
    infer_main(args)


def test_dataset(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(
        model='qwen/Qwen2-7B-Instruct',
        infer_backend=infer_backend,
        val_dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100'])
    infer_main(args)


def test_dataset_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from swift.llm import infer_main, InferArguments
    args = InferArguments(
        model='qwen/Qwen2-7B-Instruct', max_batch_size=64, val_dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000'])
    infer_main(args)


def test_dataset_mp_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from swift.llm import infer_main, InferArguments
    args = InferArguments(
        model='qwen/Qwen2-7B-Instruct', max_batch_size=64, val_dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000'])
    infer_main(args)


if __name__ == '__main__':
    test_cli('pt')
    # test_dataset('vllm')
    # test_dataset_ddp()
    # test_dataset_mp_ddp()
