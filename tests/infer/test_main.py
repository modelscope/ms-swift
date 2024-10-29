import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_cli(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(model='qwen/Qwen2-7B-Instruct', infer_backend=infer_backend)
    infer_main(args)

def test_dataset(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(model='qwen/Qwen2-7B-Instruct', infer_backend=infer_backend, val_dataset=['alpaca-zh#100'])
    infer_main(args)

if __name__== '__main__':
    # test_cli('pt')
    test_dataset('pt')
