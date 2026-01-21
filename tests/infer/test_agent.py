import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 50,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_sft():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift import sft_main, SftArguments
    sft_main(SftArguments(model='Qwen/Qwen2-7B-Instruct', dataset=['iic/ms_agent#2000'], loss_scale='react', **kwargs))


def test_infer():
    from swift import infer_main, InferArguments
    ckpt_dir = 'output/Qwen2-7B-Instruct/vx-xxx/checkpoint-xxx'
    infer_main(InferArguments(adapters=[ckpt_dir]))


if __name__ == '__main__':
    test_sft()
    # test_infer()
