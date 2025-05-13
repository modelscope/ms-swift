import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def test_dpo():
    from swift.megatron import megatron_rlhf_main, MegatronRLHFArguments
    megatron_rlhf_main(
        MegatronRLHFArguments(
            load='Qwen2.5-3B-Instruct-mcore',
            dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#1000'],
            tensor_model_parallel_size=2,
            train_iters=100,
            eval_iters=5,
            finetune=True))


if __name__ == '__main__':
    test_dpo()
