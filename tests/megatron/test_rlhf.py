import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def test_dpo():
    from swift.megatron import megatron_rlhf_main, MegatronRLHFArguments
    megatron_rlhf_main(
        MegatronRLHFArguments(
            load='Qwen2.5-3B-Instruct-mcore',
            dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#10000'],
            split_dataset_ratio=0.01,
            micro_batch_size=16,
            tensor_model_parallel_size=2,
            eval_interval=5,
            log_interval=1,
            finetune=True,
            max_epochs=1,
        ))


def test_hf():
    from swift.llm import rlhf_main, RLHFArguments
    rlhf_main(
        RLHFArguments(
            model='Qwen/Qwen2.5-3B-Instruct',
            dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#1000'],
            split_dataset_ratio=0.01,
            max_steps=100,
            padding_free=True,
            attn_impl='flash_attn',
            train_dataloader_shuffle=False,
            use_logits_to_keep=False,
        ))


if __name__ == '__main__':
    test_dpo()
    # test_hf()
