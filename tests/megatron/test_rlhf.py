import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_dpo():
    from swift.megatron import megatron_rlhf_main, MegatronRLHFArguments
    megatron_rlhf_main(
        MegatronRLHFArguments(
            load='Qwen2.5-3B-Instruct-mcore',
            dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#1000'],
            tensor_model_parallel_size=1,
            train_iters=100,
            eval_iters=5,
            finetune=True))


# {'loss': 2.58519292, 'grad_norm': 17.12728882, 'learning_rate': 9.998e-05, 'memory(GiB)': 11.29, 'train_speed(iter/s)': 0.028085, 'rewards/chosen': 0.0, 'rewards/rejected': 0.0, 'rewards/accuracies': 0.0, 'rewards/margins': 0.0, 'logps/chosen': -372.38751221, 'logps/rejected': -485.3223877, 'logits/chosen': -1.5612483, 'logits/rejected': -1.18372846, 'nll_loss': 1.89204574, 'epoch': 0.02, 'global_step/max_steps': '1/100', 'percentage': '1.00%', 'elapsed_time': '34s', 'remaining_time': '56m 45s'}
# {'loss': 2.03561759, 'grad_norm': 1.61355615, 'learning_rate': 9.938e-05, 'memory(GiB)': 11.29, 'train_speed(iter/s)': 0.04982, 'rewards/chosen': 3.43259692, 'rewards/rejected': 0.04113517, 'rewards/accuracies': 1.0, 'rewards/margins': 3.39146185, 'logps/chosen': -420.24295044, 'logps/rejected': -530.99975586, 'logits/chosen': -1.50232077, 'logits/rejected': -0.97954285, 'nll_loss': 1.89556026, 'epoch': 0.08, 'global_step/max_steps': '5/100', 'percentage': '5.00%', 'elapsed_time': '1m 39s', 'remaining_time': '31m 23s'}
# {'loss': 2.58330202, 'grad_norm': 17.195858, 'learning_rate': 9.998e-05, 'memory(GiB)': 10.31, 'train_speed(iter/s)': 0.046415, 'rewards/chosen': 0.0, 'rewards/rejected': 0.0, 'rewards/accuracies': 0.0, 'rewards/margins': 0.0, 'logps/chosen': -372.38418579, 'logps/rejected': -485.07315063, 'logits/chosen': -1.56147575, 'logits/rejected': -1.18395948, 'nll_loss': 1.89015484, 'epoch': 0.02, 'global_step/max_steps': '1/100', 'percentage': '1.00%', 'elapsed_time': '20s', 'remaining_time': '33m 29s'}
# {'loss': 2.03590226, 'grad_norm': 1.62211466, 'learning_rate': 9.938e-05, 'memory(GiB)': 10.32, 'train_speed(iter/s)': 0.058378, 'rewards/chosen': 3.42427087, 'rewards/rejected': 0.01991111, 'rewards/accuracies': 1.0, 'rewards/margins': 3.40435982, 'logps/chosen': -420.37756348, 'logps/rejected': -531.13378906, 'logits/chosen': -1.50198746, 'logits/rejected': -0.97799051, 'nll_loss': 1.89604306, 'epoch': 0.08, 'global_step/max_steps': '5/100', 'percentage': '5.00%', 'elapsed_time': '1m 24s', 'remaining_time': '26m 43s'}


def test_hf():
    from swift.llm import rlhf_main, RLHFArguments
    rlhf_main(
        RLHFArguments(
            model='Qwen/Qwen2.5-3B-Instruct',
            dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#1000'],
            max_steps=100,
            padding_free=True,
            attn_impl='flash_attn',
            train_dataloader_shuffle=False,
            use_logits_to_keep=False,
        ))


if __name__ == '__main__':
    # test_dpo()
    test_hf()
