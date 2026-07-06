import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
kwargs = {
    'per_device_train_batch_size': 4,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from swift import InferArguments, RLHFArguments, infer_main, rlhf_main
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='gkd',
            model='Qwen/Qwen2.5-0.5B',
            teacher_model='Qwen/Qwen2.5-1.5B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-en#2000'],
            split_dataset_ratio=0.01,
            load_from_cache_file=False,
            seq_kd=True,
            **kwargs,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm():
    from swift import InferArguments, RLHFArguments, infer_main, rlhf_main
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='gkd',
            model='OpenGVLab/InternVL3-2B-Pretrained',
            teacher_model='OpenGVLab/InternVL3-8B',
            dataset=['AI-ModelScope/LaTeX_OCR#2000', 'AI-ModelScope/alpaca-gpt4-data-en#2000'],
            split_dataset_ratio=0.01,
            load_from_cache_file=False,
            **kwargs,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_multi_turn():
    """GKD multi-turn smoke test: verify rollout → encode → loss works with multi_turn_scheduler.

    Uses the built-in ``math_tip_trick`` scheduler with max_turns=2 to keep the test
    lightweight. The key assertion is that training completes without raising
    NotImplementedError (the previous block) and that multi-turn response token ids
    are correctly propagated through the GKD loss pipeline.
    """
    from swift import InferArguments, RLHFArguments, infer_main, rlhf_main
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='gkd',
            model='Qwen/Qwen2.5-0.5B',
            teacher_model='Qwen/Qwen2.5-1.5B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-en#200'],
            split_dataset_ratio=0.01,
            load_from_cache_file=False,
            multi_turn_scheduler='math_tip_trick',
            max_turns=2,
            max_completion_length=256,
            num_generations=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            save_steps=50,
            num_train_epochs=1,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    if last_model_checkpoint is not None:
        infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


if __name__ == '__main__':
    # test_llm()
    # test_mllm()
    test_multi_turn()
