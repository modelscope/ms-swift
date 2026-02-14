import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 5,
    'logging_steps': 1,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
    'model': 'Qwen/Qwen2-0.5B',
    'dataset': ['AI-ModelScope/alpaca-gpt4-data-zh#2000'],
    'val_dataset': ['AI-ModelScope/alpaca-gpt4-data-zh#10'],
    'max_steps': 10,
    'dataset_num_proc': 4,
    'dataloader_num_workers': 4,
    'max_length': 2048,
    # optional
    # 'padding_free': True,
    'packing': True,
    'attn_impl': 'flash_attn',
    # 'streaming': True,
    'sequence_parallel_size': 2,
}


def test_resume_from_checkpoint():
    from swift import sft_main, SftArguments, infer_main, InferArguments
    result = sft_main(SftArguments(**kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    last_model_checkpoint = last_model_checkpoint.replace('checkpoint-10', 'checkpoint-5')
    result2 = sft_main(SftArguments(**kwargs, resume_from_checkpoint=last_model_checkpoint))
    diff = abs(result['log_history'][6]['loss'] - result2['log_history'][6]['loss'])
    print(f'diff: {diff}')
    assert diff < 0.01


def test_resume_from_checkpoint_true():
    """Test that resume_from_checkpoint='true' auto-detects the last checkpoint."""
    from swift import sft_main, SftArguments
    # First run: train and produce checkpoints.
    result = sft_main(SftArguments(**kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    output_dir = result['output_dir']
    # Second run: resume using 'true' with the same output_dir.
    result2 = sft_main(SftArguments(**kwargs, output_dir=output_dir, add_version=False, resume_from_checkpoint='true'))
    # The resumed run should have loaded the last checkpoint (checkpoint-10 here)
    # and therefore its resolved path should match what the first run produced.
    assert result2['last_model_checkpoint'] == last_model_checkpoint
    print('test_resume_from_checkpoint_true passed')


if __name__ == '__main__':
    test_resume_from_checkpoint()
    test_resume_from_checkpoint_true()
