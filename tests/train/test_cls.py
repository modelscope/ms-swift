import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 50,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from swift import SftArguments, sft_main, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2.5-1.5B-Instruct',
            tuner_type='lora',
            num_labels=2,
            dataset=['DAMO_NLP/jd:cls#2000'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_bert():

    from swift import SftArguments, sft_main, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='answerdotai/ModernBERT-base',
            # model='iic/nlp_structbert_backbone_base_std',
            tuner_type='full',
            num_labels=2,
            dataset=['DAMO_NLP/jd:cls#2000'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(model=last_model_checkpoint, load_data_args=True))


def test_mllm():
    from swift import SftArguments, sft_main, infer_main, InferArguments
    result = sft_main(
        SftArguments(
            model='OpenGVLab/InternVL2-1B',
            tuner_type='lora',
            num_labels=2,
            dataset=['DAMO_NLP/jd:cls#500'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


if __name__ == '__main__':
    # test_llm()
    # test_bert()
    test_mllm()
