import os
from typing import Literal

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'


def test_llm_quant(quant_method: Literal['gptq', 'awq'] = 'awq'):
    from swift import ExportArguments, export_main
    export_main(
        ExportArguments(
            model='Qwen/Qwen2-7B-Instruct',
            quant_bits=4,
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#1000'],
            quant_method=quant_method))


def test_vlm_quant(quant_method: Literal['gptq', 'awq'] = 'awq'):
    from swift import ExportArguments, export_main
    export_main(
        ExportArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            quant_bits=4,
            dataset=['modelscope/coco_2014_caption:validation#1000'],
            quant_method=quant_method))


def test_audio_quant(quant_method: Literal['gptq', 'awq'] = 'awq'):
    from swift import ExportArguments, export_main
    export_main(
        ExportArguments(
            model='Qwen/Qwen2-Audio-7B-Instruct',
            quant_bits=4,
            dataset=['speech_asr/speech_asr_aishell1_trainsets:validation#1000'],
            quant_method=quant_method))


def test_vlm_bnb_quant():
    from swift import ExportArguments, InferArguments, export_main, infer_main
    export_main(ExportArguments(model='Qwen/Qwen2-VL-7B-Instruct', quant_bits=4, quant_method='bnb'))

    # infer_main(InferArguments(ckpt_dir='Qwen/Qwen2-VL-7B-Instruct-bnb-int4'))


def test_bert():
    from swift import ExportArguments, export_main
    output_dir = 'output/swift_test_bert_merged'
    export_main(ExportArguments(adapters='swift/test_bert', merge_lora=True, output_dir=output_dir))
    export_main(
        ExportArguments(model=output_dir, load_data_args=True, quant_bits=4, quant_method='gptq', max_length=512))


def test_reward_model():
    from swift import ExportArguments, export_main

    export_main(
        ExportArguments(
            model='Shanghai_AI_Laboratory/internlm2-1_8b-reward',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#1000'],
            quant_bits=4,
            quant_method='gptq'))


def test_fp8():
    from swift import ExportArguments, InferArguments, export_main, infer_main
    export_main(ExportArguments(model='Qwen/Qwen2.5-3B-Instruct', quant_method='fp8'))
    infer_main(InferArguments(model='Qwen2.5-3B-Instruct-fp8'))


def test_lora_merge_export_minimal():
    from swift import ExportArguments, InferArguments, SftArguments, export_main, infer_main, sft_main
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#20'],
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            save_steps=2,
            split_dataset_ratio=0.01,
            tuner_type='lora',
            logging_steps=1,
            output_dir='output/test_lora_merge_export'))
    last_model_checkpoint = result['last_model_checkpoint']
    merge_output_dir = 'output/test_lora_merge_export_merged'
    export_main(
        ExportArguments(
            adapters=last_model_checkpoint,
            merge_lora=True,
            output_dir=merge_output_dir,
            exist_ok=True,
        ))
    infer_main(InferArguments(model=merge_output_dir, load_data_args=True, max_batch_size=2))


if __name__ == '__main__':
    # test_llm_quant('gptq')
    # test_vlm_quant('gptq')
    # test_audio_quant('gptq')
    # test_vlm_bnb_quant()
    # test_bert()
    # test_reward_model()
    test_fp8()
    # test_lora_merge_export_minimal()
