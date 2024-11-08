import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_llm_quant():
    from swift.llm import export_main, ExportArguments
    model = 'qwen/Qwen2-7B-Instruct'
    # model_type = 'qwen-7b-chat'
    export_main(
        ExportArguments(
            model=model,
            quant_bits=4,
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#1000'],
            quant_method='awq'))
    print()


def test_vlm_quant():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import export_main, ExportArguments
    # model_type = 'florence-2-large-ft'
    # model_type = 'got-ocr2'
    # model_type = 'llama3_2-11b-vision-instruct'
    # model_type = 'llava-next-video-7b-instruct'
    # model_type = 'pixtral-12b'
    # model_type = 'internvl2-8b'
    # model_type = 'qwen-vl-chat'
    # model_type = 'yi-vl-6b-chat'
    # model_type = 'mplug-owl3-7b-chat'
    # model_type = 'minicpm-v-v2_6-chat'
    # model_type = 'llava1_6-yi-34b-instruct'
    # model_type = 'qwen2-vl-2b-instruct'
    # model_type = 'qwen2-audio-7b-instruct'
    model_type = 'ovis1_6-gemma2-9b'
    # model_type = 'llama3_2-8b-omni'
    # model_type = 'llava1_6-vicuna-7b-instruct'
    # model_type = 'llava1_5-7b-instruct'
    # model_type = 'qwen2-vl-2b-instruct'
    # model_type = 'qwen-7b-chat'
    # model_type = 'emu3-chat'
    # export_main(
    #     ExportArguments(
    #         model_type=model_type, quant_bits=4, dataset=['alpaca-zh#1000'], quant_method='awq'))
    export_main(
        ExportArguments(  # , 'coco-en-mini#1000'
            model_type=model_type,
            quant_bits=4,
            dataset=['alpaca-zh#1000', 'coco-en-mini#200'],
            quant_method='gptq',
            quant_device_map='auto'))
    print()



if __name__ == '__main__':
    test_llm_quant()
