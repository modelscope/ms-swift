# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (InferArguments, ModelType, TemplateType,
                       gradio_chat_demo, gradio_generation_demo)

if __name__ == '__main__':
    # Please refer to the `infer.sh` for setting the parameters.
    # text-generation
    args = InferArguments(
        model_type=ModelType.chatglm3_6b_base,
        sft_type='lora',
        template_type=None,
        ckpt_dir=None,
        quantization_bit=4)
    # or chat
    args = InferArguments(
        model_type=ModelType.qwen_7b_chat_int4,
        sft_type='lora',
        template_type=None,
        ckpt_dir=None,
        quantization_bit=0)
    # or load_from_ckpt_dir
    args = InferArguments(ckpt_dir='xxx', load_args_from_ckpt_dir=True)
    if args.template_type.endswith('generation'):
        gradio_generation_demo(args)
    else:
        gradio_chat_demo(args, history_length=10)
