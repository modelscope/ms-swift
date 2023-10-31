# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import InferArguments, ModelType
from swift.llm.run import web_ui_main

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
    args = InferArguments(ckpt_dir='xxx', load_args_from_ckpt_dir=True)
    web_ui_main(args)
