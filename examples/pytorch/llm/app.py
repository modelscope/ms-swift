# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import InferArguments, ModelType, TemplateType, gradio_demo

if __name__ == '__main__':
    # Please refer to the `infer.sh` for setting the parameters.
    args = InferArguments(
        model_type=ModelType.qwen_7b_chat_int4,
        sft_type='lora',
        template_type=None,
        ckpt_dir=None,
        eval_human=True,
        quantization_bit=0)
    gradio_demo(args, history_length=10)
