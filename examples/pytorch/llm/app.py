# Copyright (c) Alibaba, Inc. and its affiliates.
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import custom

from swift.llm import InferArguments, ModelType, app_ui_main

if __name__ == '__main__':
    # Please refer to the `infer.sh` for setting the parameters.
    # text-generation
    # args = InferArguments(model_type=ModelType.chatglm3_6b_base)
    # or chat
    args = InferArguments(model_type=ModelType.qwen_7b_chat_int4)
    # or load from ckpt dir
    # args = InferArguments(ckpt_dir='xxx/vx_xxx/checkpoint-xxx')
    app_ui_main(args)
