# Copyright (c) Alibaba, Inc. and its affiliates.
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import custom

from swift.llm import AppUIArguments, ModelType, app_ui_main

if __name__ == '__main__':
    # Please refer to the `infer.sh` for setting the parameters.
    # text-generation
    # args = AppUIArguments(model_type=ModelType.chatglm3_6b_base)
    # or chat
    args = AppUIArguments(model_type=ModelType.qwen_7b_chat_int4)
    # or load from ckpt dir
    # args = AppUIArguments(ckpt_dir='xxx/vx-xxx/checkpoint-xxx')
    app_ui_main(args)
