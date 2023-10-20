if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import ModelType
from swift.llm.run import infer_main, sft_main

if __name__ == '__main__':
    ckpt_dir = sft_main([
        '--model_type', ModelType.qwen_7b_chat_int4, '--eval_steps', '10',
        '--train_dataset_sample', '400', '--predict_with_generate', 'true'
    ])
    print(ckpt_dir)
    infer_main(
        ['--model_type', ModelType.qwen_7b_chat_int4, '--ckpt_dir', ckpt_dir])
