import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_vit_lr():
    # https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune
    from swift.llm import sft_main, TrainArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-VL-7B-Instruct',
            dataset=['AI-ModelScope/LaTeX_OCR#20000'],
            split_dataset_ratio=0.01,
            vit_lr=2e-5,
            learning_rate=1e-5,
            aligner_lr=1e-4,
            freeze_llm=False,
            freeze_vit=False,
            freeze_aligner=False))


if __name__ == '__main__':
    test_vit_lr()
