import os


def test_eval_llm():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import eval_main, EvalArguments
    eval_main(EvalArguments(model_type='qwen1half-7b-chat', eval_dataset='ARC_c', infer_backend='lmdeploy'))


def test_eval_vlm():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import eval_main, EvalArguments
    eval_main(EvalArguments(model_type='internvl2-2b', eval_dataset='COCO_VAL', infer_backend='lmdeploy'))


def test_pt():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import pt_main, PtArguments
    pt_main(PtArguments(model_type='qwen-1_8b-chat', dataset='alpaca-zh', sft_type='lora', tuner_backend='swift'))


if __name__ == '__main__':
    # test_eval_llm()
    # test_eval_vlm()
    test_pt()
