import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from swift.llm import deploy_main, DeployArguments, infer_main, InferArguments, eval_main, EvalArguments

def test_eval_llm():
    eval_main(EvalArguments(model_type='qwen1half-7b-chat', eval_dataset='ARC_c', infer_backend='lmdeploy'))

def test_eval_vlm():
    eval_main(EvalArguments(model_type='internvl2-2b', eval_dataset='COCO_VAL', infer_backend='lmdeploy'))

if __name__ == '__main__':
    test_eval_vlm()
