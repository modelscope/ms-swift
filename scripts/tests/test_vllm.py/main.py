import os
import subprocess

from swift.llm import ModelType

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    model_name_list = ModelType.get_model_name_list()
    success_model_list = []
    fpath = os.path.join(os.path.dirname(__file__), 'utils.py')
    for model_name in model_name_list:
        code = subprocess.run(['python', fpath, '--model_type', model_name])
        if code.returncode == 0:
            success_model_list.append(model_name)
        else:
            print(f'model_name: {model_name} not support vllm.')
    print(success_model_list)
