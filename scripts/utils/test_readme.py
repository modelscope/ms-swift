import os
import re

import torch
from modelscope import snapshot_download

from swift.llm import MODEL_MAPPING


def test_readme():
    for model_type in MODEL_MAPPING.keys():
        model_id = MODEL_MAPPING[model_type]['model_id_or_path']
        model_dir = snapshot_download(model_id, revision='master')
        readme_path = os.path.join(model_dir, 'README.md')
        assert os.path.exists(readme_path)
        with open(readme_path, 'r') as f:
            text = f.read()

        code_list = re.findall(r'```python\n(.+?)\n```', text, re.M | re.S)
        print(f'model_type: {model_type}')
        for code in code_list:
            if 'import' not in code or 'modelscope' not in code:
                continue
            try:
                exec(code)
            except Exception:
                print(code)
                input('[ENTER')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    test_readme()
