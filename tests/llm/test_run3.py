import os
import shutil
import tempfile
import unittest

import json
import numpy as np

from swift.llm import MODEL_MAPPING, load_dataset


class TestRun3(unittest.TestCase):

    def setUp(self):
        print(f'Testing {type(self).__name__}.{self._testMethodName}')
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def load_ds(self, ds):
        train_dataset, val_dataset = load_dataset(
            ds,
            split_dataset_ratio=0.0,
            strict=False,
            num_proc=1,
            model_name=['小黄', 'Xiao Huang'],
            model_author=['魔搭', 'ModelScope'])
        return train_dataset.select(range(min(50, len(train_dataset))))

    # def test_model_load(self):
    #     if os.path.exists('./models.txt'):
    #         with open('./models.txt', 'r') as f:
    #             models = json.load(f)
    #     else:
    #         models = []
    #     for model_name, model_meta in MODEL_MAPPING.items():
    #         meta_requires = model_meta.requires or []
    #         for group in model_meta.model_groups:
    #             model = group.models[0]
    #             if 'skip_test' in (group.tags or []) or model.ms_model_id in models:
    #                 break
    #             requires = meta_requires + (group.requires or [])
    #             for req in requires:
    #                 os.system(f'pip install "{req}"')
    #             if not any(['transformers' in req for req in requires]):
    #                 os.system('pip install transformers -U')
    #             if not any(['accelerate' in req for req in requires]):
    #                 os.system('pip install accelerate -U')
    #             try:
    #                 model_arch_args = ''
    #                 if model_meta.model_arch:
    #                     model_arch_args = f'--model_arch {model_meta.model_arch}'
    #                 cmd = ('PYTHONPATH=. python tests/llm/load_model.py '
    #                        f'--ms_model_id {model.ms_model_id} {model_arch_args}')
    #                 if os.system(cmd) != 0:
    #                     raise RuntimeError()
    #             except Exception:
    #                 passed = False
    #             else:
    #                 passed = True
    #                 models.append(model.ms_model_id)
    #             finally:
    #                 if passed:
    #                     with open('./models.txt', 'w') as f:
    #                         json.dump(models, f)

    # def test_template_load(self):
    #     if os.path.exists('./templates.txt'):
    #         with open('./templates.txt', 'r') as f:
    #             templates = json.load(f)
    #     else:
    #         templates = []
    #     for model_name, model_meta in MODEL_MAPPING.items():
    #         template = model_meta.template
    #         meta_requires = model_meta.requires or []
    #         for group in model_meta.model_groups:
    #             model = group.models[0]
    #             if 'skip_test' in (group.tags or []) or template in templates:
    #                 break
    #             requires = meta_requires + (group.requires or [])
    #             for req in requires:
    #                 os.system(f'pip install "{req}"')
    #             if not any(['transformers' in req for req in requires]):
    #                 os.system('pip install transformers -U')
    #             if not any(['accelerate' in req for req in requires]):
    #                 os.system('pip install accelerate -U')
    #             try:
    #                 cmd = ('PYTHONPATH=. python tests/llm/load_template.py '
    #                        f'--ms_model_id {model.ms_model_id} --template {template}')
    #                 if os.system(cmd) != 0:
    #                     raise RuntimeError()
    #             except Exception:
    #                 import traceback
    #                 print(traceback.format_exc())
    #                 passed = False
    #             else:
    #                 passed = True
    #                 templates.append(template)
    #             finally:
    #                 if passed:
    #                     with open('./templates.txt', 'w') as f:
    #                         json.dump(templates, f)

    @unittest.skip('skip')
    def test_template_compare(self):
        if os.path.exists('./templates.txt'):
            with open('./templates.txt', 'r') as f:
                templates = json.load(f)
        else:
            templates = []
        skip_model_type = {
            'grok', 'deepseek_moe', 'deepseek_v2', 'deepseek_v2_5', 'llama3_1_omni', 'llava_next_qwen_hf',
            'llava1_6_yi', 'llava_next_qwen', 'mixtral', 'codefuse_codellama', 'wizardlm2', 'wizardlm2_awq',
            'openbuddy_deepseek', 'sus', 'openbuddy_mixtral', 'openbuddy_llama', 'dbrx', 'nenotron', 'reflection',
            'xverse_moe', 'qwen2_moe', 'yuan2', 'wizardlm2_moe', 'emu3_gen', 'llava1_6_mistral', 'mplug_owl3_241101',
            'llava1_6_yi_hf'
        }
        for model_name, model_meta in MODEL_MAPPING.items():
            if model_name in skip_model_type:
                continue
            template = model_meta.template
            meta_requires = model_meta.requires or []
            for group in model_meta.model_groups:
                model = group.models[0]
                if 'awq' in model.ms_model_id.lower() or 'gptq' in model.ms_model_id.lower():
                    break
                if template in templates:
                    break
                requires = meta_requires + (group.requires or [])
                for req in requires:
                    os.system(f'pip install "{req}"')
                if not any(['transformers' in req for req in requires]):
                    os.system('pip install transformers -U')
                if not any(['accelerate' in req for req in requires]):
                    os.system('pip install accelerate -U')
                try:
                    cmd = ('CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/llm/load_template.py '
                           f'--ms_model_id {model.ms_model_id} --template {template}')
                    if os.system(cmd) != 0:
                        raise RuntimeError()
                    cmd = (
                        'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/workspace/yzhao/tastelikefeet/swift python tests/llm/load_template.py '  # noqa
                        f'--ms_model_id {model.ms_model_id} --template {template} --new 0')
                    if os.system(cmd) != 0:
                        raise RuntimeError()
                    with open('new_input_ids.txt', 'r') as f:
                        input_ids_new = json.load(f)
                    with open('old_input_ids.txt', 'r') as f:
                        input_ids_old = json.load(f)
                    print('model_id', model.ms_model_id, 'new:', input_ids_new, 'old:', input_ids_old)
                    self.assertTrue(np.allclose(input_ids_new['input_ids'], input_ids_old['input_ids']))
                except Exception:
                    import traceback
                    print(traceback.format_exc())
                    passed = False
                else:
                    passed = True
                    templates.append(template)
                finally:
                    if passed:
                        with open('./templates.txt', 'w') as f:
                            json.dump(templates, f)
                    if os.path.exists('new_input_ids.txt'):
                        os.remove('new_input_ids.txt')
                    if os.path.exists('old_input_ids.txt'):
                        os.remove('old_input_ids.txt')


if __name__ == '__main__':
    unittest.main()
