import os
import torch
import unittest

from swift.utils import get_device

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def test_qwen2():
    import os

    from swift.model import get_model_processor
    model, tokenizer = get_model_processor('Qwen/Qwen2-7B-Instruct', load_model=False)
    print(f'model: {model}, tokenizer: {tokenizer}')
    # test hf
    model, tokenizer = get_model_processor('Qwen/Qwen2-7B-Instruct', load_model=False, use_hf=True)

    model, tokenizer = get_model_processor(
        'Qwen/Qwen2-7B-Instruct', torch_dtype=torch.float32, device_map=get_device(), attn_impl='flash_attn')
    print(f'model: {model}, tokenizer: {tokenizer}')


def test_modelscope_hub():
    from swift.model import get_model_processor
    model, tokenizer = get_model_processor('Qwen/Qwen2___5-Math-1___5B-Instruct/', load_model=False)


class TestMolmo2Registration(unittest.TestCase):

    def test_registration(self):
        from swift.model import MODEL_MAPPING, MLLMModelType
        from swift.template import TEMPLATE_MAPPING, TemplateType

        model_meta = MODEL_MAPPING[MLLMModelType.molmo2]
        self.assertEqual(model_meta.template, TemplateType.molmo2)
        self.assertEqual(model_meta.model_arch.arch_name, 'molmo')
        self.assertIn('Molmo2ForConditionalGeneration', model_meta.architectures)

        hf_model_ids = []
        for group in model_meta.model_groups:
            for model in group.models:
                hf_model_ids.append(model.hf_model_id)

        self.assertIn('allenai/Molmo2-4B', hf_model_ids)
        self.assertIn('allenai/Molmo2-8B', hf_model_ids)
        self.assertIn('allenai/Molmo2-O-7B', hf_model_ids)
        self.assertIn(TemplateType.molmo2, TEMPLATE_MAPPING)


if __name__ == '__main__':
    test_qwen2()
    # test_modelscope_hub()
