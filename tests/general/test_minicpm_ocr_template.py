# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
import torch
from PIL import Image
from types import SimpleNamespace

from swift.model import MODEL_MAPPING, get_model_processor
from swift.template import TEMPLATE_MAPPING, get_template


@pytest.mark.parametrize('model_type, expected_mode', [('minicpm_ocr', '4x'), ('minicpmv4_6', '16x')])
def test_default_visual_compression(model_type, expected_mode, monkeypatch):
    monkeypatch.delenv('DOWNSAMPLE_MODE', raising=False)
    meta = TEMPLATE_MAPPING[model_type]
    template = meta.template_cls(processor=None, template_meta=meta)
    template.model_meta = MODEL_MAPPING[model_type]
    template.init_env_args()
    assert template.downsample_mode == expected_mode


@pytest.mark.parametrize('downsample_mode', ['4x', '16x'])
@pytest.mark.parametrize('adapter', [False, True])
def test_training_freezes_unused_vit_merger(downsample_mode, adapter, monkeypatch):
    from swift.pipelines.train import tuner

    model = torch.nn.Module()
    model.vision_tower = torch.nn.Module()
    model.vision_tower.vit_merger = torch.nn.Linear(2, 2)
    model.language_model = torch.nn.Linear(2, 2)

    def prepare_adapter(args, model, **kwargs):
        model.vision_tower.vit_merger.lora = torch.nn.Linear(2, 2)
        model.language_model.lora = torch.nn.Linear(2, 2)
        wrapper = torch.nn.Module()
        wrapper.base_model = model
        return wrapper

    monkeypatch.setattr(tuner, 'prepare_adapter', prepare_adapter)
    monkeypatch.setattr(tuner, 'is_deepspeed_zero3_enabled', lambda: False)
    template_meta = TEMPLATE_MAPPING['minicpm_ocr']
    template = template_meta.template_cls(processor=None, template_meta=template_meta)
    template.downsample_mode = downsample_mode
    args = SimpleNamespace(
        use_liger_kernel=False,
        is_adapter=adapter,
        tuner_type='lora' if adapter else 'full',
        tuner_backend='peft',
        resume_from_checkpoint=None,
        adapters=[],
        freeze_parameters_ratio=0,
        freeze_parameters=[],
        freeze_parameters_regex=None,
        trainable_parameters=[],
        trainable_parameters_regex=None,
        use_galore=False)
    prepared = tuner.TunerMixin.prepare_model(args, model, template=template)
    parameters = dict(prepared.named_parameters())
    merger_parameters = [p for n, p in parameters.items() if 'vision_tower.vit_merger' in n]
    assert merger_parameters
    if adapter:
        merger_parameters = [p for n, p in parameters.items() if 'vision_tower.vit_merger.lora' in n]
    assert all(p.requires_grad == (downsample_mode == '16x') for p in merger_parameters)
    assert any(p.requires_grad for n, p in parameters.items() if 'language_model' in n)


def test_native_ocr_prompt_and_supervision(monkeypatch):
    model_path = os.getenv('MINICPM_OCR_MODEL_PATH')
    if not model_path:
        pytest.skip('Set MINICPM_OCR_MODEL_PATH to test local processor assets without loading weights.')
    monkeypatch.delenv('DOWNSAMPLE_MODE', raising=False)
    _, processor = get_model_processor(model_path, model_type='minicpm_ocr', load_model=False, use_hf=True)
    template = get_template(processor, loss_scale='ignore_empty_think', add_non_thinking_prefix=True)
    template.set_mode('train')
    image = Image.new('RGB', (224, 112))
    encoded = template.encode({
        'messages': [{
            'role': 'user',
            'content': '<image>Text Recognition:'
        }, {
            'role': 'assistant',
            'content': 'ABC'
        }],
        'images': [image]
    })
    prompt = processor.tokenizer.apply_chat_template([{
        'role': 'user',
        'content': '<|image_pad|>\nText Recognition:'
    }],
                                                     tokenize=False,
                                                     add_generation_prompt=True,
                                                     enable_thinking=False)
    expected = processor(images=[image], text=prompt + 'ABC<|im_end|>\n', add_special_tokens=False, return_tensors='pt')
    assert encoded['input_ids'] == expected['input_ids'][0].tolist()
    assert torch.equal(encoded['target_sizes'], expected['target_sizes'])
    assert torch.equal(encoded['pixel_values'], expected['pixel_values'])
    prefix = processor(images=[image], text=prompt, add_special_tokens=False, return_tensors='pt')
    prefix_length = prefix['input_ids'].shape[1]
    assert encoded['labels'][:prefix_length] == [-100] * prefix_length
    assert encoded['labels'][prefix_length:] == encoded['input_ids'][prefix_length:]
    assert template.data_collator([encoded])['downsample_mode'] == '4x'
