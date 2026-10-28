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


@pytest.mark.parametrize('model_type', ['minicpm_ocr', 'minicpmv4_6'])
@pytest.mark.parametrize('downsample_mode', [None, '4x', '16x'])
@pytest.mark.parametrize('adapter', [False, True])
def test_training_freezes_unused_vit_merger(model_type, downsample_mode, adapter, monkeypatch):
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
    if downsample_mode is None:
        monkeypatch.delenv('DOWNSAMPLE_MODE', raising=False)
    else:
        monkeypatch.setenv('DOWNSAMPLE_MODE', downsample_mode)
    args = SimpleNamespace(
        model_type=model_type,
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
    prepared = tuner.TunerMixin.prepare_model(args, model)
    parameters = dict(prepared.named_parameters())
    merger_parameters = [p for n, p in parameters.items() if 'vision_tower.vit_merger' in n]
    assert merger_parameters
    if adapter:
        merger_parameters = [p for n, p in parameters.items() if 'vision_tower.vit_merger.lora' in n]
    should_freeze = model_type == 'minicpm_ocr' and downsample_mode != '16x'
    assert all(p.requires_grad != should_freeze for p in merger_parameters)
    assert any(p.requires_grad for n, p in parameters.items() if 'language_model' in n)


@pytest.mark.parametrize('template_type, frozen', [('minicpm_ocr', True), ('minicpmv4_6', False)])
def test_ocr_hook_uses_selected_template_default(template_type, frozen, monkeypatch):
    from swift.tuner_plugin.minicpm_ocr import prepare_minicpm_ocr_model

    monkeypatch.delenv('DOWNSAMPLE_MODE', raising=False)
    args = SimpleNamespace(model_type='minicpm_ocr', template_meta=TEMPLATE_MAPPING[template_type])
    model = torch.nn.Module()
    model.vision_tower = torch.nn.Module()
    model.vision_tower.vit_merger = torch.nn.Linear(2, 2)
    prepare_minicpm_ocr_model(args, model)
    assert all(p.requires_grad != frozen for p in model.vision_tower.vit_merger.parameters())


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
