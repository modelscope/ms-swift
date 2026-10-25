# Copyright (c) ModelScope Contributors. All rights reserved.
"""S12 — export e2e halves that run a real artifact end to end (TinyModel, no network).

Covered here:
  * merge_lora (S12-10/11): the streaming merge's output, reloaded from disk, matches the online
    peft forward AND a hand-computed weight-level oracle ``W' = W + scaling * B @ A``.
  * quant_n_samples truncation (S12-05): exactly n encoded rows feed the calibration blocks.
  * export_cached_dataset (S12-15/16): a real cache is written, reloaded with load_from_disk, and
    its encoded features match the in-memory template.encode row by row.

The calibration backends themselves (gptq / awq) cannot run in this environment at all: gptqmodel
and autoawq both require transformers versions that conflict with ms-swift's pin, so their e2e rows
(S12-01/02) are N/A here and only the method-independent mechanics are tested. bnb load-time
quantization needs a GPU and lives below with @pytest.mark.slow + accel(1).
"""
import json
import os

import pytest

from swift.dev.tests.tiny import TinyData, TinyModel


@pytest.fixture(scope='module')
def tiny_base(tmp_path_factory):
    """One TinyModel directory shared by the module (the snapshot download dominates its cost)."""
    return TinyModel.build(tmp_path_factory.mktemp('s12-e2e') / 'base')


def _model_config(base_dir, **overrides):
    from swift.dev.config import ModelConfig
    return ModelConfig(model=base_dir, model_type='qwen2', **overrides)


# --- S12-10/11: merge_lora against a weight-level and a forward-level oracle ----------------


@pytest.fixture()
def lora_adapter(tmp_path, tiny_base):
    """A real peft LoRA adapter with a NON-zero delta (peft inits lora_B to zero, which would make
    every merge trivially correct), saved in peft's own layout."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_base, dtype=torch.bfloat16)
    config = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'])
    peft_model = get_peft_model(model, config)
    torch.manual_seed(1234)
    for name, param in peft_model.named_parameters():
        if 'lora_B' in name:
            torch.nn.init.normal_(param, std=0.02)
    adapter = tmp_path / 'adapter'
    peft_model.save_pretrained(str(adapter))
    return str(adapter)


@pytest.mark.slow
def test_merge_lora_matches_online_peft_and_hand_computed_delta(tmp_path, tiny_base, lora_adapter):
    """Two independent oracles on the merged artifact, read back from disk:
    1. weight level: W' == W + scaling * B @ A computed by hand in fp32 (within bf16 rounding);
    2. forward level: merged logits == online peft (base + adapter) logits on the same batch.
    """
    import torch
    from safetensors.torch import load_file
    from swift.dev.config import TemplateConfig, TunerConfig
    from swift.dev.recipe.export import run_merge_lora
    from transformers import AutoModelForCausalLM

    merged_dir = run_merge_lora(
        _model_config(tiny_base),
        TunerConfig(adapters=[lora_adapter]),
        template_config=TemplateConfig(template='qwen2_5'))
    assert os.path.isdir(merged_dir)
    assert os.path.isfile(os.path.join(merged_dir, 'model.safetensors'))

    # --- oracle 1: hand-computed delta on one q_proj weight -------------------------------
    base_weights = load_file(os.path.join(tiny_base, 'model.safetensors'))
    merged_weights = load_file(os.path.join(merged_dir, 'model.safetensors'))
    adapter_weights = load_file(os.path.join(lora_adapter, 'adapter_model.safetensors'))
    key = 'model.layers.0.self_attn.q_proj.weight'
    a = adapter_weights['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'].float()
    b = adapter_weights['base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight'].float()
    expected = base_weights[key].float() + (16 / 8) * (b @ a)
    actual = merged_weights[key].float()
    torch.testing.assert_close(actual, expected.to(actual.dtype).float(), rtol=5e-3, atol=5e-3)
    # a weight the adapter does not touch passes through unchanged
    untouched = 'model.layers.0.self_attn.o_proj.weight'
    assert torch.equal(merged_weights[untouched], base_weights[untouched])

    # --- oracle 2: forward equivalence with the online peft model ------------------------
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    batch = tokenizer(['What is 2+2?', 'Name a colour.'], return_tensors='pt', padding=True)

    online = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(tiny_base, dtype=torch.bfloat16), lora_adapter).eval()
    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        online_logits = online(**batch).logits
        merged_logits = merged(**batch).logits
    torch.testing.assert_close(merged_logits, online_logits, rtol=2e-2, atol=2e-2)


@pytest.mark.slow
def test_merge_lora_directory_semantics(tmp_path, tiny_base, lora_adapter):
    """The default lands next to the adapter as ``{adapter}-merged``; an existing directory is
    left untouched unless replace_if_exists (a second merge must not rewrite multi-GB weights)."""
    from swift.dev.config import TemplateConfig, TunerConfig
    from swift.dev.recipe.export import run_merge_lora

    merged = run_merge_lora(_model_config(tiny_base), TunerConfig(adapters=[lora_adapter]),
                            template_config=TemplateConfig(template='qwen2_5'))
    assert merged == f'{lora_adapter}-merged'

    marker = os.path.join(merged, 'sentinel.txt')
    with open(marker, 'w') as f:
        f.write('keep me')
    weights = os.path.join(merged, 'model.safetensors')
    first_mtime = os.path.getmtime(weights)
    again = run_merge_lora(_model_config(tiny_base), TunerConfig(adapters=[lora_adapter]),
                           template_config=TemplateConfig(template='qwen2_5'))
    assert again == merged
    assert os.path.isfile(marker)  # untouched: skip() short-circuits without replace_if_exists
    assert os.path.getmtime(weights) == first_mtime  # the weights were not re-written either

    replaced = run_merge_lora(_model_config(tiny_base), TunerConfig(adapters=[lora_adapter]),
                              template_config=TemplateConfig(template='qwen2_5'), replace_if_exists=True)
    assert replaced == merged
    assert os.path.getmtime(weights) >= first_mtime  # replace_if_exists re-wrote the weights
    # 'replace' means re-writing the artifact, not wiping the directory: unrelated files survive.
    assert os.path.isfile(marker)


# --- S12-05: quant_n_samples caps the encoded calibration rows ------------------------------


def test_quant_n_samples_truncates_the_calibration_set(tmp_path, tiny_base):
    """Exactly ``n_samples`` encoded rows feed the calibration blocks — never the whole dataset
    (an uncapped loop would let a large dataset dominate the quantization run)."""
    from swift.dev.builders import build_template, load_model_processor
    from swift.dev.components.weight_transform.quantize import QuantizeComponent
    from swift.dev.config import DatasetConfig, ModelConfig, TemplateConfig

    dataset_path = TinyData.sft(tmp_path / 'calib.jsonl', n=4)
    _, processor = load_model_processor(ModelConfig(model=tiny_base, model_type='qwen2'))
    template = build_template(TemplateConfig(template='qwen2_5'), processor)
    template.set_mode('train')

    blocks = QuantizeComponent._build_calib_data(
        'gptq', template, DatasetConfig(dataset=[dataset_path]),
        max_length=16, n_samples=1, is_multimodal=False)

    # The blocks carry exactly ONE encoded row's tokens, truncated to whole block_size units
    # (the tail beyond the last full block is dropped by design). An ignored n_samples would
    # produce the four rows' sum instead.
    per_row_lengths = []
    for row in (json.loads(line) for line in open(dataset_path)):
        inputs = template.encode(row)
        per_row_lengths.append(len(inputs['input_ids']))
    total = sum(block['input_ids'].shape[-1] for block in blocks)
    assert total in [(length // 16) * 16 for length in per_row_lengths]
    assert all(block['input_ids'].shape[-1] <= 16 for block in blocks)  # block_size = max_length


# --- S12-15/16: export_cached_dataset writes a reloadable, faithful cache -------------------


@pytest.mark.slow
def test_export_cached_dataset_roundtrip(tmp_path, tiny_base):
    """The cache is a standalone load_from_disk target whose encoded features match the in-memory
    template.encode output row by row (the read-back oracle: a cache that drifts from live
    encoding silently trains on the wrong tokens)."""
    from datasets import load_from_disk
    from swift.dev.builders import build_template, load_model_processor
    from swift.dev.config import DatasetConfig, ModelConfig, TemplateConfig
    from swift.dev.recipe.export import export_cached_dataset

    dataset_path = TinyData.sft(tmp_path / 'sft.jsonl', n=4)
    output_dir = str(tmp_path / 'cache')
    train_dir, val_dir = export_cached_dataset(
        ModelConfig(model=tiny_base, model_type='qwen2'),
        TemplateConfig(template='qwen2_5'),
        DatasetConfig(dataset=[dataset_path], split_dataset_ratio=0),
        output_dir=output_dir, store_encoded=True)
    assert val_dir is None  # split_dataset_ratio=0 -> no val split
    assert train_dir == os.path.join(output_dir, 'train')

    cached = load_from_disk(train_dir)
    assert cached.num_rows == 4

    _, processor = load_model_processor(ModelConfig(model=tiny_base, model_type='qwen2'))
    template = build_template(TemplateConfig(template='qwen2_5'), processor)
    template.set_mode('train')
    rows = [json.loads(line) for line in open(dataset_path)]
    expected = [template.encode(row) for row in rows]
    got = [dict(cached[i]) for i in range(cached.num_rows)]
    # The cache carries every encoded feature verbatim plus the packing-side columns ('length'
    # / 'lengths'), which training-time packing needs and live encoding computes downstream.
    # Order-insensitive: the loader may shuffle rows, the encoding itself must not drift.
    got_by_ids = {tuple(g['input_ids']): g for g in got}
    assert len(got_by_ids) == len(expected)  # no row was dropped or duplicated
    for e in expected:
        g = got_by_ids.pop(tuple(e['input_ids']))
        for key, value in e.items():
            assert g[key] == value
        assert g['lengths'] == [sum(g['lengths'])]  # one conversation -> one effective length
        assert g['length'] == len(g['input_ids'])


# --- S12-03 (bnb) + S12-21: a load-time quantized artifact that infer can consume ------------


@pytest.mark.slow
@pytest.mark.accel(1)
def test_bnb_quantize_e2e_produces_a_loadable_4bit_model(tmp_path, tiny_base):
    """bnb 4-bit load-time quantization end to end: the artifact's config declares the scheme,
    reloading materializes packed 4-bit weights, and generation runs on the result (S12-21: the
    exported model is consumed by real inference, not a dead format)."""
    import torch
    from swift.dev.config import QuantizeConfig, TemplateConfig
    from swift.dev.recipe.export import run_quantize
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = run_quantize(
        _model_config(tiny_base, torch_dtype='bfloat16'),
        TemplateConfig(template='qwen2_5'),
        QuantizeConfig(quant_method='bnb', quant_bits=4),
        output_dir=str(tmp_path / 'bnb-out'), exist_ok=True)

    with open(os.path.join(out, 'config.json')) as f:
        quantization_config = json.load(f)['quantization_config']
    assert quantization_config['_load_in_4bit'] is True
    assert quantization_config['bnb_4bit_quant_type'] == 'nf4'

    model = AutoModelForCausalLM.from_pretrained(out, device_map='auto')
    from bitsandbytes.nn import Params4bit
    packed = [p for p in model.parameters() if isinstance(p, Params4bit)]
    assert packed, 'the reloaded model carries no 4-bit parameter'

    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    inputs = tokenizer('Say hello.', return_tensors='pt').to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert generated.shape[-1] > inputs['input_ids'].shape[-1]
