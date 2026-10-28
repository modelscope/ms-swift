# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline OLMo 3 registration, masking and SFT checks using randomly initialized tiny models.

Run the same file on Ascend to enable the optional BF16 LoRA training smoke test.
No model weights or tokenizers are downloaded by these tests.
"""
import copy
import pytest
import torch
from packaging.requirements import Requirement
from packaging.version import Version
from transformers import AutoConfig
from unittest.mock import Mock

from swift.model import MODEL_MAPPING, LLMModelType, ModelArch, get_model_processor
from swift.model.models import olmo
from swift.model.models.olmo import Olmo3Loader

_olmo3_modeling = pytest.importorskip(
    'transformers.models.olmo3.modeling_olmo3', reason='OLMo 3 requires transformers>=4.57')
Olmo3Config = _olmo3_modeling.Olmo3Config
Olmo3ForCausalLM = _olmo3_modeling.Olmo3ForCausalLM


def _tiny_config(attn_impl='sdpa', *, hidden_size=32, num_heads=4):
    config = Olmo3Config(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=4,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=64,
        layer_types=['sliding_attention'] * 3 + ['full_attention'],
        sliding_window=4,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        rope_theta=500000,
        rope_scaling={
            'rope_type': 'yarn',
            'factor': 8.0,
            'original_max_position_embeddings': 8,
            'attention_factor': 1.2079441541679836,
            'beta_fast': 32.0,
            'beta_slow': 1.0,
        },
    )
    config._attn_implementation = attn_impl
    return config


def _batch(length, padding='none', device='cpu'):
    input_ids = torch.arange(3, 3 + 2 * length, device=device).reshape(2, length)
    attention_mask = torch.ones_like(input_ids)
    if padding != 'none':
        pad_len = max(1, length // 3)
        pad_slice = slice(-pad_len, None) if padding == 'right' else slice(0, pad_len)
        attention_mask[1, pad_slice] = 0
        input_ids[1, pad_slice] = 0
    # Match the collator's per-example position IDs, including left padding.
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    labels = input_ids.clone().masked_fill(attention_mask == 0, -100)
    # A padded query must not predict the first real token of a left-padded sample.
    first_real_token = attention_mask.argmax(-1)
    labels[torch.arange(labels.shape[0], device=device), first_real_token] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'labels': labels,
    }


def _capture_attention_masks(model, batch):
    captured = {}
    handles = []

    def capture(layer_idx):

        def hook(module, args, kwargs):
            mask = kwargs.get('attention_mask')
            captured[layer_idx] = mask.detach().clone() if mask is not None else None

        return hook

    for layer_idx in (0, 3):
        handles.append(model.model.layers[layer_idx].self_attn.register_forward_pre_hook(
            capture(layer_idx), with_kwargs=True))
    try:
        with torch.no_grad():
            model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def test_olmo3_registration():
    meta = MODEL_MAPPING[LLMModelType.olmo3]
    assert meta.loader is Olmo3Loader
    assert meta.template == 'olmo3'
    assert meta.model_arch.arch_name == ModelArch.llama
    assert 'Olmo3ForCausalLM' in meta.architectures
    requirements = [Requirement(requirement) for requirement in meta.requires]
    assert any(requirement.name == 'transformers' and Version('4.57.0') in requirement.specifier
               and Version('4.56.9') not in requirement.specifier for requirement in requirements)
    model_ids = [model.hf_model_id for group in meta.model_groups for model in group.models]
    assert 'allenai/Olmo-3-7B-Think' in model_ids


def test_auto_config_recognizes_local_olmo3(tmp_path):
    config = _tiny_config()
    config.architectures = ['Olmo3ForCausalLM']
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path, local_files_only=True, trust_remote_code=False)
    assert isinstance(loaded, Olmo3Config)
    assert loaded.model_type == 'olmo3'
    assert loaded.layer_types == config.layer_types
    assert loaded.sliding_window == 4
    assert loaded.rope_scaling == config.rope_scaling


@pytest.mark.parametrize('feature', ['padding_free', 'packing'])
def test_sft_arguments_require_flash_attention_for_padding_free(tmp_path, feature):
    from swift import SftArguments

    model_dir = tmp_path / 'model'
    config = _tiny_config()
    config.architectures = ['Olmo3ForCausalLM']
    config.save_pretrained(model_dir)
    with pytest.raises(ValueError, match='requires a flash attention implementation'):
        SftArguments(
            model=str(model_dir),
            template='olmo3',
            dataset=['unused.jsonl'],
            output_dir=str(tmp_path / 'output'),
            torch_dtype='float32',
            fp16=False,
            bf16=False,
            attn_impl='sdpa',
            **{feature: True})


@pytest.mark.parametrize('model_type,sp_size,template', [('olmo3', 1, 'olmo3'), ('olmo3', 2, 'olmo3'),
                                                         ('olmo3', 2, 'default'), ('llama', 2, 'default')])
def test_sft_arguments_keep_supported_model_configurations(tmp_path, model_type, sp_size, template):
    from transformers import LlamaConfig

    from swift import SftArguments

    model_dir = tmp_path / 'model'
    config = _tiny_config() if model_type == 'olmo3' else LlamaConfig()
    config.architectures = ['Olmo3ForCausalLM' if model_type == 'olmo3' else 'LlamaForCausalLM']
    config.save_pretrained(model_dir)
    args = SftArguments(
        model=str(model_dir),
        model_type=model_type,
        template=template,
        dataset=['unused.jsonl'],
        output_dir=str(tmp_path / 'output'),
        torch_dtype='float32',
        fp16=False,
        bf16=False,
        sequence_parallel_size=sp_size)
    assert args.model_type == model_type
    assert args.sequence_parallel_size == sp_size
    assert not args.padding_free
    assert not args.packing


@pytest.mark.parametrize('source', ['local', 'modelscope', 'huggingface'])
def test_swift_loads_tiny_olmo3_checkpoint(tmp_path, monkeypatch, source):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(10)
    model = Olmo3ForCausalLM(_tiny_config()).eval()
    model.save_pretrained(tmp_path)
    vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    vocab.update({f'token_{token_id}': token_id for token_id in range(4, model.config.vocab_size)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='<unk>')),
        pad_token='<pad>',
        bos_token='<bos>',
        eos_token='<eos>',
        unk_token='<unk>')
    tokenizer.save_pretrained(tmp_path)
    model_id_or_path = str(tmp_path)
    use_hf = source == 'huggingface'
    if source != 'local':
        # Exercise Hub selection and checkpoint loading without network downloads.
        model_id_or_path = 'allenai/Olmo-3-7B-Think'
        hub = Mock()
        hub.download_model.return_value = str(tmp_path)
        get_hub = Mock(return_value=hub)
        monkeypatch.setattr('swift.hub.get_hub', get_hub)
    loaded, processor = get_model_processor(
        model_id_or_path,
        use_hf=use_hf,
        torch_dtype=torch.float32,
        device_map='cpu',
        attn_impl='sdpa',
        download_model=source != 'local',
        model_kwargs={'local_files_only': True})
    if source != 'local':
        get_hub.assert_called_once_with(use_hf)
        hub.download_model.assert_called_once()
        assert hub.download_model.call_args.args[0] == model_id_or_path
        assert '*.safetensors' not in hub.download_model.call_args.args[2]
    assert isinstance(loaded, Olmo3ForCausalLM)
    assert loaded.model_info.model_type == LLMModelType.olmo3
    assert loaded.config._attn_implementation == 'sdpa'
    assert processor.pad_token_id == 0
    batch = _batch(8, 'right')
    with torch.no_grad():
        torch.testing.assert_close(loaded(**batch).logits, model(**batch).logits, rtol=0, atol=0)


@pytest.mark.parametrize('requested,expected', [(None, 'sdpa'), ('sdpa', 'sdpa'), ('eager', 'eager')])
def test_npu_loader_keeps_supported_attention(monkeypatch, requested, expected):
    monkeypatch.setattr(olmo, 'is_torch_npu_available', lambda: True)
    loader = object.__new__(Olmo3Loader)
    loader.attn_impl = requested
    loader.attn_impl_keys = None
    config = _tiny_config('eager')
    loader._update_attn_impl(config)
    assert config._attn_implementation == expected


@pytest.mark.parametrize('attn_impl', ['flash_attn', 'flash_attention_2', 'flash_attention_3', 'flash_attention_4'])
def test_npu_loader_rejects_attention_that_loses_sliding_window(monkeypatch, attn_impl):
    monkeypatch.setattr(olmo, 'is_torch_npu_available', lambda: True)
    loader = object.__new__(Olmo3Loader)
    loader.attn_impl = attn_impl
    loader.attn_impl_keys = None
    with pytest.raises(ValueError):
        loader._update_attn_impl(_tiny_config())


def test_non_npu_loader_keeps_flash_attention(monkeypatch):
    monkeypatch.setattr(olmo, 'is_torch_npu_available', lambda: False)
    loader = object.__new__(Olmo3Loader)
    loader.attn_impl = 'flash_attn'
    loader.attn_impl_keys = None
    config = _tiny_config()
    loader._update_attn_impl(config)
    assert config._attn_implementation == 'flash_attention_2'


@pytest.mark.parametrize('length', [3, 4, 5, 8])
@pytest.mark.parametrize('padding', ['none', 'right', 'left'])
def test_eager_and_sdpa_forward_backward_match(length, padding):
    torch.manual_seed(42)
    eager = Olmo3ForCausalLM(_tiny_config('eager')).eval()
    sdpa = Olmo3ForCausalLM(_tiny_config('sdpa')).eval()
    sdpa.load_state_dict(eager.state_dict())
    batch = _batch(length, padding)
    eager_output = eager(**batch)
    sdpa_output = sdpa(**batch)
    valid = batch['attention_mask'].bool()
    torch.testing.assert_close(sdpa_output.logits[valid], eager_output.logits[valid], rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(sdpa_output.loss, eager_output.loss, rtol=2e-5, atol=2e-6)
    eager_output.loss.backward()
    sdpa_output.loss.backward()
    for (name, eager_param), (sdpa_name, sdpa_param) in zip(eager.named_parameters(), sdpa.named_parameters()):
        assert name == sdpa_name
        assert eager_param.grad is not None, name
        assert sdpa_param.grad is not None, name
        assert torch.isfinite(sdpa_param.grad).all(), name
        torch.testing.assert_close(sdpa_param.grad, eager_param.grad, rtol=3e-4, atol=3e-6, msg=name)


@pytest.mark.parametrize('attn_impl', ['eager', 'sdpa'])
@pytest.mark.parametrize('length', [3, 4, 5, 8])
@pytest.mark.parametrize('padding', ['none', 'right', 'left'])
def test_native_masks_preserve_sliding_window_and_padding(attn_impl, length, padding):
    model = Olmo3ForCausalLM(_tiny_config(attn_impl)).eval()
    batch = _batch(length, padding)
    masks = _capture_attention_masks(model, batch)
    indices = torch.arange(length)
    causal = indices[None, :] <= indices[:, None]
    for layer_idx, window in ((0, 4), (3, None)):
        expected = causal
        if window is not None:
            expected = expected & (indices[None, :] > indices[:, None] - window)
        expected = expected[None, None, :, :] & batch['attention_mask'][:, None, None, :].bool()
        mask = masks[layer_idx]
        if mask is None:
            # SDPA may omit a purely causal mask when every token is valid.
            assert attn_impl == 'sdpa'
            assert batch['attention_mask'].all()
            assert window is None or length <= window
            actual = causal[None, None, :, :].expand(2, 1, -1, -1)
        else:
            actual = mask if mask.dtype == torch.bool else mask == 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if length > 4:
        sliding = masks[0] if masks[0].dtype == torch.bool else masks[0] == 0
        assert not sliding[0, 0, -1, length - 5]
        assert sliding[0, 0, -1, length - 4]


@pytest.mark.parametrize('attn_impl', ['eager', 'sdpa'])
def test_sliding_attention_excludes_distant_tokens_but_full_attention_keeps_them(attn_impl):
    torch.manual_seed(7)
    model = Olmo3ForCausalLM(_tiny_config(attn_impl)).eval()
    batch = _batch(8)
    masks = _capture_attention_masks(model, batch)
    hidden_states = torch.randn(2, 8, model.config.hidden_size)
    perturbed = hidden_states.clone()
    # Token 3 is just outside the last query's window [4, 5, 6, 7].
    perturbed[:, 3, :] += torch.randn_like(perturbed[:, 3, :]) * 3
    for layer_idx, attention_type in ((0, 'sliding_attention'), (3, 'full_attention')):
        attention = model.model.layers[layer_idx].self_attn
        if hasattr(model.model, 'rotary_embs'):
            position_embeddings = model.model.rotary_embs[attention_type](hidden_states, batch['position_ids'])
        else:
            position_embeddings = model.model.rotary_emb(hidden_states, batch['position_ids'], attention_type)
        kwargs = {
            'position_embeddings': position_embeddings,
            'attention_mask': masks[layer_idx],
            'position_ids': batch['position_ids'],
        }
        with torch.no_grad():
            original = attention(hidden_states=hidden_states, **kwargs)[0][:, -1]
            changed = attention(hidden_states=perturbed, **kwargs)[0][:, -1]
        if attention_type == 'sliding_attention':
            torch.testing.assert_close(original, changed, rtol=0, atol=0)
        else:
            assert not torch.allclose(original, changed, rtol=1e-5, atol=1e-7)


def _lora_model(config, device='cpu', dtype=torch.float32, gradient_checkpointing=False):
    from peft import LoraConfig, TaskType, get_peft_model

    base_model = Olmo3ForCausalLM(copy.deepcopy(config)).to(device=device, dtype=dtype)
    model = get_peft_model(
        base_model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0.0, target_modules='all-linear'))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    return model.train()


def _run_lora_steps(model, batch, steps):
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all('lora_' in name for name in trainable)
    assert any('q_proj' in name for name in trainable)
    assert any('gate_proj' in name for name in trainable)
    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-3, foreach=False)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        gradients = [parameter.grad for parameter in trainable.values() if parameter.grad is not None]
        assert len(gradients) == len(trainable)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
        assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
        optimizer.step()
    assert any(not torch.equal(before[name], parameter.detach()) for name, parameter in trainable.items())


def test_tiny_olmo3_lora_sft_step():
    torch.manual_seed(8)
    model = _lora_model(_tiny_config())
    _run_lora_steps(model, _batch(8, 'right'), steps=1)


def _npu_available():
    try:
        import torch_npu  # noqa: F401
        return hasattr(torch, 'npu') and torch.npu.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _npu_available(), reason='Ascend NPU is not available')
def test_npu_sdpa_matches_eager_with_sliding_window_and_padding():
    """Check the actual NPU SDPA backend respects both local attention and padded keys."""
    torch.manual_seed(42)
    eager = Olmo3ForCausalLM(_tiny_config('eager', hidden_size=64, num_heads=2))
    sdpa = Olmo3ForCausalLM(_tiny_config('sdpa', hidden_size=64, num_heads=2))
    sdpa.load_state_dict(eager.state_dict())
    eager = eager.to(device='npu:0', dtype=torch.bfloat16).eval()
    sdpa = sdpa.to(device='npu:0', dtype=torch.bfloat16).eval()
    batch = _batch(8, 'right', device='npu:0')
    eager_output = eager(**batch)
    sdpa_output = sdpa(**batch)
    valid = batch['attention_mask'].bool()
    # Fused SDPA and eager differ in BF16 accumulation and softmax rounding.
    torch.testing.assert_close(
        sdpa_output.logits[valid].float(), eager_output.logits[valid].float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sdpa_output.loss, eager_output.loss, rtol=2e-2, atol=2e-2)
    eager_output.loss.backward()
    sdpa_output.loss.backward()
    for model in (eager, sdpa):
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    torch.npu.synchronize()


@pytest.mark.skipif(not _npu_available(), reason='Ascend NPU is not available')
def test_npu_bf16_gradient_checkpointing_lora_sft_two_steps():
    torch.manual_seed(8)
    config = _tiny_config(hidden_size=64, num_heads=2)
    model = _lora_model(config, device='npu:0', dtype=torch.bfloat16, gradient_checkpointing=True)
    assert model.is_gradient_checkpointing
    _run_lora_steps(model, _batch(8, 'right', device='npu:0'), steps=2)
    torch.npu.synchronize()
