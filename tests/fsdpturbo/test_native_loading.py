# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import json
import pytest
import torch
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from swift.fsdpturbo.loading import _native_factory, create_meta_model

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('fsdp_turbo') is None, reason='requires the optional FSDPTurbo package')


@pytest.fixture
def native_source(tmp_path):
    from fsdp_turbo.models.deepseek.deepseek_v41.model_args import ModelArgs
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({
                '[UNK]': 0,
                'a': 1,
                'b': 2,
                'c': 3,
                'd': 4,
                'e': 5,
                'f': 6
            }, unk_token='[UNK]')),
        unk_token='[UNK]')
    tokenizer.save_pretrained(tmp_path)
    config = asdict(
        ModelArgs(
            max_seq_len=16,
            max_batch_size=4,
            vocab_size=7,
            dim=8,
            moe_inter_dim=4,
            n_layers=1,
            n_mtp_layers=0,
            n_heads=2,
            n_routed_experts=2,
            n_activated_experts=1,
            q_lora_rank=4,
            head_dim=4,
            rope_head_dim=2,
            o_groups=1,
            o_lora_rank=4,
            window_size=4,
            compress_ratios=(0, ),
            kv_source_layers=(),
            index_source_layers=(),
            index_n_heads=2,
            index_head_dim=4,
            index_topk=2,
            norm_eps=1e-6,
            hc_mult=2,
            hc_sinkhorn_iters=3,
            engram_layer_ids=(0, ),
            engram_num_embeddings=(17, ),
            engram_max_ngram_size=2,
            engram_vocab_size=7,
            engram_n_heads=1,
            engram_head_dim=4,
            engram_pad_id=0,
            engram_compressed_vocab_size=7,
        ))
    config = json.loads(json.dumps(config))
    (tmp_path / 'config.json').write_text(json.dumps(config))
    args = SimpleNamespace(
        model=str(tmp_path),
        model_type='deepseek_v41',
        torch_dtype=torch.bfloat16,
        max_length=8,
        training_args=SimpleNamespace(per_device_train_batch_size=2),
    )
    return args, tokenizer, config


def test_native_partial_uses_existing_parser_builder_and_source_tokenizer(native_source):
    from fsdp_turbo.models.deepseek.deepseek_v41.adapter import build_deepseek_v41_model

    args, source_tokenizer, config = native_source
    factory = _native_factory(args)

    assert factory.func is build_deepseek_v41_model
    assert set(factory.keywords) == {'model_args', 'tokenizer'}
    assert factory.keywords['model_args'].max_seq_len == args.max_length
    assert factory.keywords['model_args'].max_batch_size == 2
    assert json.loads(json.dumps(asdict(factory.keywords['model_args']))) == {
        **config, 'max_seq_len': args.max_length,
        'max_batch_size': 2
    }
    assert json.loads(source_tokenizer.backend_tokenizer.to_str()) == json.loads(
        factory.keywords['tokenizer'].backend_tokenizer.to_str())
    assert json.loads((Path(args.model) / 'config.json').read_text()) == config


def test_native_meta_build_preserves_source_hash_and_sensitive_dtypes(native_source):
    from accelerate import init_empty_weights

    args, _, _ = native_source
    with init_empty_weights(include_buffers=False):
        expected = _native_factory(args)()
        model = create_meta_model(args)

    assert model.model.head.weight.is_meta
    assert model.model.head.weight.dtype == torch.bfloat16
    assert model.model.head.compute_dtype == torch.float32
    assert model.model.layers[0].attn.attn_sink.dtype == torch.float32
    assert model.model.engram_hash.hash_fingerprint == expected.model.engram_hash.hash_fingerprint
    assert model.config.dim == 8


def test_native_fp16_is_rejected_without_reading_weights(native_source):
    args, _, _ = native_source
    args.torch_dtype = torch.float16
    with pytest.raises(ValueError, match='requires BF16'):
        _native_factory(args)


def test_native_builder_options_are_forwarded_and_unknown_options_fail(native_source):
    args, _, _ = native_source
    args.model_kwargs = {'chunk_loss': True, 'chunk_loss_size': 128, 'head_compute_dtype': 'bf16'}
    factory = _native_factory(args)
    assert factory.keywords['chunk_loss'] is True
    assert factory.keywords['chunk_loss_size'] == 128
    assert factory.keywords['head_compute_dtype'] == 'bf16'
    args.model_kwargs['chunk_los'] = True
    with pytest.raises(ValueError, match='chunk_los'):
        _native_factory(args)


@pytest.mark.parametrize('length,ratio,expected', [(7, 0, 8), (9, 4, 16)])
def test_native_cp_cache_covers_compression_aligned_padding(native_source, length, ratio, expected):
    args, _, config = native_source
    config['compress_ratios'] = [ratio]
    (Path(args.model) / 'config.json').write_text(json.dumps(config))
    args.cp_size = 2
    args.max_length = length

    assert _native_factory(args).keywords['model_args'].max_seq_len == expected
    assert args.max_length == length


def test_upstream_uniform_bf16_preparation_preserves_buffer_dtypes_and_deferred_tables(native_source):
    from fsdp_turbo.models.deepseek.deepseek_v41.adapter import prepare_deepseek_v41_model_for_fsdp
    from fsdp_turbo.models.deepseek.deepseek_v41.engram.host_offload_engram import host_engram_tables

    args, _, _ = native_source
    model = _native_factory(args)()
    buffer_dtypes = {name: value.dtype for name, value in model.named_buffers()}

    assert prepare_deepseek_v41_model_for_fsdp(model, device='cpu', parameter_dtype=torch.bfloat16) is model
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert {name: value.dtype for name, value in model.named_buffers()} == buffer_dtypes
    assert host_engram_tables(model)
    assert all(table.weight.is_meta for table in host_engram_tables(model))


def test_native_architecture_config_can_be_separate_from_hf_metadata(native_source):
    args, _, config = native_source
    native_config = Path(args.model) / 'native_model_args.json'
    native_config.write_text(json.dumps(config))
    metadata = {'model_type': 'deepseek_v41', 'text_config': {'hidden_size': config['dim']}}
    (Path(args.model) / 'config.json').write_text(json.dumps(metadata))
    args.model_kwargs = {'config_path': str(native_config)}

    factory = _native_factory(args)

    assert factory.keywords['model_args'].dim == config['dim']
    assert factory.keywords['model_args'].max_seq_len == args.max_length
    assert json.loads(native_config.read_text()) == config
    assert json.loads((Path(args.model) / 'config.json').read_text()) == metadata


def test_native_metadata_namespace_is_owned_by_swift(native_source):
    from torch import nn

    args, _, _ = native_source
    metadata = {
        'model_type': 'deepseek_v41',
        'text_config': {
            'hidden_size': 8
        },
        'vision_config': {
            'hidden_size': 4
        },
        'use_cache': True
    }
    (Path(args.model) / 'config.json').write_text(json.dumps(metadata))
    model = nn.Module()
    model.model = nn.Module()
    model.model.head = nn.Linear(8, 7, bias=False, device='meta')
    with mock.patch('swift.fsdpturbo.loading._native_factory', return_value=lambda: model):
        assert create_meta_model(args) is model

    assert model.config.model_type == metadata['model_type']
    assert model.config.text_config.hidden_size == 8
    assert model.config.vision_config.hidden_size == 4
    assert model.config.use_cache


@pytest.mark.parametrize('offload,device', [(True, 'cpu'), (False, 'cuda:0')])
@pytest.mark.parametrize('host_pin', [None, False, True])
def test_native_trainer_loads_after_wrapping_without_rank0_source(offload, device, host_pin):
    from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig
    from torch import nn

    from swift.fsdpturbo.trainer import FSDPTurboTrainer
    from swift.pipelines.train.tuner import TunerMixin

    events = []

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.config = SimpleNamespace(use_cache=True)

        def from_pretrained(self, path, *, strict, engram_rows_per_chunk):
            events.append(('load', path, strict, engram_rows_per_chunk))

    model = Model()
    trainer = object.__new__(FSDPTurboTrainer)
    trainer.args = SimpleNamespace(
        model='/source/native',
        tuner_type='full',
        cp_size=1,
        tp_size=1,
        ep_size=4,
        gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
        offload_params=offload,
        pin_memory=False,
        host_pin_memory=host_pin,
        resume_from_checkpoint=None,
        router_aux_loss_coef=0)
    trainer.train_args = SimpleNamespace(device=torch.device('cuda:0'))
    trainer.template = SimpleNamespace(model=None)
    trainer.train_dataset = None
    trainer.spec = SimpleNamespace(
        native_model_factory='fsdp_turbo.models.deepseek.deepseek_v41.adapter.build_deepseek_v41_model',
        frozen_modules=(),
        matching_modules=lambda _model, _patterns: {},
        validate_model=lambda *_args, **_kwargs: None)
    trainer.config = FSDPTurboConfig()

    def wrap(_config, candidate):
        events.append(('wrap', ))
        return SimpleNamespace(model=candidate)

    def materialize(candidate, *, device):
        assert candidate is model
        events.append(('materialize', device))

    def prepare(candidate, *, device, parameter_dtype, initialize_host_weights, pin_host_memory):
        assert parameter_dtype == torch.bfloat16
        assert initialize_host_weights is False
        assert pin_host_memory is (False if host_pin is None else host_pin)
        events.append(('prepare', device))
        return candidate

    with mock.patch.object(FSDPTurboTrainer, '_initialize_native_groups',
                           side_effect=lambda: events.append(('warmup', ))), \
            mock.patch.object(TunerMixin, 'prepare_model', return_value=model), \
            mock.patch('fsdp_turbo.models.deepseek.deepseek_v41.adapter.prepare_deepseek_v41_model_for_fsdp',
                       side_effect=prepare), \
            mock.patch('fsdp_turbo.models.deepseek.deepseek_v41.adapter.materialize_deepseek_v41_model',
                       side_effect=materialize), \
            mock.patch('swift.fsdpturbo.loading.create_meta_model', return_value=model), \
            mock.patch('swift.fsdpturbo.loading.load_rank0_model', side_effect=AssertionError('native rank0 load')), \
            mock.patch('swift.fsdpturbo.loading.materialize_sharded_model',
                       side_effect=AssertionError('native full-state broadcast')), \
            mock.patch('fsdp_turbo.fsdp_turbo.FSDPTurbo', side_effect=wrap):
        assert trainer.build_model().model is model

    assert events == [('prepare', torch.device('cpu')), ('warmup', ), ('wrap', ), ('materialize', torch.device(device)),
                      ('load', '/source/native', True, 65_536)]
    assert not model.config.use_cache
