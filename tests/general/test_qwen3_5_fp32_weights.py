import importlib
import inspect
import json
import tempfile
import torch
import transformers
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from threading import Event, RLock, Thread
from types import SimpleNamespace
from unittest.mock import patch

from swift.model.models import qwen as qwen_module
from swift.model.models.qwen import (_QWEN3_5_KEEP_IN_FP32_MODULES, Qwen3_5EmbLoader, Qwen3_5Loader, Qwen3_5MoeLoader,
                                     _get_qwen3_5_keep_in_fp32_modules, _patch_qwen3_5_keep_in_fp32_modules)

_POLICY_ATTR = '_keep_in_fp32_modules_strict'


@contextmanager
def _use_qwen3_5_torch_kernels():
    module_names = (
        'transformers.models.qwen3_5.modeling_qwen3_5',
        'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe',
    )
    with ExitStack() as stack:
        for module_name in module_names:
            try:
                modeling_module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(modeling_module, 'FusedRMSNormGated'):
                optional_kernels = (
                    'FusedRMSNormGated',
                    'causal_conv1d_fn',
                    'causal_conv1d_update',
                    'chunk_gated_delta_rule',
                    'fused_recurrent_gated_delta_rule',
                )
                for name in optional_kernels:
                    if hasattr(modeling_module, name):
                        stack.enter_context(patch.object(modeling_module, name, None))
            else:
                fallback_kernels = (
                    'causal_conv1d_fn',
                    'causal_conv1d_update',
                    'torch_chunk_gated_delta_rule',
                    'torch_recurrent_gated_delta_rule',
                )
                for name in fallback_kernels:
                    function = getattr(modeling_module, name, None)
                    if function is not None:
                        stack.enter_context(patch.object(modeling_module, name, inspect.unwrap(function)))
        yield


def _create_tiny_qwen3_5(model_cls):
    from transformers import Qwen3_5Config

    text_config = {
        'vocab_size': 32,
        'hidden_size': 16,
        'intermediate_size': 32,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'num_key_value_heads': 1,
        'head_dim': 8,
        'linear_num_key_heads': 2,
        'linear_num_value_heads': 2,
        'linear_key_head_dim': 4,
        'linear_value_head_dim': 4,
        'linear_conv_kernel_dim': 2,
        'layer_types': ['linear_attention'],
        'tie_word_embeddings': False,
    }
    config_cls = Qwen3_5Config
    if 'Moe' in model_cls.__name__:
        from transformers import Qwen3_5MoeConfig
        config_cls = Qwen3_5MoeConfig
        text_config.update({
            'num_experts': 2,
            'num_experts_per_tok': 1,
            'moe_intermediate_size': 8,
            'shared_expert_intermediate_size': 8,
        })
    config = config_cls(
        text_config=text_config,
        vision_config={
            'depth': 1,
            'hidden_size': 16,
            'intermediate_size': 32,
            'num_heads': 2,
            'out_hidden_size': 16,
            'patch_size': 2,
            'spatial_merge_size': 1,
            'temporal_patch_size': 1,
            'num_position_embeddings': 16,
        },
        tie_word_embeddings=False,
    )
    return model_cls(config)


def _target_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters() if name.endswith(_QWEN3_5_KEEP_IN_FP32_MODULES)
    }


def _control_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters() if name.endswith('linear_attn.out_proj.weight')
    }


def _saved_dtypes(directory: Path):
    dtypes = {}
    for path in directory.glob('*.safetensors'):
        with safe_open(path, framework='pt', device='cpu') as tensors:
            for key in tensors.keys():
                dtypes[key] = tensors.get_slice(key).get_dtype()
    return dtypes


def _apply_loader_policy(loader_cls, pretrained_cls, load_model, model_dir='', config=None, model_kwargs=None):
    loader = SimpleNamespace(auto_model_cls=None)
    observed_policies = []
    base_loader_cls = qwen_module.ModelLoader if loader_cls is Qwen3_5EmbLoader else qwen_module.Qwen2VLLoader

    def _get_model(*args, **kwargs):
        observed_policies.append(list(getattr(pretrained_cls, _POLICY_ATTR, None) or []))
        return load_model()

    with patch.object(qwen_module, '_patch_qwen3_5_linear_attention_sequence_parallel'), \
            patch.object(base_loader_cls, 'get_model', side_effect=_get_model):
        result = loader_cls.get_model(loader, str(model_dir), config, None, model_kwargs or {})
    return result, observed_policies


@unittest.skipUnless(hasattr(transformers, 'Qwen3_5ForConditionalGeneration'), 'Qwen3.5 requires Transformers 5')
class TestQwen3_5Fp32Weights(unittest.TestCase):

    def test_detects_checkpoint_fp32_policy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            one_layer_config = SimpleNamespace(text_config=SimpleNamespace(layer_types=['linear_attention']))
            two_layer_config = SimpleNamespace(
                text_config=SimpleNamespace(layer_types=['linear_attention', 'linear_attention']))
            self.assertEqual(_get_qwen3_5_keep_in_fp32_modules(str(root), one_layer_config), ())

            unsharded = root / 'unsharded'
            unsharded.mkdir()
            save_file(
                {
                    'model.layers.0.linear_attn.A_log': torch.ones(2, dtype=torch.float32),
                    'model.layers.0.linear_attn.norm.weight': torch.ones(2, dtype=torch.float32),
                    'model.layers.0.linear_attn.out_proj.weight': torch.ones(2, dtype=torch.bfloat16),
                }, unsharded / 'model.safetensors')
            self.assertEqual(
                _get_qwen3_5_keep_in_fp32_modules(str(unsharded), one_layer_config), _QWEN3_5_KEEP_IN_FP32_MODULES)
            self.assertEqual(
                _get_qwen3_5_keep_in_fp32_modules(str(unsharded), one_layer_config, {'use_safetensors': False}), ())
            self.assertEqual(
                _get_qwen3_5_keep_in_fp32_modules(
                    str(unsharded), one_layer_config, {'key_mapping': {
                        'old.weight': 'new.weight'
                    }}), ())
            self.assertEqual(_get_qwen3_5_keep_in_fp32_modules(str(unsharded), two_layer_config), ())

            sharded = root / 'sharded'
            sharded.mkdir()
            shard_names = ('model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors')
            shard_tensors = (
                {
                    'model.layers.0.linear_attn.A_log': torch.ones(2, dtype=torch.float32),
                    'model.layers.0.linear_attn.norm.weight': torch.ones(2, dtype=torch.float32),
                },
                {
                    'model.layers.1.linear_attn.A_log': torch.ones(2, dtype=torch.bfloat16),
                    'model.layers.1.linear_attn.norm.weight': torch.ones(2, dtype=torch.float32),
                },
            )
            weight_map = {}
            for shard_name, tensors in zip(shard_names, shard_tensors):
                save_file(tensors, sharded / shard_name)
                weight_map.update(dict.fromkeys(tensors, shard_name))
            with open(sharded / 'model.safetensors.index.json', 'w', encoding='utf-8') as f:
                json.dump({'metadata': {}, 'weight_map': weight_map}, f)
            self.assertEqual(
                _get_qwen3_5_keep_in_fp32_modules(str(sharded), two_layer_config), ('linear_attn.norm.weight', ))

            dual_checkpoint = root / 'dual-checkpoint'
            dual_checkpoint.mkdir()
            bf16_tensors = {
                'model.layers.0.linear_attn.A_log': torch.ones(2, dtype=torch.bfloat16),
                'model.layers.0.linear_attn.norm.weight': torch.ones(2, dtype=torch.bfloat16),
            }
            fp32_tensors = {key: value.float() for key, value in bf16_tensors.items()}
            save_file(bf16_tensors, dual_checkpoint / 'model.safetensors')
            save_file(fp32_tensors, dual_checkpoint / 'stale.safetensors')
            with open(dual_checkpoint / 'model.safetensors.index.json', 'w', encoding='utf-8') as f:
                json.dump({'metadata': {}, 'weight_map': dict.fromkeys(fp32_tensors, 'stale.safetensors')}, f)
            self.assertEqual(_get_qwen3_5_keep_in_fp32_modules(str(dual_checkpoint), one_layer_config), ())

            explicit_config = SimpleNamespace(
                text_config=one_layer_config.text_config, transformers_weights='model.safetensors.index.json')
            self.assertEqual(
                _get_qwen3_5_keep_in_fp32_modules(
                    str(dual_checkpoint), explicit_config, {
                        'use_safetensors': False,
                        'from_tf': True
                    }), _QWEN3_5_KEEP_IN_FP32_MODULES)

    def test_serializes_overlapping_policy_contexts(self):
        from transformers import Qwen3_5PreTrainedModel

        class ObservedRLock:

            def __init__(self):
                self.lock = RLock()
                self.attempts = 0
                self.second_attempted = Event()

            def __enter__(self):
                self.attempts += 1
                if self.attempts == 2:
                    self.second_attempted.set()
                return self.lock.__enter__()

            def __exit__(self, *args):
                return self.lock.__exit__(*args)

        policy_orders = ((_QWEN3_5_KEEP_IN_FP32_MODULES, ()), ((), _QWEN3_5_KEEP_IN_FP32_MODULES))
        for first_policy, second_policy in policy_orders:
            with self.subTest(first_policy=first_policy, second_policy=second_policy):
                first_entered = Event()
                release_first = Event()
                second_entered = Event()
                errors = []
                observed_lock = ObservedRLock()

                def first_context():
                    try:
                        with _patch_qwen3_5_keep_in_fp32_modules(Qwen3_5PreTrainedModel, first_policy):
                            first_entered.set()
                            if not release_first.wait(5):
                                raise TimeoutError('first policy context was not released')
                    except BaseException as error:
                        errors.append(error)

                def second_context():
                    try:
                        with _patch_qwen3_5_keep_in_fp32_modules(Qwen3_5PreTrainedModel, second_policy):
                            second_entered.set()
                    except BaseException as error:
                        errors.append(error)

                had_local_policy = _POLICY_ATTR in Qwen3_5PreTrainedModel.__dict__
                original_policy = Qwen3_5PreTrainedModel.__dict__.get(_POLICY_ATTR)
                with patch.object(qwen_module, '_QWEN3_5_KEEP_IN_FP32_MODULES_LOCK', observed_lock):
                    first = Thread(target=first_context)
                    second = Thread(target=second_context)
                    first.start()
                    second_started = False
                    try:
                        self.assertTrue(first_entered.wait(5))
                        second.start()
                        second_started = True
                        self.assertTrue(observed_lock.second_attempted.wait(5))
                        self.assertFalse(second_entered.is_set())
                    finally:
                        release_first.set()
                        first.join(5)
                        if second_started:
                            second.join(5)

                self.assertFalse(first.is_alive())
                self.assertFalse(second.is_alive())
                self.assertEqual(errors, [])
                self.assertTrue(second_entered.is_set())
                if had_local_policy:
                    self.assertIs(Qwen3_5PreTrainedModel.__dict__[_POLICY_ATTR], original_policy)
                else:
                    self.assertNotIn(_POLICY_ATTR, Qwen3_5PreTrainedModel.__dict__)

    def test_preserves_checkpoint_fp32_weights(self):
        from transformers import AutoModel, Qwen3_5ForConditionalGeneration, Qwen3_5Model, Qwen3_5PreTrainedModel

        cases = [
            ('dense', Qwen3_5Loader, Qwen3_5ForConditionalGeneration, Qwen3_5ForConditionalGeneration,
             Qwen3_5PreTrainedModel),
            ('embedding', Qwen3_5EmbLoader, Qwen3_5Model, AutoModel, Qwen3_5PreTrainedModel),
        ]
        if hasattr(transformers, 'Qwen3_5MoeForConditionalGeneration'):
            from transformers import Qwen3_5MoeForConditionalGeneration, Qwen3_5MoePreTrainedModel
            cases.insert(1, ('moe', Qwen3_5MoeLoader, Qwen3_5MoeForConditionalGeneration,
                             Qwen3_5MoeForConditionalGeneration, Qwen3_5MoePreTrainedModel))
        for name, loader_cls, source_cls, load_cls, pretrained_cls in cases:
            with self.subTest(model=name):
                had_local_policy = _POLICY_ATTR in pretrained_cls.__dict__
                original_policy = pretrained_cls.__dict__.get(_POLICY_ATTR)
                inherited_policy = list(getattr(pretrained_cls, _POLICY_ATTR, None) or [])
                expected_policy = inherited_policy + [
                    policy for policy in _QWEN3_5_KEEP_IN_FP32_MODULES if policy not in inherited_policy
                ]

                existing_policy = ['existing']
                detector_patch = patch.object(
                    qwen_module,
                    '_get_qwen3_5_keep_in_fp32_modules',
                    return_value=_QWEN3_5_KEEP_IN_FP32_MODULES,
                )
                with patch.object(pretrained_cls, _POLICY_ATTR, existing_policy), detector_patch:
                    for _ in range(2):
                        sentinel = torch.nn.Module()
                        if loader_cls is not Qwen3_5EmbLoader:
                            sentinel.visual = torch.nn.Identity()
                        result, policies = _apply_loader_policy(loader_cls, pretrained_cls, lambda: sentinel)
                        self.assertIs(result, sentinel)
                        self.assertEqual(policies, [['existing', *_QWEN3_5_KEEP_IN_FP32_MODULES]])
                        self.assertIs(pretrained_cls.__dict__[_POLICY_ATTR], existing_policy)

                    def _raise_loader_error():
                        raise RuntimeError('expected loader failure')

                    with self.assertRaisesRegex(RuntimeError, 'expected loader failure'):
                        _apply_loader_policy(loader_cls, pretrained_cls, _raise_loader_error)
                    self.assertIs(pretrained_cls.__dict__[_POLICY_ATTR], existing_policy)

                if had_local_policy:
                    self.assertIs(pretrained_cls.__dict__[_POLICY_ATTR], original_policy)
                else:
                    self.assertNotIn(_POLICY_ATTR, pretrained_cls.__dict__)

                with _patch_qwen3_5_keep_in_fp32_modules(pretrained_cls, _QWEN3_5_KEEP_IN_FP32_MODULES):
                    outer_policy = pretrained_cls.__dict__[_POLICY_ATTR]
                    self.assertEqual(outer_policy, expected_policy)
                    with _patch_qwen3_5_keep_in_fp32_modules(pretrained_cls, _QWEN3_5_KEEP_IN_FP32_MODULES):
                        self.assertEqual(pretrained_cls.__dict__[_POLICY_ATTR], expected_policy)
                    self.assertIs(pretrained_cls.__dict__[_POLICY_ATTR], outer_policy)
                if had_local_policy:
                    self.assertIs(pretrained_cls.__dict__[_POLICY_ATTR], original_policy)
                else:
                    self.assertNotIn(_POLICY_ATTR, pretrained_cls.__dict__)

            for dtype, saved_dtype in [(torch.float16, 'F16'), (torch.bfloat16, 'BF16')]:
                with self.subTest(model=name, dtype=dtype), _use_qwen3_5_torch_kernels(), \
                        tempfile.TemporaryDirectory() as tmp_dir:
                    source_dir = Path(tmp_dir) / 'source'
                    saved_dir = Path(tmp_dir) / 'saved'
                    model = _create_tiny_qwen3_5(source_cls).to(dtype)
                    config = model.config
                    source_targets = _target_parameters(model)
                    self.assertEqual(len(source_targets), 2)
                    for parameter in source_targets.values():
                        parameter.data = parameter.data.float()
                    model.save_pretrained(source_dir, safe_serialization=True)
                    del model

                    model, policies = _apply_loader_policy(
                        loader_cls,
                        pretrained_cls,
                        lambda: load_cls.from_pretrained(source_dir, dtype=dtype),
                        source_dir,
                        config,
                    )
                    self.assertEqual(policies, [expected_policy])
                    if had_local_policy:
                        self.assertIs(pretrained_cls.__dict__[_POLICY_ATTR], original_policy)
                    else:
                        self.assertNotIn(_POLICY_ATTR, pretrained_cls.__dict__)
                    self.assertTrue(set(_QWEN3_5_KEEP_IN_FP32_MODULES).issubset(model._keep_in_fp32_modules_strict))

                    targets = _target_parameters(model)
                    controls = _control_parameters(model)
                    self.assertEqual(len(targets), 2)
                    self.assertEqual(len(controls), 1)
                    self.assertEqual({parameter.dtype for parameter in targets.values()}, {torch.float32})
                    self.assertEqual({parameter.dtype for parameter in controls.values()}, {dtype})

                    model.save_pretrained(saved_dir, safe_serialization=True)
                    saved_dtypes = _saved_dtypes(saved_dir)
                    saved_targets = {
                        key: value
                        for key, value in saved_dtypes.items() if key.endswith(_QWEN3_5_KEEP_IN_FP32_MODULES)
                    }
                    saved_controls = {
                        key: value
                        for key, value in saved_dtypes.items() if key.endswith('linear_attn.out_proj.weight')
                    }
                    self.assertEqual(len(saved_targets), 2)
                    self.assertEqual(len(saved_controls), 1)
                    self.assertEqual(set(saved_targets.values()), {'F32'})
                    self.assertEqual(set(saved_controls.values()), {saved_dtype})

    def test_does_not_upcast_checkpoint_bf16_weights(self):
        from transformers import AutoModel, Qwen3_5ForConditionalGeneration, Qwen3_5Model, Qwen3_5PreTrainedModel

        cases = [
            ('dense', Qwen3_5Loader, Qwen3_5ForConditionalGeneration, Qwen3_5ForConditionalGeneration,
             Qwen3_5PreTrainedModel),
            ('embedding', Qwen3_5EmbLoader, Qwen3_5Model, AutoModel, Qwen3_5PreTrainedModel),
        ]
        if hasattr(transformers, 'Qwen3_5MoeForConditionalGeneration'):
            from transformers import Qwen3_5MoeForConditionalGeneration, Qwen3_5MoePreTrainedModel
            cases.insert(1, ('moe', Qwen3_5MoeLoader, Qwen3_5MoeForConditionalGeneration,
                             Qwen3_5MoeForConditionalGeneration, Qwen3_5MoePreTrainedModel))

        for name, loader_cls, source_cls, load_cls, pretrained_cls in cases:
            inherited_policy = list(getattr(pretrained_cls, _POLICY_ATTR, None) or [])
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(model=name, dtype=dtype), _use_qwen3_5_torch_kernels(), \
                        tempfile.TemporaryDirectory() as tmp_dir:
                    source_dir = Path(tmp_dir) / 'source'
                    source = _create_tiny_qwen3_5(source_cls).to(torch.bfloat16)
                    config = source.config
                    self.assertEqual({parameter.dtype
                                      for parameter in _target_parameters(source).values()}, {torch.bfloat16})
                    source.save_pretrained(source_dir, safe_serialization=True)
                    del source

                    model, policies = _apply_loader_policy(
                        loader_cls,
                        pretrained_cls,
                        lambda: load_cls.from_pretrained(source_dir, dtype=dtype),
                        source_dir,
                        config,
                    )
                    self.assertEqual(policies, [inherited_policy])
                    self.assertTrue(set(_QWEN3_5_KEEP_IN_FP32_MODULES).isdisjoint(model._keep_in_fp32_modules_strict))
                    self.assertEqual({parameter.dtype for parameter in _target_parameters(model).values()}, {dtype})
                    self.assertEqual({parameter.dtype for parameter in _control_parameters(model).values()}, {dtype})

    @unittest.skipIf(transformers.utils.is_torch_npu_available(),
                     'The CPU forward/backward smoke test is not compatible with global NPU model patches')
    def test_real_loader_supports_fp16_forward_and_backward(self):
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5PreTrainedModel

        dtype = torch.float16
        had_local_policy = _POLICY_ATTR in Qwen3_5PreTrainedModel.__dict__
        original_policy = Qwen3_5PreTrainedModel.__dict__.get(_POLICY_ATTR)

        def _restore_policy():
            if had_local_policy:
                setattr(Qwen3_5PreTrainedModel, _POLICY_ATTR, original_policy)
            elif _POLICY_ATTR in Qwen3_5PreTrainedModel.__dict__:
                delattr(Qwen3_5PreTrainedModel, _POLICY_ATTR)

        self.addCleanup(_restore_policy)
        # The SP patch mutates Transformers classes globally and is unrelated to this dtype integration path.
        with _use_qwen3_5_torch_kernels(), \
                patch.object(qwen_module, '_patch_qwen3_5_linear_attention_sequence_parallel'), \
                tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / 'source'
            saved_dir = Path(tmp_dir) / 'saved'
            source = _create_tiny_qwen3_5(Qwen3_5ForConditionalGeneration).to(dtype)
            config = source.config
            for parameter in _target_parameters(source).values():
                parameter.data = parameter.data.float()
            source.save_pretrained(source_dir, safe_serialization=True)
            del source

            loader = object.__new__(Qwen3_5Loader)
            loader.auto_model_cls = None
            loader.experts_impl = None
            loader.return_dummy_model = False
            loader.default_trust_remote_code = True
            loader.model_info = SimpleNamespace(task_type='causal_lm', quant_method=None)
            loader.model_meta = SimpleNamespace(is_reward=False, is_multimodal=True)
            model = loader.get_model(str(source_dir), config=config, processor=None, model_kwargs={'dtype': dtype})
            if had_local_policy:
                self.assertIs(Qwen3_5PreTrainedModel.__dict__[_POLICY_ATTR], original_policy)
            else:
                self.assertNotIn(_POLICY_ATTR, Qwen3_5PreTrainedModel.__dict__)
            self.assertTrue(set(_QWEN3_5_KEEP_IN_FP32_MODULES).issubset(model._keep_in_fp32_modules_strict))

            targets = _target_parameters(model)
            controls = _control_parameters(model)
            self.assertEqual({parameter.dtype for parameter in targets.values()}, {torch.float32})
            self.assertEqual({parameter.dtype for parameter in controls.values()}, {dtype})

            input_ids = torch.tensor([[1, 2, 3, 4]])
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for parameter in targets.values():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

            model.save_pretrained(saved_dir, safe_serialization=True)
            saved_dtypes = _saved_dtypes(saved_dir)
            self.assertEqual(
                {value
                 for key, value in saved_dtypes.items() if key.endswith(_QWEN3_5_KEEP_IN_FP32_MODULES)}, {'F32'})


if __name__ == '__main__':
    unittest.main()
