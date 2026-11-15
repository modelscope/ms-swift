import torch
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from torch import nn
from types import SimpleNamespace
from unittest import mock

from swift.fsdpturbo.loading import (_checked_checkpoint_read, _fp8_checkpoint_conversion, create_meta_model,
                                     load_rank0_model)


class TestRank0Loading(unittest.TestCase):

    def test_qwen_meta_model_uses_the_swift_loader_without_class_injection(self):
        args = SimpleNamespace(
            model_type='qwen3_5_moe', get_model_processor=mock.Mock(return_value=(mock.sentinel.model, None)))

        self.assertIs(create_meta_model(args), mock.sentinel.model)
        args.get_model_processor.assert_called_once_with(device_map='cpu', return_dummy_model=True)

    def test_missing_checkpoint_weight_is_not_randomly_initialized(self):
        from safetensors.torch import load_file, save_file
        from transformers import LlamaConfig, LlamaForCausalLM

        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            ))
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            path = Path(directory) / 'model.safetensors'
            weights = load_file(path)
            weights.pop('model.layers.0.self_attn.q_proj.weight')
            save_file(weights, path, metadata={'format': 'pt'})
            with _checked_checkpoint_read(), self.assertRaisesRegex(ValueError, 'Incomplete pretrained checkpoint'):
                LlamaForCausalLM.from_pretrained(directory)

    def test_fp4_single_source_expert_is_dequantized_before_merge(self):
        from transformers import FineGrainedFP8Config
        from transformers.core_model_loading import MergeModulelist, WeightConverter
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(dequantize=True), pre_quantized=True)
        conversion = WeightConverter(
            source_patterns='experts.*.w2.weight',
            target_patterns='experts.down_proj$',
            operations=[MergeModulelist(dim=0)],
        )
        # Transformers 5.22 uses model parameter names to decide whether
        # quantized companions remain parameters or must be dequantized.
        import inspect
        options = {
            'model': nn.Module()
        } if 'model' in inspect.signature(quantizer.update_weight_conversions).parameters else {}
        with _fp8_checkpoint_conversion():
            converters = quantizer.update_weight_conversions([conversion], **options)
        converter = next(
            entry for entry in converters if isinstance(entry, WeightConverter)
            and 'experts.down_proj' in [pattern.removesuffix('$') for pattern in entry.target_patterns])
        converter.collected_tensors[converter.source_patterns[0]] = [
            torch.full((2, 16), 0x21, dtype=torch.int8),
        ]
        converter.collected_tensors[converter.source_patterns[1]] = [torch.full((2, 1), 0.02)]
        output = converter.convert('experts.down_proj', hf_quantizer=quantizer)['experts.down_proj']
        self.assertEqual(tuple(output.shape), (1, 2, 32))
        self.assertTrue(output.is_floating_point())
        torch.testing.assert_close(output[0, 0, :4], torch.tensor([0.01, 0.02, 0.01, 0.02]))

    def test_rank0_reads_and_broadcasts_dtypes(self):
        model = nn.Linear(2, 2, dtype=torch.bfloat16)
        args = SimpleNamespace(
            resume_from_checkpoint=None,
            model_info=SimpleNamespace(quant_method=None),
            get_model_processor=mock.Mock(return_value=(model, None)),
        )
        with mock.patch('swift.fsdpturbo.loading.dist.get_rank', return_value=0), \
                mock.patch('swift.fsdpturbo.loading.dist.broadcast_object_list') as broadcast:
            loaded, dtypes = load_rank0_model(args)
        self.assertIs(loaded, model)
        self.assertEqual(dtypes, {'weight': torch.bfloat16, 'bias': torch.bfloat16})
        args.get_model_processor.assert_called_once_with(device_map='cpu')
        broadcast.assert_called_once()

    def test_e8m0_scales_are_decoded_as_values_before_bf16_conversion(self):
        from transformers import FineGrainedFP8Config
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(dequantize=True), pre_quantized=True)
        packed = torch.full((2, 16), 0x21, dtype=torch.int8)
        scales = torch.full((2, 1), 121, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        with _fp8_checkpoint_conversion(torch.bfloat16):
            output = Fp8Dequantize(quantizer)._dequantize_one(packed, scales)
        self.assertEqual(output.dtype, torch.bfloat16)
        expected = torch.tensor([0.0078125, 0.015625, 0.0078125, 0.015625], dtype=torch.bfloat16)
        torch.testing.assert_close(output[0, :4], expected, rtol=0, atol=0)

    def test_other_ranks_never_read_checkpoint_weights(self):
        args = SimpleNamespace(get_model_processor=mock.Mock())

        def broadcast(payload, **_kwargs):
            payload[0] = ({'weight': torch.bfloat16}, None)

        with mock.patch('swift.fsdpturbo.loading.dist.get_rank', return_value=3), \
                mock.patch('swift.fsdpturbo.loading.dist.broadcast_object_list', side_effect=broadcast):
            loaded, dtypes = load_rank0_model(args)
        self.assertIsNone(loaded)
        self.assertEqual(dtypes, {'weight': torch.bfloat16})
        args.get_model_processor.assert_not_called()

    def test_loading_error_is_propagated_before_mesh_creation(self):
        args = SimpleNamespace(
            resume_from_checkpoint=None,
            model_info=SimpleNamespace(quant_method=None),
            get_model_processor=mock.Mock(side_effect=FileNotFoundError('missing shard')),
        )
        with mock.patch('swift.fsdpturbo.loading.dist.get_rank', return_value=0), \
                mock.patch('swift.fsdpturbo.loading.dist.broadcast_object_list'), \
                self.assertRaisesRegex(RuntimeError, 'missing shard'):
            load_rank0_model(args)
