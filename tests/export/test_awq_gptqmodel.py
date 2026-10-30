# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import torch
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from swift.pipelines.export import quant


class TestAwqGPTQModel(unittest.TestCase):

    def setUp(self):
        self.engine = quant.QuantEngine.__new__(quant.QuantEngine)
        self.engine.args = SimpleNamespace(
            quant_method='awq',
            quant_bits=4,
            group_size=128,
            quant_batch_size=2,
            dataset=['calibration.jsonl'],
            model_dir='/model',
            output_dir='/output',
            safe_serialization=True,
            max_shard_size='5GB',
            adapters=[])
        self.engine.model = torch.nn.Linear(2, 2)
        self.engine.model.config = SimpleNamespace(quantization_config=None)
        self.engine.model.model_meta = SimpleNamespace(is_multimodal=False, additional_saved_files=[])
        self.engine.model.model_info = SimpleNamespace(is_moe_model=False, config=SimpleNamespace())
        self.engine.template = SimpleNamespace()
        self.engine.processor = SimpleNamespace(
            model_meta=self.engine.model.model_meta, model_info=self.engine.model.model_info)
        self.engine._get_quant_dataset = MagicMock(return_value=[torch.tensor([[1, 2, 3]])])
        self.wrapper = MagicMock()
        self.engine.awq_model = self.wrapper
        self.definition = MagicMock(return_value=self.wrapper)
        self.config = MagicMock(return_value=object())
        self.resolver = MagicMock(return_value=self.definition)
        gptq = types.ModuleType('gptqmodel')
        gptq.AWQConfig = self.config
        auto = types.ModuleType('gptqmodel.models.auto')
        auto.check_and_get_model_definition = self.resolver
        importer = types.ModuleType('gptqmodel.utils.importer')
        importer.auto_select_device = MagicMock(return_value='cuda')
        self.modules = {'gptqmodel': gptq, 'gptqmodel.models.auto': auto, 'gptqmodel.utils.importer': importer}

    def test_wraps_existing_model_and_saves_awq(self):
        with patch.dict(sys.modules, self.modules), patch.object(quant, 'save_checkpoint') as save_checkpoint:
            self.engine.quantize()
        self.wrapper.quantize.assert_called_once()
        self.assertEqual(self.wrapper.quantize.call_args.kwargs['calibration'], [{'input_ids': [1, 2, 3]}])
        self.assertEqual(self.wrapper.quantize.call_args.kwargs['batch_size'], 2)
        self.wrapper.save.assert_called_once_with('/output', max_shard_size='5GB')
        save_checkpoint.assert_called_once()

    def test_unlimited_awq_batch_uses_all_calibration_blocks(self):
        self.engine.args.quant_batch_size = None  # ExportArguments converts --quant_batch_size -1 to None.
        self.engine._get_quant_dataset.return_value = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]
        self.engine.awq_model_quantize()
        self.wrapper.quantize.assert_called_once_with(
            calibration=[{
                'input_ids': [1, 2]
            }, {
                'input_ids': [3, 4]
            }], batch_size=2)

    def test_awq_loads_model_once_through_public_api(self):
        args = MagicMock()
        args.quant_method = 'awq'
        args.quant_bits = 4
        args.group_size = 128
        args.model_dir = '/merged-model'
        args.torch_dtype = torch.float16
        args.adapters = []
        args.safe_serialization = True
        args.new_special_tokens = []
        args.device_map = None
        args.task_type = 'causal_lm'
        args.tuner_backend = 'peft'
        for key in [
                'model_kwargs', 'rope_scaling', 'max_memory', 'attn_impl', 'experts_impl', 'max_model_len',
                'init_strategy'
        ]:
            setattr(args, key, None)
        args.model_meta = self.engine.model.model_meta
        args.model_info = self.engine.model.model_info
        args.get_model_processor.return_value = (None, self.engine.processor)
        args.get_template.return_value = MagicMock()
        self.engine.model.config.use_cache = True
        self.wrapper.model = self.engine.model
        self.wrapper.quantized = False
        gptq = self.modules['gptqmodel']
        gptq.GPTQModel = SimpleNamespace(load=MagicMock(return_value=self.wrapper))
        with patch.dict(sys.modules, self.modules), patch.object(quant, 'prepare_model_template') as prepare, \
                patch.object(quant.HfConfigFactory, 'set_config_attr'):
            engine = quant.QuantEngine(args)
        prepare.assert_not_called()
        args.get_model_processor.assert_called_once_with(load_model=False)
        self.config.assert_called_once_with(bits=4, group_size=128, sym=False)
        gptq.GPTQModel.load.assert_called_once_with(
            '/merged-model', quantize_config=self.config.return_value, trust_remote_code=True, dtype=torch.float16)
        self.assertIs(engine.model, self.engine.model)

    def test_awq_does_not_silently_ignore_model_kwargs(self):
        args = MagicMock()
        args.quant_method = 'awq'
        args.quant_bits = 4
        args.adapters = []
        args.safe_serialization = True
        args.new_special_tokens = []
        args.device_map = None
        args.task_type = 'causal_lm'
        args.tuner_backend = 'peft'
        args.model_kwargs = {'custom_setting': True}
        with self.assertRaisesRegex(ValueError, 'model_kwargs'):
            quant.QuantEngine(args)

    def test_rejects_already_quantized_checkpoint(self):
        self.engine.model.config.quantization_config = {'quant_method': 'fp8'}
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, 'already quantized'):
            self.engine.awq_model_quantize()
        self.definition.assert_not_called()

    def test_rejects_unmerged_adapter(self):
        self.engine.args.adapters = ['/adapter']
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, 'merge_lora'):
            self.engine.awq_model_quantize()
        self.definition.assert_not_called()

    def test_rejects_fp8_weights_without_config(self):
        self.engine.model.to(dtype=torch.float8_e4m3fn)
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, 'FP8'):
            self.engine.awq_model_quantize()
        self.definition.assert_not_called()

    def test_rejects_empty_calibration(self):
        self.engine._get_quant_dataset.return_value = [torch.tensor([[]])]
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, 'calibration'):
            self.engine.awq_model_quantize()
        self.wrapper.quantize.assert_not_called()

    def test_rejects_multimodal_calibration(self):
        self.engine.model.model_meta.is_multimodal = True
        self.engine.model.model_info.model_type = 'unsupported_vl'
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, 'multimodal'):
            self.engine.awq_model_quantize()
        self.wrapper.quantize.assert_not_called()

    def test_qwen2_vl_calibration_preserves_images_and_text(self):
        data = {
            'messages': [{
                'role': 'user',
                'content': 'Compare <image> to <image>.'
            }],
            'images': [{
                'path': '/red.png',
                'bytes': None
            }, {
                'path': '/blue.png',
                'bytes': None
            }]
        }
        messages = quant.QuantEngine._awq_qwen2_vl_conversation(data)
        self.assertEqual(messages, [{
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': 'Compare '
            }, {
                'type': 'image',
                'image': '/red.png'
            }, {
                'type': 'text',
                'text': ' to '
            }, {
                'type': 'image',
                'image': '/blue.png'
            }, {
                'type': 'text',
                'text': '.'
            }]
        }])
        self.assertEqual(data['messages'][0]['content'], 'Compare <image> to <image>.')

    def test_qwen2_vl_calibration_rejects_mismatched_images(self):
        data = {'messages': [{'role': 'user', 'content': '<image><image>'}], 'images': ['/only-one.png']}
        with self.assertRaisesRegex(ValueError, 'image'):
            quant.QuantEngine._awq_qwen2_vl_conversation(data)

    def test_qwen2_vl_calibration_rejects_unhandled_metadata(self):
        data = {
            'messages': [{
                'role': 'user',
                'content': '<image>Where?'
            }],
            'images': ['/red.png'],
            'objects': {
                'bbox': [[0, 0, 1, 1]]
            }
        }
        with self.assertRaisesRegex(ValueError, 'objects'):
            quant.QuantEngine._awq_qwen2_vl_conversation(data)

    def test_qwen2_vl_calibration_inserts_missing_image_placeholder(self):
        data = {'messages': [{'role': 'user', 'content': 'Describe it.'}], 'images': ['/red.png']}
        messages = quant.QuantEngine._awq_qwen2_vl_conversation(data)
        self.assertEqual(messages[0]['content'], [{
            'type': 'image',
            'image': '/red.png'
        }, {
            'type': 'text',
            'text': 'Describe it.'
        }])

    def test_qwen2_vl_uses_visual_calibration(self):
        self.engine.model.model_meta.is_multimodal = True
        self.engine.model.model_info.model_type = 'qwen2_vl'
        conversation = [{'role': 'user', 'content': [{'type': 'image', 'image': '/red.png'}]}]
        self.engine._get_awq_qwen2_vl_dataset = MagicMock(return_value=[conversation])
        self.engine.awq_model_quantize()
        self.wrapper.quantize.assert_called_once_with(calibration=[conversation], batch_size=2)
        self.engine._get_quant_dataset.assert_not_called()

    def test_qwen2_vl_rejects_custom_system_instead_of_ignoring_it(self):
        self.engine.model.model_meta.is_multimodal = True
        self.engine.model.model_info.model_type = 'qwen2_vl'
        self.engine.args.system = 'custom system'
        with self.assertRaisesRegex(ValueError, 'system'):
            self.engine.awq_model_quantize()
        self.wrapper.quantize.assert_not_called()

    def test_qwen2_vl_applies_max_length_to_gptqmodel_tokens(self):
        self.engine.args.dataset_shuffle = False
        self.engine.args.get_dataset_kwargs = lambda: {}
        self.engine.args.max_length = 3
        self.engine.args.quant_n_samples = 2
        self.engine.template.encode = MagicMock()
        raw = [{
            'messages': [{
                'role': 'user',
                'content': '<image>What?'
            }],
            'images': ['/red.png']
        }, {
            'messages': [{
                'role': 'user',
                'content': '<image>What?'
            }],
            'images': ['/blue.png']
        }]
        dataset = MagicMock()
        dataset.shuffle.return_value = raw
        self.wrapper.prepare_dataset.side_effect = [[{
            'input_ids': torch.tensor([[1, 2, 3, 4]])
        }], [{
            'input_ids': torch.tensor([[1, 2]])
        }]]
        with patch.object(quant, 'load_dataset', return_value=(dataset, None)):
            calibration = self.engine._get_awq_qwen2_vl_dataset()
        self.assertEqual(len(calibration), 1)
        self.assertEqual(calibration[0][0]['content'][0]['image'], '/blue.png')

    def test_requires_safetensors(self):
        self.engine.args.safe_serialization = False
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, 'safe_serialization'):
            self.engine.awq_model_quantize()
        self.wrapper.quantize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
