# Copyright (c) ModelScope Contributors. All rights reserved.
import re
import torch
import unittest
from torch import nn
from types import SimpleNamespace

try:
    from swift.megatron.trainers.base import BaseMegatronTrainer
    from swift.megatron.utils.utils import get_multimodal_target_regex
    from swift.tuners import LoraConfig, Swift
    MEGATRON_UNAVAILABLE = None
except Exception as e:
    # CI does not install megatron, and a mismatched mcore dependency can fail with anything from
    # ImportError to AttributeError, so any failure here means the tests below cannot run.
    MEGATRON_UNAVAILABLE = f'Megatron dependencies not available: {e}'


class _Merger(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.linear_fc1 = nn.Linear(8, 8)
        self.linear_fc2 = nn.Linear(8, 4)


class _VisionWrapper(nn.Module):
    """Mimics `HuggingFaceVit`: `wrapper.visual` holds the transformers vision model."""

    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.merger = _Merger()


class _Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.visual = _VisionWrapper()
        self.language_model = nn.Module()
        self.language_model.proj = nn.Linear(4, 4)


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestMultimodalAlignerLoraTarget(unittest.TestCase):
    """`--freeze_aligner false` must put LoRA on the aligner, whatever shape the aligner has."""

    @staticmethod
    def _args(aligner):
        visual_cls = SimpleNamespace(_vision_tower=['visual'], _aligner=aligner)
        return SimpleNamespace(megatron_model_meta=SimpleNamespace(visual_cls=visual_cls))

    def test_nested_aligner_is_targeted(self):
        pattern = get_multimodal_target_regex(
            self._args(['visual.merger']), _Model(), freeze_llm=True, freeze_vit=True, freeze_aligner=False)
        self.assertTrue(re.fullmatch(pattern, 'visual.visual.merger.linear_fc1'))
        self.assertTrue(re.fullmatch(pattern, 'visual.visual.merger.linear_fc2'))

    def test_standalone_linear_aligner_is_targeted(self):
        model = _Model()
        model.visual.mlp1 = nn.Linear(4, 8)
        pattern = get_multimodal_target_regex(
            self._args(['mlp1']), model, freeze_llm=True, freeze_vit=True, freeze_aligner=False)
        self.assertTrue(re.fullmatch(pattern, 'visual.mlp1'))

    def test_parameter_aligner_is_skipped(self):
        model = _Model()
        model.visual.scale = nn.Parameter(torch.zeros(1))
        pattern = get_multimodal_target_regex(
            self._args(['scale']), model, freeze_llm=True, freeze_vit=True, freeze_aligner=False)
        self.assertFalse(re.fullmatch(pattern, 'visual.scale'))


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestMultimodalAlignerLoraMerge(unittest.TestCase):
    """The aligner is a plain transformers module, so its adapter is not a `LoraParallelLinear`."""

    def test_merge_lora_adapters_covers_aligner(self):
        pattern = TestMultimodalAlignerLoraTarget._args(['visual.merger'])
        pattern = get_multimodal_target_regex(pattern, _Model(), freeze_llm=True, freeze_vit=True, freeze_aligner=False)
        peft_model = Swift.prepare_model(_Model(), LoraConfig(r=4, lora_alpha=8, target_modules=pattern))
        model = peft_model.base_model
        merger = model.visual.visual.merger
        with torch.no_grad():
            merger.linear_fc1.lora_B['default'].weight.normal_()
        base_weight = merger.linear_fc1.get_base_layer().weight.data.clone()
        delta_weight = merger.linear_fc1.get_delta_weight('default')

        trainer = SimpleNamespace(unwrapped_models=[model])
        BaseMegatronTrainer.merge_lora_adapters(trainer)
        torch.testing.assert_close(merger.linear_fc1.get_base_layer().weight.data, base_weight + delta_weight)

        BaseMegatronTrainer.unmerge_lora_adapters(trainer)
        torch.testing.assert_close(merger.linear_fc1.get_base_layer().weight.data, base_weight)


if __name__ == '__main__':
    unittest.main()
