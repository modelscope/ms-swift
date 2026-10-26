# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import inspect
import torch
import unittest
from peft import LoraConfig, get_peft_model
from torch import nn
from types import SimpleNamespace

from swift.model.patcher import patch_frozen_module, patch_get_input_embeddings
from swift.trainers.mixin import SwiftMixin


class VisionTower(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.blocks = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.aligner = nn.Linear(4, 4)
        # Reproduce an input-grad hook owned by a parent model.
        self.embed.register_forward_hook(lambda module, inputs, output: output.requires_grad_(True))

    def forward(self, pixels=None, payload=None):
        if pixels is None:
            pixels = payload['images'][0]
        return self.aligner(self.blocks(self.embed(pixels)))


class TestFrozenVision(unittest.TestCase):

    def test_frozen_encoder_saves_no_backward_tensors(self):
        tower = VisionTower().requires_grad_(False)
        pixels = torch.randn(2, 4)

        def saved_tensors():
            saved = []

            def pack(tensor):
                saved.append(tensor)
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output = tower(pixels)
            return output, saved

        expected, before = saved_tensors()
        self.assertGreater(len(before), 0)
        patch_frozen_module(tower)
        actual, after = saved_tensors()
        self.assertEqual(len(after), 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def compare_gradients(self, actual, reference):
        for (name, p), (_, expected) in zip(actual.named_parameters(), reference.named_parameters()):
            with self.subTest(parameter=name):
                if expected.grad is None:
                    self.assertIsNone(p.grad)
                else:
                    self.assertIsNotNone(p.grad)
                    torch.testing.assert_close(p.grad, expected.grad)

    def test_trainability_and_input_gradients(self):
        for mode in ('frozen', 'aligner', 'partial', 'lora', 'input', 'nested_input'):
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                tower = VisionTower().requires_grad_(False)
                if mode == 'aligner':
                    tower.aligner.requires_grad_(True)
                elif mode == 'partial':
                    tower.blocks[0].requires_grad_(True)
                elif mode == 'lora':
                    tower = get_peft_model(tower, LoraConfig(r=2, target_modules=['blocks.0']))
                reference = copy.deepcopy(tower)
                patch_frozen_module(tower)
                head = nn.Linear(4, 1)
                ref_head = copy.deepcopy(head)
                x = torch.randn(2, 4, requires_grad=mode in ('input', 'nested_input'))
                ref_x = x.detach().clone().requires_grad_(x.requires_grad)
                if mode == 'nested_input':
                    image = tower(payload={'images': [x]})
                    ref_image = reference(payload={'images': [ref_x]})
                else:
                    image, ref_image = tower(x), reference(ref_x)
                self.assertTrue(ref_image.requires_grad)
                self.assertEqual(image.requires_grad, mode != 'frozen')
                torch.testing.assert_close(image, ref_image, rtol=0, atol=0)
                head(image).square().mean().backward()
                ref_head(ref_image).square().mean().backward()
                self.compare_gradients(tower, reference)
                self.compare_gradients(head, ref_head)
                if x.requires_grad:
                    torch.testing.assert_close(x.grad, ref_x.grad)

    def test_idempotency_unfreezing_and_copy(self):
        tower = VisionTower().requires_grad_(False)
        patch_frozen_module(tower)
        forward = tower.forward
        patch_frozen_module(tower)
        self.assertIs(tower.forward, forward)
        x = torch.randn(2, 4)
        self.assertFalse(tower(x).requires_grad)
        cloned = copy.deepcopy(tower)
        cloned.aligner.requires_grad_(True)
        cloned(x).sum().backward()
        self.assertIsNotNone(cloned.aligner.weight.grad)
        self.assertIsNone(tower.aligner.weight.grad)
        self.assertFalse(tower(x).requires_grad)
        tower.aligner.requires_grad_(True)
        self.assertTrue(tower(x).requires_grad)
        with torch.no_grad():
            self.assertFalse(tower(x).requires_grad)
        tower.requires_grad_(False)
        self.assertFalse(tower(x).requires_grad)

    def test_qwen35_trainer_preparation(self):
        try:
            from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        except ImportError:
            self.skipTest('Qwen3.5 requires Transformers >= 5.2')
        for vit_gc, reentrant in ((False, True), (True, True), (True, False)):
            with self.subTest(vit_gc=vit_gc, reentrant=reentrant):
                torch.manual_seed(42)
                config = Qwen3_5Config(
                    image_token_id=30,
                    video_token_id=31,
                    vision_start_token_id=29,
                    vision_end_token_id=28,
                    text_config=dict(
                        vocab_size=32,
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        head_dim=8,
                        layer_types=['full_attention']),
                    vision_config=dict(
                        hidden_size=32,
                        intermediate_size=64,
                        depth=1,
                        num_heads=4,
                        out_hidden_size=32,
                        patch_size=2,
                        spatial_merge_size=2,
                        temporal_patch_size=2,
                        num_position_embeddings=16))
                model = Qwen3_5ForConditionalGeneration(config).train()
                model.model.visual.requires_grad_(False)
                patch_get_input_embeddings(model.model.visual, 'patch_embed')
                model.model_meta = SimpleNamespace(
                    is_multimodal=True,
                    model_arch=SimpleNamespace(vision_tower=['model.visual'], language_model=['model.language_model']))
                reference = copy.deepcopy(model)
                reference.enable_input_require_grads()
                args = SimpleNamespace(
                    gradient_checkpointing=True,
                    vit_gradient_checkpointing=vit_gc,
                    gradient_checkpointing_kwargs={'use_reentrant': reentrant})
                SwiftMixin._prepare_gradient_checkpointing(SimpleNamespace(args=args), model)
                visual_grads = []
                model.model.visual.register_forward_hook(
                    lambda module, inputs, output: visual_grads.append(output.pooler_output.requires_grad))
                tokens = torch.tensor([[1, 29, 30, 30, 30, 30, 28, 2, 3]])
                labels = tokens.clone()
                labels[:, :7] = -100
                inputs = dict(
                    input_ids=tokens,
                    labels=labels,
                    pixel_values=torch.randn(16, 24),
                    image_grid_thw=torch.tensor([[1, 4, 4]]),
                    use_cache=False)
                if 'mm_token_type_ids' in inspect.signature(model.forward).parameters:
                    # Newer processors mark image tokens as 1 and text tokens as 0 for M-RoPE.
                    inputs['mm_token_type_ids'] = (tokens == config.image_token_id).long()
                actual, expected = model(**inputs), reference(**inputs)
                torch.testing.assert_close(actual.logits, expected.logits)
                torch.testing.assert_close(actual.loss, expected.loss)
                actual.loss.backward()
                expected.loss.backward()
                self.assertEqual(visual_grads, [False])
                self.compare_gradients(model, reference)


if __name__ == '__main__':
    unittest.main()
