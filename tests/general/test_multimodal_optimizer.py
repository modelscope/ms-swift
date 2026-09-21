# Copyright (c) ModelScope Contributors. All rights reserved.
import tempfile
import torch
import unittest
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import TrainingArguments
from types import SimpleNamespace

from swift.model.model_arch import get_model_arch
from swift.optimizers.multimodal import MultimodalOptimizerCallback


class TinyMultimodalModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.model.visual = nn.Module()
        self.model.visual.proj = nn.Linear(4, 4)
        self.model.visual.merger = nn.Linear(4, 4)
        self.score = nn.Linear(4, 2)
        self.frozen = nn.Linear(4, 4).requires_grad_(False)
        self.model_meta = SimpleNamespace(model_arch=get_model_arch('qwen2_vl'))


class TestMultimodalOptimizer(unittest.TestCase):

    def test_full_and_peft_parameter_coverage(self):
        for use_peft in [False, True]:
            with self.subTest(use_peft=use_peft), tempfile.TemporaryDirectory() as tmp:
                model = TinyMultimodalModel()
                if use_peft:
                    model = get_peft_model(
                        model, LoraConfig(r=2, target_modules=['language_model.0'], modules_to_save=['score']))
                    model.model.model.visual.requires_grad_(True)
                args = TrainingArguments(
                    tmp, learning_rate=0.01, weight_decay=0.1, optim='sgd', use_cpu=True, report_to=[])
                args.vit_lr = 0.002
                args.aligner_lr = 0.003
                optimizer = MultimodalOptimizerCallback(args, SimpleNamespace(model=model)).create_optimizer()
                params = [p for group in optimizer.param_groups for p in group['params']]
                expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
                self.assertEqual({id(p) for p in params}, expected_ids)
                self.assertEqual(len(params), len(expected_ids))
                groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    p.grad = torch.ones_like(p)
                    lr = 0.003 if '.visual.merger.' in name else 0.002 if '.visual.' in name else 0.01
                    self.assertEqual(groups[id(p)]['lr'], lr)
                    if name.endswith('.bias') or 'language_model.1' in name:
                        self.assertEqual(groups[id(p)]['weight_decay'], 0)
                optimizer.step()
                for name, p in model.named_parameters():
                    with self.subTest(parameter=name):
                        if not p.requires_grad:
                            torch.testing.assert_close(p, before[name])
                        else:
                            group = groups[id(p)]
                            expected = before[name] - group['lr'] * (1 + group['weight_decay'] * before[name])
                            torch.testing.assert_close(p, expected)


if __name__ == '__main__':
    unittest.main()
