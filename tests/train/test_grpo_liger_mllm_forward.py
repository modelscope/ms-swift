# Copyright (c) ModelScope Contributors. All rights reserved.
"""The multimodal Liger GRPO loss must call the vision encoder the same number of times on every rank.

Under ZeRO-3 the ranks all-gather parameters module by module, so a rank holding an image sample that runs
the vision encoder one extra time desyncs the collectives and hangs training on mixed text/image data.
"""
import torch
import unittest
from torch import nn
from types import SimpleNamespace

from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.rlhf_trainers.utils import _ForwardRedirection
from swift.template.base import Template

HIDDEN = 4


class CountingVisual(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        self.calls = 0

    def forward(self, pixel_values):
        self.calls += 1
        return self.proj(pixel_values)


class InnerModel(nn.Module):
    """Mimics a HF multimodal backbone: it runs the vision encoder itself when given `pixel_values`."""

    def __init__(self, visual):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, HIDDEN)
        self.visual = visual
        self.received = None

    def forward(self, input_ids=None, inputs_embeds=None, pixel_values=None, attention_mask=None):
        self.received = {'input_ids': input_ids, 'inputs_embeds': inputs_embeds, 'pixel_values': pixel_values}
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if pixel_values is not None:
                inputs_embeds = inputs_embeds + self.visual(pixel_values).mean()
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class MultimodalModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.visual = CountingVisual()
        self.model = InnerModel(self.visual)
        self.lm_head = nn.Linear(HIDDEN, 16)

    @property
    def device(self):
        return torch.device('cpu')

    def forward(self, input_ids=None, inputs_embeds=None, pixel_values=None, attention_mask=None):
        raise AssertionError('replaced by _forward_redirection')


class DummyImageTemplate(Template):
    """`_post_encode` in the style of `Qwen2VLTemplate._get_inputs_embeds_hf`: text-only samples go through the
    vision encoder with a dummy image, so every rank calls it once per forward."""

    def _post_encode(self, model, inputs):
        inputs_embeds = model.model.embed_tokens(inputs['input_ids'])
        pixel_values = inputs.get('pixel_values')
        if pixel_values is None:
            inputs_embeds = inputs_embeds + model.visual(torch.zeros(1, HIDDEN)).mean() * 0.
        else:
            inputs_embeds = inputs_embeds + model.visual(pixel_values).mean()
        return {'inputs_embeds': inputs_embeds}


class Engine(nn.Module):
    """Stands in for the DeepSpeed engine that `_forward_redirection` routes the call through."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def make_trainer(model):
    template = DummyImageTemplate.__new__(DummyImageTemplate)
    template.mode = 'rlhf'
    # The same registration as `Template.register_post_encode_hook`.
    model.register_forward_pre_hook(template.pre_forward_hook, with_kwargs=True)

    trainer = object.__new__(GRPOTrainer)
    trainer.model = Engine(model)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapped: wrapped.module, is_main_process=True)
    trainer.args = SimpleNamespace(report_to=[])
    trainer._forward_redirection = _ForwardRedirection()
    trainer.use_liger_loss = True
    trainer.is_multimodal = True
    trainer.chord_sft_iterator = None
    trainer.model_kwarg_keys = []
    trainer.liger_inputs = []

    def compute_liger_loss(unwrapped_model, model_inputs, grpo_batch):
        trainer.liger_inputs.append(model_inputs)
        # `compute_liger_loss` selects the completion tokens from `input_ids`.
        assert model_inputs['input_ids'] is not None
        return trainer._get_last_hidden_state(unwrapped_model, model_inputs, logits_to_keep=1).sum()

    trainer.compute_liger_loss = compute_liger_loss
    return trainer


class TestGRPOLigerMultimodalForward(unittest.TestCase):

    def test_liger_loss_runs_the_vision_encoder_once_per_forward(self):
        for with_image in (False, True):
            with self.subTest(with_image=with_image):
                model = MultimodalModel()
                trainer = make_trainer(model)
                model_inputs = {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.ones(1, 3, dtype=torch.long)
                }
                if with_image:
                    model_inputs['pixel_values'] = torch.ones(2, HIDDEN)

                loss = trainer.compute_loss(trainer.model, {'model_inputs': model_inputs, 'grpo_batch': None})

                self.assertTrue(torch.is_tensor(loss))
                # The same count with and without an image keeps the ZeRO-3 all-gathers aligned across ranks.
                self.assertEqual(model.visual.calls, 1)
                received = model.model.received
                self.assertIsNotNone(received['inputs_embeds'])
                self.assertIsNone(received['input_ids'])
                self.assertIsNone(received['pixel_values'])
                self.assertTrue(torch.equal(trainer.liger_inputs[0]['input_ids'], model_inputs['input_ids']))


if __name__ == '__main__':
    unittest.main()
