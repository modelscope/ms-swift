"""Tests for swift-specific extensions retained after the twinkle-inheritance refactor.

Model self-building (no nn.Module injection) is satisfied by inheriting twinkle's Model
(its __init__ builds the model from model_id/config and never accepts an nn.Module). The old
injection-based tests are therefore removed.

What remains swift-specific and worth testing here is the collate hook on the swift
``InputProcessor`` subclass, which is exercised without constructing a Model.
"""
import torch
from unittest.mock import MagicMock


class TestTemplateCollateHook:
    """swift InputProcessor.collate_fn invokes the template collate_mm_data hook."""

    def test_collate_stage_calls_template_hook(self):
        from swift.dev.processor import InputProcessor

        template = MagicMock()
        template.collate_mm_data.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}
        template.post_collate.side_effect = lambda x: x

        processor = InputProcessor()
        processor._template = template

        inputs = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1])
            },
            {
                'input_ids': torch.tensor([4, 5, 6]),
                'attention_mask': torch.tensor([1, 1, 1])
            },
        ]

        result = processor.collate_fn(inputs)
        template.collate_mm_data.assert_called()
        assert 'pixel_values' in result[0]

    def test_external_collate_fn_takes_priority(self):
        """An explicit collate_fn short-circuits the template hook."""
        from swift.dev.processor import InputProcessor

        template = MagicMock()
        called = {'external': False}

        def external(inputs):
            called['external'] = True
            return inputs[0]

        processor = InputProcessor(collate_fn=external)
        processor._template = template

        inputs = [{'input_ids': torch.tensor([1, 2, 3])}]
        processor.collate_fn(inputs)

        assert called['external'] is True
        template.collate_mm_data.assert_not_called()
