# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest

from swift.template.templates.gemma import DiffusionGemmaTemplate, Gemma4Template


def _row(input_ids, **kwargs):
    return {
        'input_ids': input_ids,
        'labels': input_ids,
        'length': len(input_ids),
        'mm_token_type_ids': torch.zeros(len(input_ids), dtype=torch.long),
        **kwargs,
    }


class TestGemma4TextPacking(unittest.TestCase):

    def setUp(self):
        # `packing_row` only reads the rows, so the processor is not needed.
        self.template = object.__new__(Gemma4Template)
        self.template.use_megatron = True

    def test_support_padding_free(self):
        self.assertTrue(Gemma4Template.support_padding_free)
        self.assertFalse(DiffusionGemmaTemplate.support_padding_free)

    def test_text_only_rows_are_packed(self):
        packed = self.template.packing_row([_row([2, 10, 11]), _row([2, 20])])
        self.assertEqual(packed['input_ids'], [2, 10, 11, 2, 20])
        self.assertEqual(packed['position_ids'], [0, 1, 2, 0, 1])
        self.assertEqual(packed['mm_token_type_ids'].tolist(), [0, 0, 0, 0, 0])
        self.assertNotIn('pixel_values', packed)

    def test_media_rows_are_rejected(self):
        for key in ['pixel_values', 'pixel_values_videos', 'input_features']:
            with self.subTest(key=key):
                rows = [_row([2, 10, 11]), _row([2, 20], **{key: torch.zeros(1, 4)})]
                with self.assertRaisesRegex(ValueError, 'text-only'):
                    self.template.packing_row(rows)

    def test_transformers_backend_is_rejected(self):
        self.template.use_megatron = False
        with self.assertRaisesRegex(ValueError, 'only with Megatron'):
            self.template.packing_row([_row([2, 10, 11]), _row([2, 20])])


if __name__ == '__main__':
    unittest.main()
