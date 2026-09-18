import torch
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swift.template import TEMPLATE_MAPPING, TemplateType
from swift.template.templates.gemma import Gemma3Template, Gemma3VisionTemplate


class TestGemma3VisionTemplate(unittest.TestCase):

    def test_text_only_encode_has_token_type_ids(self):
        template = object.__new__(Gemma3VisionTemplate)
        encoded = {'input_ids': [2, 10, 20], 'labels': [-100, 10, 20]}
        inputs = SimpleNamespace(images=[])

        with patch.dict('sys.modules', {'transformers.models.gemma3.processing_gemma3': None}):
            with patch.object(Gemma3Template, '_encode', return_value=encoded):
                result = template._encode(inputs)

        self.assertEqual(result['token_type_ids'], [0, 0, 0])

    def test_truncation_keeps_token_type_ids_aligned(self):
        # BOI=99, image=90, EOI=98; text appears before and after each image.
        single_image = [10, 11, 12, 13, 99, 90, 90, 98, 14, 15, 16, 17]
        two_images = [10, 11, 99, 90, 90, 98, 12, 13, 99, 90, 90, 98, 14, 15]
        cases = [
            (single_image, 'left', list(range(4, 12))),
            (single_image, 'right', list(range(8))),
            # Protected BOI tokens make these retention sets non-contiguous.
            (two_images, 'left', [2, 8, 11, 12, 13]),
            (two_images, 'right', [0, 1, 2, 3, 8]),
        ]
        for input_ids, strategy, kept in cases:
            with self.subTest(strategy=strategy, kept=kept):
                template = Gemma3VisionTemplate(
                    None,
                    TEMPLATE_MAPPING[TemplateType.gemma3_vision],
                    max_length=len(kept),
                    truncation_strategy=strategy)
                template.placeholder_tokens = [99]
                template.processor = SimpleNamespace(pad_token_id=0)
                template.mode = 'train'
                encoded = {
                    'input_ids': input_ids,
                    'labels': input_ids.copy(),
                    'loss_scale': list(range(1,
                                             len(input_ids) + 1)),
                    'token_type_ids': [int(token == 90) for token in input_ids],
                    'mm_token_type_ids': torch.arange(len(input_ids)),
                }
                with patch.object(template, '_preprocess_inputs'), patch.object(
                        template, '_encode', return_value=encoded):
                    result = template._encode_truncated(None)

                expected_ids = [input_ids[i] for i in kept]
                expected = {
                    'input_ids': expected_ids,
                    'labels': [-100] + expected_ids[1:],
                    'loss_scale': [0] + [i + 1 for i in kept[1:]],
                    'token_type_ids': [int(token == 90) for token in expected_ids],
                    'mm_token_type_ids': kept,
                }
                self.assertEqual(result['length'], len(kept))
                for key, value in expected.items():
                    self.assertEqual(torch.as_tensor(result[key]).tolist(), value)

                batch = template._data_collator([result])
                for key, value in expected.items():
                    self.assertEqual(tuple(batch[key].shape), (1, len(kept)))
                    self.assertEqual(batch[key].tolist(), [value])
