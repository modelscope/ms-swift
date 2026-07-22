import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swift.template.templates.gemma import Gemma3Template, Gemma3VisionTemplate


class TestGemma3VisionTemplate(unittest.TestCase):

    def test_text_only_encode_has_token_type_ids(self):
        template = object.__new__(Gemma3VisionTemplate)
        encoded = {'input_ids': [2, 10, 20], 'labels': [-100, 10, 20]}
        inputs = SimpleNamespace(images=[])

        with patch.object(Gemma3Template, '_encode', return_value=encoded):
            result = template._encode(inputs)

        self.assertEqual(result['token_type_ids'], [0, 0, 0])
