import unittest
from types import SimpleNamespace

from swift.template.templates.internvl import InternvlTemplate


class TestInternvlTemplate(unittest.TestCase):

    @staticmethod
    def _template(mode):
        template = object.__new__(InternvlTemplate)
        template.mode = mode
        template.max_num = 6
        return template

    def test_vllm_receives_max_dynamic_patch(self):
        template = self._template('vllm')
        inputs = SimpleNamespace(mm_processor_kwargs={})

        context = template.replace_tag('image', 0, inputs)

        self.assertEqual(context, ['<image>\n'])
        self.assertEqual(inputs.mm_processor_kwargs, {'max_dynamic_patch': 6})

    def test_vllm_preserves_explicit_max_dynamic_patch(self):
        template = self._template('vllm')
        inputs = SimpleNamespace(mm_processor_kwargs={'max_dynamic_patch': 4})

        template.replace_tag('image', 0, inputs)

        self.assertEqual(inputs.mm_processor_kwargs, {'max_dynamic_patch': 4})

    def test_transformers_does_not_receive_vllm_processor_kwargs(self):
        template = self._template('transformers')
        inputs = SimpleNamespace(mm_processor_kwargs={})

        context = template.replace_tag('image', 0, inputs)

        self.assertEqual(context, ['<img>', [-100], '</img>\n'])
        self.assertEqual(inputs.mm_processor_kwargs, {})


if __name__ == '__main__':
    unittest.main()
