import sys
import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from swift.utils import unwrap_model_for_generation


class TestUnwrapModelForGeneration(unittest.TestCase):

    @staticmethod
    def _make_dependencies(events):

        @contextmanager
        def gathered_parameters(_parameters):
            events.append('gather-enter')
            try:
                yield
            finally:
                events.append('gather-exit')

        deepspeed = ModuleType('deepspeed')
        deepspeed.zero = SimpleNamespace(GatheredParameters=gathered_parameters)
        trl_models_utils = ModuleType('trl.models.utils')
        trl_models_utils.remove_hooks = lambda _model: events.append('remove-hooks')
        trl_models_utils.add_hooks = lambda _model: events.append('add-hooks')
        return {'deepspeed': deepspeed, 'trl.models.utils': trl_models_utils}

    @staticmethod
    def _make_model_and_accelerator():
        model = Mock()
        model.named_parameters.return_value = []
        accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=3)),
            unwrap_model=Mock(return_value=model),
        )
        return model, accelerator

    def test_restores_zero3_hooks_after_normal_exit(self):
        events = []
        model, accelerator = self._make_model_and_accelerator()

        with patch.dict(sys.modules, self._make_dependencies(events)):
            with unwrap_model_for_generation(model, accelerator):
                events.append('generate')

        self.assertEqual(events, ['gather-enter', 'remove-hooks', 'generate', 'add-hooks', 'gather-exit'])

    def test_restores_zero3_hooks_after_error(self):
        events = []
        model, accelerator = self._make_model_and_accelerator()
        with (
                patch.dict(sys.modules, self._make_dependencies(events)),
                self.assertRaisesRegex(RuntimeError, 'generation failed'),
        ):
            with unwrap_model_for_generation(model, accelerator):
                events.append('generate')
                raise RuntimeError('generation failed')

        self.assertEqual(events, ['gather-enter', 'remove-hooks', 'generate', 'add-hooks', 'gather-exit'])


if __name__ == '__main__':
    unittest.main()
