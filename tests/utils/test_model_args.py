from types import SimpleNamespace
import unittest

from swift.arguments.base_args.model_args import ModelArguments


class TestModelArguments(unittest.TestCase):

    @staticmethod
    def _build_args(rope_scaling, max_model_len=8192):
        args = ModelArguments.__new__(ModelArguments)
        args.rope_scaling = rope_scaling
        args.max_model_len = max_model_len
        args.model_info = SimpleNamespace(rope_scaling={}, max_model_len=4096)
        return args

    def test_init_rope_scaling_adds_rope_type_for_string_value(self):
        args = self._build_args('yarn')

        args._init_rope_scaling()

        self.assertEqual(args.rope_scaling['type'], 'yarn')
        self.assertEqual(args.rope_scaling['rope_type'], 'yarn')
        self.assertEqual(args.rope_scaling['original_max_position_embeddings'], 4096)
        self.assertEqual(args.rope_scaling['factor'], 2.0)

    def test_init_rope_scaling_preserves_legacy_type_and_backfills_rope_type(self):
        args = self._build_args('{"type": "dynamic", "factor": 1.5}')

        args._init_rope_scaling()

        self.assertEqual(args.rope_scaling['type'], 'dynamic')
        self.assertEqual(args.rope_scaling['rope_type'], 'dynamic')
        self.assertEqual(args.rope_scaling['factor'], 1.5)


if __name__ == '__main__':
    unittest.main()