import os
import tempfile
import unittest

from swift.cli.main import parse_yaml_args


class TestYamlConfigArgv(unittest.TestCase):
    """`swift sft config.yaml` (and the JSON form) rewrites argv in place, and the launcher then
    prints and execs it as one command line, so every value has to arrive as a string."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        # parse_yaml_args records the config path for the entry point that runs after it
        self._saved_config = os.environ.pop('SWIFT_CONFIG_FILE', None)
        self.addCleanup(self._restore_config)

    def _restore_config(self):
        os.environ.pop('SWIFT_CONFIG_FILE', None)
        if self._saved_config is not None:
            os.environ['SWIFT_CONFIG_FILE'] = self._saved_config

    def parse(self, name, content):
        path = os.path.join(self._dir.name, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        argv = [path]
        parse_yaml_args(argv)
        return argv

    def assert_launchable(self, argv):
        try:
            ' '.join(argv)
        except TypeError as exc:
            self.fail(f'the launcher cannot build a command line from {argv!r}: {exc}')

    def test_float_list_is_passed_as_strings(self):
        # e.g. `interleave_prob: [0.5, 0.5]`, declared Optional[List[float]] in DataArguments
        argv = self.parse('sft.yaml', 'dataset:\n- a\n- b\ninterleave_prob: [0.5, 0.5]\n')
        self.assert_launchable(argv)
        self.assertEqual(argv[-3:], ['--interleave_prob', '0.5', '0.5'])

    def test_int_list_is_passed_as_strings(self):
        # e.g. `data_range: [0, 2]`, declared List[int] in SamplingArguments
        argv = self.parse('sample.json', '{"num_samples": 10, "data_range": [0, 2]}')
        self.assert_launchable(argv)
        self.assertEqual(argv, ['--num_samples', '10', '--data_range', '0', '2'])

    def test_string_list_is_unchanged(self):
        argv = self.parse('datasets.yaml', 'dataset:\n- a#100\n- b#100\n')
        self.assertEqual(argv, ['--dataset', 'a#100', 'b#100'])

    def test_dict_value_is_still_serialized(self):
        argv = self.parse('engine.yaml', 'model: m\nvllm_engine_kwargs:\n  gpu_memory_utilization: 0.8\n')
        self.assert_launchable(argv)
        self.assertEqual(argv, ['--model', 'm', '--vllm_engine_kwargs', '{"gpu_memory_utilization": 0.8}'])


if __name__ == '__main__':
    unittest.main()
