# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import tempfile
import unittest
from pathlib import Path

from swift.arguments import BaseArguments
from swift.cli.main import parse_yaml_args


class TestSaveArgs(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.saved_config = os.environ.pop('SWIFT_CONFIG_FILE', None)
        self.addCleanup(self.restore_env)

    def restore_env(self):
        os.environ.pop('SWIFT_CONFIG_FILE', None)
        if self.saved_config is not None:
            os.environ['SWIFT_CONFIG_FILE'] = self.saved_config

    def run_case(self, filename, same_directory, path_type, link_type=None):
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir(exist_ok=True)
        (checkpoint / 'args.json').write_text(
            json.dumps({
                'swift_version': '4.0.0',
                'model': 'local-placeholder',
                'tuner_type': 'full'
            }),
            encoding='utf-8')
        output = self.root / ('output' if link_type is None else f'output-{link_type}')
        output.mkdir(exist_ok=True)
        config = (output if same_directory else self.root) / filename
        content = 'model: local-placeholder\n' if filename.endswith('.yaml') else '{"model": "local-placeholder"}\n'
        config.write_text(content, encoding='utf-8')
        if link_type == 'symlink':
            (output / filename).symlink_to(config)
        elif link_type == 'hardlink':
            os.link(config, output / filename)
        argv = [str(config)]
        parse_yaml_args(argv)
        self.assertEqual(argv, ['--model', 'local-placeholder'])
        args = BaseArguments.from_pretrained(str(checkpoint))
        args.swift_version = '4.0.0'
        args.model_type = 'qwen2_5'
        args.max_model_len = 4096
        expected = dict(args.__dict__)
        args.save_args(path_type(output))
        self.assertEqual(json.loads((output / 'args.json').read_text(encoding='utf-8')), expected)
        saved_name = 'config_args.json' if filename == 'args.json' else filename
        self.assertEqual((output / saved_name).read_text(encoding='utf-8'), content)
        if not (same_directory and filename == 'args.json'):
            self.assertEqual(config.read_text(encoding='utf-8'), content)
        recovered = BaseArguments.from_pretrained(str(output))
        self.assertEqual(recovered.tuner_type, 'full')
        self.assertEqual(recovered.model_type, 'qwen2_5')
        self.assertEqual(recovered.max_model_len, 4096)

    def test_config_saved_without_overwriting_resolved_args(self):
        for filename in ['args.json', 'config.json', 'config.yaml']:
            for same_directory in [False, True]:
                for path_type in [str, Path]:
                    with self.subTest(filename=filename, same_directory=same_directory, path_type=path_type):
                        self.run_case(filename, same_directory, path_type)

    def test_save_without_config(self):
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'args.json').write_text('{"model": "local-placeholder"}', encoding='utf-8')
        args = BaseArguments.from_pretrained(str(checkpoint))
        args.output_dir = str(self.root / 'output')
        args.save_args()
        self.assertEqual(json.loads((Path(args.output_dir) / 'args.json').read_text(encoding='utf-8')), args.__dict__)

    def test_config_already_linked_in_output_directory(self):
        for link_type in ['symlink', 'hardlink']:
            with self.subTest(link_type=link_type):
                self.run_case('config.json', False, str, link_type=link_type)


if __name__ == '__main__':
    unittest.main()
