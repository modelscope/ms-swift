# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import tempfile
import unittest
from unittest import mock

from swift.dataset import (DatasetMeta, DatasetSyntax, dataset_syntax, load_dataset, register_dataset,
                           register_dataset_info)
from swift.dataset.dataset_meta import DATASET_MAPPING
from swift.dataset.preprocessor import ResponsePreprocessor


class TestDatasetRegistrationCache(unittest.TestCase):

    def setUp(self):
        for patch in [
                mock.patch.dict(DATASET_MAPPING, clear=True),
                mock.patch.object(dataset_syntax, '_dataset_meta_mapping', None)
        ]:
            patch.start()
            self.addCleanup(patch.stop)

    def test_registration_after_lookup(self):
        DatasetSyntax.parse('org/initial').get_dataset_meta(use_hf=True)
        meta = DatasetMeta(dataset_name='custom', hf_dataset_id='org/new', ms_dataset_id='org/new-ms')
        register_dataset(meta)
        self.assertIs(DatasetSyntax.parse('org/new').get_dataset_meta(use_hf=True), meta)
        self.assertIs(DatasetSyntax.parse('org/new-ms').get_dataset_meta(use_hf=False), meta)

    def test_replacement_removes_old_aliases(self):
        original = DatasetMeta(dataset_name='custom', hf_dataset_id='org/old', ms_dataset_id='org/shared')
        register_dataset(original)
        self.assertIs(DatasetSyntax.parse('org/old').get_dataset_meta(use_hf=True), original)
        replacement = DatasetMeta(dataset_name='custom', hf_dataset_id='org/new', ms_dataset_id='org/shared')
        register_dataset(replacement, exist_ok=True)
        self.assertIs(DatasetSyntax.parse('org/new').get_dataset_meta(use_hf=True), replacement)
        self.assertIs(DatasetSyntax.parse('org/shared').get_dataset_meta(use_hf=False), replacement)
        self.assertIsNot(DatasetSyntax.parse('org/old').get_dataset_meta(use_hf=True), original)
        with self.assertRaises(ValueError):
            register_dataset(original)
        self.assertIs(DatasetSyntax.parse('org/new').get_dataset_meta(use_hf=True), replacement)

    def test_late_registration_changes_loaded_preprocessor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'custom.jsonl')
            with open(path, 'w') as stream:
                stream.write(json.dumps({'custom_query': 'question', 'first_answer': 'first', 'next_answer': 'next'}))
            # A prior lookup, as happens after loading another dataset in a notebook.
            DatasetSyntax.parse('org/initial').get_dataset_meta(use_hf=True)
            registered = register_dataset_info([{
                'dataset_path': path,
                'columns': {
                    'custom_query': 'query',
                    'first_answer': 'response'
                },
            }])
            first, _ = load_dataset(path, strict=True, num_proc=1, shuffle=False, load_from_cache_file=False)
            self.assertEqual(first[0]['messages'], [{
                'role': 'user',
                'content': 'question'
            }, {
                'role': 'assistant',
                'content': 'first'
            }])
            register_dataset(
                DatasetMeta(
                    dataset_path=path,
                    preprocess_func=ResponsePreprocessor(columns={
                        'custom_query': 'query',
                        'next_answer': 'response'
                    })),
                exist_ok=True)
            second, _ = load_dataset(path, strict=True, num_proc=1, shuffle=False, load_from_cache_file=False)
            self.assertEqual(second[0]['messages'][-1]['content'], 'next')
            self.assertIsNot(DatasetSyntax.parse(path).get_dataset_meta(use_hf=True), registered[0])


if __name__ == '__main__':
    unittest.main()
