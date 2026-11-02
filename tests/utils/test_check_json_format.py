# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import numpy as np
import unittest

from swift.utils import check_json_format


class TestCheckJsonFormat(unittest.TestCase):

    def test_numpy_scalars(self):
        self.assertEqual(check_json_format(np.float32(1.5)), 1.5)
        self.assertEqual(check_json_format(np.int64(3)), 3)
        self.assertIs(check_json_format(np.bool_(True)), True)
        self.assertIsInstance(check_json_format(np.float32(1.5)), float)

    def test_numpy_arrays(self):
        self.assertEqual(check_json_format(np.array([1, 2, 3])), [1, 2, 3])
        self.assertEqual(check_json_format(np.array([[1, 2], [3, 4]])), [[1, 2], [3, 4]])

    def test_numpy_values_inside_metrics(self):
        metrics = {
            'mean_neg': np.float32(-0.25),
            'mean_pos': np.float32(0.75),
            'margin': 1.0,
            'per_class': np.array([0.1, 0.2]),
        }
        result = check_json_format(metrics)
        self.assertEqual(result, {'mean_neg': -0.25, 'mean_pos': 0.75, 'margin': 1.0, 'per_class': [0.1, 0.2]})
        json.dumps(result)

    def test_regular_values_are_unchanged(self):
        obj = {'a': 1, 'b': 1.5, 'c': 'text', 'd': None, 'e': [True, False]}
        self.assertEqual(check_json_format(obj), obj)


if __name__ == '__main__':
    unittest.main()
