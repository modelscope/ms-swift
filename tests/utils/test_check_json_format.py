# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import numpy as np
import unittest

from swift.utils import check_json_format


class TestCheckJsonFormat(unittest.TestCase):

    def test_numpy_scalar_becomes_a_python_scalar(self):
        self.assertEqual(check_json_format(np.float32(1.5)), 1.5)
        self.assertEqual(check_json_format(np.float64(1.5)), 1.5)
        self.assertEqual(check_json_format(np.int64(3)), 3)
        self.assertIs(check_json_format(np.bool_(True)), True)

    def test_numpy_array_becomes_a_list(self):
        self.assertEqual(check_json_format(np.array([1, 2], dtype=np.int64)), [1, 2])
        self.assertEqual(check_json_format(np.array([[1.0, 2.0]], dtype=np.float32)), [[1.0, 2.0]])

    def test_numpy_values_nested_in_containers(self):
        res = check_json_format({'loss': np.float32(0.5), 'logits': [np.int64(1), np.float32(2.0)]})
        self.assertEqual(res, {'loss': 0.5, 'logits': [1, 2.0]})

    def test_object_without_dict_attribute_does_not_raise(self):
        # `set` has no `__dict__`, the previous implementation raised an AttributeError here.
        self.assertEqual(check_json_format({1, 2}), '{1, 2}')

    def test_result_is_json_serializable(self):
        res = check_json_format({'metric': np.float32(0.5), 'values': np.array([1, 2], dtype=np.int64)})
        self.assertEqual(json.loads(json.dumps(res)), {'metric': 0.5, 'values': [1, 2]})

    def test_token_fields_are_still_masked(self):

        class Dummy:

            def __init__(self):
                self.input_token_ids = 'secret'
                self.value = 1

            def __repr__(self):
                return f'Dummy({self.input_token_ids}, {self.value})'

        dummy = Dummy()
        self.assertEqual(check_json_format(dummy), 'Dummy(None, 1)')
        self.assertEqual(dummy.input_token_ids, 'secret')  # the object is restored


if __name__ == '__main__':
    unittest.main()
