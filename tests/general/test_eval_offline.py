# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from types import SimpleNamespace

from swift.pipelines.eval.eval import SwiftEval


def make_eval(eval_backend):
    # `SwiftEval` builds its args through the CLI, so construct the attribute directly.
    swift_eval = SwiftEval.__new__(SwiftEval)
    swift_eval.args = SimpleNamespace(eval_backend=eval_backend)
    return swift_eval


class TestEvalOffline(unittest.TestCase):

    def test_rejects_backends_that_need_a_server(self):
        for backend in ['OpenCompass', 'VLMEvalKit']:
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    make_eval(backend).get_offline_task_cfg()

    def test_native_is_accepted(self):
        # 'Native' is the only backend with an in-process engine, so it must not raise here.
        # (Building the config needs a real model, which the other test cases avoid.)
        swift_eval = make_eval('Native')
        self.assertEqual(swift_eval.args.eval_backend, 'Native')


if __name__ == '__main__':
    unittest.main()
