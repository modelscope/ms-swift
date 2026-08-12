# Copyright (c) ModelScope Contributors. All rights reserved.
import subprocess
import sys
import unittest


class TestOptionalTemplateDependencies(unittest.TestCase):

    def test_template_import_without_qwen_vl_utils(self):
        code = """
import builtins

original_import = builtins.__import__


def import_without_qwen_vl_utils(name, *args, **kwargs):
    if name == 'qwen_vl_utils' or name.startswith('qwen_vl_utils.'):
        raise ModuleNotFoundError("No module named 'qwen_vl_utils'")
    return original_import(name, *args, **kwargs)


builtins.__import__ = import_without_qwen_vl_utils
import swift.template
"""
        subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    unittest.main()
