import ast
import unittest
from pathlib import Path


class TestRlhfUtilsMsgspecImport(unittest.TestCase):

    def test_msgspec_is_imported_only_when_vllm_is_available(self):
        path = Path(__file__).parents[2] / 'swift' / 'rlhf_trainers' / 'utils.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'msgspec':
                self.fail('msgspec is imported while loading swift.rlhf_trainers.utils')
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                if any(name == 'msgspec' or name.startswith('msgspec.') for name in names):
                    self.fail('msgspec is imported while loading swift.rlhf_trainers.utils')

        imported_inside_vllm_guard = False
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            call = node.test
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not isinstance(func, ast.Name) or func.id != 'is_vllm_available':
                continue
            for child in node.body:
                if isinstance(child, ast.ImportFrom) and child.module == 'msgspec':
                    imported_inside_vllm_guard = True
        self.assertTrue(imported_inside_vllm_guard)


if __name__ == '__main__':
    unittest.main()
