import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestFSDPTurboImports(unittest.TestCase):

    def test_public_package_and_cli_import_without_optional_backend(self):
        repository = Path(__file__).resolve().parents[2]
        script = textwrap.dedent("""
            import builtins
            import dataclasses
            import sys
            import types

            real_import = builtins.__import__
            optional_roots = {'fsdp_turbo', 'fsdpturbo'}

            def guarded_import(name, *args, **kwargs):
                if name.partition('.')[0] in optional_roots:
                    raise AssertionError(f'eager optional import: {name}')
                return real_import(name, *args, **kwargs)

            class GuardedFinder:

                def find_spec(self, fullname, path=None, target=None):
                    if fullname.partition('.')[0] in optional_roots:
                        raise AssertionError(f'eager optional import: {fullname}')

            builtins.__import__ = guarded_import
            sys.meta_path.insert(0, GuardedFinder())
            import swift.fsdpturbo
            import swift.utils

            swift.utils.get_logger = lambda: None

            arguments = types.ModuleType('swift.arguments')

            @dataclasses.dataclass
            class SftArguments:
                pass

            arguments.SftArguments = SftArguments
            sys.modules['swift.arguments'] = arguments

            import swift.cli.fsdpturbo
            from swift.fsdpturbo.arguments import FSDPTurboArguments
            from swift.fsdpturbo.model_specs import get_model_spec

            FSDPTurboArguments().validate_fsdpturbo(world_size=1)
            assert callable(get_model_spec)
            """)

        result = subprocess.run([sys.executable, '-c', script],
                                cwd=repository,
                                capture_output=True,
                                text=True,
                                check=False)

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
