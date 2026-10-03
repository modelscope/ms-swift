# Copyright (c) ModelScope Contributors. All rights reserved.
"""Import external plugin files by name, including during unpickling in fresh workers."""
import importlib.abc
import importlib.util
import os
import sys


class _PluginAliasLoader(importlib.abc.Loader):

    def __init__(self, name):
        self.name = name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Reuse the canonical module without overwriting its name, spec or loader.
        sys.modules[module.__name__] = importlib.import_module(self.name)


class _ExternalPluginFinder(importlib.abc.MetaPathFinder):

    def find_spec(self, fullname, path=None, target=None):
        root, _, suffix = fullname.partition('.')
        if root.startswith('_swift_external_') and suffix:
            package = sys.modules.get(root)
            if package is not None and package.__name__.startswith(__name__ + '.'):
                name = f'{package.__name__}.{suffix}'
                spec = importlib.util.find_spec(name)
                if spec is not None:
                    return importlib.util.spec_from_loader(
                        fullname, _PluginAliasLoader(name), is_package=spec.submodule_search_locations is not None)
        parent, _, name = fullname.rpartition('.')
        if parent != __name__ or not name.startswith('_'):
            return None
        try:
            file_path = os.fsdecode(bytes.fromhex(name[1:]))
        except ValueError:
            return None
        if not os.path.isabs(file_path):
            return None
        return importlib.util.spec_from_file_location(fullname, file_path)


# Resolve legacy aliases before the path finder can load a second copy from the package directory.
sys.meta_path.insert(0, _ExternalPluginFinder())
