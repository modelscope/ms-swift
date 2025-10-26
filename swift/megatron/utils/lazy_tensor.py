import os
from functools import partial

import json
import safetensors.torch


class LazyTensor:

    def __init__(self, tensor=None, loader=None):
        """You need to provide a tensor or loader"""
        self.tensor = tensor
        self.loader = loader

    def load(self):
        if self.tensor is None:
            self.tensor = self.loader()
            self.loader = None
        return self.tensor


class SafetensorsLazyLoader:

    def __init__(self, hf_model_dir: str):
        self.hf_model_dir = hf_model_dir
        self._weight_map = {}
        self._file_handles = {}
        self._load_index()

    def _open_file(self, filename: str):
        """Open a safetensors file if not already open."""
        if filename not in self._file_handles:
            file_path = os.path.join(self.hf_model_dir, filename)
            self._file_handles[filename] = safetensors.torch.safe_open(file_path, framework='pt')
        return self._file_handles[filename]

    def _load_index(self):
        """Load the model index file to get weight map."""
        index_path = os.path.join(self.hf_model_dir, 'model.safetensors.index.json')

        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self._index_file = json.load(f)
                self._weight_map = self._index_file.get('weight_map', {})
        else:
            # Single file model
            safetensors_file = os.path.join(self.hf_model_dir, 'model.safetensors')
            if os.path.exists(safetensors_file):
                # All weights are in single file
                with safetensors.torch.safe_open(safetensors_file, framework='pt') as f:
                    for key in f.keys():
                        self._weight_map[key] = 'model.safetensors'

    def get_state_dict(self):
        res = {}
        for k in self._weight_map.keys():
            res[k] = LazyTensor(loader=partial(self._load_tensor, key=k))
        return res

    def _load_tensor(self, key):
        filename = self._weight_map[key]
        file_handle = self._open_file(filename)
        return file_handle.get_tensor(key)

    def close(self):
        for f in self._file_handles:
            f.close()
        self._file_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
