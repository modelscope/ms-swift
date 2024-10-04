import inspect
import os
from typing import List, Union

import json


class BaseArguments:

    def load_json_or_path(self: Union['SftArguments', 'InferArguments'], key) -> None:
        """Convert a JSON string or JSON file into a dict"""
        value = getattr(self, key)
        if value is None:
            value = {}
        elif isinstance(value, str):
            if os.path.exists(value):  # local path
                with open(value, 'r') as f:
                    value = json.load(f)
            else:  # json str
                value = json.loads(value)
        setattr(self, key, value)

    @staticmethod
    def check_path_validity(path: Union[str, List[str]], check_path_exist: bool = False) -> Union[str, List[str]]:
        """Check the path for validity and convert it to an absolute path.

        Args:
            path: The path to be checked/converted
            check_path_exist: Whether to check if the path exists

        Returns:
            Absolute path
        """
        if isinstance(path, str):
            # Remove user path prefix and convert to absolute path.
            path = os.path.expanduser(path)
            path = os.path.abspath(path)
            if check_path_exist and not os.path.exists(path):
                raise FileNotFoundError(f"path: '{path}'")
            return path
        assert isinstance(path, list), f'path: {path}'
        res = []
        for v in path:
            res.append(ArgumentsBase.check_path_validity(v, check_path_exist))
        return res

    def handle_path(self: Union['SftArguments', 'InferArguments']) -> None:
        """Check all paths in the args correct and exist"""
        check_exist_path = {'ckpt_dir', 'resume_from_checkpoint', 'custom_register_path', 'deepspeed_config_path'}
        other_path = {'output_dir', 'logging_dir'}
        # If it is a path, check it.
        maybe_check_exist_path = ['model_id_or_path', 'custom_dataset_info', 'deepspeed']
        for k in maybe_check_exist_path:
            v = getattr(self, k, None)
            if isinstance(v, str) and (v.startswith('~') or v.startswith('/') or os.path.exists(v)):
                check_exist_path.add(k)
        # check path
        for k in check_exist_path | other_path:
            value = getattr(self, k, None)
            if value is None:
                continue
            value = self.check_path_validity(value, k in check_exist_path)
            setattr(self, k, value)
