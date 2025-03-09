# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import BaseArguments
from .megatron_args import MegatronArguments
import sys
from typing import Tuple, List, Any, Dict
from dataclasses import dataclass, asdict

@dataclass
class MegatronTrainArguments(MegatronArguments, BaseArguments):
    def __post_init__(self):
        BaseArguments.__post_init__(self)
        MegatronArguments.__post_init__(self)
        self.seq_length = self.model_info.max_model_len

    def _args_to_argv(self) -> Tuple[List[Any], Dict[str, Any]]:
        new_args = []
        args_dict = asdict(self)
        extra_args = {}
        for k, value in args_dict.items():
            if k in self.add_prefix_no:
                k = f'no_{k}'
                value = not value
            if k not in MegatronArguments.__annotations__:
                extra_args[k] = value
                continue
            if value is None or value is False:
                continue
            new_args.append(f"--{k.replace('_', '-')}")
            if isinstance(value, list):
                new_args += [str(v) for v in value]
            elif value is not True:
                new_args.append(str(value))

        return new_args, extra_args

    def parse_to_megatron(self):
        new_args, extra_args = self._args_to_argv()
        sys._old_argv = sys.argv
        sys.argv = sys.argv[:1] + new_args

        return extra_args
