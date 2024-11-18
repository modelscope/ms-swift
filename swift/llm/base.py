# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Generic, List, Optional, Type, TypeVar, Union

from swift.llm import BaseArguments
from swift.utils import get_logger, parse_args, seed_everything

logger = get_logger()


class SwiftPipeline(ABC):
    args_class = BaseArguments

    def __init__(self, args: Union[List[str], args_class, None] = None):
        self.args = self._parse_args(args)

    def _parse_args(self, args: Union[List[str], args_class, None] = None) -> args_class:
        if isinstance(args, self.args_class):
            return args
        assert self.args_class is not None
        args, remaining_argv = parse_args(self.args_class, args)
        if len(remaining_argv) > 0:
            if getattr(args, 'ignore_args_error', False):
                logger.warning(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        return args

    @staticmethod
    def _compat_dsw_gradio(args) -> None:
        from swift.llm import WebUIArguments
        if (isinstance(args, WebUIArguments) and 'JUPYTER_NAME' in os.environ and 'dsw-' in os.environ['JUPYTER_NAME']
                and 'GRADIO_ROOT_PATH' not in os.environ):
            os.environ['GRADIO_ROOT_PATH'] = f"/{os.environ['JUPYTER_NAME']}/proxy/{args.port}"

    def main(self):
        args = self.args
        self._compat_dsw_gradio(args)
        logger.info(f'Start time of running main: {datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        logger.info(f'args: {args}')
        seed_everything(args.seed)
        result = self.run()
        logger.info(f'End time of running main: {datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
        return result

    @abstractmethod
    def run(self):
        pass
