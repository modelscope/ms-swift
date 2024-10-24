import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Optional, Type, TypeVar, Union

from swift.utils import get_logger, parse_args, seed_everything
from .argument import BaseArguments

logger = get_logger()

T_Args = TypeVar('T_Args', bound=BaseArguments)


class Pipeline(ABC):
    args_class = None

    def parse_args(self, args: Union[List[str], T_Args, None] = None) -> T_Args:
        if isinstance(args, BaseArguments):
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
        from swift.llm import AppUIArguments, WebuiArguments
        if (isinstance(args, (AppUIArguments, WebuiArguments)) and 'JUPYTER_NAME' in os.environ
                and 'dsw-' in os.environ['JUPYTER_NAME'] and 'GRADIO_ROOT_PATH' not in os.environ):
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
