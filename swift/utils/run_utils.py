from typing import Callable, List, Type, TypeVar, Union

from .logger import get_logger
from .utils import parse_args

logger = get_logger()
_TArgsClass = TypeVar('_TArgsClass')
_T = TypeVar('_T')
NoneType = type(None)


def get_main(
    args_class: Type[_TArgsClass], llm_x: Callable[[_TArgsClass], _T]
) -> Callable[[Union[List[str], _TArgsClass, NoneType]], _T]:

    def x_main(argv: Union[List[str], _TArgsClass, NoneType] = None,
               **kwargs) -> _T:
        if isinstance(argv, args_class):
            args, remaining_argv = argv, []
        else:
            args, remaining_argv = parse_args(args_class, argv)
        if len(remaining_argv) > 0:
            if getattr(args, 'ignore_args_error', False):
                logger.warning(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        return llm_x(args, **kwargs)

    return x_main
