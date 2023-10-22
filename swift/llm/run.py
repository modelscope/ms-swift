from typing import Callable, List, Optional, Type, TypeVar

from swift.llm import InferArguments, SftArguments, llm_infer, llm_sft
from swift.utils import get_logger, parse_args

logger = get_logger()

_TArgsClass = TypeVar('_TArgsClass')
_T = TypeVar('_T')


def get_main(
        args_class: Type[_TArgsClass],
        llm_x: Callable[[_TArgsClass],
                        _T]) -> Callable[[Optional[List[str]]], _T]:

    def x_main(argv: Optional[List[str]] = None) -> _T:
        args, remaining_argv = parse_args(args_class, argv)
        args.init_argument()
        if len(remaining_argv) > 0:
            if args.ignore_args_error:
                logger.warning(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        return llm_x(args)

    return x_main


sft_main = get_main(SftArguments, llm_sft)
infer_main = get_main(InferArguments, llm_infer)
