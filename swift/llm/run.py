from typing import List, Optional

from swift.llm import InferArguments, SftArguments, llm_infer, llm_sft
from swift.utils import get_logger, parse_args

logger = get_logger()


def sft_main(argv: Optional[List[str]] = None) -> str:
    args, remaining_argv = parse_args(SftArguments, argv)
    args.init_argument()
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    return llm_sft(args)


def infer_main(argv: Optional[List[str]] = None) -> None:
    args, remaining_argv = parse_args(InferArguments, argv)
    args.init_argument()
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    llm_infer(args)
