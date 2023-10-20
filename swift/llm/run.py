from typing import List, Optional

from swift.llm import InferArguments, SftArguments, llm_infer, llm_sft
from swift.utils import parse_args


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


if __name__ == '__main__':
    ckpt_dir = sft_main()
    print(ckpt_dir)
    infer_main(['--ckpt_dir', ckpt_dir])
