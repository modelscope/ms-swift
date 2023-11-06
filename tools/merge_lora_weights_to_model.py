from swift.llm import InferArguments, merge_lora
from swift.utils import parse_args

if __name__ == '__main__':
    args, remaining_argv = parse_args(InferArguments, None)
    merge_lora(args, replace_if_exists=True)
