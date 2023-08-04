# ### Setting up experimental environment.
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
from transformers import GenerationConfig, TextStreamer
from utils import (DATASET_MAPPING, DEFAULT_PROMPT, MODEL_MAPPING, get_dataset,
                   get_model_tokenizer, inference, process_dataset,
                   select_dtype)

from swift import Swift, get_logger
from swift.utils import parse_args, seed_everything
from swift.utils.llm_utils import tokenize_function

logger = get_logger()


@dataclass
class InferArguments:
    model_type: str = field(
        default='qwen-7b', metadata={'choices': list(MODEL_MAPPING.keys())})
    sft_type: str = field(
        default='lora', metadata={'choices': ['lora', 'full']})
    ckpt_dir: str = '/path/to/your/vx_xxx/checkpoint-xxx'
    eval_human: bool = False  # False: eval test_dataset

    seed: int = 42
    dtype: Optional[str] = field(
        default=None, metadata={'choices': {'bf16', 'fp16', 'fp32'}})
    ignore_args_error: bool = False  # True: notebook compatibility

    dataset: str = field(
        default='alpaca-en,alpaca-zh',
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: int = 42
    dataset_sample: int = 20000  # -1: all dataset
    dataset_test_size: float = 0.01
    prompt: str = DEFAULT_PROMPT
    max_length: Optional[int] = 2048

    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.9

    def __post_init__(self):
        if not os.path.isdir(self.ckpt_dir):
            raise ValueError(f'Please enter a valid ckpt_dir: {self.ckpt_dir}')
        self.torch_dtype, _, _ = select_dtype(self.dtype, self.model_type)


def llm_infer(args: InferArguments) -> None:
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)
    # ### Loading Model and Tokenizer
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype)

    # ### Preparing lora
    if args.sft_type == 'lora':
        model = Swift.from_pretrained(model, args.ckpt_dir)
    elif args.sft_type == 'full':
        state_dict = torch.load(args.ckpt_dir, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f'args.sft_type: {args.sft_type}')

    # ### Inference
    tokenize_func = partial(
        tokenize_function,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length)
    streamer = TextStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')

    if args.eval_human:
        while True:
            instruction = input('<<< ')
            data = {'instruction': instruction}
            input_ids = tokenize_func(data)['input_ids']
            inference(input_ids, model, tokenizer, streamer, generation_config)
            print('-' * 80)
    else:
        dataset = get_dataset(args.dataset.split(','))
        _, test_dataset = process_dataset(dataset, args.dataset_test_size,
                                          args.dataset_sample,
                                          args.dataset_seed)
        mini_test_dataset = test_dataset.select(range(10))
        del dataset
        for data in mini_test_dataset:
            output = data['output']
            data['output'] = None
            input_ids = tokenize_func(data)['input_ids']
            inference(input_ids, model, tokenizer, streamer, generation_config)
            print()
            print(f'[LABELS]{output}')
            print('-' * 80)
            # input('next[ENTER]')


if __name__ == '__main__':
    args, remaining_argv = parse_args(InferArguments)
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    llm_infer(args)
