# Copyright (c) Alibaba, Inc. and its affiliates.
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import BitsAndBytesConfig, GenerationConfig, TextStreamer
from utils import (DATASET_MAPPING, MODEL_MAPPING, TEMPLATE_MAPPING,
                   get_dataset, get_model_tokenizer, get_preprocess, inference,
                   process_dataset, select_bnb, select_dtype, show_layers)

from swift import Swift, get_logger
from swift.utils import parse_args, print_model_info, seed_everything

logger = get_logger()


@dataclass
class InferArguments:
    model_type: str = field(
        default='qwen-7b-chat',
        metadata={'choices': list(MODEL_MAPPING.keys())})
    sft_type: str = field(
        default='lora', metadata={'choices': ['lora', 'full']})
    template_type: str = field(
        default=None, metadata={'choices': list(TEMPLATE_MAPPING.keys())})
    ckpt_dir: str = '/path/to/your/vx_xxx/checkpoint-xxx'
    eval_human: bool = False  # False: eval test_dataset

    seed: int = 42
    dtype: str = field(
        default='bf16', metadata={'choices': {'bf16', 'fp16', 'fp32'}})
    ignore_args_error: bool = False  # True: notebook compatibility

    dataset: str = field(
        default='alpaca-en,alpaca-zh',
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: int = 42
    dataset_sample: int = -1  # -1: all dataset
    dataset_test_size: float = 0.01
    system: str = 'you are a helpful assistant!'
    max_length: Optional[int] = 2048

    quantization_bit: Optional[int] = field(
        default=None, metadata={'choices': {4, 8}})
    bnb_4bit_comp_dtype: str = field(
        default=None, metadata={'choices': {'fp16', 'bf16', 'fp32'}})
    bnb_4bit_quant_type: str = field(
        default='nf4', metadata={'choices': {'fp4', 'nf4'}})
    bnb_4bit_use_double_quant: bool = True

    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.9
    skip_prompt: Optional[bool] = None

    # other
    use_flash_attn: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            "This parameter is used only when model_type.startswith('qwen-7b')"
        })

    def __post_init__(self):
        if not os.path.isdir(self.ckpt_dir):
            raise ValueError(f'Please enter a valid ckpt_dir: {self.ckpt_dir}')
        if self.template_type is None:
            self.template_type = MODEL_MAPPING[self.model_type].get(
                'template', 'default')
            logger.info(f'Setting template_type: {self.template_type}')

        self.torch_dtype, _, _ = select_dtype(self.dtype)
        if self.bnb_4bit_comp_dtype is None:
            self.bnb_4bit_comp_dtype = self.dtype
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self.quantization_bit, self.bnb_4bit_comp_dtype)
        if self.skip_prompt is None:
            self.skip_prompt = self.eval_human

        if self.use_flash_attn is None:
            self.use_flash_attn = 'auto'


def llm_infer(args: InferArguments) -> None:
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)

    # ### Loading Model and Tokenizer
    kwargs = {'low_cpu_mem_usage': True, 'device_map': 'auto'}
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        kwargs['quantization_config'] = quantization_config
    if args.model_type.startswith('qwen'):
        kwargs['use_flash_attn'] = args.use_flash_attn

    if args.sft_type == 'full':
        kwargs['model_dir'] = args.ckpt_dir
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, **kwargs)

    # ### Preparing lora
    if args.sft_type == 'lora':
        model = Swift.from_pretrained(
            model, args.ckpt_dir, inference_mode=True)

    show_layers(model)
    print_model_info(model)

    # ### Inference
    preprocess_func = get_preprocess(args.template_type, tokenizer,
                                     args.system, args.max_length)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')

    if args.eval_human:
        while True:
            query = input('<<< ')
            data = {'query': query}
            input_ids = preprocess_func(data)['input_ids']
            streamer.decode_kwargs['skip_special_tokens'] = True
            inference(input_ids, model, tokenizer, streamer, generation_config,
                      args.skip_prompt)
    else:
        dataset = get_dataset(args.dataset.split(','))
        _, test_dataset = process_dataset(dataset, args.dataset_test_size,
                                          args.dataset_sample,
                                          args.dataset_seed)
        mini_test_dataset = test_dataset.select(
            range(min(10, test_dataset.shape[0])))
        del dataset
        for data in mini_test_dataset:
            response = data['response']
            data['response'] = None
            input_ids = preprocess_func(data)['input_ids']
            inference(input_ids, model, tokenizer, streamer, generation_config,
                      args.skip_prompt)
            print()
            print(f'[LABELS]{response}')
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
