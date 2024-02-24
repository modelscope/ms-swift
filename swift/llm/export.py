# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Tuple, Union

import torch
from modelscope import GenerationConfig
from transformers import AwqConfig, PreTrainedModel

from swift.utils import (get_logger, get_main, get_model_info, seed_everything,
                         show_layers)
from .infer import merge_lora, save_checkpoint
from .utils import (InferArguments, Template, get_dataset, get_model_tokenizer,
                    get_template, set_generation_config)

logger = get_logger()


def prepare_awq_model_template(
        args: InferArguments) -> Tuple[PreTrainedModel, Template]:
    from awq import AutoAWQForCausalLM
    logger.info(f'args: {args}')
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)

    # Loading Model and Tokenizer
    model_kwargs = {'low_cpu_mem_usage': True, 'device_map': 'auto'}
    kwargs = {}
    if args.sft_type == 'full' and args.ckpt_dir is not None:
        kwargs['model_dir'] = args.ckpt_dir
    elif args.model_cache_dir is not None:
        kwargs['model_dir'] = args.model_cache_dir
    kwargs['automodel_class'] = AutoAWQForCausalLM
    model, tokenizer = get_model_tokenizer(args.model_type, args.torch_dtype,
                                           model_kwargs, **kwargs)
    logger.info(f'model_config: {model.config}')
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')
    set_generation_config(model, generation_config)

    logger.info(get_model_info(model))
    show_layers(model)

    template: Template = get_template(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        args.truncation_strategy,
        model=model)
    args.system = template.default_system
    logger.info(f'system: {args.system}')
    return model, template


_args = None


def _get_calib_dataset(
        data: Union[str, List[str], List[List[int]]] = 'pileval',  # not use
        tokenizer=None,
        n_samples=512,  # not use
        block_size=512,  # not use
        split='train',  # not use
        text_column='text',  # not use
) -> List[torch.Tensor]:
    global _args
    assert _args is not None
    data = _args.quant_dataset
    n_samples = _args.quant_n_samples
    block_size = _args.quant_seqlen

    if isinstance(data, str):
        data = [data]
    dataset = get_dataset(data)[0]
    dataset = dataset.shuffle(seed=42)

    samples = []
    n_run = 0
    for data in dataset:
        line = data['response']
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    logger.debug(f' * Split into {n_split} blocks')
    return [
        cat_samples[:, i * block_size:(i + 1) * block_size]
        for i in range(n_split)
    ]


def awq_model_quantize(awq_model, template: Template) -> None:
    from awq.quantize import quantizer
    assert _args is not None
    _raw_get_calib_dataset = quantizer.get_calib_dataset
    quantizer.get_calib_dataset = _get_calib_dataset
    group_size = 128
    quant_config = {
        'zero_point': True,
        'q_group_size': group_size,
        'w_bit': _args.export_quant_bits,
        'version': 'GEMM'
    }
    awq_model.quantize(template.tokenizer, quant_config=quant_config)
    quantizer.get_calib_dataset = _raw_get_calib_dataset  # recover
    awq_model.model.config.quantization_config = AwqConfig(
        bits=_args.export_quant_bits,
        group_size=group_size,
        zero_point=True,
        version='GEMM')


def llm_export(args: InferArguments) -> None:
    global _args
    _args = args
    if args.merge_lora_and_save is False and args.export_quant_bits <= 0:
        info = 'Nothing is being done.'
        if args.sft_type == 'lora':
            info += ' You can set `--merge_lora_and_save true` to merge LoRA.'
        info += ' You can set `--export_quant_bits 4` to perform AWQ-4bits quantization on the model.'
        logger.info(info)

    if args.merge_lora_and_save:
        merge_lora(args, device_map='cpu')
    if args.export_quant_bits > 0:
        assert args.quantization_bit == 0
        assert args.sft_type == 'full', 'you need to merge lora'
        awq_model, template = prepare_awq_model_template(args)
        awq_model_quantize(awq_model, template)

        if args.ckpt_dir is None:
            quant_path = f'{args.model_type}-quant'
        else:
            ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
            quant_path = os.path.join(ckpt_dir, f'{ckpt_name}-quant')
        logger.info(f'Setting quant_path: {quant_path}')
        awq_model.save_quantized(quant_path)
        save_checkpoint(None, template.tokenizer, awq_model.model_dir, None,
                        quant_path)


export_main = get_main(InferArguments, llm_export)
