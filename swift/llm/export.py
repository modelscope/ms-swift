# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Tuple, Union

import torch
from modelscope import GenerationConfig
from transformers import AwqConfig, PreTrainedModel
from datasets import concatenate_datasets
from swift.utils import (get_logger, get_main, get_model_info, push_to_ms_hub,
                         seed_everything, show_layers)
from .infer import merge_lora, save_checkpoint
from .utils import (ExportArguments, Template, get_dataset,
                    get_model_tokenizer, get_template, set_generation_config)

logger = get_logger()


def prepare_awq_model_template(
        args: ExportArguments) -> Tuple[PreTrainedModel, Template]:
    from awq import AutoAWQForCausalLM
    logger.info(f'args: {args}')
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)

    # Loading Model and Tokenizer
    model_kwargs = {
        'low_cpu_mem_usage': True,
        'device_map': args.quant_device_map
    }
    model_id_or_path = None
    if args.sft_type == 'full' and args.ckpt_dir is not None:
        model_id_or_path = args.ckpt_dir
    elif args.model_id_or_path is not None:
        model_id_or_path = args.model_id_or_path
    model, tokenizer = get_model_tokenizer(
        args.model_type,
        args.torch_dtype,
        model_kwargs,
        model_id_or_path=model_id_or_path,
        automodel_class=AutoAWQForCausalLM)
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
template = None

def _get_calib_dataset(
        data: Union[str, List[str], List[List[int]]] = 'pileval',  # not use
        tokenizer=None,
        n_samples=512,  # not use
        block_size=512,  # not use
        split='train',  # not use
        text_column='text',  # not use
) -> List[torch.Tensor]:
    global _args, template
    assert _args is not None
    assert template is not None
    data = _args.quant_dataset
    n_samples = _args.quant_n_samples
    block_size = _args.quant_seqlen

    if isinstance(data, str):
        data = [data]
    dataset, val_dataset = get_dataset(data)
    if val_dataset is not None:
        dataset = concatenate_datasets([dataset, val_dataset])
    dataset = dataset.shuffle(seed=42)

    samples = []
    n_run = 0
    for data in dataset:
        input_ids = template.encode(data)[0].get('input_ids')
        if input_ids is None or  len(input_ids) == 0:
            continue
        sample = torch.tensor(input_ids)
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=0)
    n_split = cat_samples.shape[0] // block_size
    logger.info(f'AWQ: Split into {n_split} blocks')
    return [
        cat_samples[None, i * block_size:(i + 1) * block_size]
        for i in range(n_split)
    ]


def awq_model_quantize(awq_model, tokenizer) -> None:
    from awq.quantize import quantizer
    assert _args is not None
    _raw_get_calib_dataset = quantizer.get_calib_dataset
    quantizer.get_calib_dataset = _get_calib_dataset
    group_size = 128
    quant_config = {
        'zero_point': True,
        'q_group_size': group_size,
        'w_bit': _args.quant_bits,
        'version': 'GEMM'
    }
    logger.info('Start quantizing the model...')
    awq_model.quantize(tokenizer, quant_config=quant_config)
    quantizer.get_calib_dataset = _raw_get_calib_dataset  # recover
    awq_model.model.config.quantization_config = AwqConfig(
        bits=_args.quant_bits,
        group_size=group_size,
        zero_point=True,
        version='GEMM')


def llm_export(args: ExportArguments) -> None:
    global _args, template
    if args.merge_lora:
        merge_lora(args, device_map='cpu')
    if args.quant_bits > 0:
        _args = args
        assert args.quantization_bit == 0
        assert args.sft_type == 'full', 'you need to merge lora'
        if args.dtype == 'AUTO' and args.torch_dtype is None:
            args.dtype, args.torch_dtype = 'fp16', torch.float16
            logger.info(f'Setting args.torch_dtype: {args.torch_dtype}')
        if args.ckpt_dir is None:
            quant_path = f'{args.model_type}-int{args.quant_bits}'
        else:
            ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
            quant_path = os.path.join(ckpt_dir,
                                      f'{ckpt_name}-int{args.quant_bits}')
        logger.info(f'Setting quant_path: {quant_path}')
        assert not os.path.exists(quant_path)
        awq_model, template = prepare_awq_model_template(args)
        awq_model_quantize(awq_model, template.tokenizer)
        logger.info(get_model_info(awq_model))
        show_layers(awq_model)

        awq_model.save_quantized(quant_path)
        save_checkpoint(None, template.tokenizer, awq_model.model_dir,
                        args.ckpt_dir, quant_path)
        args.ckpt_dir = quant_path

    if args.push_to_hub:
        assert args.ckpt_dir is not None, 'You need to specify `ckpt_dir`.'
        push_to_ms_hub(args.ckpt_dir, args.hub_model_id, args.hub_token,
                       args.hub_private_repo, args.commit_message)


export_main = get_main(ExportArguments, llm_export)
