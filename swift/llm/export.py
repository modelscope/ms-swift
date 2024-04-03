# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Tuple, Union

import torch
from datasets import concatenate_datasets
from modelscope import GenerationConfig
from transformers import AwqConfig, PreTrainedModel

from swift.utils import (get_logger, get_main, get_model_info, push_to_ms_hub,
                         seed_everything, show_layers)
from .infer import merge_lora, prepare_model_template, save_checkpoint
from .utils import (ExportArguments, Template, get_dataset,
                    get_model_tokenizer, get_template, set_generation_config,
                    swift_to_peft_format)

logger = get_logger()


def prepare_awq_model_template(
        args: ExportArguments) -> Tuple[PreTrainedModel, Template]:
    from awq import AutoAWQForCausalLM
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


def _get_dataset(*args, **kwargs):
    global _args, template
    assert _args is not None
    assert template is not None
    data = _args.dataset
    n_samples = _args.quant_n_samples
    block_size = _args.quant_seqlen

    # only use train_dataset
    dataset = get_dataset(data)[0]
    dataset = dataset.shuffle()

    samples = []
    n_run = 0
    for data in dataset:
        input_ids = template.encode(data)[0].get('input_ids')
        if input_ids is None or len(input_ids) == 0:
            continue
        sample = torch.tensor(input_ids)
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=0)  # shape: [X]
    n_split = cat_samples.shape[0] // block_size
    logger.info(f'Split into {n_split} blocks')
    if _args.quant_method == 'awq':
        return [
            cat_samples[None, i * block_size:(i + 1) * block_size]
            for i in range(n_split)
        ]
    else:  # gptq
        res = []
        for i in range(n_split):
            input_ids = cat_samples[None, i * block_size:(i + 1) * block_size]
            attention_mask = torch.ones_like(input_ids)
            res.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
        return res


def awq_model_quantize(awq_model, tokenizer) -> None:
    from awq.quantize import quantizer
    assert _args is not None
    logger.info(f'Quantization dataset: {_args.dataset}')
    _origin_get_calib_dataset = quantizer.get_calib_dataset
    quantizer.get_calib_dataset = _get_dataset
    group_size = 128
    quant_config = {
        'zero_point': True,
        'q_group_size': group_size,
        'w_bit': _args.quant_bits,
        'version': 'GEMM'
    }
    logger.info('Start quantizing the model...')
    awq_model.quantize(tokenizer, quant_config=quant_config)
    quantizer.get_calib_dataset = _origin_get_calib_dataset  # recover
    awq_model.model.config.quantization_config = AwqConfig(
        bits=_args.quant_bits,
        group_size=group_size,
        zero_point=True,
        version='GEMM')


def gptq_model_quantize(model, tokenizer):
    from optimum.gptq import GPTQQuantizer, quantizer
    global _args
    logger.info(f'Quantization dataset: {_args.dataset}')
    gptq_quantizer = GPTQQuantizer(
        bits=_args.quant_bits, dataset=_args.dataset)
    _origin_get_dataset = quantizer.get_dataset
    quantizer.get_dataset = _get_dataset
    logger.info('Start quantizing the model...')
    logger.warning(
        'The process of packing the model takes a long time and there is no progress bar. '
        'Please be patient and wait...')
    gptq_quantizer.quantize_model(model, tokenizer)
    quantizer.get_dataset = _origin_get_dataset  # recover
    return gptq_quantizer


def llm_export(args: ExportArguments) -> None:
    global _args, template
    logger.info(f'args: {args}')
    #if args.to_peft_format:
    #    assert args.sft_type == 'lora'
    #    args.ckpt_dir = swift_to_peft_format(args.ckpt_dir)
    if args.merge_lora:
        merge_lora(args, device_map=args.merge_device_map)
    if args.quant_bits > 0:
        _args = args
        assert args.quantization_bit == 0
        assert args.sft_type == 'full', 'you need to merge lora'
        if args.dtype == 'AUTO' and args.torch_dtype is None:
            args.dtype, args.torch_dtype = 'fp16', torch.float16
            logger.info(f'Setting args.torch_dtype: {args.torch_dtype}')
        if args.ckpt_dir is None:
            quant_path = f'{args.model_type}-{args.quant_method}-int{args.quant_bits}'
        else:
            ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
            quant_path = os.path.join(
                ckpt_dir,
                f'{ckpt_name}-{args.quant_method}-int{args.quant_bits}')
        logger.info(f'Setting quant_path: {quant_path}')
        assert not os.path.exists(quant_path)
        if args.quant_method == 'awq':
            awq_model, template = prepare_awq_model_template(args)
            awq_model_quantize(awq_model, template.tokenizer)
            logger.info(get_model_info(awq_model))
            show_layers(awq_model)
            logger.info('Saving quantized weights...')
            awq_model.save_quantized(quant_path)
            model_cache_dir = awq_model.model_dir
        else:  # gptq
            model, template = prepare_model_template(
                args, device_map=args.quant_device_map)
            gptq_quantizer = gptq_model_quantize(model, template.tokenizer)
            logger.info(get_model_info(model))
            show_layers(model)
            logger.info('Saving quantized weights...')
            gptq_quantizer.save(model, quant_path)
            model_cache_dir = model.model_dir

        save_checkpoint(None, template.tokenizer, model_cache_dir,
                        args.ckpt_dir, quant_path)
        logger.info(
            f'Successfully quantized the model and saved in {quant_path}.')
        args.ckpt_dir = quant_path

    if args.push_to_hub:
        assert args.ckpt_dir is not None, 'You need to specify `ckpt_dir`.'
        push_to_ms_hub(args.ckpt_dir, args.hub_model_id, args.hub_token,
                       args.hub_private_repo, args.commit_message)


export_main = get_main(ExportArguments, llm_export)
