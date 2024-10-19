# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import contextmanager
from types import MethodType
from typing import Dict, List, Optional

import json
import torch
import torch.nn as nn

from swift.llm import get_model_tokenizer, get_template
from swift.utils import (check_json_format, get_logger, get_main, get_model_info, push_to_ms_hub, seed_everything,
                         show_layers)
from .infer import merge_lora, prepare_model_template, save_checkpoint
from .utils import ExportArguments, Template, deep_getattr, get_dataset, get_mllm_arch, swift_to_peft_format

logger = get_logger()

_args: Optional[ExportArguments] = None
template: Optional[Template] = None


def _prepare_dataset(examples: List[Dict[str, torch.LongTensor]], batch_size: int = 1, *args, **kwargs):
    global _args, template
    assert template is not None
    examples = [
        template.data_collator(examples[start:start + batch_size]) for start in range(0, len(examples), batch_size)
    ]
    return examples


def _get_dataset(*args, **kwargs):
    global _args, template
    assert _args is not None
    assert template is not None
    data = _args.dataset
    n_samples = _args.quant_n_samples
    block_size = _args.quant_seqlen

    # only use train_dataset
    dataset = get_dataset(
        data,
        0,
        _args.dataset_seed,
        check_dataset_strategy=_args.check_dataset_strategy,
        model_name=_args.model_name,
        model_author=_args.model_author)[0]
    logger.info(f'quant_dataset: {dataset}')
    dataset = dataset.shuffle()

    samples = []
    n_run = 0
    for data in dataset:
        inputs = template.encode(data)[0]
        input_ids = inputs['input_ids']
        if input_ids is None or len(input_ids) == 0:
            continue
        if _args.is_multimodal and _args.quant_method == 'gptq':
            inputs.pop('labels', None)
            samples.append(inputs)
        else:
            samples += input_ids
        n_run += 1
        if n_run == n_samples:
            break
    if _args.is_multimodal and _args.quant_method == 'gptq':
        return samples
    # now concatenate all samples and split according to block size
    n_split = len(samples) // block_size
    logger.info(f'Split into {n_split} blocks')
    res = []
    for i in range(n_split):
        input_ids = samples[i * block_size:(i + 1) * block_size]
        if _args.quant_method == 'awq':
            res.append(torch.tensor(input_ids)[None])
        else:
            res.append({'input_ids': input_ids})
    return res


def awq_model_quantize(awq_model, tokenizer, batch_size) -> None:

    from awq.quantize import quantizer
    from transformers import AwqConfig

    assert _args is not None
    logger.info(f'Quantization dataset: {_args.dataset}')
    _origin_get_calib_dataset = quantizer.get_calib_dataset
    quantizer.get_calib_dataset = _get_dataset
    group_size = 128
    quant_config = {'zero_point': True, 'q_group_size': group_size, 'w_bit': _args.quant_bits, 'version': 'GEMM'}
    logger.info('Start quantizing the model...')
    awq_model.quantize(tokenizer, quant_config=quant_config, n_parallel_calib_samples=batch_size)
    quantizer.get_calib_dataset = _origin_get_calib_dataset  # recover
    awq_model.model.config.quantization_config = AwqConfig(
        bits=_args.quant_bits, group_size=group_size, zero_point=True, version='GEMM')


@contextmanager
def _patch_gptq():
    from optimum.gptq import quantizer
    _get_dataset_origin = quantizer.get_dataset
    _prepare_dataset_origin = quantizer.prepare_dataset
    quantizer.get_dataset = _get_dataset
    quantizer.prepare_dataset = _prepare_dataset
    yield
    quantizer.get_dataset = _get_dataset_origin
    quantizer.prepare_dataset = _prepare_dataset_origin


def _patch_model_forward(module_list):

    def _new_forward(self, *args, **kwargs):
        if 'use_cache' in kwargs:
            kwargs['use_cache'] = False
        layer_ret = self.__old_forward(*args, **kwargs)
        return layer_ret + args[len(layer_ret):]

    for module in module_list:
        if hasattr(module, '_old_forward'):  # device_map
            __old_forward = module._old_forward
            module._old_forward = MethodType(_new_forward, module)
        else:
            __old_forward = module.forward
            module.forward = MethodType(_new_forward, module)
        module.__old_forward = __old_forward


def get_block_name_to_quantize(model: nn.Module, model_type: str) -> Optional[str]:
    mllm_arch = get_mllm_arch(model_type)
    prefix = ''
    if mllm_arch is not None:
        assert len(mllm_arch.language_model) == 1, f'mllm_arch.language_model: {mllm_arch.language_model}'
        prefix = mllm_arch.language_model[0]
        model = deep_getattr(model, prefix)

    module_lists = []
    for n, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 10:
            module_lists.append((n, m))
    if module_lists:
        module_list = max(module_lists, key=lambda x: len(x[1]))
        _patch_model_forward(module_list[1])
        return f'{prefix}.{module_list[0]}'.strip('.')


def gptq_model_quantize(model, tokenizer, batch_size):
    from optimum.gptq import GPTQQuantizer
    global _args
    logger.info(f'Quantization dataset: {_args.dataset}')
    with _patch_gptq():
        gptq_quantizer = GPTQQuantizer(
            bits=_args.quant_bits,
            dataset=','.join(_args.dataset),
            batch_size=batch_size,
            block_name_to_quantize=get_block_name_to_quantize(model, _args.model_type))
        logger.info('Start quantizing the model...')
        logger.warning('The process of packing the model takes a long time and there is no progress bar. '
                       'Please be patient and wait...')
        if not hasattr(model.config, 'use_cache'):
            model.config.use_cache = None
        gptq_quantizer.quantize_model(model, tokenizer)
    return gptq_quantizer


def replace_and_concat(template: Template, template_list: List, placeholder: str, keyword: str):
    final_str = ''
    for t in template_list:
        if isinstance(t, str):
            final_str += t.replace(placeholder, keyword)
        elif isinstance(t, (tuple, list)):
            if isinstance(t[0], int):
                final_str += template.tokenizer.decode(t)
            else:
                for attr in t:
                    if attr == 'bos_token_id':
                        final_str += template.tokenizer.bos_token
                    elif attr == 'eos_token_id':
                        final_str += template.tokenizer.eos_token
                    else:
                        raise ValueError(f'Unknown token: {attr}')
    return final_str


def llm_export(args: ExportArguments) -> None:
    global _args, template
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    if args.to_peft_format:
        assert args.sft_type == 'lora', f'args.sft_type: {args.sft_type}'
        args.ckpt_dir = swift_to_peft_format(args.ckpt_dir)

    if args.merge_lora:
        # fix parameter conflict
        quant_method = args.quant_method
        args.quant_method = None
        merge_lora(args, device_map=args.merge_device_map)
        args.quant_method = quant_method

    if args.to_ollama:

        logger.info('Exporting to ollama:')
        logger.info('If you have a gguf file, try to pass the file by :--gguf_file /xxx/xxx.gguf, '
                    'else SWIFT will use the original(merged) model dir')
        os.makedirs(args.ollama_output_dir, exist_ok=True)
        if args.ckpt_dir is not None:
            model_dir = args.ckpt_dir
        else:
            model_dir = args.model_id_or_path
        logger.info(f'Using model_dir: {model_dir}')
        _, tokenizer = get_model_tokenizer(
            args.model_type, model_id_or_path=model_dir, revision=args.model_revision, load_model=False)
        model_dir = tokenizer.model_dir
        template = get_template(
            args.template_type,
            tokenizer,
            args.system,
            args.max_length,
            args.truncation_strategy,
            tools_prompt=args.tools_prompt)
        with open(os.path.join(args.ollama_output_dir, 'Modelfile'), 'w') as f:
            f.write(f'FROM {model_dir}\n')
            f.write(f'TEMPLATE """{{{{ if .System }}}}'
                    f'{replace_and_concat(template, template.system_prefix, "{{SYSTEM}}", "{{ .System }}")}'
                    f'{{{{ else }}}}{replace_and_concat(template, template.prefix, "", "")}'
                    f'{{{{ end }}}}')
            f.write(f'{{{{ if .Prompt }}}}'
                    f'{replace_and_concat(template, template.prompt, "{{QUERY}}", "{{ .Prompt }}")}'
                    f'{{{{ end }}}}')
            f.write('{{ .Response }}')
            f.write(replace_and_concat(template, template.suffix, '', '') + '"""\n')
            f.write(f'PARAMETER stop "{replace_and_concat(template, template.suffix, "", "")}"\n')
            if args.stop_words:
                for stop_word in args.stop_words:
                    f.write(f'PARAMETER stop "{stop_word}"\n')
            if args.temperature:
                f.write(f'PARAMETER temperature {args.temperature}\n')
            if args.top_k:
                f.write(f'PARAMETER top_k {args.top_k}\n')
            if args.top_p:
                f.write(f'PARAMETER top_p {args.top_p}\n')
            if args.repetition_penalty:
                f.write(f'PARAMETER repeat_penalty {args.repetition_penalty}\n')

        logger.info('Save Modelfile done, you can start ollama by:')
        logger.info('> ollama serve')
        logger.info('In another terminal:')
        logger.info('> ollama create my-custom-model ' f'-f {os.path.join(args.ollama_output_dir, "Modelfile")}')
        logger.info('> ollama run my-custom-model')
    elif args.quant_bits > 0:
        assert args.quant_output_dir is not None
        _args = args
        assert args.quantization_bit == 0, f'args.quantization_bit: {args.quantization_bit}'
        assert args.sft_type == 'full', 'you need to merge lora'
        if args.quant_method == 'awq':
            from awq import AutoAWQForCausalLM
            model, template = prepare_model_template(
                args, device_map=args.quant_device_map, task='export', automodel_class=AutoAWQForCausalLM)
            awq_model_quantize(model, template.tokenizer, args.quant_batch_size)
            model.save_quantized(args.quant_output_dir)
        elif args.quant_method == 'gptq':
            model, template = prepare_model_template(args, device_map=args.quant_device_map, task='export')
            gptq_quantizer = gptq_model_quantize(model, template.tokenizer, args.quant_batch_size)
            model.config.quantization_config.pop('dataset', None)
            gptq_quantizer.save(model, args.quant_output_dir)
        elif args.quant_method == 'bnb':
            args.quantization_bit = args.quant_bits
            args.bnb_4bit_compute_dtype, args.load_in_4bit, args.load_in_8bit = args.select_bnb()
            model, template = prepare_model_template(args, device_map=args.quant_device_map, task='export')
            model.save_pretrained(args.quant_output_dir)
        else:
            raise ValueError(f'args.quant_method: {args.quant_method}')

        logger.info(get_model_info(model))
        show_layers(model)
        logger.info('Saving quantized weights...')
        model_cache_dir = model.model_dir
        save_checkpoint(
            None,
            template.tokenizer,
            model_cache_dir,
            args.ckpt_dir,
            args.quant_output_dir,
            sft_args_kwargs={
                'dtype': args.dtype,
                'quant_method': args.quant_method
            })
        logger.info(f'Successfully quantized the model and saved in {args.quant_output_dir}.')
        args.ckpt_dir = args.quant_output_dir
    elif args.to_megatron:
        if os.path.exists(args.megatron_output_dir):
            logger.info(f'The file in Megatron format already exists in the directory: {args.megatron_output_dir}. '
                        'Skipping the conversion process.')
        else:
            from swift.llm.megatron import MegatronArguments, convert_hf_to_megatron, patch_megatron
            model, tokenizer = get_model_tokenizer(
                args.model_type,
                torch.float32, {'device_map': 'auto'},
                model_id_or_path=args.model_id_or_path,
                revision=args.model_revision)
            res = MegatronArguments.load_megatron_config(tokenizer.model_dir)
            res['model_type'] = args.model_type
            res['target_tensor_model_parallel_size'] = args.tp
            res['target_pipeline_model_parallel_size'] = args.pp
            res['load'] = model.model_dir
            res['save'] = args.megatron_output_dir
            res['seed'] = args.seed
            res['use_cpu_initialization'] = True
            megatron_args = MegatronArguments(**res)
            extra_args = megatron_args.parse_to_megatron()
            patch_megatron(tokenizer)
            convert_hf_to_megatron(model, extra_args, args.torch_dtype)
            fpath = os.path.join(args.megatron_output_dir, 'export_args.json')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(args.__dict__), f, ensure_ascii=False, indent=2)
            logger.info('Successfully converted HF format to Megatron format and '
                        f'saved it in the {args.megatron_output_dir} directory.')
    elif args.to_hf:
        if os.path.exists(args.hf_output_dir):
            logger.info(f'The file in HF format already exists in the directory: {args.hf_output_dir}. '
                        'Skipping the conversion process.')
        else:
            from swift.llm.megatron import MegatronArguments, convert_megatron_to_hf, patch_megatron
            hf_model, tokenizer = get_model_tokenizer(
                args.model_type,
                torch.float32, {'device_map': 'auto'},
                model_id_or_path=args.model_id_or_path,
                revision=args.model_revision)
            res = MegatronArguments.load_megatron_config(tokenizer.model_dir)
            res['model_type'] = args.model_type
            res['target_tensor_model_parallel_size'] = args.tp
            res['target_pipeline_model_parallel_size'] = args.pp
            res['load'] = args.ckpt_dir
            res['save'] = args.hf_output_dir
            res['use_cpu_initialization'] = True
            megatron_args = MegatronArguments(**res)
            extra_args = megatron_args.parse_to_megatron()
            extra_args['hf_ckpt_path'] = hf_model.model_dir
            patch_megatron(tokenizer)
            convert_megatron_to_hf(hf_model, extra_args)
            if args.torch_dtype is not None:
                hf_model.to(args.torch_dtype)
            save_checkpoint(hf_model, tokenizer, hf_model.model_dir, args.ckpt_dir, args.hf_output_dir)
            logger.info('Successfully converted Megatron format to HF format and '
                        f'saved it in the {args.hf_output_dir} directory.')
    if args.push_to_hub:
        ckpt_dir = args.ckpt_dir
        if ckpt_dir is None:
            ckpt_dir = args.model_id_or_path
        assert ckpt_dir is not None, 'You need to specify `ckpt_dir`.'
        push_to_ms_hub(ckpt_dir, args.hub_model_id, args.hub_token, args.hub_private_repo, args.commit_message)


export_main = get_main(ExportArguments, llm_export)
