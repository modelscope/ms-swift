# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict, List, Optional

import json
import torch
import transformers
from packaging import version

from swift.llm import get_model_tokenizer, get_template
from swift.utils import (check_json_format, get_logger, get_main, get_model_info, push_to_ms_hub, seed_everything,
                         show_layers)
from .infer import merge_lora, prepare_model_template, save_checkpoint
from .utils import ExportArguments, Template, get_dataset, swift_to_peft_format

logger = get_logger()

_args: Optional[ExportArguments] = None
template: Optional[Template] = None


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
        return [cat_samples[None, i * block_size:(i + 1) * block_size] for i in range(n_split)]
    else:  # gptq
        res = []
        for i in range(n_split):
            input_ids = cat_samples[None, i * block_size:(i + 1) * block_size]
            attention_mask = torch.ones_like(input_ids)
            res.append({'input_ids': input_ids, 'attention_mask': attention_mask})
        return res


def awq_model_quantize(awq_model, tokenizer, batch_size) -> None:

    def _llama_rotary_emb_forward(self, x, position_ids):
        with torch.no_grad():
            if 'dynamic' in self.rope_type:
                self._dynamic_frequency_update(position_ids, device=x.device)

            # Core RoPE block
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
            device_type = x.device.type
            device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
            with torch.autocast(device_type=device_type, enabled=False):
                inv_freq_expanded = inv_freq_expanded.to(position_ids_expanded.device)
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()

            # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
            cos = cos * self.attention_scaling
            sin = sin * self.attention_scaling

            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @torch.no_grad()
    def _module_forward(self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict) -> torch.Tensor:
        # The original code of awq.AwqQuantizer._module_forward has a bug with n_parallel_calib_samples
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for idx, x_partial in enumerate(partitioned_inputs):
                tmp_module_kwargs = {**module_kwargs}
                if tmp_module_kwargs.get('attention_mask'):
                    tmp_module_kwargs['attention_mask'] = tmp_module_kwargs['attention_mask'][idx:idx + self.
                                                                                              n_parallel_calib_samples]
                partial_output = module(x_partial, **tmp_module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    import awq
    from awq.quantize import quantizer
    from transformers import AwqConfig
    if version.parse(awq.__version__) >= version.parse('0.2.6'):
        quantizer.AwqQuantizer._module_forward = _module_forward

    if version.parse(transformers.__version__) >= version.parse('4.43.0'):
        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward = _llama_rotary_emb_forward

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


def gptq_model_quantize(model, tokenizer, batch_size):
    from optimum.gptq import GPTQQuantizer, quantizer
    global _args
    logger.info(f'Quantization dataset: {_args.dataset}')
    gptq_quantizer = GPTQQuantizer(bits=_args.quant_bits, dataset=','.join(_args.dataset), batch_size=batch_size)
    _origin_get_dataset = quantizer.get_dataset
    quantizer.get_dataset = _get_dataset
    logger.info('Start quantizing the model...')
    logger.warning('The process of packing the model takes a long time and there is no progress bar. '
                   'Please be patient and wait...')
    gptq_quantizer.quantize_model(model, tokenizer)
    quantizer.get_dataset = _origin_get_dataset  # recover
    return gptq_quantizer


def replace_and_concat(template: 'Template', template_list: List, placeholder: str, keyword: str):
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
                args, device_map=args.quant_device_map, verbose=False, automodel_class=AutoAWQForCausalLM)
            awq_model_quantize(model, template.tokenizer, args.quant_batch_size)
            model.save_quantized(args.quant_output_dir)
        elif args.quant_method == 'gptq':
            model, template = prepare_model_template(args, device_map=args.quant_device_map, verbose=False)
            gptq_quantizer = gptq_model_quantize(model, template.tokenizer, args.quant_batch_size)
            model.config.quantization_config.pop('dataset', None)
            gptq_quantizer.save(model, args.quant_output_dir)
        elif args.quant_method == 'bnb':
            args.quant_device_map = 'auto'  # cannot use cpu on bnb
            args.quantization_bit = args.quant_bits
            args.bnb_4bit_compute_dtype, args.load_in_4bit, args.load_in_8bit = args.select_bnb()
            model, template = prepare_model_template(args, device_map=args.quant_device_map, verbose=False)
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
            model, tokenizer = get_model_tokenizer(args.model_type, torch.float32, {'device_map': 'auto'})
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
            hf_model, tokenizer = get_model_tokenizer(args.model_type, torch.float32, {'device_map': 'auto'})
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
