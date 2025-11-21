# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import transformers
from packaging import version
from tqdm import tqdm

from swift.llm import (ExportArguments, HfConfigFactory, MaxLengthError, ProcessorMixin, deep_getattr, load_dataset,
                       prepare_model_template, save_checkpoint, to_device)
from swift.utils import get_logger, get_model_parameter_info

logger = get_logger()


class QuantEngine(ProcessorMixin):

    def __init__(self, args: ExportArguments):
        self.args = args
        kwargs = {}
        if args.quant_method == 'awq':
            from awq import AutoAWQForCausalLM
            kwargs['automodel_class'] = AutoAWQForCausalLM
        self.model, self.template = prepare_model_template(args, **kwargs)
        self.template.set_mode('train')
        self.model.config.use_cache = False
        HfConfigFactory.set_model_config_attr(self.model, 'use_cache', False)
        self.processor = self.template.processor
        args.save_args()

    def quantize(self):
        args = self.args
        if args.quant_bits is None and args.quant_method != 'fp8':
            raise ValueError(f'Please set the quant_bits. args.quant_bits: {args.quant_bits}')
        if args.quant_method == 'awq':
            self.template.model = self.model.model
            self.awq_model_quantize()
            self.model.save_quantized(
                args.output_dir, safetensors=args.safe_serialization, shard_size=args.max_shard_size)
        elif args.quant_method in {'gptq', 'gptq_v2'}:
            self.template.model = self.model
            gptq_quantizer = self.gptq_model_quantize(v2=(args.quant_method == 'gptq_v2'))
            if args.quant_method == 'gptq_v2':
                if not getattr(self.model, '_dynamic_tied_weights_keys', None):
                    self.model._dynamic_tied_weights_keys = []
                self.model._dynamic_tied_weights_keys += ['wf_unsqueeze_zero', 'wf_unsqueeze_neg_one']
            gptq_quantizer.save(
                self.model,
                args.output_dir,
                safe_serialization=args.safe_serialization,
                max_shard_size=args.max_shard_size)
        elif args.quant_method in {'bnb', 'fp8'}:
            self.model.save_pretrained(
                args.output_dir, safe_serialization=args.safe_serialization, max_shard_size=args.max_shard_size)
        else:
            raise ValueError(f'args.quant_method: {args.quant_method}')

        logger.info(f'model: {self.model}')
        logger.info(f'model_parameter_info: {get_model_parameter_info(self.model)}')
        save_checkpoint(
            None,
            self.processor,
            args.output_dir,
            model_dirs=[args.model_dir],
            additional_saved_files=self.model.model_meta.additional_saved_files)
        logger.info(f'Successfully quantized the model and saved in {args.output_dir}.')

    @torch.inference_mode()
    def _prepare_gptq_dataset(self, examples: List[Dict[str, torch.LongTensor]], batch_size: int = 1, *args, **kwargs):
        res = []
        for start in tqdm(range(0, len(examples), batch_size)):
            batched_inputs = examples[start:start + batch_size]
            inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
            if self.model.model_meta.is_multimodal:
                _, inputs = self.template.pre_forward_hook(self.model, None, inputs)
            res.append(to_device(inputs, 'cpu'))
        return res

    @torch.inference_mode()
    def _get_quant_dataset(self, *args, **kwargs):
        args = self.args
        assert args.quant_method in {'awq', 'gptq', 'gptq_v2'}
        template = self.template
        n_samples = args.quant_n_samples
        block_size = args.max_length

        # only use train_dataset
        dataset = load_dataset(
            args.dataset, split_dataset_ratio=0, shuffle=args.dataset_shuffle, **args.get_dataset_kwargs())[0]
        logger.info(f'quant_dataset: {dataset}')
        dataset = dataset.shuffle()

        samples = []
        i = 0
        prog_bar = tqdm(total=n_samples, dynamic_ncols=True)
        is_multimodal = self.model.model_meta.is_multimodal
        for data in dataset:
            try:
                inputs = template.encode(data)
            except MaxLengthError:
                continue
            if is_multimodal and args.quant_method in {'gptq', 'gptq_v2'}:
                inputs.pop('labels', None)
                samples.append(inputs)
            else:
                input_ids = inputs['input_ids']
                samples += input_ids
            i += 1
            prog_bar.update()
            if i == n_samples:
                break
        prog_bar.close()
        if is_multimodal and args.quant_method in {'gptq', 'gptq_v2'}:
            return samples
        # now concatenate all samples and split according to block size
        n_split = max(len(samples) // block_size, 1)
        logger.info(f'Split into {n_split} blocks')
        res = []
        for i in range(n_split):
            input_ids = samples[i * block_size:(i + 1) * block_size]
            if args.quant_method in {'gptq', 'gptq_v2'}:
                res.append({'input_ids': input_ids})
            else:
                res.append(torch.tensor(input_ids)[None])
        return res

    @staticmethod
    @contextmanager
    def _patch_awq_move_embed(awq_model):
        _origin_move_embed = awq_model.move_embed

        def _move_embed(model, device: str):
            if hasattr(model, '_hf_hook') and device != 'cpu':
                return
            _origin_move_embed(model, device)

        awq_model.move_embed = _move_embed
        try:
            yield
        finally:
            awq_model.move_embed = _origin_move_embed

    def awq_model_quantize(self) -> None:
        from awq.quantize import quantizer

        args = self.args
        logger.info(f'Quantization dataset: {args.dataset}')
        _origin_get_calib_dataset = quantizer.get_calib_dataset
        quantizer.get_calib_dataset = self._get_quant_dataset
        quant_config = {
            'zero_point': True,
            'q_group_size': args.group_size,
            'w_bit': args.quant_bits,
            'version': 'GEMM'
        }
        if self.model.model_info.is_moe_model:
            quant_config['modules_to_not_convert'] = self.args.get_modules_to_not_convert()
        logger.info(f'quant_config: {quant_config}')
        logger.info('Start quantizing the model...')
        with self._patch_awq_move_embed(self.model):
            self.model.quantize(
                self.tokenizer, quant_config=quant_config, n_parallel_calib_samples=args.quant_batch_size)
        quantizer.get_calib_dataset = _origin_get_calib_dataset  # recover
        if self.model.quant_config.modules_to_not_convert:
            model_arch = args.model_meta.model_arch
            lm_head_key = getattr(model_arch, 'lm_head', None) or 'lm_head'
            if lm_head_key not in self.model.quant_config.modules_to_not_convert:
                self.model.quant_config.modules_to_not_convert.append(lm_head_key)

    @contextmanager
    def _patch_gptq(self):
        from optimum.gptq import quantizer
        _get_dataset_origin = quantizer.get_dataset
        _prepare_dataset_origin = quantizer.prepare_dataset
        quantizer.get_dataset = self._get_quant_dataset
        quantizer.prepare_dataset = self._prepare_gptq_dataset
        try:
            yield
        finally:
            quantizer.get_dataset = _get_dataset_origin
            quantizer.prepare_dataset = _prepare_dataset_origin

    @staticmethod
    def get_block_name_to_quantize(model: nn.Module) -> Optional[str]:
        model_arch = model.model_meta.model_arch
        prefix = ''
        if hasattr(model_arch, 'language_model'):
            assert len(model_arch.language_model) == 1, f'mllm_arch.language_model: {model_arch.language_model}'
            prefix = model_arch.language_model[0]
            model = deep_getattr(model, prefix)

        module_lists = []
        for n, m in model.named_modules():
            if (isinstance(m, (nn.ModuleList, nn.Sequential)) and len(m) >= 10
                    and 'mlp' not in m[0].__class__.__name__.lower()):  # fix moe
                module_lists.append((n, m))
        if module_lists:
            module_list = max(module_lists, key=lambda x: len(x[1]))
            return f'{prefix}.{module_list[0]}'.strip('.')

    @staticmethod
    def _get_experts(block):
        for n, m in block.named_modules():
            if isinstance(m, (nn.ModuleList, nn.Sequential)):
                return n, m

    @staticmethod
    def get_modules_in_block_to_quantize(model, block_name: str):
        if not model.model_info.is_moe_model:
            return
        from optimum.gptq.utils import get_layers
        # Do not quantize the gate part.
        block = deep_getattr(model, block_name)[-1]
        prefix, experts = QuantEngine._get_experts(block)
        layers = get_layers(block)
        res = []
        experts = defaultdict(list)
        experts_idx = None
        for name, layer in layers.items():
            if model.model_info.model_type == 'qwen3_next' and name.startswith('self_attn.'):
                # ignore attn
                continue
            if name.startswith(prefix):
                suffix = name.rsplit('.', 1)[-1]
                experts[suffix].append(name)
                experts_idx = len(res)
            elif 'mlp.gate' not in name:
                res.append([name])
        res[experts_idx:experts_idx] = experts.values()
        return res

    @contextmanager
    def _patch_gptq_block(self, model, block_name_to_quantize):
        if version.parse(transformers.__version__) < version.parse('4.54'):
            yield
            return
        # compat transformers>=4.54
        blocks = deep_getattr(model, block_name_to_quantize)
        hooks = []

        def _to_tuple(module, input, output):
            if not isinstance(output, (list, tuple)):
                output = (output, )
            return output

        for block in blocks:
            hooks.append(block.register_forward_hook(_to_tuple))

        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def gptq_model_quantize(self, v2: bool = False):
        from optimum.gptq import GPTQQuantizer
        args = self.args
        logger.info(f'Quantization dataset: {args.dataset}')
        block_name_to_quantize = self.get_block_name_to_quantize(self.model)
        modules_in_block_to_quantize = self.get_modules_in_block_to_quantize(self.model, block_name_to_quantize)
        logger.info(f'block_name_to_quantize: {block_name_to_quantize}')
        logger.info(f'modules_in_block_to_quantize: {modules_in_block_to_quantize}')
        with self._patch_gptq():
            gptq_quantizer = GPTQQuantizer(
                bits=args.quant_bits,
                group_size=args.group_size,
                dataset=','.join(args.dataset),
                batch_size=args.quant_batch_size,
                block_name_to_quantize=block_name_to_quantize,
                modules_in_block_to_quantize=modules_in_block_to_quantize,
                checkpoint_format='gptq_v2' if v2 else 'gptq')
            gptq_quantizer.serialization_keys.append('block_name_to_quantize')
            logger.info('Start quantizing the model...')
            logger.warning('The process of packing the model takes a long time and there is no progress bar. '
                           'Please be patient and wait...')
            with self._patch_gptq_block(self.model, block_name_to_quantize):
                gptq_quantizer.quantize_model(self.model, self.tokenizer)
            self.model.config.quantization_config.pop('dataset', None)
        return gptq_quantizer


def quantize_model(args: ExportArguments):
    QuantEngine(args).quantize()
