# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn as nn
import transformers
from collections import defaultdict
from contextlib import contextmanager
from packaging import version
from tqdm import tqdm
from typing import Dict, List, Optional

from swift.arguments import ExportArguments
from swift.dataset import load_dataset
from swift.model import save_checkpoint
from swift.template import MaxLengthError
from swift.utils import HfConfigFactory, ProcessorMixin, deep_getattr, get_logger, get_model_parameter_info, to_device
from ..utils import prepare_model_template

logger = get_logger()


class QuantEngine(ProcessorMixin):

    def __init__(self, args: ExportArguments):
        self.args = args
        if args.quant_method == 'awq':
            if args.adapters:
                raise ValueError('AWQ requires merged adapters. Use --merge_lora true first.')
            if not args.safe_serialization:
                raise ValueError('GPTQModel AWQ export requires --safe_serialization true.')
            if args.quant_bits is None:
                raise ValueError('Please set quant_bits for AWQ export.')
            if args.new_special_tokens:
                raise ValueError('AWQ does not support adding special tokens during export.')
            if args.device_map is not None:
                raise ValueError('GPTQModel AWQ selects its own device; --device_map is not supported.')
            if args.task_type != 'causal_lm' or args.tuner_backend == 'unsloth':
                raise ValueError('GPTQModel AWQ supports causal language models without the unsloth backend.')
            for name in ('model_kwargs', 'rope_scaling', 'max_memory', 'attn_impl', 'experts_impl', 'max_model_len',
                         'init_strategy'):
                if getattr(args, name, None):
                    raise ValueError(f'GPTQModel AWQ does not support --{name} during export.')
            if getattr(args.model_info.config, 'quantization_config', None):
                raise ValueError('AWQ requires an unquantized model; the input is already quantized.')
            try:
                from gptqmodel import AWQConfig, GPTQModel
            except ImportError as exc:
                raise ImportError('AWQ export requires GPTQModel >= 7.3.2: pip install "gptqmodel>=7.3.2"') from exc
            _, processor = args.get_model_processor(load_model=False)
            self.template = args.get_template(processor)
            quant_config = AWQConfig(bits=args.quant_bits, group_size=args.group_size, sym=False)
            self.awq_model = GPTQModel.load(
                args.model_dir, quantize_config=quant_config, trust_remote_code=True, dtype=args.torch_dtype)
            if self.awq_model.quantized:
                raise ValueError('AWQ requires an unquantized model; the input is already quantized.')
            self.model = self.awq_model.model
            self.model.model_meta = processor.model_meta
            self.model.model_info = processor.model_info
            self.model.model_dir = args.model_dir
        else:
            self.model, self.template = prepare_model_template(args)
        self.template.set_mode('train')
        self.model.config.use_cache = False
        HfConfigFactory.set_config_attr(self.model.config, 'use_cache', False)
        self.processor = self.template.processor
        args.save_args()

    def quantize(self):
        args = self.args
        if args.quant_bits is None and args.quant_method not in {'fp8', 'entropack'}:
            raise ValueError(f'Please set the quant_bits. args.quant_bits: {args.quant_bits}')
        if args.quant_method == 'awq':
            self.template.model = self.model
            awq_model = self.awq_model_quantize()
            awq_model.save(args.output_dir, max_shard_size=args.max_shard_size)
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
            # Optimum may convert the checkpoint format while saving the weights.
            # Persist its final configuration so reloading uses the same format.
            self.model.config.quantization_config = gptq_quantizer.to_dict()
            self.model.config.quantization_config.pop('dataset', None)
            self.model.config.save_pretrained(args.output_dir)
        elif args.quant_method in {'bnb', 'fp8', 'entropack'}:
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
        logger.info(f'Successfully quantized the model and saved in `{args.output_dir}`.')

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

    def awq_model_quantize(self):
        args = self.args
        if getattr(self.model.config, 'quantization_config', None):
            raise ValueError('AWQ requires an unquantized model; the input is already quantized.')
        if any(str(param.dtype).startswith('torch.float8') for param in self.model.parameters()):
            raise ValueError('AWQ requires an unquantized model; FP8 weights are not supported.')
        if args.adapters:
            raise ValueError('AWQ requires merged adapters. Use --merge_lora true first.')
        if not args.safe_serialization:
            raise ValueError('GPTQModel AWQ export requires --safe_serialization true.')

        logger.info(f'Quantization dataset: {args.dataset}')
        if self.model.model_meta.is_multimodal:
            raise ValueError('GPTQModel AWQ export does not currently support multimodal calibration.')
        calibration = [{
            'input_ids': block.reshape(-1).tolist()
        } for block in self._get_quant_dataset() if block.numel()]
        if not calibration:
            raise ValueError('AWQ calibration dataset contains no usable tokens.')
        logger.info('Start quantizing the model...')
        batch_size = args.quant_batch_size if args.quant_batch_size is not None else len(calibration)
        self.awq_model.quantize(calibration=calibration, batch_size=batch_size)
        return self.awq_model

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
            language_model = [lm for lm in model_arch.language_model if not lm.endswith('lm_head')]
            assert len(language_model) == 1, f'model_arch.language_model: {language_model}'
            prefix = language_model[0]
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
            if not hasattr(self.model, 'hf_device_map'):
                self.model.hf_device_map = {'': torch.device('cuda:0')}
            with self._patch_gptq_block(self.model, block_name_to_quantize):
                gptq_quantizer.quantize_model(self.model, self.tokenizer)
            self.model.config.quantization_config.pop('dataset', None)
        return gptq_quantizer


def quantize_model(args: ExportArguments):
    QuantEngine(args).quantize()
