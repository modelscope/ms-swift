from typing import Any, Dict, List, Union

from transformers import IntervalStrategy

from swift.utils import get_logger, get_model_parameter_info
from ..argument import SftArguments
from ..base import SwiftPipeline
from ..dataset import load_dataset
from ..infer import RequestConfig, prepare_generation_config
from ..model import ModelInfo, ModelMeta, get_model_arch, get_model_tokenizer
from ..template import Template, get_template
from ..tuner import prepare_tuner
from ..utils import deep_getattr, dynamic_gradient_checkpointing

logger = get_logger()


class SwiftSft(SwiftPipeline[SftArguments]):
    args_class = SftArguments

    def __init__(self, args: Union[List[str], SftArguments, None] = None) -> None:
        super().__init__(args)
        self.train_msg = {}
        self._prepare_model_tokenizer()

        self._prepare_generation_config()
        self._prepare_template()
        self._prepare_gradient_checkpointing()

        prepare_tuner(self.model, args)
        logger.info(self.model)
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

    def _prepare_gradient_checkpointing(self):
        args = self.args
        dynamic_gradient_checkpointing(self.model)

        if args.gradient_checkpointing:
            self.model.config.use_cache = False  # fix transformers==4.36
            logger.info('Setting model.config.use_cache: False')
            self.model.enable_input_require_grads()
        model_meta = self.model.model_meta
        model_arch = get_model_arch(model_meta.model_arch)
        if model_meta.is_multimodal and model_arch:
            for vision_tower_name in model_arch.vision_tower:
                vision_tower = deep_getattr(self.model, vision_tower_name)
                if args.vit_gradient_checkpointing:
                    if hasattr(vision_tower, 'enable_input_require_grads'):
                        try:
                            vision_tower.enable_input_require_grads()
                        except NotImplementedError:
                            pass
                else:
                    self.model.gradient_checkpointing_disable()

    def _prepare_generation_config(self):
        args = self.args
        self.model.generation_config = prepare_generation_config(self.model.generation_config,
                                                                 args.get_request_config(False))
        logger.info(f'model.generation_config: {self.model.generation_config}')

    def _prepare_model_tokenizer(self):
        args = self.args
        model_kwargs = {}
        if args.quantization_config:
            model_kwargs['quantization_config'] = quantization_config

        model, tokenizer = get_model_tokenizer(
            args.model,
            args.torch_dtype,
            args.device_map,
            model_type=args.model_type,
            revision=args.model_revision,
            model_kwargs=model_kwargs,
            attn_impl=args.attn_impl,
            rope_scaling=args.rope_scaling,
            use_unsloth=args.tuner_backend == 'unsloth',
            is_training=True)

        self.model = model
        self.tokenizer = tokenizer
        if hasattr(self.model, 'hf_device_map'):
            logger.info(f'model.hf_device_map: {self.model.hf_device_map}')

        logger.info(f'model_config: {self.model.config}')

    def _prepare_template(self) -> None:
        args = self.args
        template = get_template(
            args.template,
            self.tokenizer,
            args.system,
            args.max_length,
            truncation_strategy=args.truncation_strategy,
            max_pixels=args.max_pixels,
            loss_scale=args.loss_scale,
            tools_prompt=args.tools_prompt,
            sequence_parallel_size=args.sequence_parallel_size,
        )
        logger.info(f'default_system: {template.default_system}')
        self.template = template

    def _prepare_dataset(self):
        args = self.args
        dataset_kwargs = {
            'dataset_seed': args.dataset_seed,
            'num_proc': args.num_proc,
            'load_from_cache_file': args.load_from_cache_file,
            'download_mode': args.download_mode,
            'model_name': args.model_name,
            'model_author': args.model_author,
            'streaming': args.streaming,
            'streaming_val_size': args.streaming_val_size,
            'streaming_buffer_size': args.streaming_buffer_size,
            'strict': False
        }

        if len(args.val_dataset) > 0:
            # Loading val dataset
            _, val_dataset = load_dataset(args.val_dataset, 1.0, **dataset_kwargs)
            args.split_dataset_ratio = 0
        train_dataset, val_dataset = load_dataset(args.dataset, args.split_dataset_ratio, **dataset_kwargs)
        logger.info(f'train_dataset: {train_dataset}')
        logger.info(f'val_dataset: {val_dataset}')

        return train_dataset, val_dataset

    def run(self):
        args = self.args
        self._prepare_dataset()
        self._print_one_sample()

    def _print_one_sample(self):
        pass


def sft_main(args: Union[List[str], SftArguments, None] = None) -> List[Dict[str, Any]]:
    return SwiftSft(args).main()
