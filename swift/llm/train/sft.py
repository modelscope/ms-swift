from functools import partial
from typing import Any, Dict, List, Union

from transformers import IntervalStrategy

from swift.plugin.callback import extra_callbacks
from swift.plugin.optimizer import optimizers_map
from swift.trainers import TrainerFactory
from swift.utils import (append_to_jsonl, check_json_format, compute_acc_metrics, compute_nlg_metrics, get_dist_setting,
                         get_logger, get_model_parameter_info, is_ddp_plus_mp, is_dist, is_master, plot_images,
                         preprocess_logits_for_acc, seed_everything, show_layers, use_torchacc)
from ..argument import SftArguments
from ..base import SwiftPipeline
from ..dataset import EncodePreprocessor, load_dataset, stat_dataset
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
        self._prepare_callbacks()

        self.model = prepare_tuner(self.model, args)
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

        model, tokenizer = get_model_tokenizer(
            args.model,
            args.torch_dtype,
            args.device_map,
            model_type=args.model_type,
            revision=args.model_revision,
            quantization_config=args.quantization_config,
            attn_impl=args.attn_impl,
            rope_scaling=args.rope_scaling,
            use_unsloth=args.tuner_backend == 'unsloth')

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
        template = self.template

        train_dataset, val_dataset = self._prepare_dataset()
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)

        padding_to = args.max_length if args.train_type == 'longlora' else None
        data_collator = partial(template.data_collator, padding_to=padding_to, model=self.model)
        optimizers = self._prepare_optimizers(train_dataset)

        if args.predict_with_generate:
            compute_metrics = partial(compute_nlg_metrics, tokenizer=tokenizer)
            preprocess_logits_for_metrics = None
        else:
            compute_metrics = partial(
                compute_acc_metrics,
                acc_strategy=args.acc_strategy,
                is_encoder_decoder=self.model.config.is_encoder_decoder)
            compute_metrics = compute_metrics
            preprocess_logits_for_metrics = preprocess_logits_for_acc

        trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=self.callbacks,
            optimizers=optimizers,
            tokenizer=self.tokenizer,
        )
        template.register_post_encode_hook([self.model])
        trainer.train(args.training_args.resume_from_checkpoint)

    def _prepare_optimizers(self, train_dataset):
        args = self.args
        optimizer_callback = optimizers_map['default']
        if args.lorap_lr_ratio:
            optimizer_callback = optimizers_map['lorap']
        if args.use_galore:
            if args.galore_target_modules is None:
                args.galore_target_modules = find_all_linears(model, 0, args.model_type, args.quant_method)
            if args.galore_with_embedding:
                args.galore_target_modules += find_embedding(model)
            optimizer_callback = optimizers_map['galore']

        return optimizer_callback(self.model, train_dataset, args)

    def _prepare_callbacks(self):
        args = self.args
        callbacks = []
        if args.lisa_activated_layers > 0:
            assert args.train_type == 'full', 'LISA only supports full parameter training.'
            lisa_callback = DynamicLayerActivationCallback(
                n_layers=args.lisa_activated_layers,  # Number of layers to activate
                step_interval=args.lisa_step_interval,  # Step interval to update active layers
                model=model)
            lisa_callback.switch_active_layers()  # Make trainable parameters printing a correct value
            callbacks.append(lisa_callback)

        if args.is_adapter and args.tuner_backend == 'swift':
            callbacks.append(TrainerAdapterCallback(args))
        callbacks += extra_callbacks
        self.callbacks = callbacks

    def _encode_dataset(self, train_dataset, val_dataset):
        template = self.template
        args = self.args
        if args.packing:
            from swift.llm.utils.utils import ConstantLengthDataset
            train_dataset = ConstantLengthDataset.get_packed_dataset(
                template, train_dataset, args.max_length, lazy_tokenize=args.lazy_tokenize)
            if val_dataset is not None:
                val_dataset = ConstantLengthDataset.get_packed_dataset(
                    template, val_dataset, args.max_length, lazy_tokenize=args.lazy_tokenize)
            if not args.lazy_tokenize:
                template.print_inputs(train_dataset[0], template, {})
                self.train_msg['train_dataset'] = stat_dataset(train_dataset)
                if val_dataset is not None:
                    self.train_msg['val_dataset'] = stat_dataset(val_dataset)
        elif not args.lazy_tokenize:
            inputs = template.encode(next(iter(train_dataset)) if args.streaming else train_dataset[0])
            template.print_inputs(inputs)
            model = None if args.num_proc > 1 else self.model
            train_dataset = EncodePreprocessor(
                template, model=model)(
                    train_dataset, num_proc=args.num_proc, load_from_cache_file=args.load_from_cache_file)
            if val_dataset is not None:
                val_dataset = EncodePreprocessor(
                    template, model=model)(
                        val_dataset, num_proc=args.num_proc, load_from_cache_file=args.load_from_cache_file)

            if not args.streaming:
                self.train_msg['train_dataset'] = stat_dataset(train_dataset)
                if val_dataset is not None:
                    self.train_msg['val_dataset'] = stat_dataset(val_dataset)
        else:
            inputs = template.encode(train_dataset[0])
            template.print_inputs(inputs)
            train_dataset = LazyLLMDataset(train_dataset, template.encode)
            if val_dataset is not None:
                val_dataset = LazyLLMDataset(val_dataset, template.encode)
        return train_dataset, val_dataset


def sft_main(args: Union[List[str], SftArguments, None] = None) -> List[Dict[str, Any]]:
    return SwiftSft(args).main()
