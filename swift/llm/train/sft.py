# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import List, Union

from datasets import Dataset as HfDataset

from swift.plugin import extra_callbacks, get_loss_func, get_metric
from swift.trainers import IntervalStrategy, TrainerFactory
from swift.utils import (append_to_jsonl, get_logger, get_model_parameter_info, is_master, plot_images, stat_array,
                         use_torchacc)
from ..argument import TrainArguments
from ..base import SwiftPipeline
from ..dataset import (EncodePreprocessor, GetLengthPreprocessor, IterablePackingDataset, LazyLLMDataset,
                       PackingDataset, load_dataset)
from ..infer import prepare_generation_config
from ..model import HfConfigFactory, get_model_arch
from ..utils import deep_getattr, dynamic_gradient_checkpointing
from .tuner import TunerMixin

logger = get_logger()


class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = TrainArguments
    args: args_class

    def __init__(self, args: Union[List[str], TrainArguments, None] = None) -> None:
        super().__init__(args)
        self.train_msg = {}
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_callbacks()

    def _prepare_gradient_checkpointing(self):
        args = self.args
        HfConfigFactory.set_model_config_attr(self.model, 'use_cache', False)
        if args.gradient_checkpointing:
            self.model.supports_gradient_checkpointing = True
            dynamic_gradient_checkpointing(self.model)
            self.model.enable_input_require_grads()
        model_meta = self.model.model_meta
        model_arch = get_model_arch(model_meta.model_arch)
        if model_meta.is_multimodal and model_arch:
            for vision_tower_name in model_arch.vision_tower:
                vision_tower = deep_getattr(self.model, vision_tower_name)
                if hasattr(vision_tower, 'enable_input_require_grads'):
                    try:
                        vision_tower.enable_input_require_grads()
                    except NotImplementedError:
                        pass

    def _prepare_generation_config(self):
        args = self.args
        self.model.origin_generation_config = self.model.generation_config
        self.model.generation_config = prepare_generation_config(self.model.generation_config,
                                                                 args.get_request_config(), self.tokenizer)
        logger.info(f'model.generation_config: {self.model.generation_config}')

    def _prepare_model_tokenizer(self):
        args = self.args
        self.model, self.processor = args.get_model_processor()

        if hasattr(self.model, 'hf_device_map'):
            logger.info(f'model.hf_device_map: {self.model.hf_device_map}')

        logger.info(f'model_info: {self.model.model_info}')

        self._prepare_generation_config()
        self._prepare_gradient_checkpointing()

    def _prepare_template(self) -> None:
        template = self.args.get_template(self.processor)
        if self.args.task_type == 'causal_lm':
            template.set_mode('train')
        if template.use_model:
            template.model = self.model
        self.template = template

    def _get_dataset(self):
        # The random shuffling of the training set occurs in the dataloader of the trainer.
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = load_dataset(
            args.dataset, split_dataset_ratio=args.split_dataset_ratio, **dataset_kwargs)
        if len(args.val_dataset) > 0:
            # Loading val dataset
            _, val_dataset = load_dataset(args.val_dataset, split_dataset_ratio=1.0, **dataset_kwargs)
            assert args.split_dataset_ratio == 0.
        logger.info(f'train_dataset: {train_dataset}')
        logger.info(f'val_dataset: {val_dataset}')

        return train_dataset, val_dataset

    def _get_loss_func(self):
        args = self.args
        loss_type = args.loss_type
        if loss_type is None and args.loss_scale != 'default':
            loss_type = 'loss_scale'
        return get_loss_func(loss_type)

    def _get_data_collator(self):
        args = self.args
        template = self.template
        padding_to = args.max_length if args.train_type == 'longlora' else None
        return partial(template.data_collator, padding_to=padding_to)

    @staticmethod
    def _save_val_dataset(output_dir: str, val_dataset):
        if isinstance(val_dataset, HfDataset):
            os.makedirs(output_dir, exist_ok=True)
            val_dataset_path = os.path.join(output_dir, 'val_dataset.jsonl')
            append_to_jsonl(val_dataset_path, val_dataset.to_list())
            logger.info(f'The split dataset from the training set will be saved at: {val_dataset_path}.')

    def run(self):
        args = self.args

        train_dataset, val_dataset = self._get_dataset()
        self._save_val_dataset(args.output_dir, val_dataset)
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        args.save_args()

        data_collator = self._get_data_collator()
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)

    def _get_trainer_kwargs(self):
        args = self.args
        if args.metric is not None:
            compute_metrics, preprocess_logits_for_metrics = get_metric(args.metric)
        elif args.predict_with_generate:
            compute_metrics, preprocess_logits_for_metrics = get_metric('nlg')
        else:
            compute_metrics, preprocess_logits_for_metrics = get_metric('acc')
            compute_metrics = partial(
                compute_metrics, acc_strategy=args.acc_strategy, is_encoder_decoder=self.template.is_encoder_decoder)
        return {
            'compute_metrics': compute_metrics,
            'preprocess_logits_for_metrics': preprocess_logits_for_metrics,
            'compute_loss_func': self._get_loss_func()
        }

    def _save_trainer_state(self, trainer):
        training_args = trainer.args
        state = trainer.state
        if not hasattr(state, 'last_model_checkpoint'):
            logger.warning('No training was carried out, which may be due to the dataset being too small '
                           'or incorrect usage of resume_from_checkpoint.')
            return
        if self.args.create_checkpoint_symlink:
            last_checkpoint = os.path.join(self.args.output_dir, 'last')
            best_checkpoint = os.path.join(self.args.output_dir, 'best')
            os.symlink(state.last_model_checkpoint, last_checkpoint)
            os.symlink(state.best_model_checkpoint, best_checkpoint)
            state.last_model_checkpoint = last_checkpoint
            state.best_model_checkpoint = best_checkpoint
        logger.info(f'last_model_checkpoint: {state.last_model_checkpoint}')
        logger.info(f'best_model_checkpoint: {state.best_model_checkpoint}')

        # Visualization
        if is_master() and not use_torchacc():
            if 'tensorboard' in training_args.report_to:
                images_dir = os.path.join(training_args.output_dir, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)
            if training_args.push_to_hub:
                trainer.push_to_hub()

        self.train_msg.update({
            'last_model_checkpoint': state.last_model_checkpoint,
            'best_model_checkpoint': state.best_model_checkpoint,
            'best_metric': state.best_metric,
            'global_step': state.global_step,
            'log_history': state.log_history,
            'memory': trainer.max_memory,
        })
        if is_master():
            jsonl_path = os.path.join(training_args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, self.train_msg)
        return self.train_msg

    def train(self, trainer):
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        trainer.train(trainer.args.resume_from_checkpoint)

        return self._save_trainer_state(trainer)

    def _prepare_callbacks(self):
        from .callback import DynamicLayerActivationCallback, TrainerAdapterCallback
        args = self.args
        callbacks = []
        if args.lisa_activated_layers > 0:
            assert args.train_type == 'full', 'LISA only supports full parameter training.'
            lisa_callback = DynamicLayerActivationCallback(
                n_layers=args.lisa_activated_layers,  # Number of layers to activate
                step_interval=args.lisa_step_interval,  # Step interval to update active layers
                model=self.model)
            lisa_callback.switch_active_layers()  # Make trainable parameters printing a correct value
            callbacks.append(lisa_callback)

        if args.is_adapter and args.train_type == 'adalora':
            callbacks.append(TrainerAdapterCallback(args))
        callbacks += extra_callbacks
        self.callbacks = callbacks

    def _stat_dataset(self, dataset: HfDataset):
        args = self.args
        if isinstance(dataset, HfDataset):
            dataset = GetLengthPreprocessor()(dataset, num_proc=args.dataset_num_proc)
            length = dataset['length']
        else:
            length = []
            for row in dataset:
                length.append(max([len(row[k]) for k in row.keys() if k.endswith('input_ids')]))
        _, stat_str = stat_array(length)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str

    def _encode_dataset(self, train_dataset, val_dataset):
        template = self.template
        args = self.args
        is_grpo = hasattr(args, 'rlhf_type') and args.rlhf_type == 'grpo'
        predict_with_generate = getattr(args, 'predict_with_generate', False)
        if not is_grpo:
            if args.packing:
                packing_dataset_cls = IterablePackingDataset if args.streaming else PackingDataset
                train_dataset = packing_dataset_cls(
                    self.template, train_dataset, num_workers=args.dataset_num_proc, strict=args.strict)
                if val_dataset is not None:
                    val_dataset = packing_dataset_cls(
                        self.template, val_dataset, num_workers=args.dataset_num_proc, strict=args.strict)
            elif args.lazy_tokenize:
                train_dataset = LazyLLMDataset(
                    train_dataset, template.encode, strict=args.strict, random_state=args.data_seed)
                if val_dataset is not None and not predict_with_generate:
                    val_dataset = LazyLLMDataset(
                        val_dataset, template.encode, strict=args.strict, random_state=args.data_seed)
            else:
                preprocessor = EncodePreprocessor(template=template)
                train_dataset = preprocessor(train_dataset, num_proc=args.dataset_num_proc, strict=args.strict)
                if val_dataset is not None and not predict_with_generate:
                    val_dataset = preprocessor(val_dataset, num_proc=args.dataset_num_proc, strict=args.strict)

            if is_master():
                inputs = train_dataset[0] if hasattr(train_dataset, '__len__') else next(iter(train_dataset))
                template.print_inputs(inputs, tokenizer_kwargs=inputs.pop('tokenizer_kwargs', None) or {})
            if isinstance(train_dataset, (HfDataset, PackingDataset)):
                self.train_msg['train_dataset'] = self._stat_dataset(train_dataset)
                if val_dataset is not None and not predict_with_generate:
                    self.train_msg['val_dataset'] = self._stat_dataset(val_dataset)

        if val_dataset is None and hasattr(args, 'training_args'):
            args.training_args.evaluation_strategy = IntervalStrategy.NO
            args.training_args.eval_strategy = IntervalStrategy.NO

        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        return train_dataset, val_dataset


def sft_main(args: Union[List[str], TrainArguments, None] = None):
    return SwiftSft(args).main()
