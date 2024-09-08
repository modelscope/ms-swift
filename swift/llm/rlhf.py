# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import json
import torch
from transformers import IntervalStrategy

from swift.trainers import RLHFTrainerFactory, get_preprocess_func, get_preprocessed_rlhf_dataset
from swift.utils import (append_to_jsonl, check_json_format, get_logger, get_main,
                         is_master, plot_images, seed_everything)
from . import LazyLLMDataset, print_example
from .sft import prepare_dataset, prepare_train_model_template
from .utils import RLHFArguments, TEMPLATE_MAPPING, get_time_info

logger = get_logger()


def llm_rlhf(args: RLHFArguments) -> Dict[str, Any]:
    logger.info(f'args: {args}')
    training_args = args.training_args
    streaming = args.streaming
    seed_everything(args.seed)

    is_generation = TEMPLATE_MAPPING[args.template_type].get('is_generation', False)
    if is_generation:
        logger.warning(f"Please check if args.template_type: '{args.template_type}' is correct. ")

    if args.gpu_memory_fraction is not None:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(max(min(args.gpu_memory_fraction, 1.0), 0.01), device=device_id)

    model, ref_model, template, callbacks = prepare_train_model_template(args)
    tokenizer = template.tokenizer

    train_dataset, val_dataset = prepare_dataset(args)

    # tokenize dataset
    is_encoder_decoder = model.config.is_encoder_decoder

    if args.lazy_tokenize:
        preprocess_func = get_preprocess_func(
            template=template, rlhf_type=args.rlhf_type, streaming=streaming)
        td0, tkwargs0 = preprocess_func(train_dataset[0]), {}
        print_example(td0, tokenizer, tkwargs0)
        train_dataset = LazyLLMDataset(train_dataset, template, encode_func=preprocess_func)
        if val_dataset is not None:
            val_dataset = LazyLLMDataset(val_dataset, template, encode_func=preprocess_func)
    else:
        preprocess_kwargs = {}
        if not streaming:
            preprocess_kwargs = dict(
                num_proc=args.preprocess_num_proc,
                desc='tokenizing paired dataset',
            )
        train_dataset, val_dataset = get_preprocessed_rlhf_dataset(
            train_dataset,
            val_dataset,
            template=template,
            rlhf_type=args.rlhf_type,
            streaming=streaming,
            **preprocess_kwargs)

    # Trainer
    logger.info(f'training_args: {training_args}')

    trainer_kwargs = RLHFTrainerFactory.get_training_args(args)

    trainer_kwargs['args'].generation_config = generation_config
    trainer_cls = RLHFTrainerFactory.get_trainer(args.rlhf_type)

    trainer_kwargs['streaming'] = streaming
    trainer_kwargs['is_encoder_decoder'] = is_encoder_decoder
    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        lazy_tokenize=args.lazy_tokenize,
        **trainer_kwargs)

    trainer.sft_args = args
    if is_master():
        for args_obj, fname in zip([args, training_args], ['sft_args.json', 'training_args.json']):
            fpath = os.path.join(args.output_dir, fname)
            logger.info(f'The {args_obj.__class__.__name__} will be saved in: {fpath}')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(args_obj.__dict__), f, ensure_ascii=False, indent=2)
    logging_path = os.path.join(args.output_dir, 'logging.jsonl')
    logger.info(f'The logging file will be saved in: {logging_path}')
    with template.training_context():
        trainer.train(training_args.resume_from_checkpoint)
    last_model_checkpoint = getattr(trainer.state, 'last_model_checkpoint', None)
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
    logger.info(f'best_model_checkpoint: {trainer.state.best_model_checkpoint}')
    if not streaming:
        train_time = get_time_info(trainer.state.log_history, len(train_dataset))
    # Visualization
    if is_master():
        if 'tensorboard' in args.training_args.report_to:
            images_dir = os.path.join(args.output_dir, 'images')
            logger.info(f'images_dir: {images_dir}')
            plot_images(images_dir, args.logging_dir, ['train/loss'], 0.9)
        if args.push_to_hub:
            trainer._add_patterns_to_gitignore(['images/'])
            trainer.push_to_hub()
    run_info = {
        'memory': trainer.perf['memory'],
        'last_model_checkpoint': last_model_checkpoint,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric': trainer.state.best_metric,
        'global_step': trainer.state.global_step,
        'log_history': trainer.state.log_history,
        'model_info': model_info
    }
    if not streaming:
        run_info.update({'train_time': train_time})
        run_info.update({'dataset_info': trainer.dataset_info})
    if is_master():
        jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
        append_to_jsonl(jsonl_path, run_info)
    return run_info


rlhf_main = get_main(RLHFArguments, llm_rlhf)
