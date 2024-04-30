# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import json
import numpy as np
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig
from transformers import IntervalStrategy
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_npu_available

from swift.trainers.orpo_trainers import ORPOTrainer
from swift.utils import (check_json_format, get_dist_setting, get_logger, get_main, get_model_info, is_ddp_plus_mp,
                         is_dist, is_master, plot_images, seed_everything, show_layers)
from .tuner import prepare_model
from .utils import (ORPOArguments, Template, get_dataset, get_model_tokenizer, get_template, get_time_info,
                    set_generation_config)

logger = get_logger()


def llm_orpo(args: ORPOArguments) -> str:
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    training_args = args.training_args
    if is_torch_npu_available():
        print(f'device_count: {torch.npu.device_count()}')
    else:
        print(f'device_count: {torch.cuda.device_count()}')
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(f'rank: {rank}, local_rank: {local_rank}, ' f'world_size: {world_size}, local_world_size: {local_world_size}')

    if args.gpu_memory_fraction is not None:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(max(min(args.gpu_memory_fraction, 1.0), 0.01), device=device_id)

    # Loading Model and Tokenizer
    if is_deepspeed_zero3_enabled():
        model_kwargs = {'device_map': None}
    elif is_torch_npu_available():
        model_kwargs = {'device_map': local_rank if local_rank >= 0 else 0}
    else:
        model_kwargs = {'low_cpu_mem_usage': True}
        if is_dist() and not is_ddp_plus_mp():
            model_kwargs['device_map'] = {'': local_rank}
        else:
            model_kwargs['device_map'] = 'auto'

    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        model_kwargs['quantization_config'] = quantization_config

    kwargs = {
        'max_length': args.max_length,
        'use_unsloth': args.tuner_backend == 'unsloth',
        'load_in_4bit': args.quantization_bit == 4
    }
    if args.use_flash_attn is not None:
        kwargs['use_flash_attn'] = args.use_flash_attn
    model, tokenizer = get_model_tokenizer(
        args.model_type,
        args.torch_dtype,
        model_kwargs,
        model_id_or_path=args.model_id_or_path,
        revision=args.model_revision,
        **kwargs)

    logger.info(f'model_config: {model.config}')
    if hasattr(model, 'hf_device_map'):
        logger.info(f'model device_map {model.hf_device_map}')
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
    training_args.generation_config = generation_config

    model, _ = prepare_model(model, args)

    show_layers(model)
    model_info = None
    if not is_deepspeed_zero3_enabled():
        model_info = get_model_info(model)
        logger.info(model_info)
    logger.info(model)

    if args.gradient_checkpointing:
        model.config.use_cache = False  # fix transformers==4.36
        logger.info('Setting model.config.use_cache: False')
        model.enable_input_require_grads()

    # Loading Dataset
    random_state = np.random.RandomState(args.dataset_seed)
    train_dataset, val_dataset = get_dataset(
        args.dataset, args.dataset_test_ratio, random_state, check_dataset_strategy=args.check_dataset_strategy)
    val_dataset_sample = args.val_dataset_sample
    if train_dataset is not None and args.train_dataset_sample >= 0:
        train_dataset_sample = min(args.train_dataset_sample, train_dataset.shape[0])
        if train_dataset.shape[0] > train_dataset_sample:
            logger.info(f'train_dataset_sample: {train_dataset_sample}')
            train_idxs = random_state.permutation(train_dataset_sample)
            train_dataset = train_dataset.select(train_idxs)
        if val_dataset_sample is None:
            val_dataset_sample = max(int(train_dataset_sample * args.dataset_test_ratio), 1)
    if val_dataset is not None and val_dataset_sample is not None and val_dataset_sample >= 0:
        if val_dataset.shape[0] > val_dataset_sample:
            logger.info(f'val_dataset_sample: {val_dataset_sample}')
            val_idxs = random_state.permutation(val_dataset_sample)
            val_dataset = val_dataset.select(val_idxs)

    if val_dataset is None:
        training_args.evaluation_strategy = IntervalStrategy.NO
        training_args.do_eval = False
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    template: Template = get_template(
        args.template_type, tokenizer, args.system, args.max_length, args.truncation_strategy, model=model)
    args.system = template.default_system
    logger.info(f'system: {args.system}')

    # Trainer
    logger.info(f'training_args: {training_args}')

    trainer_kwargs = {}
    if args.check_model_is_latest is False:
        trainer_kwargs['check_model'] = False

    if not hasattr(training_args, 'model_init_kwargs'):
        training_args.model_init_kwargs = None
    if not hasattr(training_args, 'generate_during_eval'):
        training_args.generate_during_eval = False
    if not hasattr(training_args, 'max_completion_length'):  # encoder-decoder
        training_args.max_completion_length = None
    if not hasattr(training_args, 'padding_value'):
        training_args.padding_value = 0
    if not hasattr(training_args, 'dataset_num_proc'):
        training_args.dataset_num_proc = None
    if not hasattr(training_args, 'label_pad_token_id'):
        training_args.label_pad_token_id = -100
    if not hasattr(training_args, 'disable_dropout'):
        training_args.disable_dropout = True
    training_args.truncation_mode = 'keep_end' if args.truncation_left else 'keep_start'
    training_args.max_length = args.max_length
    training_args.max_prompt_length = args.max_prompt_length
    training_args.beta = args.beta

    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        template=template,
        test_oom_error=args.test_oom_error,
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
    trainer.train(training_args.resume_from_checkpoint)
    last_model_checkpoint = getattr(trainer.state, 'last_model_checkpoint', None)
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
    logger.info(f'best_model_checkpoint: {trainer.state.best_model_checkpoint}')
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
        'train_time': train_time,
        'last_model_checkpoint': last_model_checkpoint,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric': trainer.state.best_metric,
        'global_step': trainer.state.global_step,
        'log_history': trainer.state.log_history,
        'model_info': model_info,
        'dataset_info': trainer.dataset_info,
    }
    jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(run_info) + '\n')
    return run_info


orpo_main = get_main(ORPOArguments, llm_orpo)
