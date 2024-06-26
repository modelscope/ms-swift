# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import json
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig
from transformers import IntervalStrategy
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_npu_available

from swift.trainers import RLHFTrainerFactory
from swift.utils import (check_json_format, get_dist_setting, get_logger, get_main, get_model_info, is_ddp_plus_mp,
                         is_dist, is_master, plot_images, seed_everything, show_layers)
from .tuner import prepare_model
from .utils import (TEMPLATE_MAPPING, RLHFArguments, Template, get_dataset, get_model_tokenizer, get_template,
                    get_time_info, set_generation_config)

logger = get_logger()


def llm_rlhf(args: RLHFArguments) -> Dict[str, Any]:
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

    # device map
    if is_deepspeed_zero3_enabled():
        model_kwargs = {'device_map': None}
    elif is_torch_npu_available():
        model_kwargs = {'device_map': local_rank if local_rank >= 0 else 0}
    elif args.device_map_config_path is not None:
        cwd = os.getcwd()
        config_path = args.device_map_config_path if os.path.isabs(args.device_map_config_path) else os.path.join(
            cwd, args.device_map_config_path)
        with open(config_path, 'r') as json_file:
            model_kwargs = {'device_map': json.load(json_file)}
    else:
        model_kwargs = {'low_cpu_mem_usage': True}
        if is_dist() and not is_ddp_plus_mp():
            model_kwargs['device_map'] = {'': local_rank}
        elif torch.cuda.device_count() == 1:
            model_kwargs['device_map'] = 'cuda:0'
        else:
            model_kwargs['device_map'] = 'auto'

    # quantization
    if args.quant_method == 'hqq':
        from transformers import HqqConfig
        if args.hqq_dynamic_config_path is not None:
            cwd = os.getcwd()
            config_path = args.hqq_dynamic_config_path if os.path.isabs(args.hqq_dynamic_config_path) else os.path.join(
                cwd, args.hqq_dynamic_config_path)
            with open(config_path, 'r') as json_file:
                quantization_config = HqqConfig(dynamic_config=json.load(json_file))
        else:
            if args.quantization_bit == 0:
                logger.info("You haven't set the quantization_bit parameter; set it to 8.")
                args.quantization_bit = 8
            quantization_config = HqqConfig(nbits=args.quantization_bit, axis=args.hqq_axis)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        model_kwargs['quantization_config'] = quantization_config
    elif args.quant_method == 'eetq':
        from transformers import EetqConfig
        quantization_config = EetqConfig('int8')
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        model_kwargs['quantization_config'] = quantization_config
    elif args.load_in_8bit or args.load_in_4bit:
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
    if args.local_repo_path:
        kwargs['local_repo_path'] = args.local_repo_path

    if args.rope_scaling:
        kwargs['rope_scaling'] = args.rope_scaling
        kwargs['max_length'] = args.max_length

    model, tokenizer = get_model_tokenizer(
        args.model_type,
        args.torch_dtype,
        model_kwargs,
        model_id_or_path=args.model_id_or_path,
        revision=args.model_revision,
        quant_method=args.quant_method,
        is_training=True,
        **kwargs)
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

    # Preparing LoRA
    model, _ = prepare_model(model, args)

    show_layers(model)
    logger.info(model)
    model_info = None
    if not is_deepspeed_zero3_enabled():
        model_info = get_model_info(model)
        logger.info(model_info)

    if args.gradient_checkpointing:
        model.config.use_cache = False  # fix transformers==4.36
        logger.info('Setting model.config.use_cache: False')
        model.enable_input_require_grads()

    if args.ref_model_type is not None:
        if args.ref_model_free:
            logger.warning(f"{args.rlhf_type} algorithm don't require ref model,\
                     therefore the ref model will not be loaded here.")
            ref_model = None
        else:
            ref_model, _ = get_model_tokenizer(
                args.ref_model_type,
                args.torch_dtype,
                model_kwargs,
                model_id_or_path=args.ref_model_id_or_path,
                revision=args.model_revision,
                quant_method=args.quant_method,
                **kwargs)
    else:
        ref_model = None

    if hasattr(model, 'hf_device_map'):
        logger.info(f'model device_map {model.hf_device_map}')

    # Loading Dataset
    train_dataset, val_dataset = get_dataset(
        args.dataset,
        args.dataset_test_ratio,
        args.dataset_seed,
        check_dataset_strategy=args.check_dataset_strategy,
        model_name=args.model_name,
        model_author=args.model_author)

    if len(args.val_dataset) > 0:
        # Loading val dataset
        _, val_dataset = get_dataset(
            args.val_dataset,
            1.0,
            args.dataset_seed,
            check_dataset_strategy=args.check_dataset_strategy,
            model_name=args.model_name,
            model_author=args.model_author)

    train_dataset, val_dataset = args._handle_dataset_compat(train_dataset, val_dataset)
    if val_dataset is None:
        training_args.evaluation_strategy = IntervalStrategy.NO
        training_args.do_eval = False
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')

    template_kwargs = {}
    template_info = TEMPLATE_MAPPING[args.template_type]
    use_model = template_info.get('use_model', False)
    if use_model:
        template_kwargs['model'] = model

    if args.sequence_parallel_size and args.sequence_parallel_size > 1:
        template_kwargs['sequence_parallel_size'] = args.sequence_parallel_size

    template: Template = get_template(
        args.template_type, tokenizer, args.system, args.max_length, args.truncation_strategy, model=model)
    if not template.support_multi_round and 'history' in train_dataset[0]:
        logger.info(
            'The current template does not support multi-turn dialogue. The chatml template is used by default. \
You can also use the --model_type parameter to specify the  template.')
        template: Template = get_template(
            'chatml', tokenizer, args.system, args.max_length, args.truncation_strategy, model=model)
    args.system = template.default_system
    logger.info(f'system: {args.system}')

    # Trainer
    logger.info(f'training_args: {training_args}')

    trainer_kwargs = RLHFTrainerFactory.get_training_args(args)

    if ref_model is not None:
        trainer_kwargs['ref_model'] = ref_model

    trainer_kwargs['args'].generation_config = generation_config
    trainer_cls = RLHFTrainerFactory.get_trainer(args.rlhf_type)

    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        template=template,
        **trainer_kwargs)

    trainer.sft_args = args
    if is_master():
        for args_obj, fname in zip([args, training_args], ['sft_args.json', 'training_args.json']):
            fpath = os.path.join(args.output_dir, fname)
            logger.info(f'The {args_obj.__class__.__name__} will be saved in: {fpath}')
            args_dict = args_obj.__dict__
            args_dict.pop('hub_token', None)
            args_dict.pop('push_to_hub_token', None)
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(args_dict), f, ensure_ascii=False, indent=2)
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


rlhf_main = get_main(RLHFArguments, llm_rlhf)
