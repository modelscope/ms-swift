# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial

import json
import numpy as np
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig

from swift.trainers import (IntervalStrategy, Seq2SeqTrainer,
                            Seq2SeqTrainingArguments)
from swift.tuners import (LongLoRAConfig, LongLoRAModelType, LoraConfig,
                          LoRAConfig, NEFTuneConfig, Swift)
from swift.utils import (check_json_format, compute_acc_metrics,
                         compute_nlg_metrics, freeze_model_parameters,
                         get_dist_setting, get_logger, is_ddp_plus_mp, is_dist,
                         is_master, plot_images, preprocess_logits_for_metrics,
                         print_model_info, seed_everything, show_layers)
from .utils import (SftArguments, Template, add_self_cognition_dataset,
                    data_collate_fn, dataset_map, find_all_linear_for_lora,
                    get_additional_saved_files, get_dataset,
                    get_model_tokenizer, get_template, print_example,
                    set_generation_config, sort_by_max_length, stat_dataset)

logger = get_logger()


def llm_sft(args: SftArguments) -> str:
    logger.info(f'args: {args}')
    print(f'device_count: {torch.cuda.device_count()}')
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(f'rank: {rank}, local_rank: {local_rank}, '
          f'world_size: {world_size}, local_world_size: {local_world_size}')
    seed_everything(args.seed)

    # Loading Model and Tokenizer
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

    kwargs = {}
    if args.use_flash_attn is not None:
        kwargs['use_flash_attn'] = args.use_flash_attn
    if args.model_cache_dir is not None:
        kwargs['model_dir'] = args.model_cache_dir
    model, tokenizer = get_model_tokenizer(args.model_type, args.torch_dtype,
                                           model_kwargs, **kwargs)
    logger.info(f'model_config: {model.config}')

    # Preparing LoRA
    if args.sft_type in ('lora', 'qalora', 'longlora'):
        if args.resume_from_checkpoint is None:
            if 'ALL' in args.lora_target_modules:
                assert len(args.lora_target_modules) == 1
                args.lora_target_modules = find_all_linear_for_lora(
                    model, args.quantization_bit, args.model_type)
                logger.info(
                    f'Setting lora_target_modules: {args.lora_target_modules}')
            if args.sft_type == 'lora':
                lora_kwargs = {}
                if args.tuner_backend == 'swift':
                    lora_config_cls = LoRAConfig
                elif args.tuner_backend == 'peft':
                    lora_config_cls = LoraConfig
                    lora_kwargs['task_type'] = 'CAUSAL_LM'
                lora_config = lora_config_cls(
                    r=args.lora_rank,
                    target_modules=args.lora_target_modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout_p,
                    **lora_kwargs)
                model = Swift.prepare_model(model, lora_config)
                logger.info(f'lora_config: {lora_config}')
            elif args.sft_type == 'longlora':
                assert args.tuner_backend != 'peft'
                assert LongLoRAModelType.LLAMA in args.model_type
                longlora_config = LongLoRAConfig(
                    r=args.lora_rank,
                    target_modules=args.lora_target_modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout_p,
                    model_type=LongLoRAModelType.LLAMA,
                    use_flash_attn=args.use_flash_attn)
                model = Swift.prepare_model(model, longlora_config)
                logger.info(f'longlora_config: {longlora_config}')
            elif args.sft_type == 'qalora':
                assert getattr(
                    model, 'quantization_method',
                    None) == 'gptq', 'qalora must be used with auto_gptq'
                lora_kwargs = {}
                lora_config = LoRAConfig(
                    r=args.lora_rank,
                    target_modules=args.lora_target_modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout_p,
                    use_qa_lora=True,
                    **lora_kwargs)
                model = Swift.prepare_model(model, lora_config)
                logger.info(f'lora_config: {lora_config}')
        else:
            model = Swift.from_pretrained(
                model, args.resume_from_checkpoint, is_trainable=True)
    elif args.sft_type == 'full':
        if args.freeze_parameters > 0:
            freeze_model_parameters(model, args.freeze_parameters)
    else:
        raise ValueError(f'args.sft_type: {args.sft_type}')

    if args.neftune_alpha > 0.001:
        neftune_config = NEFTuneConfig(noise_alpha=args.neftune_alpha)
        model = Swift.prepare_model(model, neftune_config)
        logger.info(f'neftune_config: {neftune_config}')

    show_layers(model)
    print_model_info(model)
    logger.info(model)

    # Loading Dataset
    random_state = np.random.RandomState(args.dataset_seed)
    train_dataset, val_dataset = get_dataset(
        args.dataset,
        args.dataset_test_ratio,
        random_state,
        check_dataset_strategy=args.check_dataset_strategy)
    val_dataset_sample = args.val_dataset_sample
    if train_dataset is not None and args.train_dataset_sample >= 0:
        train_dataset_sample = min(args.train_dataset_sample,
                                   train_dataset.shape[0])
        if train_dataset.shape[0] > train_dataset_sample:
            logger.info(f'train_dataset_sample: {train_dataset_sample}')
            train_idxs = random_state.permutation(train_dataset_sample)
            train_dataset = train_dataset.select(train_idxs)
        if val_dataset_sample is None:
            val_dataset_sample = max(
                int(train_dataset_sample * args.dataset_test_ratio), 1)
    if val_dataset is not None and val_dataset_sample is not None and val_dataset_sample >= 0:
        if val_dataset.shape[0] > val_dataset_sample:
            logger.info(f'val_dataset_sample: {val_dataset_sample}')
            val_dataset = val_dataset.select(range(val_dataset_sample))
    # add self-cognition dataset
    if args.self_cognition_sample > 0:
        train_dataset = add_self_cognition_dataset(train_dataset,
                                                   args.self_cognition_sample,
                                                   args.model_name,
                                                   args.model_author)

    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    template: Template = get_template(args.template_type, tokenizer,
                                      args.system, args.max_length,
                                      args.truncation_strategy)
    args.system = template.default_system
    logger.info(f'system: {args.system}')
    train_dataset = dataset_map(train_dataset, template.encode)
    if val_dataset is not None:
        val_dataset = dataset_map(val_dataset, template.encode)
    if args.test_oom_error:
        train_dataset = sort_by_max_length(train_dataset, 20000)
    # Data analysis
    data_collator = partial(
        data_collate_fn,
        tokenizer=tokenizer,
        padding_to=args.max_length if args.sft_type == 'longlora' else None)
    print_example(train_dataset[0], tokenizer)
    stat_dataset(train_dataset)
    if val_dataset is not None:
        stat_dataset(val_dataset)

    # Setting training_args
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')
    set_generation_config(model, generation_config)
    evaluation_strategy = IntervalStrategy.STEPS
    load_best_model_at_end = True
    if val_dataset is None:
        evaluation_strategy = IntervalStrategy.NO
        load_best_model_at_end = False
    additional_saved_files = []
    if args.sft_type == 'full':
        additional_saved_files = get_additional_saved_files(args.model_type)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=evaluation_strategy,
        logging_dir=args.logging_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model='rouge-l'
        if args.predict_with_generate else 'loss',
        greater_is_better=args.predict_with_generate,
        sortish_sampler=True,
        optim=args.optim,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        push_hub_strategy=args.push_hub_strategy,
        hub_token=args.hub_token,
        push_to_hub=args.push_to_hub,
        resume_from_checkpoint=args.resume_from_checkpoint,
        ddp_backend=args.ddp_backend,
        gradient_checkpointing=args.gradient_checkpointing,
        predict_with_generate=args.predict_with_generate,
        generation_config=generation_config,
        local_rank=local_rank,
        only_save_model=args.only_save_model,
        train_sampler_random=args.train_sampler_random,
        report_to=args.report_to,
        deepspeed=args.deepspeed,
        additional_saved_files=additional_saved_files,
        save_on_each_node=args.save_on_each_node)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    if is_dist():
        # Compatible with https://github.com/huggingface/transformers/pull/25903
        training_args._frozen = False
        if args.gradient_checkpointing:
            training_args.ddp_find_unused_parameters = False
            training_args.ddp_broadcast_buffers = False
        else:
            training_args.ddp_find_unused_parameters = True
            training_args.ddp_broadcast_buffers = True

    logger.info(f'training_args: {training_args}')

    trainer_kwargs = {}
    if args.predict_with_generate:
        trainer_kwargs['compute_metrics'] = partial(
            compute_nlg_metrics, tokenizer=tokenizer)
    else:
        trainer_kwargs['compute_metrics'] = compute_acc_metrics
        trainer_kwargs[
            'preprocess_logits_for_metrics'] = preprocess_logits_for_metrics
    if args.check_model_is_latest is False:
        trainer_kwargs['check_model'] = False

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        **trainer_kwargs)
    trainer.sft_args = args
    if is_master():
        for args_obj, fname in zip([args, training_args],
                                   ['sft_args.json', 'training_args.json']):
            fpath = os.path.join(args.output_dir, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(
                    check_json_format(args_obj.__dict__),
                    f,
                    ensure_ascii=False,
                    indent=2)

    trainer.train(training_args.resume_from_checkpoint)
    last_model_checkpoint = getattr(trainer.state, 'last_model_checkpoint',
                                    None)
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
    logger.info(
        f'best_model_checkpoint: {trainer.state.best_model_checkpoint}')
    # Visualization
    if is_master():
        images_dir = os.path.join(args.output_dir, 'images')
        logger.info(f'images_dir: {images_dir}')
        tb_dir = os.path.join(args.output_dir, 'runs')
        plot_images(images_dir, tb_dir, ['train/loss'], 0.9)
        if args.push_to_hub:
            trainer._add_patterns_to_gitignores(['images/'])
            trainer.push_to_hub()
    return {
        'last_model_checkpoint': last_model_checkpoint,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric': trainer.state.best_metric,
        'global_step': trainer.state.global_step,
        'log_history': trainer.state.log_history,
    }
