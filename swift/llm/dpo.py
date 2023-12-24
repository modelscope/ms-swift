# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig

from swift.trainers import (IntervalStrategy, Seq2SeqTrainingArguments, DPOTrainer)
from swift.utils import (check_json_format, compute_acc_metrics,
                         compute_nlg_metrics, get_dist_setting, get_logger, get_model_info,
                         is_ddp_plus_mp, is_dist, is_master, plot_images,
                         preprocess_logits_for_metrics, seed_everything,
                         show_layers)
from .utils import History
from .tuner import prepare_model
from .utils import (LazyLLMDataset, DPOArguments, Template,
                    data_collate_dpo_fn, dataset_map,
                    get_additional_saved_files, get_dataset,
                    get_model_tokenizer, get_template, set_generation_config, sort_by_max_length)
from .utils.template import Context, _concat_context_list, _simplify_context_list

logger = get_logger()


def llm_dpo(args: DPOArguments) -> str:
    logger.info(f'args: {args}')
    print(f'device_count: {torch.cuda.device_count()}')
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(f'rank: {rank}, local_rank: {local_rank}, '
          f'world_size: {world_size}, local_world_size: {local_world_size}')
    seed_everything(args.seed)

    # Loading Model and Tokenizer
    model_kwargs = {'low_cpu_mem_usage': True}
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
    model_kwargs['device_map'] = 'auto'
    model, tokenizer = get_model_tokenizer(args.model_type, args.torch_dtype,
                                           model_kwargs, **kwargs)
    if args.ref_model_type is not None:
        model_kwargs['device_map'] = 'auto'
        ref_model, ref_tokenizer = get_model_tokenizer(args.ref_model_type, args.torch_dtype,
                                               model_kwargs, **kwargs)
    else:
        ref_model = None
        ref_tokenizer = None

    logger.info(f'model_config: {model.config}')
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

    model = prepare_model(model, args)

    show_layers(model)
    model_info = get_model_info(model)
    logger.info(model_info)
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

    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    template: Template = get_template(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        args.truncation_strategy,
        model=model)
    args.system = template.default_system
    logger.info(f'system: {args.system}')

    class DPOTemplate(Template):

        def __init__(self, template):
            self.template = template

        def __getattr__(self, name):
            return getattr(self.template, name)

        def encode(self, example: Dict[str,
                                    Any]) -> Dict[str, Optional[List[int]]]:
            query: Optional[str] = example.get('query', None)
            history: Optional[History] = example.get('history', None)
            system: Optional[str] = example.get('system', None)
            if query is None:
                query = ''
            if history is None:
                history = []
            if len(history) > 0:
                assert self.support_multi_round, 'the template not support multi-round chat'
            if system is None: 
                if self.use_default_system:
                    system = self.default_system
            else:
                assert self.prefix_has_system is not None, 'not support `system`'
            res_context_list: List[Context] = []
            compute_loss_idx: List[int] = []
            if system is None:
                assert template.prefix != template.prefix_has_system, f'template.prefix: {template.prefix}'
                prefix = template.prefix
            else:
                prefix = template.prefix_has_system
            _concat_context_list(
                prefix, res_context_list, compute_loss_idx, system=system)
            for i, (q, r) in enumerate(history):
                _concat_context_list(
                    [*template.prompt, '{{RESPONSE}}', *template.chat_sep],
                    res_context_list,
                    compute_loss_idx,
                    query=q,
                    response=r,
                    round0=i)
            _concat_context_list(
                template.prompt,
                res_context_list,
                compute_loss_idx,
                query=query,
                round0=len(history))
            res_context_list, compute_loss_idx = _simplify_context_list(
                res_context_list, compute_loss_idx)
            return {
                'prompt': res_context_list[0],
                'chosen': example['response'],
                'rejected': example['rejected_response'],
            }

    if not args.lazy_tokenize:
        logger.info(f'Using num_proc: {args.preprocess_num_proc}')
        # train_dataset = train_dataset.map(template.encode)
        train_dataset = dataset_map(train_dataset, DPOTemplate(template).encode,
                                    args.preprocess_num_proc)
        if val_dataset is not None:
            val_dataset = dataset_map(val_dataset, DPOTemplate(template).encode,
                                      args.preprocess_num_proc)
        if args.test_oom_error:
            train_dataset = sort_by_max_length(train_dataset, 20000)
        # Data analysis
        # print_example(train_dataset[0], tokenizer)

        def stat_dataset(llm_dataset) -> None:
            """Statistical analysis was performed on the dataset"""
            _token_len = []
            from datasets import Dataset as HfDataset
            from swift.utils.np_utils import stat_array
            if isinstance(llm_dataset, HfDataset):
                prompt = llm_dataset['prompt']
                chosen = llm_dataset['chosen']
                rejected = llm_dataset['rejected']
                for ii, cc, rr in zip(prompt, chosen, rejected):
                    _token_len.append(len(ii) + max(len(cc), len(rr)))
            else:
                for d in llm_dataset:
                    _token_len.append(len(d['prompt']))
            _, stat_str = stat_array(_token_len)
            logger.info(f'Dataset Token Length: {stat_str}')

        stat_dataset(train_dataset)
        if val_dataset is not None:
            stat_dataset(val_dataset)
    else:
        train_dataset = LazyLLMDataset(train_dataset, DPOTemplate(template))
        val_dataset = LazyLLMDataset(val_dataset, DPOTemplate(template))

    data_collator = partial(
        data_collate_dpo_fn,
        tokenizer=tokenizer,
        padding_to=args.max_length if args.sft_type == 'longlora' else None)
    # Setting training_args
    evaluation_strategy = IntervalStrategy.STEPS
    load_best_model_at_end = False
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
        save_strategy=args.save_strategy,
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
        disable_tqdm=args.disable_tqdm,
        save_on_each_node=args.save_on_each_node,
        acc_strategy=args.acc_strategy,
        save_safetensors=args.save_safetensors)

    if args.gradient_checkpointing:
        model.config.use_cache = False  # fix transformers==4.36
        logger.info('Setting model.config.use_cache: False')
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
        compute_metrics = partial(
            compute_acc_metrics, acc_strategy=args.acc_strategy)
        trainer_kwargs['compute_metrics'] = compute_metrics
        trainer_kwargs[
            'preprocess_logits_for_metrics'] = preprocess_logits_for_metrics
    if args.check_model_is_latest is False:
        trainer_kwargs['check_model'] = False

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_prompt_length=2048,
        max_length=4096,
        **trainer_kwargs)
    trainer.sft_args = args
    if is_master():
        for args_obj, fname in zip([args, training_args],
                                   ['sft_args.json', 'training_args.json']):
            fpath = os.path.join(args.output_dir, fname)
            logger.info(
                f'The {args_obj.__class__.__name__} will be saved in: {fpath}')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(
                    check_json_format(args_obj.__dict__),
                    f,
                    ensure_ascii=False,
                    indent=2)
    logging_path = os.path.join(args.output_dir, 'logging.jsonl')
    logger.info(f'The logging file will be saved in: {logging_path}')
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
        'model_info': model_info,
    }
