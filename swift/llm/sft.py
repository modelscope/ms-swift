# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

import json
import torch
import transformers
from datasets import Dataset as HfDataset
from packaging import version
from transformers import BitsAndBytesConfig, GenerationConfig, IntervalStrategy
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_npu_available, strtobool

from swift.torchacc_utils import patch_acc_model
from swift.trainers import TrainerFactory
from swift.trainers.utils import can_return_loss, find_labels
from swift.utils import (append_to_jsonl, check_json_format, compute_acc_metrics, compute_nlg_metrics, get_dist_setting,
                         get_logger, get_main, get_model_info, is_ddp_plus_mp, is_dist, is_master, plot_images,
                         preprocess_logits_for_metrics, seed_everything, show_layers, use_torchacc)
from .accelerator import ta_accelerate
from .tuner import prepare_model
from .utils import (TEMPLATE_MAPPING, LazyLLMDataset, PtArguments, RLHFArguments, SftArguments, Template, dataset_map,
                    deep_getattr, dynamic_vit_gradient_checkpointing, get_dataset, get_mllm_arch, get_model_tokenizer,
                    get_template, get_time_info, print_example, set_generation_config, sort_by_max_length, stat_dataset)

logger = get_logger()


def _get_train_val_dataset(args: SftArguments) -> Tuple[HfDataset, Optional[HfDataset]]:
    # Loading Dataset
    train_dataset, val_dataset = get_dataset(
        args.dataset,
        args.dataset_test_ratio,
        args.dataset_seed,
        check_dataset_strategy=args.check_dataset_strategy,
        model_name=args.model_name,
        model_author=args.model_author,
        streaming=args.streaming,
        streaming_val_size=args.streaming_val_size,
        streaming_buffer_size=args.streaming_buffer_size)
    if len(args.val_dataset) > 0:
        # Loading val dataset
        _, val_dataset = get_dataset(
            args.val_dataset,
            1.0,
            args.dataset_seed,
            check_dataset_strategy=args.check_dataset_strategy,
            model_name=args.model_name,
            model_author=args.model_author,
            streaming=args.streaming,
            streaming_val_size=args.streaming_val_size,
            streaming_buffer_size=args.streaming_buffer_size)

    train_dataset, val_dataset = args._handle_dataset_compat(train_dataset, val_dataset)
    if args.train_type == 'ppo':  # Remove response columns from dataset
        existing_columns = list(next(iter(train_dataset)).keys())
        columns_to_remove = [col for col in ['response', 'rejected_response'] if col in existing_columns]
        train_dataset = train_dataset.map(remove_columns=columns_to_remove)
        logger.info(f'remove columns: {columns_to_remove} in PPO')
        if val_dataset is not None:
            val_dataset = val_dataset.map(remove_columns=columns_to_remove)
    # The random shuffling of the training set occurs in the dataloader of the trainer.
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    return train_dataset, val_dataset


def llm_sft_megatron(args: SftArguments) -> Dict[str, Any]:
    assert os.path.exists(args.resume_from_checkpoint), (
        f'Please run `CUDA_VISIBLE_DEVICES=0 swift export --model_type {args.model_type} --tp {args.tp} --pp {args.pp} '
        f'--megatron_output_dir {args.resume_from_checkpoint} --to_megatron true` '
        'to convert the weights to Megatron format.')
    from swift.llm.megatron import (MegatronArguments, patch_megatron, get_megatron_model_convert, forward_step,
                                    train_valid_test_datasets_provider as _train_valid_test_datasets_provider)
    from megatron.core.enums import ModelType
    from megatron.training import pretrain
    _, tokenizer = get_model_tokenizer(
        args.model_type, model_id_or_path=args.model_id_or_path, revision=args.model_revision, load_model=False)

    # Loading Dataset
    template: Template = get_template(args.template_type, tokenizer, args.system, args.max_length,
                                      args.truncation_strategy)

    train_dataset, val_dataset = _get_train_val_dataset(args)
    td0, tkwargs0 = template.encode(train_dataset[0])
    print_example(td0, tokenizer, tkwargs0)
    train_dataset = LazyLLMDataset(train_dataset, template.encode)
    if val_dataset is not None:
        val_dataset = LazyLLMDataset(val_dataset, template.encode)

    res = MegatronArguments.load_megatron_config(tokenizer.model_dir)
    res.update(MegatronArguments.from_sft_args(args, train_dataset, val_dataset))
    megatron_args = MegatronArguments(**res)
    extra_args = megatron_args.parse_to_megatron()

    model_provider, _ = get_megatron_model_convert(args.model_type)
    train_valid_test_datasets_provider = partial(
        _train_valid_test_datasets_provider, train_dataset=train_dataset, val_dataset=val_dataset, template=template)
    train_valid_test_datasets_provider.is_distributed = True
    patch_megatron(tokenizer)
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults=extra_args)
    logger.info(f'output_dir: {args.output_dir}')
    if is_master():
        fpath = os.path.join(args.output_dir, 'sft_args.json')
        logger.info(f'The {args.__class__.__name__} will be saved in: {fpath}')
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(check_json_format(args.__dict__), f, ensure_ascii=False, indent=2)
    logging_path = os.path.join(args.output_dir, 'logging.jsonl')
    logger.info(f'The logging file will be saved in: {logging_path}')
    # Visualization
    if is_master():
        images_dir = os.path.join(args.output_dir, 'images')
        logger.info(f'images_dir: {images_dir}')
        plot_images(images_dir, args.logging_dir, ['train/loss'], 0.9)
    return {}


def get_default_device_map():
    if is_deepspeed_zero3_enabled() or os.environ.get('ACCELERATE_USE_FSDP', 'False') == 'true':
        return None
    local_rank = get_dist_setting()[1]
    if is_torch_npu_available():
        if local_rank >= 0:
            return f'npu:{local_rank}'
        else:
            return 'npu:0'
    if torch.cuda.device_count() == 0:
        return 'cpu'
    elif torch.cuda.device_count() == 1:
        return 'cuda:0'
    elif is_dist() and not is_ddp_plus_mp():
        return f'cuda:{local_rank}'
    else:
        return 'auto'


def prepare_model_template_train(args, msg: Optional[Dict[str, Any]] = None):

    if args.gpu_memory_fraction is not None:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(max(min(args.gpu_memory_fraction, 1.0), 0.01), device=device_id)

    if is_torch_npu_available():
        print(f'device_count: {torch.npu.device_count()}')
    else:
        print(f'device_count: {torch.cuda.device_count()}')
    print(f'rank: {args.rank}, local_rank: {args.local_rank}, '
          f'world_size: {args.world_size}, local_world_size: {args.local_world_size}')

    # Loading Model and Tokenizer
    model_kwargs = {}
    if not use_torchacc():
        if args.device_map_config is not None:
            device_map = args.device_map_config
        else:
            device_map = get_default_device_map()
        model_kwargs['device_map'] = device_map
        if device_map == 'auto':
            model_kwargs['low_cpu_mem_usage'] = True
    if args.device_max_memory:
        n_gpu = torch.cuda.device_count()
        assert len(args.device_max_memory) == n_gpu // args.local_world_size
        model_kwargs['max_memory'] = {
            i: mem
            for i, mem in zip(range(max(args.local_rank, 0), n_gpu, args.local_world_size), args.device_max_memory)
        }

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
    elif args.load_in_8bit or args.load_in_4bit:  # bnb
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

    model, tokenizer = get_model_tokenizer(
        args.model_type,
        args.torch_dtype,
        model_kwargs,
        model_id_or_path=args.model_id_or_path,
        revision=args.model_revision,
        quant_method=args.quant_method,
        is_training=True,
        **kwargs)

    if hasattr(model, 'hf_device_map'):
        logger.info(f'model.hf_device_map: {model.hf_device_map}')
    for k in ['gptq', 'awq', 'aqlm']:
        if getattr(model, f'is_{k}', None):
            args.quant_method = k
            logger.info(f'Setting args.quant_method: {args.quant_method}')
            break
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
    set_generation_config(model, generation_config)
    logger.info(f'model.generation_config: {model.generation_config}')
    args.training_args.generation_config = model.generation_config

    if use_torchacc():
        import torchacc as ta
        # Get `label` and `return_loss` before 'ta_accelerate' because it will
        # wrapper the model and make these properties wrong.
        label_names = find_labels(model)
        return_loss = can_return_loss(model)
        model = patch_acc_model(model, args)

    if args.is_multimodal and args.gradient_checkpointing and args.vit_use_gc:
        dynamic_vit_gradient_checkpointing(model, args.model_type)

    if args.gradient_checkpointing:
        model.config.use_cache = False  # fix transformers==4.36
        logger.info('Setting model.config.use_cache: False')
        model.enable_input_require_grads()
        mllm_arch = get_mllm_arch(args.model_type)
        if mllm_arch is not None:
            for vision_tower_name in mllm_arch.vision_tower:
                vision_tower = deep_getattr(model, vision_tower_name)
                if hasattr(vision_tower, 'enable_input_require_grads'):
                    try:
                        vision_tower.enable_input_require_grads()
                    except NotImplementedError:
                        pass

    # Preparing LoRA
    model, callbacks = prepare_model(model, args)

    show_layers(model)
    logger.info(model)
    model_info = get_model_info(model)
    logger.info(model_info)
    if isinstance(msg, dict):
        msg['model_info'] = model_info

    if use_torchacc():
        model.config.use_cache = False
        logger.info('Setting model.config.use_cache: False')
        model = ta_accelerate(
            model,
            args.fsdp_num,
            args.model_layer_cls_name,
            args.bf16,
            args.fp16,
            gradient_checkpointing=True,
            fsdp_flatten_parameters=(args.sft_type == 'full'))
        model.label_names = label_names
        model.return_loss = return_loss

    template_kwargs = {}
    template_kwargs['use_loss_scale'] = args.use_loss_scale
    if args.loss_scale_config_path is not None:
        cwd = os.getcwd()
        config_path = args.loss_scale_config_path if os.path.isabs(args.loss_scale_config_path) else os.path.join(
            cwd, args.loss_scale_config_path)
        with open(config_path, 'r') as json_file:
            template_kwargs['loss_scale_map'] = json.load(json_file)
    template_kwargs['tools_prompt'] = args.tools_prompt
    if args.sequence_parallel_size and args.sequence_parallel_size > 1:
        template_kwargs['sequence_parallel_size'] = args.sequence_parallel_size
    template_kwargs['rescale_image'] = args.rescale_image
    template: Template = get_template(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        args.truncation_strategy,
        model=model,
        **template_kwargs)
    template._is_training = True
    if args.streaming:
        template.encode = partial(template.encode, streaming=args.streaming)
    logger.info(f'system: {template.default_system}')
    logger.info(f'args.lazy_tokenize: {args.lazy_tokenize}')

    if not isinstance(args, RLHFArguments):
        return model, template, callbacks

    # ref_model
    ref_model = None
    if not args.ref_model_free and (args.ref_model_type or args.sft_type == 'full' or args.rlhf_type == 'ppo'):
        if args.ref_model_type:
            kwargs['model_id_or_path'] = args.ref_model_id_or_path
            kwargs['revision'] = args.ref_model_revision
        else:
            kwargs['model_id_or_path'] = args.model_id_or_path
            kwargs['revision'] = args.model_revision

        # Be aware of the unexpected behavior caused by double monkey patching.
        ref_model, _ = get_model_tokenizer(
            args.ref_model_type or args.model_type,
            args.torch_dtype,
            model_kwargs,
            quant_method=args.quant_method,
            **kwargs)
        ref_model.requires_grad_(False).eval()

    template.ref_model = ref_model
    return model, ref_model, template, callbacks


def prepare_dataset(args, template: Template, msg: Optional[Dict[str, Any]] = None):
    training_args = args.training_args
    train_dataset, val_dataset = _get_train_val_dataset(args)
    if use_torchacc():
        training_args.train_dataset_sample = train_dataset.shape[0] if train_dataset is not None else 0

    if val_dataset is None:
        training_args.evaluation_strategy = IntervalStrategy.NO
        training_args.eval_strategy = IntervalStrategy.NO
        training_args.do_eval = False

    tokenizer = template.tokenizer
    dataset_info = {}
    if args.packing:
        from swift.llm.utils.utils import ConstantLengthDataset
        train_dataset = ConstantLengthDataset.get_packed_dataset(
            template, train_dataset, args.max_length, lazy_tokenize=args.lazy_tokenize)
        if val_dataset is not None:
            val_dataset = ConstantLengthDataset.get_packed_dataset(
                template, val_dataset, args.max_length, lazy_tokenize=args.lazy_tokenize)
        if not args.lazy_tokenize:
            print_example(train_dataset[0], tokenizer, {})
            dataset_info['train_dataset'] = stat_dataset(train_dataset)
            if val_dataset is not None:
                dataset_info['val_dataset'] = stat_dataset(val_dataset)
    elif not args.lazy_tokenize:
        model = template.model
        if not args.streaming:
            if args.preprocess_num_proc > 1:
                use_model = TEMPLATE_MAPPING[args.template_type].get('use_model', False)
                if use_model:
                    args.preprocess_num_proc = 1
                    logger.warning('The current Template does not support num_proc. '
                                   f'Setting args.preprocess_num_proc to: {args.preprocess_num_proc}')
                else:
                    template.model = None
        td0, tkwargs0 = template.encode(train_dataset[0])
        print_example(td0, tokenizer, tkwargs0)
        train_dataset = dataset_map(train_dataset, template.encode, args.preprocess_num_proc, streaming=args.streaming)
        if val_dataset is not None:
            val_dataset = dataset_map(val_dataset, template.encode, args.preprocess_num_proc, streaming=args.streaming)
        template.model = model  # recover
        if args.test_oom_error:
            train_dataset = sort_by_max_length(train_dataset, 20000)
        # Data analysis
        if train_dataset is None:
            logger.error('Error accessing train_dataset properties. '
                         'Please ensure that the dataset is properly initialized,'
                         'and every sample of the train_dataset not empty.')
            raise AttributeError('Failed to access dataset attributes,train_dataset is None. This might be because:\n'
                                 '(1) The dataset contains None for input or labels;\n'
                                 "(2) The 'max_length' setting is too short causing data truncation.")
        if not args.streaming:
            dataset_info['train_dataset'] = stat_dataset(train_dataset)
            if val_dataset is not None:
                dataset_info['val_dataset'] = stat_dataset(val_dataset)
    else:
        td0, tkwargs0 = template.encode(train_dataset[0])
        print_example(td0, tokenizer, tkwargs0)
        train_dataset = LazyLLMDataset(train_dataset, template.encode)
        if val_dataset is not None:
            val_dataset = LazyLLMDataset(val_dataset, template.encode)
    if isinstance(msg, dict):
        msg['dataset_info'] = dataset_info
    return train_dataset, val_dataset


def trainer_train(
    args,
    model,
    template,
    train_dataset,
    val_dataset,
    callbacks=None,
    msg=None,
    ref_model=None,
    reward_model=None,
    value_model=None,
) -> Dict[str, Any]:
    if msg is None:
        msg = {}
    training_args = args.training_args
    padding_to = args.max_length if args.sft_type == 'longlora' else None
    tokenizer = template.tokenizer
    data_collator = partial(template.data_collator, padding_to=padding_to)

    if use_torchacc():
        train_batch_size = args.batch_size
        eval_batch_size = args.eval_batch_size
        train_batch_size *= args.world_size
        eval_batch_size *= args.world_size
        training_args.per_device_train_batch_size = train_batch_size
        training_args.per_device_eval_batch_size = eval_batch_size
        training_args.group_by_length = use_torchacc()

    logger.info(f'training_args: {training_args}')

    trainer_cls, trainer_kwargs = TrainerFactory.get_trainer_info(args)
    if not hasattr(model.config, 'is_encoder_decoder'):
        model.config.is_encoder_decoder = False
    is_encoder_decoder = model.config.is_encoder_decoder
    trainer_kwargs['is_encoder_decoder'] = is_encoder_decoder
    if args.check_model_is_latest is False:
        trainer_kwargs['check_model'] = False
    if isinstance(args, RLHFArguments):
        trainer_kwargs['ref_model'] = ref_model
    elif args.predict_with_generate:
        trainer_kwargs['compute_metrics'] = partial(compute_nlg_metrics, tokenizer=tokenizer)
    else:
        compute_metrics = partial(
            compute_acc_metrics, acc_strategy=args.acc_strategy, is_encoder_decoder=is_encoder_decoder)
        trainer_kwargs['compute_metrics'] = compute_metrics
        trainer_kwargs['preprocess_logits_for_metrics'] = preprocess_logits_for_metrics
    if args.train_type == 'ppo':
        trainer_kwargs['reward_model'] = reward_model
        trainer_kwargs['value_model'] = value_model
    trainer = trainer_cls(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs)
    trainer.is_multimodal = args.is_multimodal
    trainer.sft_args = args
    if use_torchacc():
        trainer.label_names = model.label_names
        trainer.can_return_loss = model.return_loss
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
    # Visualization
    if is_master() and not use_torchacc():
        if 'tensorboard' in training_args.report_to:
            images_dir = os.path.join(args.output_dir, 'images')
            logger.info(f'images_dir: {images_dir}')
            plot_images(images_dir, args.logging_dir, ['train/loss'], 0.9)
        if args.push_to_hub:
            trainer.push_to_hub()
    run_info = {
        'memory': trainer.perf['memory'],
        'last_model_checkpoint': last_model_checkpoint,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric': trainer.state.best_metric,
        'global_step': trainer.state.global_step,
        'log_history': trainer.state.log_history,
        **msg
    }
    if not args.streaming:
        train_time = get_time_info(trainer.state.log_history, len(train_dataset))
        run_info.update({'train_time': train_time})
    for key in ['gen_time', 'gen_len']:
        if key in trainer.perf and trainer.perf[key] != 0:
            run_info[key] = trainer.perf[key]
    if is_master():
        jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
        append_to_jsonl(jsonl_path, run_info)
    return run_info


def llm_sft(args: SftArguments) -> Dict[str, Any]:
    logger.info(f'args: {args}')
    seed_everything(args.seed)

    is_generation = TEMPLATE_MAPPING[args.template_type].get('is_generation', False)
    if is_generation and type(args) is SftArguments:
        logger.warning(f"Please check if args.template_type: '{args.template_type}' is correct. "
                       'Currently, SFT is in progress, but the template is used for PT.')
    elif not is_generation and type(args) is PtArguments:
        logger.warning(f"Please check if args.template_type: '{args.template_type}' is correct. "
                       'Currently, PT is in progress, but the template is used for SFT.')

    if args.train_backend == 'megatron':
        return llm_sft_megatron(args)
    msg = {}
    model, template, callbacks = prepare_model_template_train(args, msg)
    train_dataset, val_dataset = prepare_dataset(args, template, msg)
    return trainer_train(args, model, template, train_dataset, val_dataset, callbacks=callbacks, msg=msg)


def get_sft_main(args, llm):
    if use_torchacc():
        import torchacc as ta
        import torch_xla.runtime as xr
        xla_cache_path = os.getenv('TORCHACC_CACHE_PATH')
        read_only = strtobool(os.getenv('TORCHACC_CACHE_PATH_READ_ONLY', '0'))
        suffix = f'_rank{xr.global_ordinal()}'
        if xla_cache_path and not xla_cache_path.endswith(suffix):
            xr.initialize_cache(xla_cache_path + suffix, readonly=read_only)
        if version.parse(transformers.__version__) < version.parse('4.41.0'):
            # This patch should be called before `llm_sft`.
            ta.accelerate_hf_trainer()
    return get_main(args, llm)


sft_main = get_sft_main(SftArguments, llm_sft)
pt_main = get_sft_main(PtArguments, llm_sft)
