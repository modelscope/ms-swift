import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List
from typing import Optional

import jieba
import numpy as np
import torch
import torch.distributed as dist
from nltk.translate.bleu_score import (SmoothingFunction, sentence_bleu)
from rouge import Rouge
from rouge.rouge import Rouge
from transformers import BitsAndBytesConfig, GenerationConfig

from swift import (AdapterConfig, HubStrategy, LoRAConfig, Seq2SeqTrainer,
                   Seq2SeqTrainingArguments, Swift, SwiftConfig, ResTuningConfig, get_logger)
from swift.hub import HubApi, ModelScopeConfig
from swift.utils import (add_version_to_work_dir, parse_args, print_model_info,
                         seed_everything)
from swift.utils.llm_utils import data_collate_fn, print_example, stat_dataset
from utils import (DATASET_MAPPING, MODEL_MAPPING, TEMPLATE_MAPPING,
                   broadcast_string, find_all_linear_for_lora, get_dataset,
                   get_dist_setting, get_model_tokenizer, get_preprocess,
                   is_dist, is_master, plot_images, process_dataset,
                   select_bnb, select_dtype, show_layers)

logger = get_logger()


@dataclass
class SftArguments:
    model_type: str = field(
        default='qwen-7b-chat',
        metadata={'choices': list(MODEL_MAPPING.keys())})
    # qwen-7b: lora+4bitQ: 10G, lora+8bitQ: 14G, lora: 22G; full: 95G
    sft_type: str = field(
        default='lora')
    template_type: str = field(
        default=None, metadata={'choices': list(TEMPLATE_MAPPING.keys())})
    output_dir: str = 'runs'
    # DDP + MP(device_map) is not supported
    ddp_backend: Optional[str] = field(
        default=None, metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl']})

    seed: int = 42
    resume_from_ckpt: Optional[str] = None
    dtype: str = field(
        default='bf16', metadata={'choices': {'bf16', 'fp16', 'fp32'}})
    ignore_args_error: bool = False  # True: notebook compatibility

    dataset: str = field(
        default='alpaca-en,alpaca-zh',
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: int = 42
    dataset_sample: int = -1  # -1: all dataset
    dataset_test_size: float = 0.01
    system: str = 'you are a helpful assistant!'
    max_length: Optional[int] = 1024

    # If you want to use qlora, set the quantization_bit to 8 or 4.
    # And you need to install bitsandbytes: `pip install bitsandbytes -U`
    # note: bf16 and quantization have requirements for gpu architecture
    quantization_bit: Optional[int] = field(
        default=None, metadata={'choices': {4, 8}})
    bnb_4bit_comp_dtype: str = field(
        default=None, metadata={'choices': {'fp16', 'bf16', 'fp32'}})
    bnb_4bit_quant_type: str = field(
        default='nf4', metadata={'choices': {'fp4', 'nf4'}})
    bnb_4bit_use_double_quant: bool = True

    lora_target_modules: Optional[List[str]] = None
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.1
    adapter_length: int = 32

    gradient_checkpointing: bool = True
    batch_size: int = 1
    eval_batch_size: int = 1
    num_train_epochs: int = 1
    # if max_steps >= 0, override num_train_epochs
    max_steps: int = -1
    optim: str = 'adamw_torch'
    learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.
    predict_with_generate: bool = False
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.05

    eval_steps: int = 50
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    logging_steps: int = 5
    dataloader_num_workers: int = 1

    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True
    hub_strategy: HubStrategy = HubStrategy.EVERY_SAVE
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None,
        metadata={
            'help':
                'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
        })

    # other
    use_flash_attn: Optional[bool] = field(
        default=None,
        metadata={
            'help':
                "This parameter is used only when model_type.startswith('qwen-7b')"
        })

    def __post_init__(self):
        if is_dist():
            rank, local_rank, _, _ = get_dist_setting()
            torch.cuda.set_device(local_rank)
            self.seed += rank  # Avoid the same dropout
            if self.ddp_backend is None:
                self.ddp_backend = 'nccl'
            if self.ddp_backend == 'gloo' and self.quantization_bit is not None:
                raise ValueError('not supported, please use `nccl`')

            # Initialize in advance
            dist.init_process_group(backend=self.ddp_backend)

        if self.sft_type == 'full':
            assert self.quantization_bit is None, 'not supported'
            assert self.dtype != 'fp16', 'please use bf16 or fp32'
            if self.learning_rate is None:
                self.learning_rate = 1e-5
            if self.save_steps is None:
                # Saving the model takes a long time
                self.save_steps = self.eval_steps * 4
        else:
            if self.learning_rate is None:
                self.learning_rate = 1e-4
            if self.save_steps is None:
                self.save_steps = self.eval_steps

        if self.template_type is None:
            self.template_type = MODEL_MAPPING[self.model_type].get(
                'template', 'default')
            logger.info(f'Setting template_type: {self.template_type}')

        self.output_dir = os.path.join(self.output_dir, self.model_type)

        if self.lora_target_modules is None:
            self.lora_target_modules = MODEL_MAPPING[
                self.model_type]['lora_TM']
        self.torch_dtype, self.fp16, self.bf16 = select_dtype(self.dtype)
        if self.bnb_4bit_comp_dtype is None:
            self.bnb_4bit_comp_dtype = self.dtype
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self.quantization_bit, self.bnb_4bit_comp_dtype)

        if self.hub_model_id is None:
            self.hub_model_id = f'{self.model_type}-{self.sft_type}'
            logger.info(f'Setting hub_model_id: {self.hub_model_id}')
        if self.use_flash_attn is None:
            self.use_flash_attn = 'auto'
        if self.push_to_hub:
            api = HubApi()
            if self.hub_token is None:
                self.hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
            if self.hub_token is not None:
                api.login(self.hub_token)
            else:
                assert ModelScopeConfig.get_token(
                ) is not None, 'Please enter hub_token'
            logger.info('hub login successful!')


def llm_sft(args: SftArguments) -> None:
    print(f'device_count: {torch.cuda.device_count()}')
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(f'rank: {rank}, local_rank: {local_rank}, '
          f'world_size: {world_size}, local_world_size: {local_world_size}')
    seed_everything(args.seed)

    # ### Loading Model and Tokenizer
    kwargs = {'low_cpu_mem_usage': True}
    if is_dist():
        kwargs['device_map'] = {'': local_rank}
    else:
        kwargs['device_map'] = 'auto'
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        kwargs['quantization_config'] = quantization_config
    if args.model_type.startswith('qwen-7b'):
        kwargs['use_flash_attn'] = args.use_flash_attn

    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, **kwargs)

    if args.resume_from_ckpt is None:
        swift_config: Dict[str, SwiftConfig] = dict()
        for sft_type in args.sft_type.split(','):
            if sft_type == 'lora':
                if 'ALL' in args.lora_target_modules:
                    assert len(args.lora_target_modules) == 1
                    args.lora_target_modules = find_all_linear_for_lora(
                        model, args.quantization_bit, args.model_type)
                    logger.info(
                        f'Setting lora_target_modules: {args.lora_target_modules}'
                    )

                lora_config = LoRAConfig(
                    r=args.lora_rank,
                    target_modules=args.lora_target_modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout_p)
                logger.info(f'lora_config: {lora_config}')
                swift_config['lora'] = lora_config
            elif sft_type == 'adapter':
                adapter_config = AdapterConfig(
                    dim=model.config.hidden_size,
                    target_modules=MODEL_MAPPING[args.model_type].get(
                        'adapter_TM', 'mlp'),
                    method_name='forward',
                    hidden_pos=0,
                    adapter_length=args.adapter_length,
                )
                logger.info(f'adapter_config: {adapter_config}')
                swift_config['adapter'] = adapter_config
            elif sft_type == 'restuner':
                restuner_config = ResTuningConfig(
                    dims=model.config.hidden_size,
                    **MODEL_MAPPING[args.model_type]['restuner_TM']
                )
                logger.info(f'restuner_config: {restuner_config}')
                swift_config['restuner'] = restuner_config
        model = Swift.prepare_model(model, swift_config)
    else:
        model = Swift.from_pretrained(
            model, args.resume_from_ckpt, is_trainable=True)

    show_layers(model)
    print_model_info(model)
    logger.info(str(model))

    # ### Loading Dataset
    dataset = get_dataset(args.dataset.split(','))
    if isinstance(dataset, tuple):
        train_dataset, val_dataset = dataset
    else:
        train_dataset, val_dataset = process_dataset(dataset,
                                                     args.dataset_test_size,
                                                     args.dataset_sample,
                                                     args.dataset_seed)

    generation_config = {
            'do_sample': True,
            'top_p': 0.7,
            'max_length': args.max_length,
            'temperature': 0.95
    }

    # args.max_source_length = 64
    # args.max_target_length = 64
    # prompt_column = 'query'
    # response_column = 'response'
    # history_column = None
    # prefix = ''
    # max_target_length = 128
    #
    # def preprocess_function_eval(examples):
    #     inputs, targets = [], []
    #     for i in range(len(examples[prompt_column])):
    #         if examples[prompt_column][i] and examples[response_column][i]:
    #             query = examples[prompt_column][i]
    #             if history_column is None or len(examples[history_column][i]) == 0:
    #                 prompt = query
    #             else:
    #                 prompt = ''
    #                 history = examples[history_column][i]
    #                 for turn_idx, (old_query, response) in enumerate(history):
    #                     prompt += '[Round {}]\n问：{}\n答：{}\n'.format(
    #                         turn_idx, old_query, response)
    #                 prompt += '[Round {}]\n问：{}\n答：'.format(len(history), query)
    #             inputs.append(prompt)
    #             targets.append(examples[response_column][i])
    #
    #     inputs = [prefix + inp for inp in inputs]
    #     model_inputs = tokenizer(
    #         inputs,
    #         max_length=args.max_source_length,
    #         truncation=True,
    #         padding=True)
    #     labels = tokenizer(
    #         text_target=targets, max_length=max_target_length, truncation=True)
    #
    #     if True:
    #         labels['input_ids'] = [[(lb if lb != tokenizer.pad_token_id else -100)
    #                                 for lb in label]
    #                                for label in labels['input_ids']]
    #     model_inputs['labels'] = labels['input_ids']
    #
    #     return model_inputs
    #
    # def preprocess_function_train(examples):
    #     max_seq_length = args.max_source_length + args.max_target_length
    #
    #     model_inputs = {
    #         'input_ids': [],
    #         'labels': [],
    #     }
    #     for i in range(len(examples[prompt_column])):
    #         if examples[prompt_column][i] and examples[response_column][i]:
    #             query, answer = examples[prompt_column][i], examples[
    #                 response_column][i]
    #
    #             if history_column is None:
    #                 prompt = query
    #             else:
    #                 prompt = ''
    #                 history = examples[history_column][i]
    #                 for turn_idx, (old_query, response) in enumerate(history):
    #                     prompt += '[Round {}]\n问：{}\n答：{}\n'.format(
    #                         turn_idx, old_query, response)
    #                 prompt += '[Round {}]\n问：{}\n答：'.format(len(history), query)
    #
    #             prompt = prefix + prompt
    #             a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
    #             b_ids = tokenizer.encode(text=answer, add_special_tokens=False)
    #
    #             if len(a_ids) > args.max_source_length - 1:
    #                 a_ids = a_ids[:args.max_source_length - 1]
    #
    #             if len(b_ids) > args.max_target_length - 2:
    #                 b_ids = b_ids[:args.max_target_length - 2]
    #
    #             input_ids = tokenizer.build_inputs_with_special_tokens(
    #                 a_ids, b_ids)
    #
    #             if False:
    #                 context_length = input_ids.index(tokenizer.bos_token_id)
    #             else:
    #                 context_length = len(a_ids) + 2
    #             mask_position = context_length - 1
    #             labels = [-100] * context_length + input_ids[mask_position + 1:]
    #
    #             pad_len = max_seq_length - len(input_ids)
    #             input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    #             labels = labels + [tokenizer.pad_token_id] * pad_len
    #             if True:
    #                 labels = [(lb if lb != tokenizer.pad_token_id else -100)
    #                           for lb in labels]
    #
    #             model_inputs['input_ids'].append(input_ids)
    #             model_inputs['labels'].append(labels)
    #
    #     return model_inputs

    preprocess_func = get_preprocess(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        batched=True,
        validate_generation=False)
    train_dataset = train_dataset.map(preprocess_func, batched=True)
    preprocess_func = get_preprocess(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        batched=True,
        validate_generation=True)
    val_dataset = val_dataset.map(preprocess_func, batched=True)
    del dataset
    # Data analysis
    stat_dataset(train_dataset)
    stat_dataset(val_dataset)
    data_collator = partial(data_collate_fn, tokenizer=tokenizer)
    print_example(train_dataset[0], tokenizer)

    # ### Setting trainer_args
    output_dir = None
    if is_master():
        output_dir = add_version_to_work_dir(args.output_dir)
    if is_dist():
        # Make sure to set the same output_dir when using DDP.
        output_dir = broadcast_string(output_dir)
    trainer_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
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
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        sortish_sampler=True,
        optim=args.optim,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        hub_strategy=args.hub_strategy,
        hub_token=args.hub_token,
        push_to_hub=args.push_to_hub,
        resume_from_checkpoint=args.resume_from_ckpt,
        ddp_backend=args.ddp_backend,
        gradient_checkpointing=args.gradient_checkpointing,
        predict_with_generate=args.predict_with_generate,
        generation_config=GenerationConfig.from_dict(generation_config),
        local_rank=local_rank)

    if args.gradient_checkpointing:
        # fix: gradients will be None
        model.config.use_cache = True
        model.enable_input_require_grads()
        if is_dist():
            trainer_args._frozen = False  # Compatible with transformers==4.32.0
            trainer_args.ddp_find_unused_parameters = False
            trainer_args.ddp_broadcast_buffers = False
            trainer_args._frozen = True
    logger.info(f'trainer_args: {trainer_args}')

    def compute_metrics(prediction):
        preds, labels = prediction[0], prediction[1]

        score_dict = {
            'rouge-1': [],
            'rouge-2': [],
            'rouge-l': [],
            'bleu-4': []
        }

        def _decode(tokens, ignore_pad_token_for_loss=False):
            if ignore_pad_token_for_loss:
                tokens = np.where(tokens != -100, tokens,
                                  tokenizer.pad_token_id)
            tokens = np.where(tokens < tokenizer.vocab_size, tokens,
                              tokenizer.pad_token_id)
            return [
                t for t in tokenizer.batch_decode(
                    tokens, skip_special_tokens=True)
            ]

        for pred, label in zip(preds, labels):
            pred = ''.join(_decode(pred, False))
            label = ''.join(_decode(label, True))
            hypothesis = list(jieba.cut(pred))
            if len(hypothesis) == 0 or ''.join(hypothesis) == '.':
                hypothesis = [tokenizer.decode(tokenizer.eos_token_id)]
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis),
                                      ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v['f'] * 100, 4))
            bleu_score = sentence_bleu(
                [list(label)],
                list(pred),
                smoothing_function=SmoothingFunction().method3)
            score_dict['bleu-4'].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if args.predict_with_generate else None,
    )

    trainer.train(trainer_args.resume_from_checkpoint)

    # ### Visualization
    if is_master():
        images_dir = os.path.join(output_dir, 'images')
        logger.info(f'images_dir: {images_dir}')
        tb_dir = os.path.join(output_dir, 'runs')
        folder_name = os.listdir(tb_dir)[0]
        tb_dir = os.path.join(tb_dir, folder_name)
        plot_images(images_dir, tb_dir, ['train/loss'], 0.9)
        if args.push_to_hub:
            trainer._add_patterns_to_gitignores(['images/'])
            trainer.push_to_hub()


if __name__ == '__main__':
    args, remaining_argv = parse_args(SftArguments)
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    llm_sft(args)
