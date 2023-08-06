import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import torch
from transformers import BitsAndBytesConfig
from utils import (DATASET_MAPPING, DEFAULT_PROMPT, MODEL_MAPPING, get_dataset,
                   get_model_tokenizer, plot_images, process_dataset,
                   select_bnb, select_dtype, show_layers)

from swift import (HubStrategy, LoraConfig, Seq2SeqTrainer,
                   Seq2SeqTrainingArguments, Swift, get_logger)
from swift.utils import (add_version_to_work_dir, parse_args, print_model_info,
                         seed_everything)
from swift.utils.llm_utils import (data_collate_fn, print_example,
                                   stat_dataset, tokenize_function)

logger = get_logger()


@dataclass
class SftArguments:
    model_type: str = field(
        default='qwen-7b', metadata={'choices': list(MODEL_MAPPING.keys())})
    # qwen-7b: lora+4bitQ: 10G, lora+8bitQ: 14G, lora: 22G; full: 95G
    sft_type: str = field(
        default='lora', metadata={'choices': ['lora', 'full']})
    output_dir: str = 'runs'

    seed: int = 42
    resume_from_ckpt: Optional[str] = None
    dtype: str = field(
        default='fp16', metadata={'choices': {'bf16', 'fp16', 'fp32'}})
    ignore_args_error: bool = False  # True: notebook compatibility

    dataset: str = field(
        default='alpaca-en,alpaca-zh',
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: int = 42
    dataset_sample: int = 20000  # -1: all dataset
    dataset_test_size: float = 0.01
    prompt: str = DEFAULT_PROMPT
    max_length: Optional[int] = 2048

    # If you want to use qlora, set the quantization_bit to 8 or 4.
    # And you need to install bitsandbytes: `pip install bitsandbytes -U`
    # note: bf16 and quantization have requirements for gpu architecture
    quantization_bit: Optional[int] = field(
        default=None, metadata={'choices': {4, 8}})
    bnb_4bit_comp_dtype: str = field(
        default='fp16', metadata={'choices': {'fp16', 'bf16', 'fp32'}})
    bnb_4bit_quant_type: str = field(
        default='nf4', metadata={'choices': {'fp4', 'nf4'}})
    bnb_4bit_use_double_quant: bool = True

    lora_target_modules: Optional[List[str]] = None
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.1

    gradient_checkpoint: bool = True
    batch_size: int = 1
    num_train_epochs: int = 1
    optim: str = 'adamw_torch'
    learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.1

    eval_steps: int = 50
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    logging_steps: int = 5

    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True
    hub_strategy: HubStrategy = HubStrategy.EVERY_SAVE
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = None

    def __post_init__(self):
        if self.sft_type == 'lora':
            if self.learning_rate is None:
                self.learning_rate = 1e-4
            if self.save_steps is None:
                self.save_steps = self.eval_steps
        elif self.sft_type == 'full':
            assert self.quantization_bit is None, 'not supported'
            if self.learning_rate is None:
                self.learning_rate = 1e-5
            if self.save_steps is None:
                # Saving the model takes a long time
                self.save_steps = self.eval_steps * 4
        else:
            raise ValueError(f'sft_type: {self.sft_type}')

        self.output_dir = os.path.join(self.output_dir, self.model_type)

        if self.lora_target_modules is None:
            self.lora_target_modules = MODEL_MAPPING[
                self.model_type]['lora_TM']
        self.torch_dtype, self.fp16, self.bf16 = select_dtype(self.dtype)
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self.quantization_bit, self.bnb_4bit_comp_dtype)

        if self.hub_model_id is None:
            self.hub_model_id = f'{self.model_type}-sft'


def llm_sft(args: SftArguments) -> None:
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)

    # ### Loading Model and Tokenizer
    model_kwargs = {'device_map': 'auto'}
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        model_kwargs['quantization_config'] = quantization_config

    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, **model_kwargs)

    if args.gradient_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ### Preparing lora
    if args.sft_type == 'lora':
        if args.resume_from_ckpt is None:
            lora_config = LoraConfig(
                r=args.lora_rank,
                target_modules=args.lora_target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout_p,
                task_type='CAUSAL_LM')
            logger.info(f'lora_config: {lora_config}')
            model = Swift.prepare_model(model, lora_config)
        else:
            model = Swift.from_pretrained(
                model, args.resume_from_ckpt, is_trainable=True)

    show_layers(model)
    print_model_info(model)

    # ### Loading Dataset
    dataset = get_dataset(args.dataset.split(','))
    train_dataset, val_dataset = process_dataset(dataset,
                                                 args.dataset_test_size,
                                                 args.dataset_sample,
                                                 args.dataset_seed)
    tokenize_func = partial(
        tokenize_function,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length)
    train_dataset = train_dataset.map(tokenize_func)
    val_dataset = val_dataset.map(tokenize_func)
    del dataset
    # Data analysis
    stat_dataset(train_dataset)
    stat_dataset(val_dataset)
    data_collator = partial(data_collate_fn, tokenizer=tokenizer)
    print_example(train_dataset[0], tokenizer)

    # ### Setting trainer_args
    output_dir = add_version_to_work_dir(args.output_dir)
    trainer_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_steps=args.eval_steps,
        dataloader_num_workers=1,
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
        resume_from_checkpoint=args.resume_from_ckpt)

    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ### Visualization
    images_dir = os.path.join(output_dir, 'images')
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
