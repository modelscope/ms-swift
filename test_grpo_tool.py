# Copyright (c) Alibaba, Inc. and its affiliates.
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from typing import List, Union

from swift.utils import get_logger, get_model_parameter_info
from swift.llm.argument import RLHFArguments
from swift.llm.train.kto import prepare_kto_dataset
from swift.llm.train.sft import SwiftSft
logger = get_logger()


class SwiftRLHF(SwiftSft):
    args_class = RLHFArguments
    args: args_class

    def _prepare_model_tokenizer(self):
        from swift.llm.infer.utils import prepare_adapter
        args = self.args
        for key in ['ref', 'reward', 'value']:
            origin_key = key
            setattr(self, f'{key}_model', None)
            if key == 'value':
                if args.rlhf_type == 'ppo':
                    key = 'reward'
                else:
                    continue
            model_id_or_path = getattr(args, f'{key}_model')
            if model_id_or_path is None:
                continue
            model_type = getattr(args, f'{key}_model_type')
            model_revision = getattr(args, f'{key}_model_revision')
            adapters = args.adapters if key == 'ref' else args.reward_adapters
            if origin_key == 'ref':
                task_type = args.task_type
                num_labels = None
            else:
                task_type = 'seq_cls'
                num_labels = 1
            # Be aware of the unexpected behavior caused by double monkey patching.
            model, processor = args.get_model_processor(
                model=model_id_or_path,
                model_type=model_type,
                model_revision=model_revision,
                task_type=task_type,
                num_labels=num_labels)

            model = prepare_adapter(args, model, adapters)
            if origin_key in {'ref', 'reward'}:
                model.requires_grad_(False).eval()
            else:
                model = self.prepare_model(args, model, task_type=task_type)
                logger.info(f'value_model: {model}')
                model_parameter_info = get_model_parameter_info(model)
                self.train_msg['value_model_parameter_info'] = model_parameter_info
                logger.info(f'value_model_parameter_info: {model_parameter_info}')
            setattr(self, f'{origin_key}_model', model)
            if origin_key == 'reward' and args.rlhf_type == 'grpo':
                reward_template = self.args.get_template(processor)
                if reward_template.use_model:
                    reward_template.model = model
                self.reward_template = reward_template

        super()._prepare_model_tokenizer()

    def _prepare_template(self) -> None:
        args = self.args
        super()._prepare_template()
        model_mapping = {'kto': 'kto', 'ppo': 'pt', 'grpo': 'pt'}
        self.template.set_mode(model_mapping.get(args.rlhf_type, 'rlhf'))

        if args.rlhf_type == 'ppo':
            args.training_args.stop_token_id = self.template.template_meta.stop_token_id

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _get_trainer_kwargs(self):
        trainer_kwargs = {}
        for key in ['ref', 'reward', 'value']:
            key = f'{key}_model'
            model = getattr(self, key, None)
            if model or self.args.rlhf_type == 'ppo':
                trainer_kwargs[key] = model
        if hasattr(self, 'reward_template'):
            trainer_kwargs['reward_template'] = self.reward_template
        if self.args.rlhf_type == 'grpo':
            trainer_kwargs['reward_funcs'] = self.args.reward_funcs
        return trainer_kwargs


if __name__ == '__main__':
    """
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy format \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#1000' \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true
    """
    args = RLHFArguments(
        rlhf_type = 'grpo',
        model = '~/.cache/modelscope/hub/qwen/Qwen2.5-7B-Instruct',
        reward_funcs = ['simplereward'],
        train_type = 'lora',
        lora_rank = 8,
        lora_alpha = 32,
        target_modules = 'all-linear',
        torch_dtype = 'bfloat16',
        dataset = 'math_operations_dataset.jsonl',
        max_completion_length = 1024,
        num_train_epochs = 1,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        learning_rate = 1e-5,
        gradient_accumulation_steps = 1,
        eval_steps = 100,
        save_steps = 100,
        save_total_limit = 2,
        logging_steps = 1,
        max_length = 2048,
        output_dir = 'output',
        warmup_ratio = 0.05,
        dataloader_num_workers = 4,
        dataset_num_proc = 4,
        num_generations = 4,
        temperature = 0.9,
        log_completions = True,
        system = 'tool_system.txt',
        loss_scale="default"

    )
    SwiftRLHF(args).run()
