# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from swift.llm import RLHFArguments
from swift.llm.train import SwiftRLHF
from swift.trainers import GRPOTrainer
from swift.utils import get_logger, get_model_parameter_info

logger = get_logger()

# environments
# pip install math_verify
# pip install git+https://github.com/huggingface/trl.git


class CustomGRPO(SwiftRLHF):
    args_class = RLHFArguments
    args: args_class

    def run(self):
        train_dataset, val_dataset = self._get_dataset()
        data_collator = self._get_data_collator()
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer = GRPOTrainer(
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


if __name__ == '__main__':
    model_id_or_path = 'Qwen/Qwen2.5-1.5B-Instruct'  # model_id or model_path
    output_dir = 'output'

    # dataset
    dataset = ['AI-MO/NuminaMath-TIR#5000']  # dataset_id or dataset_path
    data_seed = 42
    split_dataset_ratio = 0.01  # Split validation set
    num_proc = 4  # The number of processes for data loading.

    # GRPO hyperarguments
    num_generations = 3  # G in GRPO paper
    max_completion_length = 1024
    use_vllm = True
    vllm_gpu_memory_utilization = 0.3
    reward_funcs = ['accuracy', 'format']  # see details in swift/plugin/orm.py
    # set system prompt in R1 paper
    SYSTEM_PROMPT = (
        'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
        'The assistant first thinks about the reasoning process in the mind and then provides the user '
        'with the answer. The reasoning process and answer are enclosed within <think> </think> '
        'and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> '
        'answer here </answer>')
    # training_args
    training_args = RLHFArguments(
        rlhf_type='grpo',
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        system=SYSTEM_PROMPT,
        model=model_id_or_path,
        dataset=dataset,
        reward_funcs=reward_funcs,
        split_dataset_ratio=split_dataset_ratio,
        output_dir=output_dir,
        learning_rate=1e-6,
        dataset_num_proc=num_proc,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        gradient_accumulation_steps=8,
        save_total_limit=2,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
    )
    CustomGRPO(training_args).main()
