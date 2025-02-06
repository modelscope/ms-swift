# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from swift.llm import RLHFArguments
from swift.llm.train import SwiftRLHF
from swift.trainers import GRPOTrainer
from swift.utils import get_logger, get_model_parameter_info

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = get_logger()


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
    model_id_or_path = 'Qwen/Qwen2.5-7B-Instruct'  # model_id or model_path
    SYSTEM_PROMPT = (
        'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant '
        'first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning '
        'process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., '
        '<think> reasoning process here </think><answer> answer here </answer>')
    output_dir = 'output'

    # dataset
    dataset = ['AI-MO/NuminaMath-TIR#100']  # dataset_id or dataset_path
    data_seed = 42
    max_length = 2048
    split_dataset_ratio = 0.01  # Split validation set
    num_proc = 4  # The number of processes for data loading.

    # lora
    lora_rank = 8
    lora_alpha = 32

    # GRPO hyperarguments
    num_generations = 8  # G in GRPO paper
    max_prompt_length = 1024  # truncation to avoid OOM

    # reward_model_id_or_path = ''

    # reward function
    def accuracy_reward(completions, solution, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        contents = [completion[0]['content'] for completion in completions]
        rewards = []
        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
                print('Failed to parse gold solution: ', sol)
            rewards.append(reward)

        return rewards

    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think><answer>.*?</answer>$'
        completion_contents = [completion[0]['content'] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    # set system prompt in R1 paper
    SYSTEM_PROMPT = (
        'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant '
        'first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning '
        'process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., '
        '<think> reasoning process here </think><answer> answer here </answer>')
    # training_args
    training_args = RLHFArguments(
        rlhf_type='grpo',
        num_generations=num_generations,
        reward_funcs=[accuracy_reward, format_reward],
        system=SYSTEM_PROMPT,
        output_dir=output_dir,
        learning_rate=1e-4,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        gradient_accumulation_steps=16,
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
    )
    CustomGRPO(training_args).main()
