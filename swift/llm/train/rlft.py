# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from typing import List, Union
import torch
from functools import partial
from modelscope import AutoModelForSequenceClassification, GenerationConfig, AutoTokenizer
from rouge import Rouge
from trl.models.utils import unwrap_model_for_generation
from copy import deepcopy, copy
from swift.llm.train.tuner import prepare_adapter
from .mcts import monte_carlo_tree_search
from swift.llm.template.template_inputs import StdTemplateInputs
from ..argument import RLFTArguments
from .sft import SwiftSft
from swift.utils import get_logger, get_model_parameter_info
from ...plugin.orm import orms
from ...trainers import TrainerFactory

logger = get_logger()


class SwiftRLFT(SwiftSft):
    args_class = RLFTArguments
    args: args_class

    def _prepare_prm(self):
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.args.reward_model, trust_remote_code=True, num_labels=1, torch_dtype=torch.bfloat16,
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.args.reward_model)

    def get_reward_by_model(self, conv):
        import torch
        conv_formatted = self.reward_tokenizer.apply_chat_template(conv, tokenize=False)
        conv_tokenized = self.reward_tokenizer(conv_formatted, return_tensors="pt").to(self.reward_model.device)
        with torch.no_grad():
            return self.reward_model(**conv_tokenized)[0].cpu().detach().item()

    def rollout(self, data, trainer, rd):
        splitter = [
            '.', '。', '\n\n'
        ]
        with torch.no_grad():
            eos = [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]
            for s in splitter:
                eos.extend(self.tokenizer.encode(s, add_special_tokens=False))
            generation_config = GenerationConfig(
                max_new_tokens=100,
                temperature=min(0.3 + 0.07 * rd, 1.2),
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos,
            )
            queries = data["input_ids"].to(self.model.device)
            ground_truths = data["ground_truth"]
            messages = data["_messages"]
            
            os.mkdirs('mcts', exist_ok=True)
            with open(f'mcts/round_{rd}.jsonl', 'w') as f:
                with unwrap_model_for_generation(self.model, trainer.accelerator) as unwrapped_model:
                    generated = []
                    for i, (q, m, g) in enumerate(zip(queries, messages, ground_truths)):
                        # TODO get a negative path?
                        gen = monte_carlo_tree_search(self.tokenizer.decode(q),
                                                    unwrapped_model,
                                                    self.tokenizer,
                                                    partial(orms.get(self.args.orm_type), ground_truths=[g]),
                                                    self.get_reward_by_model,
                                                    generation_config,
                                                    max_depth=6,
                                                    max_children=4,
                                                    success_factor=1.0,
                                                    decay_factor=0.5,
                                                    penalty_factor=0.2,
                                                    penalty_decay=0.5,
                                                    score_threshold=-5,
                                                    history=m,
                                                    )
                        if gen[1]:
                            _data = deepcopy(data)
                            messages = _data['_messages'][i]
                            assert messages[-1]['content'] is None
                            messages[-1]['content'] = gen[0]
                            encoded = self.template.encode(StdTemplateInputs.from_dict({'messages': messages}, tools_prompt='toolbench'))
                            encoded.pop('_messages', None)
                            generated.append(encoded)
                            f.write(json.dumps({'messages': messages}) + '\n')
            return generated

    def train(self, trainer):
        self._prepare_prm()
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        # trainer.train(trainer.args.resume_from_checkpoint)
        new_dataset = []
        for i in range(50):
            train_dataloader = trainer.get_train_dataloader()
            cnt = 0
            for batch in train_dataloader:
                cnt += 1
                new_data = self.rollout(batch, trainer, i)
                new_dataset.extend(new_data)
                if cnt > 30:
                    break

            trainer._origin_dataset = trainer.train_dataset
            trainer.train_dataset = new_dataset
            trainer.args.eval_strategy = 'no'
            trainer.train(trainer.args.resume_from_checkpoint)
            trainer.train_dataset = trainer._origin_dataset
            trainer.train_dataset = trainer.train_dataset.shuffle()

        return self._save_trainer_state(trainer)


def rlft_main(args: Union[List[str], RLFTArguments, None] = None):
    return SwiftRLFT(args).main()
