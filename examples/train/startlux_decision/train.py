# Copyright (c) ModelScope Contributors. All rights reserved.
"""Train StartLux dense/MoE decision adapters with the SWIFT trainer."""
import argparse
import json
import os
import torch
from datasets import Dataset
from decision_training import (audit_splits, audit_trainable, candidate_loss, collate, copy_metadata, encode_records,
                               install_head, lora_targets)
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig
from transformers.trainer_callback import TrainerCallback

from swift.model import get_model_processor
from swift.template import get_template
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments


class DecisionTrainer(Seq2SeqTrainer):

    def _get_data_collator(self, args, template):
        from functools import partial
        return partial(collate, pad_token_id=template.tokenizer.pad_token_id)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        targets = inputs.pop('labels')
        inputs.pop('compute_loss_func', None)
        outputs = model(**inputs)
        loss = candidate_loss(outputs.logits, inputs['attention_mask'], targets).mean()
        return (loss, outputs) if return_outputs else loss


class BeforeValidation(TrainerCallback):
    """Validate after Trainer has prepared the FSDP model and optimizer together."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.metrics = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.metrics = self.trainer.evaluate()
        return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--save-steps', type=int, default=50)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora-scope', choices=['attention', 'language-linear'], default='attention')
    parser.add_argument('--resume')
    parser.add_argument('--fsdp', action='store_true')
    args = parser.parse_args()
    if Path(args.output).exists() and not args.resume:
        raise FileExistsError('Use a fresh output directory')
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(args.model)
    model_type = 'qwen3_5_moe' if 'Moe' in config.architectures[0] else 'qwen3_5'
    model, processor = get_model_processor(
        args.model,
        model_type=model_type,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cpu'} if args.fsdp else {'': int(os.environ.get('LOCAL_RANK', '0'))},
        attn_impl='sdpa')
    tokenizer = getattr(processor, 'tokenizer', processor)
    install_head(model, args.model, tokenizer)
    train, train_ids = encode_records(args.train, tokenizer)
    dev, dev_ids = encode_records(args.dev, tokenizer)
    audit_splits(train_ids, dev_ids)
    template = get_template(processor, template_type='qwen3_5', padding_free=False, padding_side='right')
    template.model = model
    model = get_peft_model(
        model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=lora_targets(model, args.lora_scope)))
    audit = audit_trainable(model)
    extra = {}
    if args.fsdp:
        extra = {
            'fsdp': 'full_shard auto_wrap',
            'fsdp_config': {
                'fsdp_version':
                2,
                'transformer_layer_cls_to_wrap':
                sorted({type(m).__name__
                        for m in model.modules() if type(m).__name__.endswith('DecoderLayer')}),
                'activation_checkpointing':
                True,
                'use_orig_params':
                True
            }
        }
    training = Seq2SeqTrainingArguments(
        output_dir=args.output,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=not args.fsdp,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        use_logits_to_keep=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        save_steps=min(args.save_steps, args.steps),
        save_total_limit=2,
        logging_steps=1,
        eval_strategy='no',
        report_to=[],
        seed=42,
        data_seed=42,
        check_model=False,
        **extra)
    trainer = DecisionTrainer(
        model=model,
        args=training,
        template=template,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(dev))
    trainer.model_accepts_loss_kwargs = False
    baseline = BeforeValidation(trainer)
    trainer.add_callback(baseline)
    trainer.train(resume_from_checkpoint=args.resume)
    after = trainer.evaluate()
    trainer.save_model(str(Path(args.output) / 'adapter'))
    if trainer.is_world_process_zero():
        copy_metadata(args.model, Path(args.output) / 'adapter')
        Path(args.output, 'validation.json').write_text(
            json.dumps(
                {
                    'before': baseline.metrics,
                    'after': after,
                    'train_questions': len(train),
                    'dev_questions': len(dev),
                    'steps': trainer.state.global_step,
                    'training_args': vars(args),
                    'trainable': audit,
                    'scope': 'held-out validation loss; test accuracy must be evaluated separately'
                },
                indent=2))


if __name__ == '__main__':
    main()
