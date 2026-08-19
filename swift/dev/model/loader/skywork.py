# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the Skywork families.

Ported from ``swift/model/models/skywork.py``. The plain ``skywork`` LLM plus the two reward
families. A reward model is a seq_cls head with ``num_labels=1`` -- declared via ``is_reward=True``
(mirrors legacy, where ``is_reward`` drives the num_labels=1 default and reward load handling).
"""
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class SkyworkLoader(ModelLoader):

    model_type = 'skywork'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['SkyworkForCausalLM']
    template = 'skywork'
    models = [
        ('skywork/Skywork-13B-base', 'skywork/Skywork-13B-base'),
        'skywork/Skywork-13B-chat',
    ]

    def process_tokenizer(self, processor):
        for token in ('[USER]', '[BOT]', '[SEP]'):
            processor.add_tokens(token)
        return processor


@register_model
class Llama3_2RewardLoader(ModelLoader):

    model_type = 'llama3_2_reward'
    model_cls = 'transformers:AutoModelForSequenceClassification'
    architectures = ['LlamaForSequenceClassification']
    template = 'llama3_2'
    is_reward = True
    requires = ['transformers>=4.43']
    models = [
        ('AI-ModelScope/Skywork-Reward-Llama-3.1-8B', 'Skywork/Skywork-Reward-Llama-3.1-8B'),
        ('AI-ModelScope/Skywork-Reward-Llama-3.1-8B-v0.2', 'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2'),
        ('AI-ModelScope/GRM_Llama3.1_8B_rewardmodel-ft', 'Ray2333/GRM_Llama3.1_8B_rewardmodel-ft'),
        ('AI-ModelScope/GRM-llama3.2-3B-rewardmodel-ft', 'Ray2333/GRM-llama3.2-3B-rewardmodel-ft'),
    ]


@register_model
class GemmaRewardLoader(ModelLoader):

    model_type = 'gemma_reward'
    model_cls = 'transformers:AutoModelForSequenceClassification'
    architectures = ['Gemma2ForSequenceClassification']
    template = 'gemma'
    is_reward = True
    requires = ['transformers>=4.42']
    models = [
        ('AI-ModelScope/Skywork-Reward-Gemma-2-27B', 'Skywork/Skywork-Reward-Gemma-2-27B'),
        ('AI-ModelScope/Skywork-Reward-Gemma-2-27B-v0.2', 'Skywork/Skywork-Reward-Gemma-2-27B-v0.2'),
    ]
