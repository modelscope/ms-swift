# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import os.path
from typing import Optional

import torch
from peft import (AdaLoraConfig, IA3Config, LoftQConfig, LoHaConfig,
                  LoKrConfig, LoraConfig, OFTConfig, PeftConfig, PeftModel,
                  PeftModelForCausalLM, PeftModelForSeq2SeqLM,
                  PeftModelForSequenceClassification,
                  PeftModelForTokenClassification, PrefixTuningConfig,
                  PromptEncoderConfig, PromptLearningConfig,
                  PromptTuningConfig, get_peft_config, get_peft_model,
                  get_peft_model_state_dict)

from swift.hub.snapshot_download import snapshot_download


def adalora_forward(self, *args, **kwargs):
    outputs = self.model.forward(*args, **kwargs)

    if (getattr(outputs, 'loss', None) is not None) and isinstance(
            outputs.loss, torch.Tensor):
        # Calculate the orthogonal regularization
        orth_reg_weight = self.peft_config[
            self.trainable_adapter_name].orth_reg_weight

        if orth_reg_weight <= 0:
            raise ValueError('orth_reg_weight should be greater than 0. ')

        regu_loss = 0
        num_param = 0
        for n, p in self.model.named_parameters():
            if ('lora_A' in n
                    or 'lora_B' in n) and self.trainable_adapter_name in n:
                para_cov = p @ p.T if 'lora_A' in n else p.T @ p
                I = torch.eye(
                    *para_cov.size(),
                    out=torch.empty_like(para_cov))  # noqa: E741
                I.requires_grad = False
                num_param += 1
                if isinstance(regu_loss, torch.Tensor):
                    regu_loss = regu_loss.to(para_cov.device)
                regu_loss += torch.norm(para_cov - I, p='fro')
        if num_param > 0:
            regu_loss = regu_loss / num_param
        else:
            regu_loss = 0
        if isinstance(regu_loss, torch.Tensor) and isinstance(
                outputs.loss, torch.Tensor):
            regu_loss = regu_loss.to(outputs.loss.device)
        outputs.loss += orth_reg_weight * regu_loss
    return outputs


def adalora_mask_to_budget(self, model, budget):
    value_ipt = {}
    vector_ipt = {}
    triplet_ipt = {}
    # Get the importance score for A, E, B
    for n, p in model.named_parameters():
        if f'lora_A.{self.adapter_name}' in n:
            entry_ipt = self._element_score(n)
            comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
            name_m = n.replace('lora_A', '%s')
            if name_m not in vector_ipt:
                vector_ipt[name_m] = [comb_ipt]
            else:
                vector_ipt[name_m].append(comb_ipt)
        if f'lora_B.{self.adapter_name}' in n:
            entry_ipt = self._element_score(n)
            comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
            name_m = n.replace('lora_B', '%s')
            if name_m not in vector_ipt:
                vector_ipt[name_m] = [comb_ipt]
            else:
                vector_ipt[name_m].append(comb_ipt)
        if f'lora_E.{self.adapter_name}' in n:
            entry_ipt = self._element_score(n)
            name_m = n.replace('lora_E', '%s')
            value_ipt[name_m] = entry_ipt

    all_score = []
    # Calculate the score for each triplet
    for name_m in vector_ipt:
        ipt_E = value_ipt[name_m]
        ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
        sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
        name_E = name_m % 'lora_E'
        triplet_ipt[name_E] = sum_ipt.view(-1, 1)
        sum_ipt = sum_ipt.view(-1)
        if all_score:
            sum_ipt = sum_ipt.to(all_score[0].device)
        all_score.append(sum_ipt)

    # Get the threshold by ranking ipt
    mask_threshold = torch.kthvalue(
        torch.cat(all_score),
        k=self.init_bgt - budget,
    )[0].item()

    rank_pattern = {}
    # Mask the unimportant triplets
    with torch.no_grad():
        for n, p in model.named_parameters():
            if f'lora_E.{self.adapter_name}' in n:
                p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                rank_pattern[n] = (
                    ~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
    return rank_pattern


def patch_adalora():
    from peft.tuners.adalora import AdaLoraModel, RankAllocator
    AdaLoraModel.forward = adalora_forward
    RankAllocator.mask_to_budget = adalora_mask_to_budget


def get_wrapped_class(module_class):
    """Get a custom wrapper class for peft classes to download the models from the ModelScope hub

    Args:
        module_class: The actual module class

    Returns:
        The wrapper
    """

    class PeftWrapper(module_class):

        @classmethod
        def from_pretrained(cls,
                            model,
                            model_id,
                            *args,
                            revision: Optional[str] = None,
                            **kwargs):
            if not os.path.exists(model_id):
                model_id = snapshot_download(model_id, revision=revision)
            return module_class.from_pretrained(model, model_id, *args,
                                                **kwargs)

    return PeftWrapper


def wrap_module(module):
    if not hasattr(module, 'from_pretrained'):
        return module

    return get_wrapped_class(module)


patch_adalora()
PeftModel = wrap_module(PeftModel)
PeftConfig = wrap_module(PeftConfig)
PeftModelForSeq2SeqLM = wrap_module(PeftModelForSeq2SeqLM)
PeftModelForSequenceClassification = wrap_module(
    PeftModelForSequenceClassification)
PeftModelForTokenClassification = wrap_module(PeftModelForTokenClassification)
PeftModelForCausalLM = wrap_module(PeftModelForCausalLM)
PromptEncoderConfig = wrap_module(PromptEncoderConfig)
PromptTuningConfig = wrap_module(PromptTuningConfig)
PrefixTuningConfig = wrap_module(PrefixTuningConfig)
PromptLearningConfig = wrap_module(PromptLearningConfig)
LoraConfig = wrap_module(LoraConfig)
AdaLoraConfig = wrap_module(AdaLoraConfig)
IA3Config = wrap_module(IA3Config)
LoHaConfig = wrap_module(LoHaConfig)
LoKrConfig = wrap_module(LoKrConfig)
LoftQConfig = wrap_module(LoftQConfig)
OFTConfig = wrap_module(OFTConfig)
get_peft_config = get_peft_config
get_peft_model_state_dict = get_peft_model_state_dict
get_peft_model = get_peft_model
