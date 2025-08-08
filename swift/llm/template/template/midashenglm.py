# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union
from swift.utils import get_env_args, is_deepspeed_enabled

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Word
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta
from ..vision_utils import load_audio, load_batch


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    


class MiDashengLMTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        if not self.use_chat_template:
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        else:
            return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>']


    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        if not inputs.audios:
            return super()._encode(inputs)
        
        base_encoded = super()._encode(inputs)
        
        messages = []
        
        system = self._get_system(inputs)
        if system:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system}]
            })
        
        audio_idx = 0
        for message in inputs.messages:
            role = message['role']
            content = message['content']
            
            if content is None:
                continue
                
            message_content = []
            
            if isinstance(content, str):
                text_parts = content.split('<audio>')
                for i, text_part in enumerate(text_parts):
                    if text_part:
                        message_content.append({"type": "text", "text": text_part})
                    
                    if i < len(text_parts) - 1 and audio_idx < len(inputs.audios):
                        audio_path = inputs.audios[audio_idx]
                        if isinstance(audio_path, str):
                            message_content.append({"type": "audio", "path": audio_path})
                        else:
                            message_content.append({"type": "audio", "audio": audio_path})
                        audio_idx += 1
            else:
                message_content.append({"type": "text", "text": str(content)})
            
            if not message_content:
                message_content = [{"type": "text", "text": ""}]
                
            messages.append({
                "role": role,
                "content": message_content
            })
        
        add_generation_prompt = True
        if messages and messages[-1]['role'] == 'assistant':
            add_generation_prompt = False
        elif self.is_training:
            add_generation_prompt = False
        
        try:
            model_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                add_special_tokens=True,
                return_dict=True,
            )

        except Exception as e:
            print(f"Processor apply_chat_template failed: {e}")
            return super()._encode(inputs)
        
        encoded = {}


        
        if 'input_ids' in model_inputs:
            input_ids = model_inputs['input_ids']
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() > 1:
                    input_ids = input_ids.squeeze(0)
                encoded['input_ids'] = input_ids.tolist()
            else:
                encoded['input_ids'] = input_ids
        
        if self.is_training and base_encoded.get('labels') is not None:
            base_input_ids = base_encoded['input_ids']
            base_labels = base_encoded['labels']
            new_input_ids = encoded['input_ids']
            
            labels = [-100] * len(new_input_ids)
            
            if messages and messages[-1]['role'] == 'assistant':
                assistant_start = len(new_input_ids) // 2
                labels[assistant_start:] = new_input_ids[assistant_start:]
            
            encoded['labels'] = labels
        else:
            encoded['labels'] = None
        
        if base_encoded.get('loss_scale') is not None and encoded['labels'] is not None:
            loss_scale = [0.0 if label == -100 else 1.0 for label in encoded['labels']]
            encoded['loss_scale'] = loss_scale
        else:
            encoded['loss_scale'] = None
        
        for key, value in model_inputs.items():
            if key not in ['input_ids'] and key not in encoded:
                encoded[key] = value
        
        if not self.is_training:
            for k in list(encoded.keys()):
                if k.endswith('labels') or k.endswith('loss_scale'):
                    encoded[k] = None
        
        
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        
        input_values = [b['input_values'] for b in batch if b.get('input_values') is not None]
        audio_lengths = [b['audio_length'] for b in batch if b.get('audio_length') is not None]
        
        if input_values:
            res['input_values'] = torch.concat(input_values)
            if audio_lengths:
                res['audio_length'] = torch.concat(audio_lengths)
                
        return res

    def get_generate_ids(self, generate_ids: Union[torch.Tensor, List[int]],
                         num_prompt_tokens: int) -> Union[torch.Tensor, List[int]]:

        return generate_ids


register_template(QwenTemplateMeta(MLLMTemplateType.midashenglm, template_cls=MiDashengLMTemplate))