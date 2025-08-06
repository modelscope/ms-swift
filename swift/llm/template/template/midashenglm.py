# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Word
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta

from typing import Any, Dict, List, Optional, Union

@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'

class MiDashengLMTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        audios = inputs.audios
        audio = audios[index]
        assert isinstance(audio, str)
        
        return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>']

    def _tokenize(self, context, **tokenizer_kwargs):
        return super()._tokenize(context, **tokenizer_kwargs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        if not inputs.audios:
            return super()._encode(inputs)
        
        audio_data = []
        for audio_path in inputs.audios:
            try:
                import librosa
                import numpy as np
                sampling_rate = getattr(self.processor, 'sampling_rate', 16000)
                audio_array, _ = librosa.load(audio_path, sr=sampling_rate)
                audio_data.append(audio_array)
            except Exception as e:
                print(f"Failed to load audio {audio_path}: {e}")
                audio_data.append(np.zeros(1600)) 
        
        if not audio_data:
            return super()._encode(inputs)
        
        original_encoded = super()._encode(inputs)

        conversation_text = ""
        if hasattr(inputs, 'conversations') and inputs.conversations:
            for conv in inputs.conversations:
                role = conv.get('from', conv.get('role', ''))
                content = conv.get('value', conv.get('content', ''))
                if isinstance(content, list):
                    text_parts = []
                    audio_index = 0
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'audio' or 'audio' in item:
                                audio_token = getattr(self.processor, 'audio_token', '<|AUDIO|>')
                                text_parts.append(f'<|audio_bos|>{audio_token}<|audio_eos|>')
                                audio_index += 1
                            elif 'text' in item:
                                text_parts.append(item['text'])
                        else:
                            text_parts.append(str(item))
                    content = ''.join(text_parts)
                
                conversation_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            conversation_text += "<|im_start|>assistant\n"
        else:
            decoded_text = self.tokenizer.decode(original_encoded['input_ids'], skip_special_tokens=False)
            audio_token = getattr(self.processor, 'audio_token', '<|AUDIO|>')
            conversation_text = decoded_text.replace('<|AUDIO|>', audio_token)
        
        try:
            max_length = getattr(self.tokenizer, 'model_max_length', None)
            if max_length is None or max_length > 1000000:  
                max_length = 2048  
            
            processor_kwargs = {
                'text': [conversation_text],
                'audio': audio_data,
                'return_tensors': "pt",
                'padding': True,
            }
            
            if max_length and max_length < 1000000:
                processor_kwargs['truncation'] = True
                processor_kwargs['max_length'] = max_length
            
            processor_result = self.processor(**processor_kwargs)

            result = {}
            for key, value in processor_result.items():
                if hasattr(value, 'squeeze'):
                    squeezed = value.squeeze(0)
                    if hasattr(squeezed, 'tolist'):
                        result[key] = squeezed.tolist()
                    else:
                        result[key] = squeezed
                else:
                    result[key] = value
            
            if 'input_ids' in result:
                input_ids = result['input_ids']
                labels = input_ids.copy()

                audio_token_id = getattr(self.processor, 'audio_token_id', None)
                if audio_token_id is not None:
                    labels = [-100 if token_id == audio_token_id else token_id for token_id in labels]

                result['labels'] = labels
            
            return result
            
        except Exception as e:
            print(f"Processor failed: {e}")

            return original_encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:

        res = super()._data_collator(batch, padding_to=padding_to)

        if batch and batch[0].get('input_features') is not None:
            values = [b.get('input_features') for b in batch if b.get('input_features') is not None]
            if values:
                import torch
                if isinstance(values[0], torch.Tensor):
                    res['input_features'] = torch.stack(values)
                else:
                    res['input_features'] = torch.tensor(values)
        
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.midashenglm, template_cls=MiDashengLMTemplate))