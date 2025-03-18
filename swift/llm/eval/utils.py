from typing import Any, Dict, List, Union
from evalscope.models.custom import CustomModel
from dataclasses import asdict
from transformers import PreTrainedModel
import torch.nn as nn

from ..template import InferRequest
from ..infer import PtEngine, RequestConfig

class EvalModel(CustomModel):
    
    def __init__(self, model:Union[PreTrainedModel, nn.Module], template, max_batch_size, model_name: str, **kwargs) -> None:
        super().__init__(config={'model_id': model_name}, **kwargs)
        self.model_name = model_name
        self.model = model
        self.template = template
        self.engine = PtEngine.from_model_template(model, template, max_batch_size=max_batch_size)
        
    def predict(self, prompts: List[dict], **kwargs) -> List[Dict[str, Any]]:
        # use origin inputs
        infer_requests = self.prepare_inputs(kwargs.get('origin_inputs', prompts))
        
        infer_cfg = kwargs['infer_cfg'].copy()
        generation_config = RequestConfig(**infer_cfg)

        response = self.engine.infer(
            infer_requests=infer_requests,
            request_config=generation_config,
            use_tqdm=False
        )
        dict_response = [asdict(item) for item in response]
        return dict_response
    
    def prepare_inputs(self, prompts: List[dict]):
        infer_requests = []
        for input_item in prompts:
            
            data: list = input_item['data']
            if isinstance(data[0], tuple):  # for truthful_qa and hellaswag
                query = '\n'.join(''.join(item) for item in data)
                system_prompt = input_item.get('system_prompt', None)
            else:
                query = data[0]
                system_prompt = input_item.get('system_prompt', None)
                
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': query})
            infer_requests.append(InferRequest(messages=messages))
        return infer_requests