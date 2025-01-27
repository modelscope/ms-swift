import os
from typing import Literal
from swift.llm import load_image
import re
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def draw_bbox(image, response):
    matchs = re.findall(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>', response)
    

def infer_grounding():
    from swift.llm import (PtEngine, RequestConfig, AdapterRequest, get_template, BaseArguments, InferRequest,
                           safe_snapshot_download, get_model_tokenizer)
    from swift.tuners import Swift
    image = load_image('http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png')
    infer_request = InferRequest(messages=[{'role': 'user', 'content': 'Task: Object Detection'}], 
                                 images=[image])

    request_config = RequestConfig(max_tokens=512, temperature=0)
    adapter_path = safe_snapshot_download('/mnt/nas2/huangjintao.hjt/work/llmscope/output/v92-20250126-173609/checkpoint-1237')
    args = BaseArguments.from_pretrained(adapter_path)

    engine = PtEngine(args.model, adapters=[adapter_path])
    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    print(f'lora-response: {response}')

    new_image = draw_bbox(image, response)
    new_image.save('animal_bbox.png')


if __name__ == '__main__':
    infer_grounding()
