import os
from typing import List

from swift.llm import BaseArguments, InferRequest, PtEngine, get_template

os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

infer_request = InferRequest(
    messages=[{
        'role':
        'user',
        'content':
        "多标签分类，类别包括：['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', "
        "'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', "
        "'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']"
    }],
    images=['xxx.jpg'])
adapter_path = 'output/vx-xxx/checkpoint-xxx'
args = BaseArguments.from_pretrained(adapter_path)

engine = PtEngine(
    args.model,
    adapters=[adapter_path],
    task_type='seq_cls',
    num_labels=args.num_labels,
    problem_type=args.problem_type)
template = get_template(args.template, engine.processor, args.system, use_chat_template=args.use_chat_template)
engine.default_template = template

resp_list = engine.infer([infer_request])
response: List[int] = resp_list[0].choices[0].message.content
print(f'response: {response}')
