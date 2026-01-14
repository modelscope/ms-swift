import os

from swift import BaseArguments, InferRequest, TransformersEngine, get_template

os.environ['MAX_PIXELS'] = '1003520'

infer_request = InferRequest(
    messages=[{
        'role': 'user',
        'content': 'Task: Classify household waste.'
    }], images=['xxx.jpg'])
adapter_path = 'output/vx-xxx/checkpoint-xxx'
args = BaseArguments.from_pretrained(adapter_path)

engine = TransformersEngine(args.model, adapters=[adapter_path], task_type='seq_cls', num_labels=args.num_labels)
template = get_template(
    engine.processor, args.system, template_type=args.template, use_chat_template=args.use_chat_template)
engine.template = template

resp_list = engine.infer([infer_request])
response: int = resp_list[0].choices[0].message.content
print(f'response: {response}')
