import os
import re
from typing import Literal

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MAX_PIXELS'] = '1003520'


def draw_bbox_qwen2_vl(image, response, norm_bbox: Literal['norm1000', 'none']):
    matches = re.findall(
        r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>',
        response)
    ref = []
    bbox = []
    for match_ in matches:
        ref.append(match_[0])
        bbox.append(list(match_[1:]))
    draw_bbox(image, ref, bbox, norm_bbox=norm_bbox)


def infer_grounding():
    # use transformers==4.51.3
    from swift.llm import PtEngine, RequestConfig, BaseArguments, InferRequest, safe_snapshot_download
    output_path = 'bbox.png'
    image = load_image('http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png')
    infer_request = InferRequest(messages=[{'role': 'user', 'content': 'Task: Object Detection'}], images=[image])

    request_config = RequestConfig(max_tokens=512, temperature=0, return_details=True)
    adapter_path = safe_snapshot_download('swift/test_grounding')
    args = BaseArguments.from_pretrained(adapter_path)

    engine = PtEngine(args.model, adapters=[adapter_path])
    resp_list = engine.infer([infer_request], request_config)
    image = image.resize(resp_list[0].images_size[0])
    response = resp_list[0].choices[0].message.content
    print(f'lora-response: {response}')

    draw_bbox_qwen2_vl(image, response, norm_bbox=args.norm_bbox)
    print(f'output_path: {output_path}')
    image.save(output_path)


if __name__ == '__main__':
    from swift.llm import draw_bbox, load_image
    infer_grounding()
