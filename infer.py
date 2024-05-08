import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

model_type = ModelType.deepseek_vl_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')
model_path="/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/model/deepseek-vl-7b-chat"
local_repo_path="/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/model/DeepSeek-VL"
dtype="fp16"
model, tokenizer = get_model_tokenizer(model_type, torch.float32,
                                       model_kwargs={'device_map': 'auto'}, 
                                       model_id_or_path=model_path,
                                       local_repo_path=local_repo_path,
                                       dtype=dtype
                                       )
model.generation_config.max_new_tokens = 1000
template = get_template(template_type, tokenizer)
seed_everything(42)

# query = '<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？'
query = '<img>/cephfs/group/teg-openrecom-openrc/starxhong/swift/road.png</img>距离各城市多远？'
query = '人设：你是一个「广告」的「SPU」标注人员。\n任务：现在提供给你某个「广告」的内容，同时提供这个「广告」对应的「SPU」。\n需要输出：「广告」表达出来的商品主体和「SPU」的匹配程度；匹配得分：0-100分；以及理由，再从理由反思输出是否正确，最后重新输出得分和匹配程度；\n规则：匹配得分越高表示匹配程度越高；\n输出格式：最终得分:{}，最终匹配程度:「高」或者「低」，理由:{}。\n需要注意：匹配程度只能输出「高」或者「低」，不可以输出「中等」或者「一般」等。\n一个「广告」可能由「广告标题」、「广告描述」、「广告OCR」、「广告视频OCR」、「广告主所属行业」、「广告图片」等几部分组成。如果根据广告标题或广告描述已经可以判断广告的商品主体，则以此为判断依据。\n「广告」：「广告描述>」:欧莱雅·爆款护肤套装爆款不止买一送一;「广告OCR」:LOREALPARiS巴黎欧莱雅唯品会LOREALLOREALPAHIB年度特卖大会唯品会LOREALPARiS巴黎欧莱雅爆款;「广告视频OCR」:源头抗初老2充盈肌底弹力深澈补水淡化细纹玻色因淡纹3熬夜不垮脸经典抗皱礼盒满载爱意奢润抗老4年轻紧·亮·弹;「广告主所属行业」:护肤彩妆;\n「广告图片」：<img>/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/images/id_13230216566_image_0.jpg</img><img>/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/frames/id_13230216566_frame_0.jpg</img><img>/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/frames/id_13230216566_frame_1.jpg</img><img>/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/frames/id_13230216566_frame_2.jpg</img>;\n「SPU」：欧莱雅复颜玻尿酸水光充盈导入水乳女士面霜'
query = '人设：你是一个「广告」的「SPU」标注人员。\n任务：现在提供给你某个「广告」的内容，同时提供这个「广告」对应的「SPU」。\n需要输出：「广告」表达出来的商品主体和「SPU」的匹配程度；匹配得分：0-100分；以及理由，再从理由反思输出是否正确，最后重新输出得分和匹配程度；\n规则：匹配得分越高表示匹配程度越高；\n输出格式：最终得分:{}，最终匹配程度:「高」或者「低」，理由:{}。\n需要注意：匹配程度只能输出「高」或者「低」，不可以输出「中等」或者「一般」等。\n一个「广告」可能由「广告标题」、「广告描述」、「广告OCR」、「广告视频OCR」、「广告主所属行业」、「广告图片」等几部分组成。如果根据广告标题或广告描述已经可以判断广告的商品主体，则以此为判断依据。\n「广告」：「广告描述>」:欧莱雅·爆款护肤套装爆款不止买一送一;「广告OCR」:LOREALPARiS巴黎欧莱雅唯品会LOREALLOREALPAHIB年度特卖大会唯品会LOREALPARiS巴黎欧莱雅爆款;「广告视频OCR」:源头抗初老2充盈肌底弹力深澈补水淡化细纹玻色因淡纹3熬夜不垮脸经典抗皱礼盒满载爱意奢润抗老4年轻紧·亮·弹;「广告主所属行业」:护肤彩妆;\n「广告图片」：<img>/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/images/id_13230216566_image_0.jpg</img>;\n「SPU」：欧莱雅复颜玻尿酸水光充盈导入水乳女士面霜'
query = '请描述下面广告图片的内容<img>/cephfs/group/teg-openrecom-openrc/starxhong/feature_eval/images/id_13230216566_image_0.jpg</img>;'

response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = '距离最远的城市是哪？'
gen = inference_stream(model, template, query, history)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f'history: {history}')
"""
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？
response: 这个标志显示了从当前位置到以下城市的距离：

- 马塔（Mata）：14公里
- 阳江（Yangjiang）：62公里
- 广州（Guangzhou）：293公里

这些信息是根据图片中的标志提供的。
query: 距离最远的城市是哪？
response: 根据图片中的标志，距离最远的城市是广州，距离为293公里。
history: [['<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？', '这个标志显示了从当前位置到以下城市的距离：\n\n- 马塔（Mata）：14公里\n- 阳江（Yangjiang）：62公里\n- 广州（Guangzhou）：293公里\n\n这些信息是根据图片中的标志提供的。'], ['距离最远的城市是哪？', '根据图片中的标志，距离最远的城市是广州，距离为293公里。']]
"""
