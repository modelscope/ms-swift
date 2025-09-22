"""
模块功能
-------
本模块集中定义多模态（视觉/音频/视频）场景下的数据集预处理器与数据集注册逻辑，
涵盖图像描述（caption/VQA）、视频理解、OCR、语音识别、工具/代理类任务等。通过 `DatasetMeta` 与
各类 `Preprocessor`（继承自 `MessagesPreprocessor`/`ResponsePreprocessor`/`RowPreprocessor`）
将原始数据样本统一转换为训练所需的标准字段，如 `messages/query/response/images/audios/videos/tools`。

典型用法
-------
1. 导入本模块即完成对若干多模态数据集的注册；
2. 上层数据加载器按 `ms_dataset_id/hf_dataset_id/subsets/split` 查找并构建 `Dataset`；
3. 预处理器在 `prepare_dataset/preprocess` 中下载媒体、重写路径、抽样与清洗字段。

说明：本文件为每一行代码添加了中文注释与必要的文档注释，便于快速理解与维护。
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明
import ast  # 抽象语法树解析：安全解析字符串字面量
import os  # 路径与文件操作
from typing import Any, Dict, Optional  # 类型注解：通用、字典、可选

import numpy as np  # 数值工具，这里用于随机选择与索引
from datasets import Dataset as HfDataset  # HuggingFace 标准数据集类型
from datasets import IterableDataset as HfIterableDataset  # 可迭代数据集类型，用于流式处理
from tqdm import tqdm  # 进度条，迭代大数据集时显示进度

from swift.utils import get_hf_endpoint, use_hf_hub  # 工具：判断是否使用 HF Hub 与获取端点
from ..media import MediaResource  # 媒体资源下载/缓存工具
from ..preprocessor import GroundingMixin, MessagesPreprocessor, ResponsePreprocessor, RowPreprocessor  # 预处理基类与 Grounding 混入
from ..register import DatasetMeta, SubsetDataset, register_dataset  # 数据集元信息、子集描述与注册入口


class ShareGPT4oPreprocessor(MessagesPreprocessor):
    """
    类说明
    -----
    ShareGPT-4o 多模态数据集的预处理器：
    - 在 `prepare_dataset` 阶段下载并定位图像根目录；
    - 在 `preprocess` 阶段拼接相对路径为绝对路径，并确保文件存在；
    - 将单张图片路径封装为列表形式，统一下游消费接口。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理单条样本：标准化并重写图像路径。

        参数
        ----
        - row: 原始样本字典，预期包含 `images` 相对路径。

        返回
        ----
        - Optional[Dict[str, Any]]: 成功则返回标准化后的记录，失败（无图像或路径不存在）则返回 None。
        """
        row = super().preprocess(row)  # 先做通用字段标准化/映射
        image = row['images']  # 取出图像相对路径
        if not image:  # 若不存在图像
            return  # 直接丢弃样本
        image = os.path.join(self.prefix_path, image)  # 拼接成绝对路径
        if not os.path.exists(image):  # 若路径不存在
            return  # 丢弃样本
        row['images'] = [image]  # 规范为单元素列表
        return row  # 返回处理后的样本

    def prepare_dataset(self, dataset):
        """
        下载 ShareGPT-4o 所需的图像压缩包并设置前缀路径。
        """
        if not use_hf_hub():  # 根据环境选择下载源
            url = ('https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/ShareGPT-4o/repo?'
                   'Revision=master&FilePath=images.zip')  # MS 源地址
        else:
            url = f'{get_hf_endpoint()}/datasets/OpenGVLab/ShareGPT-4o/blob/main/images.zip'  # HF 源地址
        local_dir = MediaResource.download(url, 'sharegpt_4o_images')  # 下载并返回本地目录
        self.prefix_path = os.path.join(local_dir, 'mnt', 'petrelfs', 'wangwenhai', 'workspace_cef', '4o', 'image')  # 组装前缀
        return super().prepare_dataset(dataset)  # 继续父类准备流程


register_dataset(
    DatasetMeta(  # 注册 ShareGPT-4o 数据集
        ms_dataset_id='AI-ModelScope/ShareGPT-4o',  # MS 数据集 ID
        hf_dataset_id='OpenGVLab/ShareGPT-4o',  # HF 数据集 ID
        preprocess_func=ShareGPT4oPreprocessor(),  # 绑定预处理器
        subsets=['image_caption'],  # 使用 image_caption 子集
        split=['images'],  # 使用 images 切分名
        tags=['vqa', 'multi-modal'],  # 标签：视觉问答/多模态
    ))


class GPT4vDataset(ResponsePreprocessor):
    """
    类说明
    -----
    GPT-4V 图像描述数据：统一设置查询为“图像的标题是什么？”，然后调用父类进行标准化。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        设置 query 并交由父类处理。

        参数
        ----
        - row: 原始记录，至少包含 `images/caption` 等字段（通过 columns 映射）。

        返回
        ----
        - Dict[str, Any]: 标准化样本。
        """
        row['query'] = 'What is the caption of this image?'  # 统一图像描述问题
        return super().preprocess(row)  # 标准化


register_dataset(
    DatasetMeta(  # 注册 gpt4v-dataset
        ms_dataset_id='swift/gpt4v-dataset',  # MS ID
        hf_dataset_id='laion/gpt4v-dataset',  # HF ID
        preprocess_func=GPT4vDataset(columns={  # 列映射：链接->images，caption->response
            'link': 'images',
            'caption': 'response'
        }),
        subsets=['train'],
        split=['train'],  # 使用 train 划分
        tags=['en', 'caption', 'multi-modal', 'quality'],  # 英文/图像描述/多模态/高质量
        huge_dataset=True,  # 数据集较大
    ))

register_dataset(
    DatasetMeta(  # 注册 RLAIF-V 视觉偏好数据
        ms_dataset_id='swift/RLAIF-V-Dataset',  # MS ID
        hf_dataset_id='openbmb/RLAIF-V-Dataset',  # HF ID
        preprocess_func=ResponsePreprocessor(columns={  # 列映射：问题/优选/劣选
            'question': 'query',
            'chosen': 'response',
            'rejected': 'rejected_response'
        }),
        tags=['rlhf', 'dpo', 'multi-modal', 'en'],  # 标签：RLHF/DPO/多模态/英文
    ))


class GarbagePreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    垃圾分类图片数据：统一设置分类任务说明后进行标准化。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        设置 query 并委托父类处理。
        """
        row['query'] = 'Task: Classify household waste.'  # 分类任务指令
        return super().preprocess(row)  # 标准化


register_dataset(
    DatasetMeta(  # 注册垃圾分类数据集
        ms_dataset_id='tany0699/garbage265',  # MS ID
        preprocess_func=GarbagePreprocessor(columns={  # 列映射：类别->label，文件->images
            'category': 'label',
            'image:FILE': 'images'
        }),
        tags=['cls', '🔥', 'multi-modal'],  # 标签：分类/热门/多模态
    ))


class SA1BPairedCaptionPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    处理成对的图像-全局描述：随机挑选一个中文提示作为 query，`global_caption` 作为 response。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        构造两段式对话消息。
        """
        prompt = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']  # 候选中文提示
        response = row['global_caption']  # 全局描述
        query = np.random.choice(prompt)  # 随机选择提示
        return {
            'messages': [{
                'role': 'user',
                'content': query,
            }, {
                'role': 'assistant',
                'content': response,
            }]
        }


register_dataset(
    DatasetMeta(  # 注册 SA1B 成对描述
        ms_dataset_id='Tongyi-DataEngine/SA1B-Paired-Captions-Images',  # MS ID
        preprocess_func=SA1BPairedCaptionPreprocessor(columns={  # 列映射：开源 URL -> images
            'opensource_url': 'images',
        }),
        tags=['zh', 'multi-modal', 'vqa'],  # 标签：中文/多模态/VQA
    ))


class SA1BDenseCaptionPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    处理 SA1B 密集描述：从 `cap_seg` 中解析 `global_caption`，并随机挑选中文提示作为 query。
    """
    column_mapping = {
        'url': 'images',  # 将 url 列映射为 images
    }

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析密集描述并构造两段式消息。
        """
        prompt = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']  # 候选提示
        response = ast.literal_eval(row['cap_seg'])  # 安全解析字符串为字典
        response = response.get('global_caption')  # 提取全局描述
        query = np.random.choice(prompt)  # 随机提示
        return {
            'messages': [{
                'role': 'user',
                'content': query,
            }, {
                'role': 'assistant',
                'content': response,
            }]
        }


register_dataset(
    DatasetMeta(  # 注册 SA1B 密集描述
        ms_dataset_id='Tongyi-DataEngine/SA1B-Dense-Caption',  # MS ID
        preprocess_func=SA1BDenseCaptionPreprocessor(columns={  # 列映射：url -> images
            'url': 'images',
        }),
        tags=['zh', 'multi-modal', 'vqa'],  # 标签
        huge_dataset=True,  # 数据量较大
    ))


class COCO2014Preprocess(ResponsePreprocessor):
    """
    类说明
    -----
    COCO-2014 图像描述：
    - 去除 `caption` 中 `&&` 之后的噪声部分；
    - 统一查询为英文描述请求。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗 caption 并标准化样本。
        """
        caption = row['caption']  # 原始描述
        if '&&' in caption:  # 若包含分隔噪声
            caption = caption.split('&&')[0]  # 仅保留前半段
        row['query'] = 'please describe the image.'  # 统一查询指令
        row['response'] = caption  # 设置响应

        return super().preprocess(row)  # 标准化


register_dataset(
    DatasetMeta(  # 注册 COCO-2014 caption 数据
        ms_dataset_id='modelscope/coco_2014_caption',
        preprocess_func=COCO2014Preprocess(),  # 绑定预处理器
        subsets=[  # 子集映射到底层 split
            SubsetDataset('train', 'coco_2014_caption', ['train']),
            SubsetDataset('validation', 'coco_2014_caption', ['validation']),
        ],
        tags=['chat', 'multi-modal', 'vision', '🔥'],  # 标签
    ))


class MantisPreprocessor(MessagesPreprocessor):
    """
    类说明
    -----
    Mantis-Instruct 多模态数据预处理：
    - prepare 阶段按子集下载图片压缩包并缓存；
    - preprocess 阶段将相对路径转为本地绝对路径，过滤缺失文件。
    """

    def __init__(self, *, subset: str, columns: Optional[Dict[str, str]] = None) -> None:
        """
        初始化所需子集与列映射。
        """
        self.subset = subset  # 记录当前子集名
        super().__init__(columns=columns)  # 初始化基类

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        """
        下载当前子集的图片压缩包并设置本地目录。
        """
        if not use_hf_hub():  # 根据环境选择数据源
            url = (f'https://www.modelscope.cn/api/v1/datasets/swift/Mantis-Instruct/repo?Revision='
                   f'master&FilePath={self.subset}/train_images.zip')  # noqa  # MS 源
        else:
            url = (f'{get_hf_endpoint()}/datasets/TIGER-Lab/Mantis-Instruct/'
                   f'resolve/main/{self.subset}/train_images.zip')  # HF 源
        self.local_dir = MediaResource.download(url, f'mantis_{self.subset}')  # 下载并缓存
        return super().prepare_dataset(dataset)  # 继续父类流程

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将相对路径列表转为绝对路径并过滤缺失项。
        """
        images = [os.path.join(self.local_dir, p['path']) for p in row['images']]  # 拼接绝对路径
        if not all([os.path.exists(d) for d in images]):  # 若存在缺失文件
            images = []  # 清空以触发丢弃

        if not images:  # 无有效图片
            return  # 丢弃样本
        row['images'] = images  # 写回图片列表
        return super().preprocess(row)  # 标准化


mantis_subsets_name = [  # Mantis 子集枚举
    'birds-to-words', 'chartqa', 'coinstruct', 'contrastive_caption', 'docvqa', 'dreamsim', 'dvqa', 'iconqa',
    'imagecode', 'llava_665k_multi', 'lrv_multi', 'multi_vqa', 'nextqa', 'nlvr2', 'spot-the-diff', 'star',
    'visual_story_telling'
]

_mantis_subsets = []  # 收集构造好的子集描述
for subset in mantis_subsets_name:
    _subset = SubsetDataset(subset=subset, split=['train'], preprocess_func=MantisPreprocessor(subset=subset))  # 构造子集
    _mantis_subsets.append(_subset)  # 加入列表

register_dataset(
    DatasetMeta(  # 注册 Mantis-Instruct
        ms_dataset_id='swift/Mantis-Instruct',  # MS ID
        subsets=_mantis_subsets,  # 使用上面构造的全部子集
        tags=['chat', 'multi-modal', 'vision'],  # 标签
    ))


class LLaVADataPreprocessor(MessagesPreprocessor):
    """
    类说明
    -----
    为 LLaVA 数据修复图片绝对路径：
    - prepare_dataset 阶段下载各图片资源根目录；
    - preprocess 阶段根据相对路径归并到本地缓存路径，校验存在性。
    """

    def prepare_dataset(self, dataset):
        """
        下载或定位所需媒体根目录并缓存到 `self.all_folders`。
        """
        self.all_folders = {}  # 存放各媒体类型的本地根路径
        for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
            self.all_folders[media_type] = MediaResource.download(media_type)  # 下载或定位缓存
        return super().prepare_dataset(dataset)  # 继续父类流程

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将相对路径归并成本地绝对路径并校验存在性。
        """
        if not row['images']:  # 无图片直接跳过
            return
        row = super().preprocess(row)  # 标准化字段
        images = [p['path'] for p in row['images']]  # 提取相对路径列表
        new_images = []  # 存放修复后的绝对路径
        for image in images:  # 针对不同前缀选择对应根目录
            if 'coco/' in image:
                image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
            elif 'gqa/' in image:
                image = os.path.join(self.all_folders['gqa'], image.replace('gqa/', ''))
            elif 'ocr_vqa/' in image:
                image = os.path.join(self.all_folders['ocr_vqa'], image)
            elif 'textvqa/' in image:
                image = os.path.join(self.all_folders['textvqa'], image.replace('textvqa/', ''))
            elif 'VG_100K/' in image:
                image = os.path.join(self.all_folders['VG_100K'], image.replace('vg/', ''))
            elif 'VG_100K_2/' in image:
                image = os.path.join(self.all_folders['VG_100K_2'], image.replace('vg/', ''))
            new_images.append(image)  # 记录修复路径
        if all(os.path.exists(image) for image in new_images):  # 确保全部存在
            row['images'] = new_images  # 写回修复后的路径
        else:
            return {'images': None}  # 任何缺失则标记为无图像，供上游过滤
        return row  # 返回


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/llava-data',
        hf_dataset_id='TIGER-Lab/llava-data',
        subsets=['llava_instruct'],
        preprocess_func=LLaVADataPreprocessor(),
        tags=['sft', 'multi-modal', 'quality'],
    ))


class PixelProsePreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        caption_prompt = [
            'Give the description of this image.', 'Describe this picture', 'What is the proper title of this image?'
        ]
        vlm_caption = row['vlm_caption']
        if vlm_caption.startswith('This image displays:'):
            vlm_caption = vlm_caption[len('This image displays:'):].strip()
        return {
            'messages': [{
                'role': 'user',
                'content': np.random.choice(caption_prompt)
            }, {
                'role': 'assistant',
                'content': vlm_caption
            }],
            'images': row['url']
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/pixelprose',
        hf_dataset_id='tomg-group-umd/pixelprose',
        preprocess_func=PixelProsePreprocessor(),
        split=['train', 'cc12m', 'commonpool', 'redcaps'],
        tags=['caption', 'multi-modal', 'vision'],
        huge_dataset=True,
    ))


class AIShell1Preprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    AIShell-1 语音识别（ASR）数据：统一设置 query 为“语音转文本”，并去掉文本中的空格。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        设置统一查询并清理标签文本空格。
        """
        row['query'] = '语音转文本'
        row['response'] = row['Text:LABEL'].replace(' ', '')  # 去除空格
        return super().preprocess(row)  # 标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='speech_asr/speech_asr_aishell1_trainsets',
        subsets=[
            SubsetDataset('train', split=['train']),
            SubsetDataset('validation', split=['validation']),
            SubsetDataset('test', split=['test']),
        ],
        preprocess_func=AIShell1Preprocessor(columns={'Audio:FILE': 'audios'}),
        tags=['chat', 'multi-modal', 'audio'],
    ))


class EmoSchemaPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    EgoSchema 视频多选题：
    - prepare_dataset 阶段下载多个分片压缩包并记录可用 mp4 文件集合；
    - preprocess 阶段将多选项拼接进 query，转换响应为 A-E 选项字母，并绑定本地视频路径。
    """

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        """
        下载 5 个视频分片并收集本地 mp4 名称集合。
        """
        for i in range(1, 6):  # 5 个分片
            if not use_hf_hub():
                url = f'https://modelscope.cn/datasets/AI-ModelScope/egoschema/resolve/master/videos_chunked_0{i}.zip'
            else:
                url = f'{get_hf_endpoint()}/datasets/lmms-lab/egoschema/resolve/main/videos_chunked_0{i}.zip'
            local_dir = MediaResource.download(url, 'egoschema')  # 下载

        self.local_dir = os.path.join(local_dir, 'videos')  # 视频目录
        self.mp4_set = [file[:-4] for file in os.listdir(self.local_dir) if file.endswith('mp4')]  # 收集可用视频名
        return super().prepare_dataset(dataset)  # 父类

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        绑定本地视频路径并将选项数字映射为字母。
        """
        if row['video_idx'] not in self.mp4_set:  # 视频不存在则跳过
            return None
        transfer_to_option = {  # 数字到字母选项映射
            '0': 'A',
            '1': 'B',
            '2': 'C',
            '3': 'D',
            '4': 'E',
        }
        row = {
            'query': row['query'] + '\n' + '\n'.join(row['option']),  # 拼接多选项到查询
            'response': transfer_to_option[row['response']],  # 响应映射为字母
            'videos': [os.path.join(self.local_dir, f"{row['video_idx']}.mp4")],  # 本地视频路径
        }
        return super().preprocess(row)  # 标准化


class EmoSchemaClsPreprocessor(EmoSchemaPreprocessor):
    """
    类说明
    -----
    EgoSchema 分类版本：与 `EmoSchemaPreprocessor` 类似，但输出数值标签 `label`。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        绑定本地视频路径并输出整型标签。
        """
        if row['video_idx'] not in self.mp4_set:
            return None
        row = {
            'query': row['query'] + '\n' + '\n'.join(row['option']),  # 拼接选项
            'label': int(row['response']),  # 转换为整型标签
            'videos': [os.path.join(self.local_dir, f"{row['video_idx']}.mp4")],  # 本地视频路径
        }
        return ResponsePreprocessor.preprocess(self, row)  # 标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/egoschema',
        hf_dataset_id='lmms-lab/egoschema',
        subsets=[
            SubsetDataset('default', 'Subset', preprocess_func=EmoSchemaPreprocessor()),
            SubsetDataset('cls', 'Subset', preprocess_func=EmoSchemaClsPreprocessor())
        ],
        split=['test'],
        tags=['chat', 'multi-modal', 'video'],
    ))


def _generate_url_list(_url, _range):
    lst = []
    for i in range(1, (_range + 1)):
        lst.append(_url.replace('{}', str(i)))
    return lst


class LLaVAVideo178KPreprocessor(MessagesPreprocessor):
    """
    类说明
    -----
    LLaVA-Video-178K 视频数据：
    - 根据子集自动下载分片压缩包；
    - 在样本级将相对视频文件名转换为本地绝对路径。
    """

    def __init__(self, *, subset: str, columns: Optional[Dict[str, str]] = None) -> None:
        """
        记录子集名并初始化列映射。
        """
        self.subset = subset  # 当前子集
        super().__init__(columns=columns)  # 初始化基类

    url_prefix = 'https://www.modelscope.cn/datasets/lmms-lab/LLaVA-Video-178K/resolve/master/'  # 默认 MS 端点
    if use_hf_hub():  # 若使用 HF Hub，则切换端点
        url_prefix = f'{get_hf_endpoint()}/datasets/lmms-lab/LLaVA-Video-178K/resolve/main/'

    video_resources = {
        '0_30_s_academic_v0_1':
        _generate_url_list(
            url_prefix + '0_30_s_academic_v0_1/0_30_s_academic_v0_1_videos_{}.tar.gz',
            8,
        ),
        '0_30_s_youtube_v0_1':
        _generate_url_list(
            url_prefix + '0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_{}.tar.gz',
            19,
        ),
        '1_2_m_academic_v0_1':
        _generate_url_list(
            url_prefix + '1_2_m_academic_v0_1/1_2_m_academic_v0_1_videos_{}.tar.gz',
            14,
        ),
        '1_2_m_youtube_v0_1':
        _generate_url_list(
            url_prefix + '1_2_m_youtube_v0_1/1_2_m_youtube_v0_1_videos_{}.tar.gz',
            50,
        ),
        '2_3_m_academic_v0_1':
        _generate_url_list(
            url_prefix + '2_3_m_academic_v0_1/2_3_m_academic_v0_1_videos_{}.tar.gz',
            18,
        ),
        '2_3_m_youtube_v0_1':
        _generate_url_list(
            url_prefix + '2_3_m_youtube_v0_1/2_3_m_youtube_v0_1_videos_{}.tar.gz',
            98,
        ),
        '30_60_s_academic_v0_1':
        _generate_url_list(
            url_prefix + '30_60_s_academic_v0_1/30_60_s_academic_v0_1_videos_{}.tar.gz',
            10,
        ),
        '30_60_s_youtube_v0_1':
        _generate_url_list(
            url_prefix + '30_60_s_youtube_v0_1/30_60_s_youtube_v0_1_videos_{}.tar.gz',
            13,
        ),
    }

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        """
        下载选定子集的视频分片并设置本地目录。
        """
        urls = self.video_resources[self.subset]  # 取出该子集所有分片 URL 列表
        self.local_dir = MediaResource.download(urls, f'llava_video_178k_{self.subset}', file_type='sharded')  # 分片下载
        return super().prepare_dataset(dataset)  # 继续父类

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将视频文件名转换为本地绝对路径并校验。
        """
        file_path = os.path.join(self.local_dir, f"{row['videos']}")  # 拼接本地路径
        if not os.path.exists(file_path):  # 文件缺失则丢弃
            return None
        return super().preprocess({'messages': row['messages'], 'videos': file_path})  # 标准化


llava_video_subsets = []
for subset in [
        '0_30_s_academic_v0_1',
        '0_30_s_youtube_v0_1',
        '1_2_m_academic_v0_1',
        '1_2_m_youtube_v0_1',
        '2_3_m_academic_v0_1',
        '2_3_m_youtube_v0_1',
        '30_60_s_academic_v0_1',
        '30_60_s_youtube_v0_1',
]:
    subset = SubsetDataset(
        subset=subset,
        split=['caption', 'open_ended', 'multi_choice'],
        preprocess_func=LLaVAVideo178KPreprocessor(subset=subset),
    )
    llava_video_subsets.append(subset)

register_dataset(
    DatasetMeta(
        hf_dataset_id='lmms-lab/LLaVA-Video-178K', subsets=llava_video_subsets, tags=['chat', 'multi-modal', 'video']))


class MovieChat1KPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    MovieChat-1K 测试数据：
    - prepare_dataset 阶段下载测试集视频文件集合；
    - preprocess 阶段将相对路径转为本地路径，并选取合适的 `query/response` 字段。
    """

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        """
        下载测试集中涉及的 mp4 文件集合。
        """
        mp4_set = [f'{i}.mp4' for i in range(1, 10)] + \
                  [f'{i}.mp4' for i in range(201, 240)] + \
                  [f'AWA-{i}.mp4' for i in range(1, 10)] + \
                  [f'AWB-{i}.mp4' for i in range(1, 16)] + \
                  [f'AWC-{i}.mp4' for i in range(1, 11)] + \
                  [f'AWD-{i}.mp4' for i in range(1, 8)] + \
                  [f'AWE-{i}.mp4' for i in range(1, 7)] + \
                  [f'AWG-{i}.mp4' for i in range(1, 12)] + \
                  [f'AWH-{i}.mp4' for i in range(1, 8)] + \
                  [f'BWA-{i}.mp4' for i in range(1, 7)] + \
                  [f'BWB-{i}.mp4' for i in range(1, 7)] + \
                  [f'BWD-{i}.mp4' for i in range(1, 6)] + \
                  [f'BWE-{i}.mp4' for i in range(1, 6)] + \
                  [f'BWG-{i}.mp4' for i in range(1, 6)] + \
                  [f'BWH-{i}.mp4' for i in range(1, 6)] + \
                  [f'TFS-{i}.mp4' for i in range(1, 13)] + \
                  [f'UWA-{i}.mp4' for i in range(1, 5)] + ['UWA-6.mp4']  # 构造需要下载的文件名集合
        for file in mp4_set:
            if not use_hf_hub():
                url = f'https://modelscope.cn/datasets/AI-ModelScope/MovieChat-1K-test/resolve/master/videos/{file}'
            else:
                url = f'{get_hf_endpoint()}/datasets/Enxin/MovieChat-1K-test/resolve/main/videos/{file}'
            self.local_dir = MediaResource.download(url, 'moviechat_1k_test', file_type='file')  # 下载到本地缓存
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将相对视频路径转换为本地路径并选择问题/答案作为 query/response。
        """
        file_path = os.path.join(self.local_dir, f"{row['info']['video_path']}")  # 拼接本地路径
        if not os.path.exists(file_path):  # 缺失则跳过
            return None
        return super().preprocess({
            'query': row['global'][0]['question'],  # 使用全局问题
            'response': row['global'][0]['answer'],  # 使用全局答案
            'videos': file_path,  # 本地视频路径
        })


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/MovieChat-1K-test',
        hf_dataset_id='Enxin/MovieChat-1K-test',
        preprocess_func=MovieChat1KPreprocessor(),
        split=['train'],
        tags=['chat', 'multi-modal', 'video']))


class VideoChatGPTPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    VideoChatGPT 测试数据：
    - prepare_dataset 阶段下载测试视频；
    - preprocess 阶段筛选 `.mp4` 并根据多个可能字段选取有效的 query。
    """

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        """
        下载 VideoChatGPT 测试视频并设置本地目录。
        """
        if not use_hf_hub():  # 选择 MS 或 HF 源
            url = 'https://modelscope.cn/datasets/swift/VideoChatGPT/resolve/master/videos.zip'
        else:
            url = f'{get_hf_endpoint()}/datasets/lmms-lab/VideoChatGPT/resolve/main/videos.zip'
        local_dir = MediaResource.download(url, 'video_chatgpt')  # 下载视频压缩包
        self.local_dir = os.path.join(local_dir, 'Test_Videos')  # 指向解压后的测试视频目录
        return super().prepare_dataset(dataset)  # 父类流程

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        仅保留存在于本地的 mp4 视频，并从多个字段中选择有效 query。
        """
        # only `.mp4`  # 仅处理 mp4
        mp4_set = [file[:-4] for file in os.listdir(self.local_dir) if file.endswith('mp4')]  # 可用视频集合
        if row['video_name'] not in mp4_set:  # 不在集合内则跳过
            return
        row['videos'] = os.path.join(self.local_dir, f"{row['video_name']}.mp4")  # 拼接视频路径
        for key in ['query', 'question_1', 'question_2']:  # 依次尝试多个字段
            query = row.get(key)
            if query is None or query == 'None':  # 忽略无效字符串
                continue
            row['query'] = query  # 使用该字段作为查询
            return super().preprocess(row)  # 标准化并返回


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/VideoChatGPT',
        hf_dataset_id='lmms-lab/VideoChatGPT',
        subsets=['Generic', 'Temporal', 'Consistency'],
        preprocess_func=VideoChatGPTPreprocessor(),
        split=['test'],
        tags=['chat', 'multi-modal', 'video', '🔥'],
    ))


def preprocess_mind2web(dataset, **kwargs):

    def preprocess_row(row: Dict[str, Any]) -> Dict[str, Any]:
        raw_html = row['cleaned_html']
        screenshot = row['screenshot']
        row['screenshot'] = MediaResource.safe_save(screenshot, row['action_uid'] + '.jpg', 'mind2web')
        action = row['target_action_reprs']
        actions = action.split('->')
        row['query'] = f'The snapshot of screen:<image>\nThe html source code:{raw_html}\n'
        action = actions[-1]
        where = actions[0] if len(actions) > 1 else ''
        what = ''
        if ':' in action:
            action, what = action[:action.find(':')], action[action.find(':') + 1:]
        row['response'] = f'Action: {action.strip()}\nAction Input: {where.strip()}{"," + what.strip()}'
        return row

    conversations = []
    tools = [{
        'function': {
            'name': 'CLICK',
            'desc': 'Choose and click an element in the web page',
            'parameter': [{
                'element': 'string, the element in the web page to click'
            }]
        }
    }, {
        'function': {
            'name':
            'TYPE',
            'desc':
            'Input some text into a web element like <input> or <textbox>',
            'parameter': [{
                'element': 'string, the element in the web page to input to',
                'content': 'string, what content to input into the textbox element'
            }]
        }
    }, {
        'function': {
            'name':
            'SELECT',
            'desc':
            'Select an element from a combobox',
            'parameter': [{
                'element': 'string, the combobox or dropdown in the web page on which the select happens',
                'content': 'string, which choices to choose'
            }]
        }
    }]

    def history_to_messages(history):
        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})
        return messages

    if isinstance(dataset, HfIterableDataset):

        def generate_example(dataset):
            history = []
            images = []
            for row in dataset:
                target_action_index = row['target_action_index']
                row = preprocess_row(row)
                query = row['query']
                if target_action_index == '0':
                    if history:
                        yield {'messages': history_to_messages(history), 'images': images, 'tools': tools}
                        images = []
                        history = []
                    query = query + '\n' + row['confirmed_task']
                history.append([query, row['response']])
                images.append(row['screenshot'])

            if history:
                yield {'messages': history_to_messages(history), 'images': images, 'tools': tools}

        return HfIterableDataset.from_generator(generate_example, gen_kwargs={'dataset': dataset})

    history = []
    images = []
    for row in tqdm(dataset):
        target_action_index = row['target_action_index']
        row = preprocess_row(row)
        query = row['query']
        if target_action_index == '0':
            if history:
                conversations.append({'messages': history_to_messages(history), 'images': images, 'tools': tools})
                images = []
                history = []
            query = query + '\n' + row['confirmed_task']
        history.append([query, row['response']])
        images.append(row['screenshot'])

    if history:
        conversations.append({'messages': history_to_messages(history), 'images': images, 'tools': tools})

    return HfDataset.from_list(conversations)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/Multimodal-Mind2Web',
        hf_dataset_id='osunlp/Multimodal-Mind2Web',
        preprocess_func=preprocess_mind2web,
        tags=['agent', 'multi-modal']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/M3IT',
        subsets=[
            'coco', 'vqa-v2', 'shapes', 'shapes-rephrased', 'coco-goi-rephrased', 'snli-ve', 'snli-ve-rephrased',
            'okvqa', 'a-okvqa', 'viquae', 'textcap', 'docvqa', 'science-qa', 'imagenet', 'imagenet-open-ended',
            'imagenet-rephrased', 'coco-goi', 'clevr', 'clevr-rephrased', 'nlvr', 'coco-itm', 'coco-itm-rephrased',
            'vsr', 'vsr-rephrased', 'mocheg', 'mocheg-rephrased', 'coco-text', 'fm-iqa', 'activitynet-qa', 'msrvtt',
            'ss', 'coco-cn', 'refcoco', 'refcoco-rephrased', 'multi30k', 'image-paragraph-captioning', 'visual-dialog',
            'visual-dialog-rephrased', 'iqa', 'vcr', 'visual-mrc', 'ivqa', 'msrvtt-qa', 'msvd-qa', 'gqa', 'text-vqa',
            'ocr-vqa', 'st-vqa', 'flickr8k-cn'
        ],
        preprocess_func=ResponsePreprocessor(columns={
            'instruction': 'system',
            'inputs': 'query',
            'image_base64_str': 'images',
            'outputs': 'response'
        }),
        split=['train'],
        huge_dataset=True,
        tags=['chat', 'multi-modal', 'vision']))


class ShareGPT4VPreprocessor(MessagesPreprocessor):
    """
    类说明
    -----
    ShareGPT4V 图像数据：
    - prepare_dataset 阶段按配置名下载所需媒体根目录；
    - preprocess 阶段根据路径前缀选择对应根目录并重写为本地绝对路径。
    """

    def prepare_dataset(self, dataset):
        """
        根据数据集配置名下载必要的图片源目录。
        """
        split = ['ShareGPT4V', 'ShareGPT4V-PT'] if dataset.config_name is None else dataset.config_name  # 选择子配置
        IMAGE_DATASET_REQUIREMENTS = {  # 子配置对应的媒体需求
            'ShareGPT4V': ['coco', 'sam', 'llava', 'wikiart', 'share_textvqa', 'web-celebrity', 'web-landmark'],
            'ShareGPT4V-PT': ['coco', 'sam', 'llava']
        }

        if isinstance(split, str):  # 统一为列表
            split = [split]
        self.all_folders = {}  # 媒体根目录映射
        for sp in split:  # 遍历子配置
            for media_type in IMAGE_DATASET_REQUIREMENTS[sp]:  # 下载所需媒体
                self.all_folders[media_type] = MediaResource.download(media_type)
        return super().prepare_dataset(dataset)  # 父类流程

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        根据路径前缀映射到本地根目录，设置 `images` 字段。
        """
        image = row['image']  # 取原始图片相对路径
        row.update(super().preprocess(row))  # 父类标准化
        if 'coco/' in image:
            image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
        elif 'sam/' in image:
            image = os.path.join(self.all_folders['sam'], image.replace('sam/images/', ''))
        elif 'llava/' in image:
            image = os.path.join(self.all_folders['llava'], image.replace('llava/llava_pretrain/images/', ''))
        elif 'wikiart/' in image:
            image = os.path.join(self.all_folders['wikiart'], image.replace('wikiart/images/', 'data/wikiart/images/'))
        elif 'share_textvqa/' in image:
            image = os.path.join(self.all_folders['share_textvqa'],
                                 image.replace('share_textvqa/images/', 'data/share_textvqa/images/'))
        elif 'web-celebrity/' in image:
            image = os.path.join(self.all_folders['web-celebrity'],
                                 image.replace('web-celebrity/images/', 'data/web-celebrity/images/'))
        elif 'web-landmark/' in image:
            image = os.path.join(self.all_folders['web-landmark'],
                                 image.replace('web-landmark/images/', 'data/web-landmark/images/'))
        if os.path.exists(image):  # 文件存在则设置 images
            row['images'] = image
        else:
            return  # 丢弃缺失样本
        return row  # 返回处理结果


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/ShareGPT4V',
        subsets=['ShareGPT4V', 'ShareGPT4V-PT'],
        preprocess_func=ShareGPT4VPreprocessor(),
        huge_dataset=True,
        tags=['chat', 'multi-modal', 'vision']))


class TextCapsPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    TextCaps 图像文字描述：统一query，过滤不存在的图片路径。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        统一 query 并校验图片路径存在。
        """
        row['query'] = 'What is the caption of this image?'
        if not os.path.exists(row['images']['path']):  # 图片缺失直接跳过
            return None
        return super().preprocess(row)  # 标准化


class TextCapsEmbPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    TextCaps Embedding 版本：不提供 query，仅用于图像-文本嵌入任务，仍需校验图片存在。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        构造空 query 并校验图片路径存在。
        """
        row['query'] = ''
        if not os.path.exists(row['images']['path']):
            return None
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/TextCaps',
        hf_dataset_id='HuggingFaceM4/TextCaps',
        subsets=[
            SubsetDataset(
                name='default',
                preprocess_func=TextCapsPreprocessor(columns={'reference_strs': 'response'}),
                split=['train', 'validation'],
            ),
            SubsetDataset(
                name='emb',
                preprocess_func=TextCapsEmbPreprocessor(columns={'reference_strs': 'response'}),
                split=['train', 'validation'],
            ),
        ],
        huge_dataset=True,
        tags=['multi-modal', 'en', 'caption', 'quality']))


class RefCOCOPreprocessor(ResponsePreprocessor, GroundingMixin):  # 结合响应式预处理与 Grounding 混入
    """
    类说明
    -----
    RefCOCO/RefCOCOg 等数据的预处理器：
    - 支持两类任务：'caption' 与 'grounding'（通过 `task_type` 控制）；
    - 在 `preprocess` 中组织 `objects`/`images` 并构造提示 (query, response)。
    """
    task_type = 'caption'  # 默认任务类型为 caption，可通过构造函数覆盖

    def __init__(self, task_type, **kwargs):
        """
        指定任务类型（caption/grounding），并初始化父类。
        """
        self.task_type = task_type  # 记录具体任务类型
        super().__init__(**kwargs)  # 初始化父类

    def prepare_dataset(self, dataset):
        """\
        下载 COCO2014 资源并设置缓存目录。
        """
        self.cache_dir = MediaResource.download(
            'https://www.modelscope.cn/api/v1/datasets/we_dont_produce_water/'
            'coco_res/repo?Revision=master&FilePath=coco_2014.zip', 'coco2014')  # 下载并缓存
        return dataset  # 不改变原始 dataset

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """\
        构造 grounding/caption 任务所需的 query/response 与 objects。
        """
        caption = row['captions'][0]  # 取第一条描述
        bbox = row['bbox']  # 边框坐标
        image_path = os.path.join(self.cache_dir, row['image_path'].replace('coco/train2014', 'train2014'))  # 修复路径
        if not os.path.exists(image_path):  # 缺失则跳过
            return

        for i in range(len(bbox)):  # 归一化为整数像素
            bbox[i] = round(float(bbox[i]))
        res = {}  # 待返回的记录

        objects = {
            'ref': [caption],  # 参照文本
            'bbox': [bbox],  # 对应边框
        }
        res['query'], res['response'] = self.construct_grounding_prompt()  # 由混入类构造提示
        res['images'] = [image_path]  # 图片路径
        res['objects'] = objects  # 目标对象
        return super().preprocess(res)  # 标准化


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/refcoco',
        hf_dataset_id='jxu124/refcoco',
        subsets=[
            SubsetDataset(
                name='caption',
                preprocess_func=RefCOCOPreprocessor('caption'),
            ),
            SubsetDataset(
                name='grounding',
                preprocess_func=RefCOCOPreprocessor('grounding'),
            )
        ],
        split=['train', 'validation'],
        tags=['multi-modal', 'en', 'grounding']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/refcocog',
        hf_dataset_id='jxu124/refcocog',
        subsets=[
            SubsetDataset(
                name='caption',
                preprocess_func=RefCOCOPreprocessor('caption'),
            ),
            SubsetDataset(
                name='grounding',
                preprocess_func=RefCOCOPreprocessor('grounding'),
            )
        ],
        split=['train', 'validation'],
        tags=['multi-modal', 'en', 'grounding']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/lnqa',
        hf_dataset_id='vikhyatk/lnqa',
        preprocess_func=MessagesPreprocessor(user_role='question', assistant_role='answer'),
        split=['train', 'validation'],
        huge_dataset=True,
        tags=['multi-modal', 'en', 'ocr-vqa', 'quality']))


class LLaVAInstructPreprocessor(MessagesPreprocessor):

    def prepare_dataset(self, dataset):
        self.all_folders = {}
        for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
            self.all_folders[media_type] = MediaResource.download(media_type)
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image = row['images']
        if 'coco/' in image:
            image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
        elif 'gqa/' in image:
            image = os.path.join(self.all_folders['gqa'], image.replace('gqa/', ''))
        elif 'ocr_vqa/' in image:
            image = os.path.join(self.all_folders['ocr_vqa'], image)
        elif 'textvqa/' in image:
            image = os.path.join(self.all_folders['textvqa'], image.replace('textvqa/', ''))
        elif 'VG_100K/' in image:
            image = os.path.join(self.all_folders['VG_100K'], image.replace('vg/', ''))
        elif 'VG_100K_2/' in image:
            image = os.path.join(self.all_folders['VG_100K_2'], image.replace('vg/', ''))
        if os.path.exists(image):
            row['images'] = image
        else:
            return

        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LLaVA-Instruct-150K',
        ms_revision='d5db3806e395c60496630a206c336932e85a2d00',
        preprocess_func=LLaVAInstructPreprocessor(),
        split=['train'],
        tags=['chat', 'multi-modal', 'vision']))


class LLaVAPretrainPreprocessor(MessagesPreprocessor):
    """
    类说明
    -----
    LLaVA 预训练数据预处理：
    - prepare_dataset 阶段下载图片压缩包并记录本地目录；
    - preprocess 阶段将相对路径映射为本地绝对路径，仅返回有效样本。
    """

    def prepare_dataset(self, dataset):
        if not use_hf_hub():  # 根据环境决定下载源（MS/HF）
            url = ('https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/LLaVA-Pretrain/repo?'
                   'Revision=master&FilePath=images.zip')  # MS 源
        else:
            url = f'{get_hf_endpoint()}/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip'  # HF 源
        self.media_dir = MediaResource.download(
            url,
            # noqa
            'llava_pretrain')  # 下载并返回本地缓存目录
        return super().prepare_dataset(dataset)  # 继续父类流程

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将相对图片路径映射为本地绝对路径，仅当文件存在时返回有效记录。

        返回
        ----
        - Optional[Dict[str, Any]]: {'images': 本地路径} 或 None 以丢弃无效样本。
        """
        row.update(super().preprocess(row))  # 先标准化原始行
        if row['image']:  # 存在图片字段
            file_path = os.path.join(self.media_dir, row['image'])  # 拼接绝对路径
            if os.path.exists(file_path):  # 文件存在
                return {'images': file_path}  # 返回仅含 images 的记录
            else:
                return  # 文件缺失丢弃
        else:
            return  # 无图片字段丢弃


register_dataset(  # 注册 LLaVA 预训练数据集
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LLaVA-Pretrain',  # MS 数据集 ID
        ms_revision='e3a3f0bfaad05e90e46745152a32bf944e0f4a63',  # 固定版本号，确保数据一致性
        hf_dataset_id='liuhaotian/LLaVA-Pretrain',  # HF 数据集 ID
        preprocess_func=LLaVAPretrainPreprocessor(),  # 绑定预处理器
        huge_dataset=True,  # 数据集体量较大
        tags=['chat', 'multi-modal', 'quality']))  # 标签：对话/多模态/质量

register_dataset(  # 注册 Midefics 医学 VQA 数据集
    DatasetMeta(
        ms_dataset_id='swift/MideficsDataset',  # MS ID
        hf_dataset_id='WinterSchool/MideficsDataset',  # HF ID
        preprocess_func=MessagesPreprocessor(inner_key='data', user_role='question', assistant_role='answer'),  # 指定消息键与角色
        tags=['medical', 'en', 'vqa']))  # 标签：医学/英文/VQA

register_dataset(  # 注册 OK-VQA 训练集
    DatasetMeta(
        ms_dataset_id='swift/OK-VQA_train',  # MS ID
        hf_dataset_id='Multimodal-Fatima/OK-VQA_train',  # HF ID
        preprocess_func=ResponsePreprocessor(),  # 使用响应式预处理
        tags=['multi-modal', 'en', 'vqa', 'quality']))  # 标签

register_dataset(  # 注册 A-OKVQA 数据集
    DatasetMeta(
        ms_dataset_id='swift/A-OKVQA',  # MS ID
        hf_dataset_id='HuggingFaceM4/A-OKVQA',  # HF ID
        split=['train', 'validation'],  # 训练/验证划分
        preprocess_func=ResponsePreprocessor(columns={'rationales': 'response'}),  # 将 rationales 作为响应
        tags=['multi-modal', 'en', 'vqa', 'quality']))  # 标签


class OcrvqaPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    OCR-VQA 任务：从 `questions/answers` 随机选择同一索引构成问答对。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        随机采样一个问答对并构造两段式消息。
        """
        idx = np.random.choice(range(len(row['questions'])))  # 随机索引
        query = row['questions'][idx]  # 对应问题
        response = row['answers'][idx]  # 对应答案
        return {
            'messages': [{
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
        }


register_dataset(  # 注册 OCR-VQA 数据集
    DatasetMeta(
        ms_dataset_id='swift/OCR-VQA',  # MS ID
        hf_dataset_id='howard-hou/OCR-VQA',  # HF ID
        split=['train', 'validation'],  # 训练/验证划分
        preprocess_func=OcrvqaPreprocessor(),  # 绑定 OCR-VQA 预处理器
        tags=['multi-modal', 'en', 'ocr-vqa']))  # 标签


class ScienceQAPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    ScienceQA：将推理过程与最终答案合并为响应，问题作为查询。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并解析过程与最终答案为响应文本。
        """
        query = row['question']  # 原始问题
        response = row['choices'][row['answer']]  # 选项中的最终答案
        solution = row['solution']  # 推理过程
        response = f'{solution}\nSo the final answer is: {response}'  # 拼接
        return {'messages': [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': response}]}


register_dataset(  # 注册 ScienceQA 数据集
    DatasetMeta(
        ms_dataset_id='swift/ScienceQA',  # MS ID
        hf_dataset_id='derek-thomas/ScienceQA',  # HF ID
        split=['train', 'validation'],  # 训练/验证划分
        preprocess_func=ScienceQAPreprocessor(),  # 绑定 ScienceQA 预处理器
        tags=['multi-modal', 'science', 'vqa', 'quality']))  # 标签


class GritPreprocessor(RowPreprocessor, GroundingMixin):
    """
    类说明
    -----
    GRIT 数据预处理器：根据 `ref_exps`（指代表达的起止位置与 bbox）
    - 切分 `caption` 中对应的对象短语，构造 `objects.ref`；
    - 提取归一化后的 bbox，构造 `objects.bbox`；
    - 检测区间是否重叠，若重叠则丢弃样本；
    - 根据 `task_type`（'grounding'/'caption'/'vqa'）构造 query/response。
    """

    def __init__(self, task_type, **kwargs):
        """
        初始化 GRIT 预处理器。

        参数
        ----
        - task_type: 任务类型，'grounding' 或 'caption' 或其他（如 'vqa'）。
        - **kwargs: 传递给父类 `RowPreprocessor` 的参数。
        """
        self.task_type = task_type  # 记录当前任务类型，影响查询/响应模板
        super().__init__(**kwargs)  # 初始化基类（列映射、随机状态等）

    @staticmethod
    def has_overlap(start_ends):
        """
        判断一组起止区间是否存在重叠（按起点排序后检查）。

        参数
        ----
        - start_ends: List[List[float]]，每个元素为 [start, end]。

        返回
        ----
        - bool: 存在重叠返回 True，否则 False。

        示例
        ----
        >>> GritPreprocessor.has_overlap([[0, 3], [2, 5]])
        True
        """
        for i in range(1, len(start_ends)):  # 从第二个区间开始与前一个比较
            if start_ends[i][0] < start_ends[i - 1][1]:  # 若当前起点 < 前一区间终点，则重叠
                return True  # 存在重叠
        return False  # 无重叠

    @staticmethod
    def replace_intervals_with_tags(response, start_ends):
        """
        将文本 `response` 中的若干区间替换为占位标签 `<ref-object><bbox>`。

        参数
        ----
        - response: 原始描述文本。
        - start_ends: 区间列表，每个元素为 [start, end]。

        返回
        ----
        - str: 替换占位后的文本。
        """
        result = []  # 保存拼接片段
        last_end = 0  # 上一次截取的结束位置
        for start, end in start_ends:  # 遍历每个区间
            result.append(response[int(last_end):int(start)])  # 追加区间前的原文
            result.append('<ref-object><bbox>')  # 用占位标签替换该区间
            last_end = end  # 更新末尾位置
        result.append(response[int(last_end):])  # 追加最后一个区间后的原文
        return ''.join(result)  # 拼接成字符串返回

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将 GRIT 原始行转为标准样本：生成 `messages/images/objects`。

        参数
        ----
        - row: 原始记录，需包含 `images`，`caption`，`ref_exps`：
          - `ref_exps[i]` 形如 [start, end, x1, y1, x2, y2, ...]。

        返回
        ----
        - Optional[Dict[str, Any]]: 标准化样本；若区间非法或缺失信息则返回 None。

        示例
        ----
        >>> sample = {'images': ['img.jpg'], 'caption': 'a cat on a mat', 'ref_exps': [[2, 5, 0.1,0.1,0.3,0.3]]}
        >>> out = GritPreprocessor('grounding').preprocess(sample)
        >>> isinstance(out, dict) and 'objects' in out
        True
        """
        images = row['images']  # 图像路径（列表或单一路径）
        caption = row['caption']  # 原始整句描述
        ref_exps = row['ref_exps']  # 指代表达与边框列表
        objects = {'ref': [], 'bbox': [], 'bbox_type': 'norm1'}  # 初始化 objects：norm1 表示 0~1 归一化坐标
        start_end_pairs = []  # 收集 [start, end] 区间以用于排序与重叠检测
        for ref_exp in ref_exps:  # 遍历每个指代表达
            start = ref_exp[0]  # 起始字符位置
            end = ref_exp[1]  # 结束字符位置
            # conf = ref_exp[6] TODO filter low confidence rows?  # 置信度可选过滤（待实现）
            start_end_pairs.append(ref_exp[0:2])  # 仅保存 [start, end]

            object_part = caption[int(start):int(end)]  # 从 caption 中切出对象短语
            objects['ref'].append(object_part)  # 记录对象短语列表
            objects['bbox'].append(ref_exp[2:6])  # 记录对应 bbox（x1,y1,x2,y2）

        start_end_pairs.sort(key=lambda x: (x[0], x[1]))  # 先按起点再按终点排序
        if self.has_overlap(start_end_pairs) or not ref_exps:  # 存在重叠或无指代表达
            return  # 丢弃该样本

        if self.task_type in ('grounding', 'caption'):  # 需要使用 grounding/caption 模板
            query, response = self.construct_grounding_prompt()  # 由 GroundingMixin 随机生成模板对
        else:  # 其他任务（如 vqa）使用通用问法
            query = 'what is the proper caption of this image?'  # 通用查询
            response = caption  # 直接返回整句 caption 作为参考答案
        return {  # 返回标准化样本
            'messages': [{  # 两段式消息：用户/助手
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
            'images': images,  # 图像路径（或列表）
            'objects': objects  # 对象短语与对应 bbox 信息
        }


register_dataset(  # 注册 GRIT 数据集（多任务）
    DatasetMeta(
        ms_dataset_id='swift/GRIT',  # MS ID
        hf_dataset_id='zzliang/GRIT',  # HF ID
        subsets=[  # 定义多个子任务子集
            SubsetDataset(
                name='caption',  # 图像描述
                preprocess_func=GritPreprocessor('caption', columns={'url': 'images'}),  # 绑定 caption 预处理
            ),
            SubsetDataset(
                name='grounding',  # 目标指代
                preprocess_func=GritPreprocessor('grounding', columns={'url': 'images'}),  # 绑定 grounding 预处理
            ),
            SubsetDataset(
                name='vqa',  # 视觉问答
                preprocess_func=GritPreprocessor('vqa', columns={'url': 'images'}),  # 绑定 vqa 预处理
            )
        ],
        huge_dataset=True,  # 数据规模较大
        tags=['multi-modal', 'en', 'caption-grounding', 'vqa', 'quality']))  # 标签


class GQAPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    GQA 问答数据：
    - prepare_dataset 阶段下载 `gqa` 媒体根目录；
    - preprocess 阶段绑定本地图片路径并生成两段式消息。
    """

    def prepare_dataset(self, dataset):
        """
        下载/定位 gqa 资源目录。
        """
        self.local_cache = MediaResource.download('gqa')  # 下载或使用本地缓存
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        构造问答消息并绑定图片路径，若图片不存在则跳过。
        """
        image_path = os.path.join(self.local_cache, 'images', row['imageId'] + '.jpg')  # 拼接图片路径
        if os.path.exists(image_path):  # 文件存在
            return {
                'messages': [{
                    'role': 'user',
                    'content': row['question']  # 问题文本
                }, {
                    'role': 'assistant',
                    'content': row['fullAnswer']  # 答案文本
                }],
                'images': image_path,  # 本地图片
            }
        else:
            return  # 缺失则跳过


register_dataset(  # 注册 GQA 数据集
    DatasetMeta(
        hf_dataset_id='lmms-lab/GQA',  # HF ID（无 MS ID）
        split=['train_all_instructions'],  # 使用特定划分
        preprocess_func=GQAPreprocessor(),  # 绑定 GQA 预处理器
        huge_dataset=True,  # 数据量大
        tags=['multi-modal', 'en', 'vqa', 'quality']))  # 标签


class CocoPreprocessor(ResponsePreprocessor):
    category = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row['query'] = 'Task: Object Detection'
        objects = row['objects']
        objects['ref'] = [self.category[c] for c in objects['category']]
        row['response'] = '\n'.join(['<ref-object><bbox>'] * len(objects['ref']))
        return super().preprocess(row)


register_dataset(  # 注册 COCO 检测格式数据
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/coco',  # MS ID
        hf_dataset_id='detection-datasets/coco',  # HF ID
        preprocess_func=CocoPreprocessor(),  # 绑定 COCO 预处理器
        huge_dataset=True,  # 数据量大
        tags=['multi-modal', 'en', 'vqa', 'quality']))  # 标签


class LLaVAMixSFTPreprocessor(RowPreprocessor):
    """
    类说明
    -----
    LLaVA 混合指令数据（视觉 SFT）预处理：
    - 将多模态消息的 content 列表组装成纯文本（图片位置以 `<image>` 占位）；
    - 输出标准 `messages` 列表。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        将多模态内容展平成文本消息列表。
        """
        messages = row['messages']  # 原始多模态消息
        rounds = []  # 输出的文本化消息
        for msg in messages:  # 遍历每轮
            role = msg['role']  # 角色
            content = msg['content']  # 内容列表（text/image）
            text = ''  # 该轮聚合文本
            for index in content:  # 遍历内容片段
                if index['type'] == 'text':  # 文本片段
                    text += index['text']
                elif index['type'] == 'image':  # 图片片段
                    text += '<image>'  # 用占位符表示

            rounds.append({'role': role, 'content': text})  # 追加该轮

        return {'messages': rounds}  # 返回结果


register_dataset(  # 注册 LLaVA 指令混合集（视觉 SFT 验证）
    DatasetMeta(
        ms_dataset_id='swift/llava-instruct-mix-vsft',  # MS ID
        hf_dataset_id='HuggingFaceH4/llava-instruct-mix-vsft',  # HF ID
        split=['test'],  # 测试划分
        preprocess_func=LLaVAMixSFTPreprocessor(),  # 绑定预处理器
        tags=['multi-modal', 'en', 'vqa', 'quality']))  # 标签


class LatexocrPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    LaTeX OCR 任务预处理：将 query 统一为英文说明，交由父类完成标准化。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """设置统一查询并生成标准 messages。"""
        row['query'] = 'Using LaTeX to perform OCR on the image.'  # 统一任务描述
        return super().preprocess(row)  # 标准化


register_dataset(  # 注册 LaTeX OCR 数据集
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LaTeX_OCR',  # MS ID
        hf_dataset_id='linxy/LaTeX_OCR',  # HF ID
        subsets=['default', 'human_handwrite', 'human_handwrite_print', 'synthetic_handwrite', 'small'],  # 多子集
        preprocess_func=LatexocrPreprocessor(),  # 绑定预处理器
        split=['train', 'validation', 'test'],  # 训练/验证/测试
        tags=['chat', 'ocr', 'multi-modal', 'vision'],  # 标签
    ))


class CapchaImagesPreprocessor(ResponsePreprocessor):
    """
    类说明
    -----
    验证码图片识别任务预处理：统一 query，保留原始 response。
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """设置统一查询并生成标准 messages。"""
        row['query'] = 'recognize the content.'  # 指定任务意图
        return super().preprocess(row)  # 标准化


register_dataset(  # 注册验证码图片数据集
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/captcha-images',  # MS ID
        split=['train', 'validation'],  # 训练/验证
        preprocess_func=CapchaImagesPreprocessor(columns={'solution': 'response'}),  # 列映射：答案为 response
        tags=['chat', 'multi-modal', 'vision']))  # 标签


class ClevrPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row.get('query', '')
        query = (f'{query} Output the thinking process in <think> </think> and '
                 'final answer (number) in <answer> </answer> tags.')
        row.update({'query': query})
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='okwinds/clevr_cogen_a_train',
        hf_dataset_id='leonardPKU/clevr_cogen_a_train',
        preprocess_func=ClevrPreprocessor(),
        tags=['qa', 'math', 'vision', 'grpo']))
