# libero_dataset.py
from email.mime import image
import random
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset, load_dataset

# --- VLA-0 核心技巧实现 ---
def random_masking(action_text: str, masking_ratio: float = 0.25) -> str:
    """
    对VLA-0的动作文本进行随机掩码（这里采用随机丢弃部分动作）。
    VLA-0论文原文是"mask out characters"，但随机丢弃整个数字token是
    一种更稳定且有效的实现方式，可以迫使模型依赖图像和指令，
    而非仅仅自回归地补全数字序列。

    Args:
        action_text: e.g., "512 488 230 910 110 450 1000"
        masking_ratio: 丢弃动作序列中token的比例。

    Returns:
        被掩码后的动作文本，e.g., "512 230 910 450 1000"
    """
    # 如果ratio不合法或为0，则不进行操作
    if not (0 < masking_ratio < 1):
        return action_text

    action_tokens = action_text.split(' ')
    num_tokens = len(action_tokens)
    # 计算需要保留的token数量
    num_to_keep = round(num_tokens * (1 - masking_ratio))

    # 随机选择要保留的token的索引，并排序以维持相对顺序
    indices_to_keep = sorted(random.sample(range(num_tokens), k=num_to_keep))

    # 构建新的token列表
    masked_tokens = [action_tokens[i] for i in indices_to_keep]

    return " ".join(masked_tokens)


class VLA0LiberoPreprocessor(ResponsePreprocessor):
    """
    data example:
    {"messages": [{"role": "system", "content": "Analyze the input image and predict robot actions for the next 1 timesteps. Each action has 7 dimensions. Output a single sequence of 7 integers (0 - 1000 each), representing the 5 timesteps sequentially. Provide only space-separated numbers. Nothing else."}, {"role": "user", "content": "task description: put the white mug on the left plate and put the yellow and white mug on the right plate<image><image>"}, {"role": "assistant", "content": "619 337 809 494 493 500 1000"}], "images": ["images/00000059_main.jpg", "images/00000059_wrist.jpg"]}

    """
    def __init__(self, masking_ratio: float = 0.25, is_training: bool = True):
        """
        Args:
            masking_ratio: 动作文本的掩码比例。
            is_training: 只有在训练时才应用掩码。
        """
        super().__init__()
        self.masking_ratio = masking_ratio
        self.is_training = is_training

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # processed_data = super().preprocess(row)
        system_message = row.get('messages', [])[0].get('content', '')
        query_message = row.get('messages', [])[1].get('content', '')
        response_message = row.get('messages', [])[2].get('content', '')
        images = row.get('images', [])

        if not system_message or not query_message or not response_message:
            print("shit")
            return None

        if self.is_training:
            response_message = random_masking(response_message, self.masking_ratio)

        # # --- VLA-0 随机masking操作 ---
        # # 仅在训练阶段对 response (即动作文本) 进行操作
        # if self.is_training and 'response' in processed_data:
        #     original_response = processed_data['response']
        #     processed_data['response'] = random_masking(original_response, self.masking_ratio)

        # # 将 images 字段传递下去，SWIFT 会自动处理
        # if 'images' in row:
        #     processed_data['images'] = row['images']
            
        return super().preprocess({
            'system': system_message,
            'query': query_message,
            'response': response_message,
            'images': images,
        })
    
    def prepare_dataset(self, dataset):
        self.prefix_path = '/home/yuquan002/ssd/libero_vl_dataset/'
        return super().prepare_dataset(dataset)

# --- 向 SWIFT 框架注册我们的自定义数据集 ---
# **如何调用**: SWIFT 命令行通过这里的 'dataset_id' 来识别
# --dataset libero-vla0=/path/to/data  <-- 'libero-vla0' 就对应下面的 dataset_id
# 
# 我们不再需要 ms_dataset_id 或 hf_dataset_id，因为我们加载的是本地文件。

register_dataset(
    DatasetMeta(
        dataset_name='libero-vla0',
        ms_dataset_id='libero_vla0',
        hf_dataset_id=None,
        dataset_path='/home/yuquan002/ssd/libero_vl_dataset/train.jsonl',
        preprocess_func=VLA0LiberoPreprocessor(masking_ratio=0.25, is_training=True),
    ))

register_dataset(
    DatasetMeta(
        dataset_name='libero-spatial-vla0',
        ms_dataset_id='libero_spatial_vla0',
        hf_dataset_id=None,
        dataset_path='/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla/train_converted.jsonl',
        preprocess_func=VLA0LiberoPreprocessor(masking_ratio=0.25, is_training=False),
    ))

if __name__ == '__main__':
    dataset = load_dataset(['libero-vla0'])[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')