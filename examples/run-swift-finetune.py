# -*- coding: utf-8 -*-
import sys

import torch

sys.path.append("../../")

from swift.llm import sft_main


if __name__ == '__main__':
    output = sft_main()



"""
--model_type internvl2-1b --model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B --dataset coco-en-2-mini --max_length 4096 --sft_type lora
--model_type internvl2-1b --model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B --dataset coco-en-2-mini --max_length 4096 --sft_type full
"""

"""
# 支持多个数据集
--model_type internvl2-1b 
--model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B 
--dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/train.jsonl,/mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/val.jsonl
--val_dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/val.jsonl
--max_length 4096 --sft_type lora
"""

"""
--model_type internvl2-1b 
--model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B 
--dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/train.jsonl
--val_dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/val.jsonl
--max_length 2500 --sft_type full
--num_train_epochs 20
--save_steps 200
--save_strategy steps
--save_total_limit 5
"""


# 自定义template

"""
--model_type internvl2-1b 
--model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B 
--dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/train_with_angle.jsonl
--val_dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/val_with_angle.jsonl
--max_length 2500 --sft_type full
--num_train_epochs 20
--save_steps 200
--save_strategy steps
--save_total_limit 5
--template internvl2-angle
"""