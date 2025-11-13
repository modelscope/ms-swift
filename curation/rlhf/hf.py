import os
import json
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# --- 1. 配置模型和处理器 ---
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_dir = '/home/yuquan002/ssd/ms-swift-robotics/output/qwen3-vl-4b-instruct-vla0-libero/v6-20251029-202919/checkpoint-105984'

print("正在加载模型...")
model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_dir)
print("模型加载完毕。")

# --- 2. 准备图像输入 (这部分可以保持不变) ---
main_dir = "/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla/images/00000000_main.jpg"
main = Image.open(main_dir).convert("RGB")
wrist_dir = "/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla/images/00000000_wrist.jpg"
wrist = Image.open(wrist_dir).convert("RGB")

# --- 3. 定义输出文件路径 ---
REWARD_DATA_FILE = 'reward_model_data.jsonl'
ACCEPTED_DATA_FILE = 'accepted_data.jsonl'

# --- 4. 主循环：生成、评估、保存 ---
while True:
    print("-" * 128)
    input_text = input("请输入你的指令 (输入 'exit' 退出): ")
    if input_text.lower() == 'exit':
        break
    
    print("正在处理输入并生成回复...")
    print('-' * 128)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that helps people find information."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": main},
                {"type": "image", "image": wrist},
                {"type": "text", "text": input_text},
            ],
        }
    ]

    # --- 准备模型输入 ---
    # 注意：这里的 text 和 inputs 只需要构建一次，可以在两次生成调用中复用
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info([messages], image_patch_size=16)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    inputs = inputs.to('cuda')

    # --- 生成两个不同的回复 ---
    # 通过设置 do_sample=True 和不同的 temperature/top_p 来增加多样性
    # 回复 A: temperature 较低，偏向于确定性
    generated_ids_a = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=0.7, top_p=0.9)
    # 回复 B: temperature 较高，更多样
    generated_ids_b = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=1.0, top_p=0.95)

    # --- 解码生成的文本 ---
    generated_ids_a_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids_a)]
    output_text_a = processor.batch_decode(generated_ids_a_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    generated_ids_b_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids_b)]
    output_text_b = processor.batch_decode(generated_ids_b_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # --- 人类反馈环节 ---
    print("\n" + "="*40 + " 请选择更好的回复 " + "="*40)
    print("\n[ 回复 A ]:\n", output_text_a)
    print("\n" + "-"*100)
    print("\n[ 回复 B ]:\n", output_text_b)
    print("\n" + "="*100)

    choice = ''
    while choice not in ['a', 'b', 'none']:
        choice = input("请选择更好的回复 ('a' 或 'b')，如果两个都不好请输入 'none': ").lower()

    if choice == 'none':
        print("两个回复都被舍弃，不保存。")
        continue
    
    # --- 确定 chosen 和 rejected 的内容 ---
    if choice == 'a':
        chosen_response = output_text_a
        rejected_response = output_text_b
    else: # choice == 'b'
        chosen_response = output_text_b
        rejected_response = output_text_a

    # --- 准备要保存的数据 ---
    # 1. 奖励模型数据
    reward_data = {
        "messages": messages,
        "chosen": chosen_response,
        "rejected": rejected_response
    }
    # 在处理 messages 前，需要将 PIL.Image 对象转换为可序列化的格式，比如路径
    # 这里我们简化处理，不将图片本身存入json，只存文本
    reward_data['messages'][1]['content'] = [{"type": "text", "text": input_text}] # 仅保留文本

    # 2. 仅包含接受回复的数据
    accepted_messages = messages.copy()
    accepted_messages[1]['content'] = [{"type": "text", "text": input_text}] # 同上，仅保留文本
    accepted_messages.append({"role": "assistant", "content": chosen_response})
    accepted_data = {"messages": accepted_messages}

    # --- 将数据追加写入JSONL文件 ---
    try:
        with open(REWARD_DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(reward_data, ensure_ascii=False) + '\n')
        
        with open(ACCEPTED_DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(accepted_data, ensure_ascii=False) + '\n')
        
        print(f"成功将选择 '{choice.upper()}' 的数据保存到 {REWARD_DATA_FILE} 和 {ACCEPTED_DATA_FILE}")
    
    except Exception as e:
        print(f"保存文件时出错: {e}")

print("程序已退出。")