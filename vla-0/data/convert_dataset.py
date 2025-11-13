import json

def convert_dataset_format(input_file_path, output_file_path):
    """
    将数据集从源格式转换为目标格式。

    Args:
        input_file_path (str): 源数据文件路径 (train.jsonl)。
        output_file_path (str): 转换后新文件的保存路径。
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                # 1. 读取并解析原始的每一行JSON数据
                original_data = json.loads(line.strip())

                # 2. 提取所有需要的内容
                system_content = original_data.get("system", "")
                prompt_content = original_data.get("prompt", "")
                response_content = original_data.get("response", "")
                image_paths = original_data.get("image", [])

                # 从原始prompt中分离出文本描述
                # 假设任务描述和图片占位符总是以 "\n\nimage:" 分隔
                user_text_prompt = prompt_content.split("\n\nimage:")[0]
                
                # 根据图片数量生成对应的<image>占位符字符串
                image_placeholders = "".join(["<image>" for _ in image_paths])
                
                # 组合成新的user content
                user_content_final = f"{user_text_prompt}{image_placeholders}"


                # 3. 构建新的目标JSON结构
                converted_data = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content
                        },
                        {
                            "role": "user",
                            "content": user_content_final
                        },
                        {
                            "role": "assistant",
                            "content": response_content
                        }
                    ],
                    "images": image_paths
                }

                # 4. 将转换后的JSON对象写入新文件，并添加换行符
                outfile.write(json.dumps(converted_data, ensure_ascii=False) + '\n')
                
        print(f"转换完成！文件已保存至: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {input_file_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 使用说明 ---
# 1. 将此脚本保存为 .py 文件（例如 convert_script.py）。
# 2. 确保您的原始数据集文件 `train.jsonl` 与此脚本在同一目录下。
# 3. 运行此脚本，它会生成一个名为 `train_converted.jsonl` 的新文件。

# 定义输入和输出文件名
source_file = '/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla/train.jsonl'
destination_file = '/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla/train_converted_4.jsonl'

# 执行转换
convert_dataset_format(source_file, destination_file)