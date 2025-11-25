"""
验证并修复 JSONL 文件中的 null 值问题

功能：
1. 检测所有可能导致类型推断错误的 null 值
2. 移除包含 null 值的样本
3. 统计并报告问题

用法：
    python fix_null_jsonl.py input.jsonl output.jsonl
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def has_null_values(data, path="root"):
    """
    递归检查数据中是否包含 null 值
    返回: (has_null, null_paths)
    """
    null_paths = []

    if data is None:
        return True, [path]

    if isinstance(data, dict):
        for key, value in data.items():
            has_null, paths = has_null_values(value, f"{path}.{key}")
            if has_null:
                null_paths.extend(paths)

    elif isinstance(data, list):
        if len(data) == 0:
            # 空列表也标记为问题
            null_paths.append(f"{path}[empty list]")
        else:
            for i, item in enumerate(data):
                has_null, paths = has_null_values(item, f"{path}[{i}]")
                if has_null:
                    null_paths.extend(paths)

    elif isinstance(data, str):
        if not data.strip():
            # 空字符串
            null_paths.append(f"{path}[empty string]")

    return len(null_paths) > 0, null_paths


def validate_sample(sample):
    """
    验证样本是否完全合规（无 null、无空值）
    返回: (is_valid, error_message)
    """
    # 检查必需字段是否存在
    required_fields = ['id', 'image', 'conversations']
    for field in required_fields:
        if field not in sample:
            return False, f"缺少字段: {field}"

    # 检查是否有 null 值
    has_null, null_paths = has_null_values(sample)
    if has_null:
        return False, f"包含 null/空值: {', '.join(null_paths[:3])}"  # 只显示前3个

    # 额外检查：确保 conversations 不为空且格式正确
    conversations = sample.get('conversations', [])
    if not isinstance(conversations, list) or len(conversations) == 0:
        return False, "conversations 为空或格式错误"

    for i, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            return False, f"conversations[{i}] 不是字典"
        if 'from' not in turn or 'value' not in turn:
            return False, f"conversations[{i}] 缺少 from/value"

    return True, None


def process_jsonl(input_file, output_file, verbose=False):
    """
    处理 JSONL 文件，移除包含 null 值的样本
    """
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print()

    valid_count = 0
    invalid_count = 0
    error_stats = {}

    # 创建输出目录
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(tqdm(f_in, desc="处理中"), start=1):
            try:
                sample = json.loads(line)

                # 验证样本
                is_valid, error_msg = validate_sample(sample)

                if is_valid:
                    # 写入有效样本
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    valid_count += 1
                else:
                    invalid_count += 1

                    # 统计错误类型
                    error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                    error_stats[error_type] = error_stats.get(error_type, 0) + 1

                    if verbose and invalid_count <= 10:
                        print(f"\n第 {line_num} 行无效: {error_msg}")
                        print(f"  数据: {json.dumps(sample, ensure_ascii=False)[:200]}...")

            except json.JSONDecodeError as e:
                invalid_count += 1
                error_stats['JSON 解析错误'] = error_stats.get('JSON 解析错误', 0) + 1
                if verbose and invalid_count <= 10:
                    print(f"\n第 {line_num} 行 JSON 解析错误: {e}")

    # 输出统计信息
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"有效样本: {valid_count:,} 条")
    print(f"无效样本: {invalid_count:,} 条")
    print(f"通过率: {valid_count / (valid_count + invalid_count) * 100:.2f}%")

    if error_stats:
        print("\n错误类型统计:")
        for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count:,} 条")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="验证并修复 JSONL 文件中的 null 值问题",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "input",
        help="输入的 JSONL 文件",
    )

    parser.add_argument(
        "output",
        help="输出的 JSONL 文件（已清理）"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细的错误信息"
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return

    process_jsonl(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()
