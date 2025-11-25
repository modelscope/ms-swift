"""
合并、打乱、采样 JSONL 文件 (命令行版本)

用法示例：
    # 合并两个文件，保留 10%
    python merge_jsonl_cli.py \
        -i file1.jsonl file2.jsonl \
        -o output.jsonl \
        -r 0.1

    # 合并所有文件，不采样，不打乱
    python merge_jsonl_cli.py \
        -i *.jsonl \
        -o output.jsonl \
        --no-shuffle

    # 指定采样数量而非比例
    python merge_jsonl_cli.py \
        -i file1.jsonl file2.jsonl \
        -o output.jsonl \
        -n 10000
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def load_jsonl(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"加载 {Path(file_path).name}", leave=False):
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """保存为 JSONL 文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"保存", leave=False):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="合并、打乱、采样 JSONL 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "-i", "--input",
        nargs='+',
        required=True,
        help="输入的 JSONL 文件列表"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出的 JSONL 文件路径"
    )

    parser.add_argument(
        "-r", "--ratio",
        type=float,
        default=None,
        help="采样比例 (0-1)，例如 0.1 表示保留 10%%"
    )

    parser.add_argument(
        "-n", "--num",
        type=int,
        default=None,
        help="采样数量，例如 10000 表示保留 10000 条"
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不打乱数据（默认会打乱）"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    print("=" * 60)
    print("合并、打乱、采样 JSONL 文件")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  输入文件: {len(args.input)} 个")
    print(f"  输出文件: {args.output}")
    print(f"  打乱: {'否' if args.no_shuffle else '是'}")
    print(f"  随机种子: {args.seed}")

    # 1. 加载所有文件
    print(f"\n步骤 1: 加载输入文件")
    all_data = []
    file_stats = []

    for file_path in args.input:
        if not Path(file_path).exists():
            print(f"  警告: 文件不存在，跳过: {file_path}")
            continue

        data = load_jsonl(file_path)
        file_stats.append((Path(file_path).name, len(data)))
        all_data.extend(data)
        print(f"  ✓ {Path(file_path).name}: {len(data):,} 条")

    print(f"\n  总计: {len(all_data):,} 条")

    if len(all_data) == 0:
        print("\n错误: 没有加载到任何数据！")
        return

    # 2. 打乱数据
    if not args.no_shuffle:
        print(f"\n步骤 2: 打乱数据")
        random.shuffle(all_data)
        print(f"  ✓ 已打乱 {len(all_data):,} 条数据")

    # 3. 采样
    if args.num is not None:
        # 按数量采样
        sample_size = min(args.num, len(all_data))
        sampled_data = all_data[:sample_size]
        print(f"\n步骤 3: 采样数据")
        print(f"  采样数量: {len(sampled_data):,} / {len(all_data):,} 条")
    elif args.ratio is not None:
        # 按比例采样
        sample_size = int(len(all_data) * args.ratio)
        sampled_data = all_data[:sample_size]
        print(f"\n步骤 3: 采样数据")
        print(f"  采样比例: {args.ratio:.2%}")
        print(f"  采样数量: {len(sampled_data):,} / {len(all_data):,} 条")
    else:
        # 不采样，使用全部
        sampled_data = all_data
        print(f"\n步骤 3: 使用全部数据")
        print(f"  数量: {len(sampled_data):,} 条")

    # 4. 保存结果
    print(f"\n步骤 4: 保存结果")
    save_jsonl(sampled_data, args.output)
    print(f"  ✓ 已保存到: {args.output}")

    print("\n" + "=" * 60)
    print(f"完成！最终数据量: {len(sampled_data):,} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
