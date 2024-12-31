from copy import copy

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

@dataclass
class Node:
    """
    定义蒙特卡洛树的节点。
    """
    text: str
    score: float
    success_count: int = 0  # 记录子节点中成功的数量
    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list)

def monte_carlo_tree_search(
    prompt: str,
    model,
    tokenizer,
    custom_func: Callable[[str], bool],
    custom_score: Callable[[str], float],
    generation_config: GenerationConfig,
    max_depth: int = 3,
    max_children: int = 5,
    success_factor: float = 1.0,
    decay_factor: float = 0.5,    # 奖励衰减因子
    penalty_factor: float = 0.2,  # 失败惩罚的初始因子
    penalty_decay: float = 0.5,    # 失败惩罚的衰减因子
    score_threshold: float = 0.0,
    history: List = [],
) -> Tuple[List[str], bool]:
    """
    使用蒙特卡洛树搜索生成文本，并返回最佳路径。

    Args:
        prompt (str): 初始提示文本。
        model: 预训练的语言模型。
        tokenizer: 对应模型的分词器。
        custom_func (Callable[[str], bool]): 自定义的条件判断函数，输入为句子，输出为布尔值。
        custom_score (Callable[[str], float]): 自定义的评分函数，输入为句子，输出为浮点数得分。
        max_depth (int): 树的最大深度。
        max_children (int): 每个节点最大子节点数量。
        max_new_tokens (int): 每次生成的最大新token数。
        decay_factor (float): 奖励衰减因子。
        penalty_factor (float): 失败节点对父节点的初始惩罚因子。
        penalty_decay (float): 失败惩罚的衰减因子。

    Returns:
        Tuple[List[str], bool]: 最佳路径的文本列表和是否成功找到满足条件的路径。
    """
    # 初始化根节点
    root = Node(text=prompt, score=0.0)
    current_nodes = [root]
    success_nodes = []  # 存储满足条件的成功路径节点
    for depth in range(max_depth):
        all_children = []
        print(f"\n--- 深度 {depth + 1} ---")
        for node in current_nodes:
            print(f"扩展节点: \"{node.text}\" (分数: {node.score:.2f})")

            # 如果当前节点已具有成功子节点，不再扩展
            if node.success_count > 0:
                print(f"节点 \"{node.text}\" 已有成功子节点，不再扩展。")
                continue

            # 使用模型生成多个可能的续句
            inputs = tokenizer(node.text, return_tensors='pt').to(model.device)
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                num_return_sequences=max_children,
            )

            for gen in outputs:
                gen_text = tokenizer.decode(gen, skip_special_tokens=True)

                # 确保生成的文本是对当前节点的自然延续
                if not gen_text.startswith(node.text):
                    continue  # 跳过不相关的生成

                # 计算生成文本的得分
                h = copy(history)
                h[-1][1] = gen_text
                score = custom_score(h)
                if score < score_threshold:
                    continue
                # 创建子节点
                child = Node(text=gen_text, score=score, parent=node)

                # 判断子节点是否满足自定义条件
                if custom_func(child.text):
                    success_nodes.append(child)

                    # 按衰减因子向所有祖先节点分配奖励
                    current_reward = success_factor
                    ancestor = node
                    while ancestor is not None:
                        ancestor.score += current_reward
                        ancestor.success_count += 1
                        print(f"生成子句: \"{child.text}\" (得分: {child.score:.2f}) [成功] 父节点 \"{ancestor.text}\" 得分增加至 {ancestor.score:.2f}")
                        current_reward *= decay_factor  # 奖励衰减
                        ancestor = ancestor.parent
                else:
                    # 如果生成结束且未满足成功条件，则视为失败
                    if tokenizer.eos_token_id and gen[-1].item() == tokenizer.eos_token_id:
                        # 按比例向所有祖先节点分配惩罚
                        current_penalty = penalty_factor
                        ancestor = node
                        while ancestor is not None:
                            ancestor.score -= current_penalty
                            print(f"生成子句: \"{child.text}\" (得分: {child.score:.2f}) [失败：生成结束] 父节点 \"{ancestor.text}\" 得分减少至 {ancestor.score:.2f}")
                            current_penalty *= penalty_decay  # 惩罚衰减
                            ancestor = ancestor.parent
                    else:
                        print(f"生成子句: \"{child.text}\" (得分: {child.score:.2f})")
                        node.children.append(child)
                        all_children.append(child)

        if success_nodes:
            print(f"在深度 {depth + 1} 找到了满足条件的路径。")
            break  # 找到成功路径后，可以选择停止搜索或继续搜索以找到更优的路径

        if not all_children:
            print("没有更多的子节点生成，提前终止搜索。")
            break

        # 按照得分降序排列所有子节点，并选择前 max_children 个
        all_children = sorted(all_children, key=lambda x: x.score, reverse=True)[:max_children]
        current_nodes = all_children

    # 如果找到成功的节点，选择得分最高的路径
    if success_nodes:
        best_success_node = max(success_nodes, key=lambda x: x.score)
        # 追溯路径
        path = []
        node = best_success_node
        while node:
            path.insert(0, node.text)
            node = node.parent
        return path, True
    else:
        print("未能找到满足条件的路径。")
        return [], False

def custom_condition(sample_text: str) -> bool:
    """
    自定义条件判断函数。

    例子：判断句子是否包含特定关键词。

    Args:
        sample_text (str): 生成的句子文本。

    Returns:
        bool: 如果满足条件，返回 True；否则，返回 False。
    """
    # 示例条件：句子中包含 "成功" 或 "拯救" 这个词
    return "成功" in sample_text or "拯救" in sample_text  # 根据需求调整

def custom_score_function(sample_text: str) -> float:
    """
    自定义评分函数。

    例子：根据句子长度和关键词出现情况打分。

    Args:
        sample_text (str): 生成的句子文本。

    Returns:
        float: 句子的得分。
    """
    score = 0.0
    # 示例评分策略：
    # - 句子越长，得分越高
    # - 句子中包含 "美丽" 或 "勇敢" 等正面词汇，得分增加
    # - 句子中包含 "失败" 或 "悲伤" 等负面词汇，得分减少

    # 基本长度得分
    score += len(sample_text) * 0.1

    # 关键词得分
    positive_keywords = ["美丽", "勇敢", "聪明", "善良", "成功", "拯救"]
    negative_keywords = ["失败", "悲伤", "恐惧", "危险"]

    for word in positive_keywords:
        if word in sample_text:
            score += 5.0

    for word in negative_keywords:
        if word in sample_text:
            score -= 5.0

    return score

def main():
    # 选择预训练模型和分词器
    model_name = 'AI-ModelScope/gpt2'  # 可以替换为其他支持的模型，如 'gpt2-medium'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 设置设备（GPU 如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义初始提示
    prompt = "从前有一个"

    # 执行蒙特卡洛树搜索
    best_path, success = monte_carlo_tree_search(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        custom_func=custom_condition,      # 自定义条件函数
        custom_score=custom_score_function,  # 自定义评分函数
        max_depth=5,       # 树的深度
        max_children=5,    # 每个节点的最大子节点数
        max_new_tokens=50, # 每次生成的最大token数
        decay_factor=0.5,  # 奖励衰减因子
        penalty_factor=5.0, # 失败惩罚的初始因子
        penalty_decay=0.5   # 失败惩罚的衰减因子
    )

    # 输出结果
    if success:
        print("\n=== 成功生成路径 ===")
        for i, sentence in enumerate(best_path, 1):
            print(f"{i}. {sentence}")
    else:
        print("\n=== 生成失败，没有找到满足条件的路径 ===")

if __name__ == "__main__":
    main()