from copy import copy, deepcopy

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
            print(f"扩展节点: \"{node.text[len(prompt):]}\" (分数: {node.score:.2f})")

            # 如果当前节点已具有成功子节点，不再扩展
            if node.success_count > 0:
                print(f"节点 \"{node.text[len(prompt):]}\" 已有成功子节点，不再扩展。")
                continue

            # 使用模型生成多个可能的续句
            inputs = tokenizer(node.text, return_tensors='pt').to(model.device)
            # inputs = {'input_ids': node.text.unsqueeze(0)}
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                num_return_sequences=max(max_children-depth, 1),
            )

            for gen in outputs:
                gen_text = tokenizer.decode(gen, skip_special_tokens=False)
                has_end = gen_text.endswith('<|im_end|>')

                # 确保生成的文本是对当前节点的自然延续
                if not gen_text.startswith(node.text):
                    continue  # 跳过不相关的生成

                gen_text = gen_text.rstrip('<|endoftext|>').rstrip('<|im_end|>')

                # 计算生成文本的得分
                h = deepcopy(history)
                h[-1]['content'] = gen_text[len(node.text):]
                score = custom_score(h)
                if score < score_threshold:
                    continue
                # 创建子节点
                child = Node(text=gen_text, score=score, parent=node)

                # 判断子节点是否满足自定义条件
                if custom_func([h[-1]['content']])[0]:
                    success_nodes.append(child)

                    # 按衰减因子向所有祖先节点分配奖励
                    current_reward = success_factor
                    ancestor = node
                    while ancestor is not None:
                        ancestor.score += current_reward
                        ancestor.success_count += 1
                        print(f"生成子句: \"{child.text[len(prompt):]}\" (得分: {child.score:.2f}) [成功] 父节点 \"{ancestor.text[len(prompt):]}\" 得分增加至 {ancestor.score:.2f}")
                        current_reward *= decay_factor  # 奖励衰减
                        ancestor = ancestor.parent
                else:
                    # 如果生成结束且未满足成功条件，则视为失败
                    if has_end:
                        # 按比例向所有祖先节点分配惩罚
                        current_penalty = penalty_factor
                        ancestor = node
                        while ancestor is not None:
                            ancestor.score -= current_penalty
                            print(f"生成子句: \"{child.text[len(prompt):]}\" (得分: {child.score:.2f}) [失败：生成结束] 父节点 \"{ancestor.text[len(prompt):]}\" 得分减少至 {ancestor.score:.2f}")
                            current_penalty *= penalty_decay  # 惩罚衰减
                            ancestor = ancestor.parent
                    else:
                        print(f"生成子句: \"{child.text[len(prompt):]}\" (得分: {child.score:.2f})")
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
        # while node:
        #     path.insert(0, node.text)
        #     node = node.parent
        return node.text[len(prompt):], True
    else:
        print("未能找到满足条件的路径。")
        return None, False
