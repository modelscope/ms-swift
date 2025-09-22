"""
模块说明：
    本模块提供基础的路径处理工具函数，主要用于校验与规范化用户传入的路径，
    将相对路径与包含用户目录符号（~）的路径统一转换为绝对路径，并可选地检查
    路径是否存在。该工具在参数解析、数据集/模型文件定位等场景中通用。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Union  # 引入类型注解：支持字符串、字符串列表或 None


def to_abspath(path: Union[str, List[str], None], check_path_exist: bool = False) -> Union[str, List[str], None]:
    """将输入路径规范化为绝对路径，并可选校验其存在性。

    参数:
        path: 待校验/转换的路径，支持字符串、字符串列表或 None。
        check_path_exist: 是否检查路径存在性；为 True 时若不存在则抛出 FileNotFoundError。

    返回:
        - 若输入为 None：返回 None；
        - 若输入为 str：返回绝对路径字符串；
        - 若输入为 List[str]：返回等长的绝对路径列表。
    """
    # 若未提供路径，直接返回 None
    if path is None:
        return
    # 若为单个字符串路径
    elif isinstance(path, str):
        # Remove user path prefix and convert to absolute path.
        # 先展开 ~ 等用户目录符号，再转为绝对路径
        path = os.path.abspath(os.path.expanduser(path))
        # 如需校验存在性且路径不存在，抛出文件不存在错误
        if check_path_exist and not os.path.exists(path):
            raise FileNotFoundError(f"path: '{path}'")
        # 返回转换后的绝对路径
        return path
    # 若不是 None 或 str，则期望应为列表
    assert isinstance(path, list), f'path: {path}'
    # 用于收集每个元素转换后的绝对路径
    res = []
    # 遍历列表中每个路径元素并递归转换
    for v in path:
        res.append(to_abspath(v, check_path_exist))
    # 返回处理后的路径列表
    return res
