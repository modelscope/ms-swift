import os
from typing import TYPE_CHECKING, Optional

import safetensors.torch
import torch

from swift.llm import deep_getattr, get_multimodal_target_regex
from swift.plugin import Tuner, extra_tuners
from swift.tuners import LoraConfig, Swift
from swift.utils import get_logger

logger = get_logger()
if TYPE_CHECKING:
    from swift.llm import TrainArguments


def is_vit_param(model_arch, parameter_name: str) -> bool:
    """
    判断给定参数名是否属于视觉分支（ViT）或对齐器（aligner）相关模块。

    函数通过在完整参数名中查找各模块前缀（字符串），判断该参数
    是否属于 `model_arch.vision_tower` 或 `model_arch.aligner` 的任一子模块。

    Args:
        model_arch: 模型结构描述对象，应包含 `vision_tower` 与 `aligner` 两个可迭代的模块前缀列表。
        parameter_name (str): 完整参数名，例如 `model.vision_tower.blocks.0.attn.q_proj.weight`。

    Returns:
        bool: 若参数属于视觉分支或对齐器模块，返回 True；否则返回 False。
    """
    for module_prefix in model_arch.vision_tower + model_arch.aligner:
        if f'.{module_prefix}.' in parameter_name:
            return True
    return False


class CustomTuner(Tuner):
    """Full-parameter training of ViT while LoRA training LLM"""

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        """
        从给定路径加载 Swift/LoRA 权重，并额外加载 ViT 全量权重后合并到模型。

        该方法首先使用 `Swift.from_pretrained` 恢复（通常是 LLM 上的）LoRA 等可训练权重，
        随后从 `model_id/vit.safetensors` 读取 ViT 的全量参数并以 `strict=False` 合并，
        从而实现“LLM 走 LoRA、ViT 走全参”的权重恢复流程。

        Args:
            model (torch.nn.Module): 待加载权重的模型实例。
            model_id (str): 已保存权重的目录路径（其中应包含 `vit.safetensors`）。
            **kwargs: 传递给 `Swift.from_pretrained` 的其他关键字参数。

        Returns:
            torch.nn.Module: 合并权重后的模型实例。
        """
        # 先恢复 Swift/LoRA 等权重（通常作用在 LLM 部分）
        model = Swift.from_pretrained(model, model_id, **kwargs)
        # 读取 ViT 全量权重文件（要求保存在 model_id/vit.safetensors）
        state_dict = safetensors.torch.load_file(os.path.join(model_id, 'vit.safetensors'))
        # 合并 ViT 参数到当前模型；strict=False 允许仅覆盖存在的键
        model.load_state_dict(state_dict, strict=False)
        # 返回加载完成的模型
        return model

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """
        保存当前可训练参数，并额外导出 ViT 专属权重为 `vit.safetensors`。

        若未显式传入 `state_dict`，则仅收集 `requires_grad=True` 的参数进行保存，
        这与本插件的训练策略一致（LLM 使用 LoRA，仅部分权重可训练；ViT 全参可训练）。
        除了常规的 `model.save_pretrained` 导出外，本方法还会筛选出属于 ViT/aligner 的参数，
        另存为 `vit.safetensors`，以便后续在 `from_pretrained` 时单独合并。

        Args:
            model (torch.nn.Module): 待保存的模型实例。
            save_directory (str): 保存目录。
            state_dict (Optional[dict]): 要保存的权重字典；若为 None，则根据 `requires_grad` 自动收集。
            safe_serialization (bool): 是否使用 safetensors 安全序列化。
            **kwargs: 传递给 `model.save_pretrained` 的其他关键字参数。

        Returns:
            None
        """
        # 若未传入 state_dict，则仅收集当前可训练参数（requires_grad=True）
        if state_dict is None:
            state_dict = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    state_dict[n] = p.detach().cpu()
        # 常规导出（含 LoRA 等），保留与平台兼容的安全序列化
        model.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization, **kwargs)
        # vit
        # 仅筛出 ViT/aligner 相关参数，单独导出为 vit.safetensors，便于后续独立加载合并
        model_arch = model.model_meta.model_arch
        state_dict = {k: v for k, v in state_dict.items() if is_vit_param(model_arch, k)}
        safetensors.torch.save_file(
            state_dict, os.path.join(save_directory, 'vit.safetensors'), metadata={'format': 'pt'})

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        """
        准备训练用模型：对 LLM 注入 LoRA，对 ViT/aligner 开启全参训练。

        该方法根据多模态模型结构与自动匹配到的目标正则，构造 LoRA 配置并注入到 LLM；
        同时将视觉塔（vision_tower）与对齐器（aligner）模块整体设置为 `requires_grad=True`，
        实现“ViT 全参训练 + LLM LoRA”的训练策略。

        Args:
            args (TrainArguments): 训练参数对象，需包含 `lora_rank`、`lora_alpha` 等字段。
            model (torch.nn.Module): 原始模型实例。

        Returns:
            torch.nn.Module: 注入 LoRA 并启用 ViT/aligner 全参训练后的模型。
        """
        # 读取模型结构描述，用于定位视觉塔与对齐器模块
        model_arch = model.model_meta.model_arch
        # 自动检索多模态 LLM 需要注入 LoRA 的目标模块（以正则表达式表达）
        target_regex = get_multimodal_target_regex(model)
        # 打印日志，便于调试查看 LoRA 目标模块范围
        logger.info(f'target_regex: {target_regex}')
        # 构造 LoRA 配置，仅作用于 LLM 目标模块
        lora_config = LoraConfig(
            task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=target_regex)
        # 将 LoRA 注入模型
        model = Swift.prepare_model(model, lora_config)
        # 将视觉塔与对齐器模块设置为全参可训练
        for module_prefix in model_arch.vision_tower + model_arch.aligner:
            deep_getattr(model, module_prefix).requires_grad_(True)
        # 返回准备好的模型
        return model


extra_tuners['custom'] = CustomTuner
