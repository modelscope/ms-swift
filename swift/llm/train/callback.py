"""
脚本用途:
- 定义基于 Hugging Face Trainer 的训练回调，用于在特定训练策略下动态调整行为。
- 包含两类回调：
  1) TrainerAdapterCallback：在 AdaLoRA 场景下设置总步数，并在每步前动态更新/分配（monkey-patch zero_grad）。
  2) DynamicLayerActivationCallback：按步间隔随机激活部分网络层的梯度，降低显存占用并进行层级轮换训练。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import types  # 标准库：用于将函数绑定为对象方法（types.MethodType）

import numpy as np  # 第三方：用于随机选择层索引
import torch  # 第三方：PyTorch 深度学习框架
from transformers import TrainerCallback  # 第三方：HF Trainer 的回调基类

from swift.utils import get_logger  # 项目工具：获取日志记录器

logger = get_logger()  # 初始化模块级日志记录器


class TrainerAdapterCallback(TrainerCallback):  # 训练器适配回调：适配 AdaLoRA 等场景
    """训练器适配回调。

    作用:
        - 在训练开始时为 AdaLoRA 设置总训练步数（total_step）。
        - 对模型的 `zero_grad` 进行 monkey-patch，使其在每步前调用 `update_and_allocate` 实现动态分配。

    适用场景:
        - 当 `args.train_type == 'adalora'` 时生效。
    """

    def __init__(self, args):
        """初始化回调。

        参数:
            args: 训练参数对象，需包含 `train_type` 等字段。
        """
        self.global_step = 0  # 记录当前全局步数，供动态分配时使用
        self.args = args  # 保存训练参数

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, _args, state, control, **kwargs):
        """训练开始时的回调。

        参数:
            _args: Trainer 的训练参数。
            state: TrainerState，包含状态信息（如 max_steps）。
            control: TrainerControl，用于影响训练流程（未使用）。
            **kwargs: 额外关键字参数，这里需要从中取得 `model`。
        """
        model = kwargs['model']  # 取得当前训练的模型实例
        if self.args.train_type == 'adalora':  # 仅在 AdaLoRA 场景下执行
            model.peft_config['default'].total_step = state.max_steps  # 设置总步数，供 AdaLoRA 调度使用

            def zero_grad(_self, *args, **kwargs):
                """替换模型的 zero_grad：先进行动态更新/分配，再调用原始 zero_grad。"""
                _self.update_and_allocate(self.global_step + 1)  # 基于下一步步数执行动态分配
                _self._zero_grad(*args, **kwargs)  # 调用原始 zero_grad 行为

            model._zero_grad = model.zero_grad  # 备份原始 zero_grad
            model.zero_grad = types.MethodType(zero_grad, model)  # 绑定并替换为自定义 zero_grad 方法

    def on_step_end(self, _args, state, control, **kwargs):
        """每步结束时更新全局步数（AdaLoRA 场景）。"""
        if self.args.train_type == 'adalora':  # 仅在 AdaLoRA 时需要记录步数
            self.global_step = state.global_step  # 同步当前全局步数


class DynamicLayerActivationCallback(TrainerCallback):  # 动态层激活回调：按间隔随机激活部分层
    """动态层激活回调。

    作用:
        - 以固定步间隔在模块列表（ModuleList）中随机选择若干层，仅对这些层开启梯度以进行训练。
        - 其它层梯度关闭以节省显存，并实现层级轮换训练（如 LISA 场景）。

    参数:
        n_layers: 每次需要激活的层数。
        step_interval: 进行层切换的步数间隔。
        model: 目标模型，需包含一个可识别的 `torch.nn.ModuleList` 子模块。
    """

    def __init__(self, n_layers: int, step_interval: int, model: torch.nn.Module):
        """初始化回调并确定需轮换的层集合。"""
        super().__init__()  # 初始化基类 TrainerCallback
        self.n_layers = n_layers  # 每次激活的层数
        self.step_interval = step_interval  # 层切换的步数间隔
        self.model = model  # 目标模型
        layers_name = None  # 将要记录的模块列表属性名
        layers = None  # 将要记录的模块列表对象
        for name, module in model.named_modules():  # 遍历命名子模块
            if isinstance(module, torch.nn.ModuleList):  # 寻找第一个 ModuleList 作为层集合
                layers_name = name  # 记录属性名
                layers = module  # 记录模块列表
                break  # 找到后立即停止
        assert layers_name is not None  # 必须找到用于轮换的层集合
        self.layers_attribute = layers_name  # 保存层集合的属性路径
        self.total_layers = len(layers)  # 记录层总数

        # Freeze all layers upon initialization
        self.freeze_all_layers()  # 初始化时冻结所有层的梯度
        self.active_layers_indices = []  # 当前激活层的索引列表

    def freeze_all_layers(self):
        """冻结所有层的梯度（requires_grad=False）。"""
        layers = self.model.get_submodule(self.layers_attribute)  # 获取层集合（ModuleList）
        for layer in layers:  # 遍历每一层
            for param in layer.parameters():  # 遍历层内参数
                param.requires_grad = False  # 关闭梯度

    def on_step_begin(self, args, state, control, **kwargs):
        """每步开始前按设定间隔切换激活层（包含起始步）。"""
        # Check if it's time to switch active layers, including at step 0
        if state.global_step % self.step_interval == 0 or state.global_step == 1:  # 到达切换步或初始化后第一次
            self.switch_active_layers()  # 执行层切换

    def switch_active_layers(self):
        """随机选择并激活若干层的梯度，其它层保持冻结。"""
        # First, disable gradients for all layers
        self.freeze_all_layers()  # 切换前先全部冻结，保证只有选中层被激活

        # Randomly select n_layers to activate
        layers = self.model.get_submodule(self.layers_attribute)  # 获取层集合
        self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False)  # 随机选层索引
        # Enable gradients only for the selected layers
        for idx in self.active_layers_indices:  # 遍历被选中的层索引
            for param in layers[idx].parameters():  # 遍历层内参数
                param.requires_grad = True  # 仅为选中层开启梯度
