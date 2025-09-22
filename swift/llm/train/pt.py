# Copyright (c) Alibaba, Inc. and its affiliates.
"""
脚本用途:
- 定义 `SwiftPt` 训练类，继承 `SwiftSft`，用于 PT（预训练/非对话 SFT）场景。
- 在模板准备阶段统一关闭对话模板、并将损失缩放应用到所有 token（loss_scale='all'）。
- 提供 `pt_main` 作为模块级简化训练入口，便于通过参数启动训练流程。

主要组件:
- 类 `SwiftPt`: 定制训练配置与模板准备逻辑。
- 函数 `pt_main`: 从入参构建 `SwiftPt` 实例并执行主流程。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union  # 导入类型提示：列表、可选与联合类型

from swift.utils import get_logger  # 导入日志记录器工厂函数
from ..argument import TrainArguments  # 导入训练参数数据结构
from .sft import SwiftSft  # 导入基类：提供通用 SFT 训练流程

logger = get_logger()  # 初始化模块级日志记录器


class SwiftPt(SwiftSft):  # 定义 SwiftPt 类，继承 SwiftSft，定制 PT 训练行为
    """SwiftPt 训练类

    作用:
        - 在 PT（预训练/非对话 SFT）场景下，覆盖并完善模板准备逻辑。
        - 统一关闭对话模板、设置损失缩放方式，并复用父类的通用训练流程。

    属性:
        args_class: 指定该类使用的参数类型（`TrainArguments`）。
        args: 运行期注入的训练参数实例，类型为 `args_class`。
    """

    args_class = TrainArguments  # 指定参数类型为 TrainArguments
    args: args_class  # 类型注解：实例属性 args 的类型为 args_class

    def _prepare_template(self) -> None:
        """准备训练模板配置。

        动作:
            - 关闭聊天模板（use_chat_template=False），适配非对话式训练。
            - 将损失缩放应用到所有 token（loss_scale='all'）。
            - 记录配置变更日志，并调用父类以完成其余准备工作。

        返回:
            None
        """
        self.args.use_chat_template = False  # 关闭对话模板，适配非对话式训练
        self.args.loss_scale = 'all'  # 将损失计算扩展到所有 token
        logger.info('Setting args.use_chat_template: False')  # 记录配置变更：关闭聊天模板
        logger.info("Setting args.loss_scale: 'all'")  # 记录配置变更：损失缩放应用于全部 token
        super()._prepare_template()  # 调用父类方法，完成其余模板准备流程


def pt_main(args: Optional[Union[List[str], TrainArguments]] = None):  # 模块级训练入口函数
    """PT 训练入口。

    参数:
        args: 训练参数，可为命令行参数列表（List[str]）或 `TrainArguments` 实例；
              为空时从环境或默认配置中解析。

    返回:
        任意类型。通常为训练主流程的返回值（由 `SwiftSft.main` 定义）。
    """
    return SwiftPt(args).main()  # 基于传入参数构建 SwiftPt 实例并执行主流程
