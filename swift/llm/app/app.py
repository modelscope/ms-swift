"""模块功能概述：
该模块实现 Swift 框架的 Web UI 应用程序入口，提供基于 Gradio 的可视化交互界面。

Gradio 是一个用于快速将机器学习模型或函数封装成交互式网页界面的工具，可轻松实现模型的可视化、测试与分享。

核心功能：
1. SwiftApp 类：继承自 SwiftPipeline，封装了 Gradio UI 的启动和配置逻辑；
2. app_main 函数：应用程序入口函数，提供便捷的启动接口；
3. 自动部署后端推理服务（如未提供 base_url）或连接到现有服务；
4. 根据 Gradio 版本自动适配并发配置参数。

应用场景：
- 快速启动可视化对话界面，用于模型测试和演示；
- 支持多模态模型的交互式推理；
- 可配置服务端口、共享链接、系统提示等参数。

典型使用：
    # 命令行启动
    $ swift app --model_id_or_path qwen/Qwen-7B-Chat
    
    # 代码启动
    >>> from swift.llm.app import app_main
    >>> app_main(['--model_id_or_path', 'qwen/Qwen-7B-Chat', '--server_port', '7860'])
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
from contextlib import nullcontext  # 引入空上下文管理器，用于条件性地跳过上下文管理（当不需要部署后端时使用）
from typing import List, Optional, Union  # 引入类型注解，用于参数和返回值的类型提示

import gradio  # 引入 Gradio 库，用于构建和启动 Web UI 界面
from packaging import version  # 引入版本解析模块，用于比较 Gradio 版本号以适配不同版本的 API

from swift.utils import get_logger  # 引入日志工具函数，用于创建模块级日志记录器
from ..argument import AppArguments  # 引入应用程序参数类，定义了启动 UI 所需的所有配置参数
from ..base import SwiftPipeline  # 引入基础管道类，提供参数解析和主流程控制功能
from ..infer import run_deploy  # 引入推理服务部署函数，用于自动启动后端推理服务
from .build_ui import build_ui  # 引入 UI 构建函数，用于创建 Gradio 界面组件

logger = get_logger()  # 创建模块级日志记录器，用于输出运行时信息和错误日志


class SwiftApp(SwiftPipeline):
    """类功能：
    定义 Swift 应用程序类，继承自 SwiftPipeline 基类。
    负责启动和管理基于 Gradio 的 Web UI 应用程序。
    
    核心职责：
        1. 参数管理：使用 AppArguments 类型管理应用程序配置；
        2. 服务部署：根据配置自动部署后端推理服务或连接到现有服务；
        3. UI 构建：调用 build_ui 函数创建 Gradio 界面；
        4. 服务启动：配置并启动 Gradio 服务器，处理版本兼容性。
    
    继承关系：
        - 继承自 SwiftPipeline，复用参数解析、日志记录等基础功能。
    
    属性：
        - args_class: 指定参数类为 AppArguments；
        - args: 应用程序参数实例，包含模型路径、服务端口、UI 配置等。
    
    实际使用示例：
        示例 1：启动本地模型的 Web UI（自动部署后端）
        >>> app = SwiftApp(['--model_id_or_path', 'qwen/Qwen-7B-Chat', 
        ...                 '--server_port', '7860'])
        >>> app.main()  # 自动部署推理服务并启动 Gradio UI
        
        示例 2：连接到已运行的推理服务
        >>> app = SwiftApp(['--base_url', 'http://localhost:8000/v1',
        ...                 '--server_port', '7860'])
        >>> app.main()  # 直接连接到指定的推理服务，不部署新服务
        
        示例 3：配置多模态模型和自定义标题
        >>> app = SwiftApp(['--model_id_or_path', 'qwen/Qwen-VL-Chat',
        ...                 '--is_multimodal', 'True',
        ...                 '--studio_title', '我的模型助手',
        ...                 '--lang', 'zh'])
        >>> app.main()  # 启动支持图像输入的多模态 UI
    """
    args_class = AppArguments  # 指定该管道使用的参数类为 AppArguments，用于参数解析和验证
    args: args_class  # 类型注解：声明 args 属性的类型为 AppArguments 实例

    def run(self):
        """函数功能：
        主运行方法，执行应用程序的主要启动流程，包括后端服务部署（可选）、UI 构建和 Gradio 服务器启动。
        
        参数：
            无（使用 self.args 中的配置参数）
        
        返回值：
            None（方法会阻塞运行 Gradio 服务器，直到用户手动停止）
        
        执行流程：
            1. 判断是否需要自动部署后端推理服务（基于 args.base_url 是否为空）；
            2. 在部署上下文中获取后端服务 URL；
            3. 调用 build_ui 函数构建 Gradio 界面组件；
            4. 根据 Gradio 版本配置并发参数；
            5. 启动 Gradio 服务器并监听指定端口。
        
        实际使用示例：
            示例 1：完整启动流程（自动部署后端）
            >>> app = SwiftApp(['--model_id_or_path', 'qwen/Qwen-7B-Chat'])
            >>> app.run()  # 自动部署推理服务 -> 构建 UI -> 启动 Gradio 服务器
            # 访问 http://127.0.0.1:7860 可看到 Web 界面
            
            示例 2：连接到外部服务
            >>> app = SwiftApp(['--base_url', 'http://10.0.0.5:8000/v1'])
            >>> app.run()  # 直接使用外部推理服务 -> 构建 UI -> 启动 Gradio 服务器
        """
        args = self.args  # 从实例属性中获取应用程序参数对象（AppArguments 实例）
        deploy_context = nullcontext() if args.base_url else run_deploy(args, return_url=True)  # 条件性地创建部署上下文：若已提供 base_url（连接到现有服务），使用空上下文；否则调用 run_deploy 启动新的推理服务并返回其 URL
        with deploy_context as base_url:  # 进入部署上下文：若自动部署，base_url 为新服务的 URL；若使用 nullcontext，base_url 为 None
            base_url = base_url or args.base_url  # 确定最终的推理服务 URL：优先使用上下文返回的 base_url（自动部署的服务），若为 None 则使用参数中指定的 args.base_url
            demo = build_ui(  # 调用 build_ui 函数构建 Gradio 界面对象（Blocks 实例）
                base_url,  # 传入推理服务的基础 URL（如 'http://localhost:8000/v1'），UI 将向此地址发送推理请求
                args.model_suffix,  # 传入模型后缀名称（用于 UI 中显示或区分多个模型），如 ['chat', 'base']
                request_config=args.get_request_config(),  # 调用参数对象的方法获取请求配置（RequestConfig 实例），包含温度、top_p、最大 token 数等推理参数
                is_multimodal=args.is_multimodal,  # 传入是否为多模态模型的标志（布尔值），若为 True，UI 将显示图像上传组件
                studio_title=args.studio_title,  # 传入 UI 标题字符串（显示在页面顶部），默认为 'SWIFT Studio'
                lang=args.lang,  # 传入界面语言设置（如 'zh' 或 'en'），控制 UI 文本的显示语言
                default_system=args.system)  # 传入默认系统提示词字符串（如 'You are a helpful assistant'），用于初始化对话上下文
            concurrency_count = 1 if args.infer_backend == 'pt' else 16  # 根据推理后端类型设置并发数：若使用 PyTorch 后端（'pt'），设为 1（避免并发导致的显存问题）；其他后端（如 'vllm'）设为 16（支持高并发）
            if version.parse(gradio.__version__) < version.parse('4'):  # 检查 Gradio 版本号：若版本低于 4.0
                queue_kwargs = {'concurrency_count': concurrency_count}  # 使用旧版参数名 'concurrency_count'（Gradio 3.x 的 API）
            else:  # 否则（Gradio 版本 >= 4.0）
                queue_kwargs = {'default_concurrency_limit': concurrency_count}  # 使用新版参数名 'default_concurrency_limit'（Gradio 4.x 的 API）
            demo.queue(**queue_kwargs).launch(  # 为 Gradio 应用启用队列机制（使用适配后的并发参数）并调用 launch 方法启动服务器
                server_name=args.server_name, server_port=args.server_port, share=args.share)  # 传入服务器配置：server_name 为监听地址（如 '0.0.0.0' 允许外部访问），server_port 为端口号（如 7860），share 为是否创建公开分享链接（布尔值，需要 Gradio 服务）


def app_main(args: Optional[Union[List[str], AppArguments]] = None):
    """函数功能：
    应用程序主入口函数，接受命令行参数列表或 AppArguments 实例，封装了 SwiftApp 的实例化和主流程执行。
    
    参数：
        args (Optional[Union[List[str], AppArguments]]): 
            - 可选参数，支持三种形式：
              1. None: 从 sys.argv 读取命令行参数；
              2. List[str]: 命令行参数列表（如 ['--model_id_or_path', 'qwen/Qwen-7B-Chat']）；
              3. AppArguments: 已实例化的参数对象。
    
    返回值：
        None（函数会阻塞运行直到 Gradio 服务器停止）
    
    实际使用示例：
        示例 1：从命令行参数启动（脚本模式）
        >>> # 在 Python 脚本中
        >>> from swift.llm.app import app_main
        >>> if __name__ == '__main__':
        ...     app_main()  # 自动读取 sys.argv
        # 运行：python script.py --model_id_or_path qwen/Qwen-7B-Chat
        
        示例 2：使用参数列表启动（编程模式）
        >>> app_main(['--model_id_or_path', 'qwen/Qwen-7B-Chat',
        ...           '--server_port', '7860',
        ...           '--lang', 'zh',
        ...           '--studio_title', '智能助手'])
        # 启动中文界面的 Gradio 应用，监听 7860 端口
        
        示例 3：使用参数对象启动（高级模式）
        >>> from swift.llm.argument import AppArguments
        >>> args = AppArguments(
        ...     model_id_or_path='qwen/Qwen-7B-Chat',
        ...     server_port=7860,
        ...     infer_backend='vllm',
        ...     temperature=0.7
        ... )
        >>> app_main(args)  # 使用预配置的参数对象启动
        
        示例 4：连接到远程推理服务
        >>> app_main(['--base_url', 'http://10.0.0.5:8000/v1',
        ...           '--server_port', '7860'])
        # 不部署本地模型，直接连接到远程推理服务
    """
    return SwiftApp(args).main()  # 实例化 SwiftApp 类（传入参数对象或参数列表），并调用继承自 SwiftPipeline 的 main 方法（该方法会执行参数解析、日志配置、调用 run 方法等完整流程）
