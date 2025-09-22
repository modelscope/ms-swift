"""
模块功能
-------
定义用于 Web UI 启动与展示的基础参数类 `WebUIArguments`，包括服务绑定地址、端口、是否公开分享
以及默认语言设置。该模块仅承载参数，不包含运行逻辑，便于在 CLI/脚本/配置文件中统一传递。

典型用法
-------
>>> args = WebUIArguments(server_name='0.0.0.0', server_port=7860, share=False, lang='zh')
>>> # 上层启动 Web UI 服务时读取这些参数以完成绑定与界面语言配置
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明
from dataclasses import dataclass  # 引入 dataclass 装饰器，便于声明仅含参数的数据类


@dataclass  # 数据类装饰器：自动生成 __init__/__repr__ 等方法
class WebUIArguments:
    """
    类说明
    -----
    Web UI 启动参数类，统一管理前端服务的绑定配置与展示语言。

    字段
    ----
    - server_name: 服务器绑定的主机名或 IP，'0.0.0.0' 表示对外网可见。
    - server_port: 服务监听端口，常用为 7860。
    - share: 是否开启公开分享（例如通过第三方隧道外网访问）。
    - lang: 界面语言，默认中文 'zh'。

    示例
    ---
    >>> WebUIArguments(server_name='127.0.0.1', server_port=8080, share=True, lang='en')
    WebUIArguments(server_name='127.0.0.1', server_port=8080, share=True, lang='en')
    """
    server_name: str = '0.0.0.0'  # 绑定的主机名或 IP 地址
    server_port: int = 7860  # Web UI 服务监听的端口号
    share: bool = False  # 是否对外分享（通常用于临时公网访问）
    lang: str = 'zh'  # Web UI 的默认语言（'zh'/'en' 等）
