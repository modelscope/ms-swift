import numpy as np
import tyro
import logging

# 假设 websocket_vla_server.py 文件与此脚本位于同一目录
# 这个模块定义了 WebsocketPolicyServer 类，用于处理网络通信
from websocket_vla_server import WebsocketPolicyServer

class DebugPolicy:
    """
    一个用于服务器调试的虚拟策略（Policy）。
    
    这个策略会忽略所有输入的观测数据（observation），并始终返回一个
    固定的、预定义的7维默认动作。这用于测试需要单步动作的环境。
    """
    def __init__(self):
        """
        初始化调试策略，并定义默认的7维动作。
        """
        # 定义单个时间步的默认动作，这是一个7维的NumPy数组
        self.default_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
        
        logging.info("DebugPolicy 已初始化。")
        logging.info(f"将为每个请求发送以下默认动作 (shape: {self.default_action.shape}):\n{self.default_action}")

    def infer(self, obs: dict):
        """
        实现策略推断接口。
        
        参数:
            obs (dict): 从客户端接收的观测数据。在这个调试策略中，此参数将被忽略。
            
        返回:
            dict: 一个包含 "actions" 键的字典，其值为默认的7维动作数组。
        """
        logging.info("已接收到观测数据（在调试模式下将被忽略），正在返回默认的7维动作。")
        
        # 返回的格式必须与真实策略的输出格式一致
        return {"actions": self.default_action}

def main(
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    主函数，用于配置并启动带有调试策略的 WebSocket 服务器。
    
    参数:
        host (str): 服务器绑定的主机地址。 "0.0.0.0" 表示监听所有可用的网络接口。
        port (int): 服务器监听的端口号。
    """
    logging.info("正在使用 DebugPolicy 初始化服务器...")
    
    # 1. 创建 DebugPolicy 的实例
    policy = DebugPolicy()
    
    # 2. 创建 WebsocketPolicyServer 的实例，并将调试策略注入
    #    服务器将使用这个策略来响应客户端请求
    server = WebsocketPolicyServer(policy=policy, host=host, port=port)
    
    # 3. 启动服务器并开始监听连接
    logging.info(f"调试服务器已在 ws://{host}:{port} 上启动")
    server.serve_forever()

if __name__ == "__main__":
    # 配置日志记录，以便在控制台看到服务器的运行信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 使用 tyro 解析命令行参数并运行 main 函数
    tyro.cli(main)