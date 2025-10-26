# websocket_vla_server.py
import asyncio
import logging
import traceback
import websockets.asyncio.server
import websockets.frames
import msgpack_numpy

# --- Websocket 服务器 (修改自 OpenPI, 移除其依赖) ---
class WebsocketPolicyServer:
    def __init__(self, policy, host: str, port: int):
        self._policy = policy
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        logging.info(f"服务器启动于 ws://{self._host}:{self._port}")
        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"客户端 {websocket.remote_address} 已连接")
        packer = msgpack_numpy.Packer()

        try:
            while True:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                action = self._policy.infer(obs)
                await websocket.send(packer.pack(action))
        except websockets.ConnectionClosed:
            logging.info(f"客户端 {websocket.remote_address} 已断开")
        except Exception:
            error_msg = traceback.format_exc()
            logging.error(f"发生错误: {error_msg}")
            await websocket.send(error_msg)
            await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR)
            raise