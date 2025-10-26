# vla0_client_policy.py
import logging
import time
from typing import Dict, Any

import websockets.sync.client

import msgpack_numpy

class VLA0ClientPolicy:
    """
    The VLA-0 WebSocket Client Policy for communicating with the VLA-0 policy server.
    This class handles sending observations to the server and receiving inferred actions.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws = self._wait_for_server()

    def _wait_for_server(self) -> websockets.sync.client.ClientConnection:
        logging.info(f"Waiting for server: {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                logging.info("Successfully connected to server!")
                return conn
            except ConnectionRefusedError:
                logging.info("Connection refused, retrying in 5 seconds...")
                time.sleep(5)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Send observation data to the server and return the inferred actions."""
        try:
            data = self._packer.pack(obs)
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                # If the response is a string, it indicates an error message
                raise RuntimeError(f"Server returned error:\n{response}")
            return msgpack_numpy.unpackb(response)
        except websockets.exceptions.ConnectionClosed:
            logging.error("Connection to server closed, attempting to reconnect...")
            self._ws = self._wait_for_server()
            # Resend the request
            return self.infer(obs)

    def reset(self) -> None:
        """Reset the client policy state if necessary."""
        pass