# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass


@dataclass
class WebUIArguments:
    """A dataclass for web UI configuration arguments.

    Args:
        server_name (str): The hostname or IP address to be bound to the Web UI server.
            Defaults to '127.0.0.1' (localhost only) for security. Set to '0.0.0.0' to allow
            access from all network interfaces — this will make the Web UI accessible to anyone
            who can reach the server, which is dangerous without additional authentication.
        server_port (int): The port number to be bound to the Web UI server. Defaults to 7860.
        share (bool): Whether to create a public, shareable link for the web UI. Defaults to False.
        lang (str): The language for the web UI, chosen from {'zh', 'en'}. Defaults to 'zh'.
    """
    server_name: str = '127.0.0.1'
    server_port: int = 7860
    share: bool = False
    lang: str = 'zh'
