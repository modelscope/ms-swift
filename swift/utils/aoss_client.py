import os
from aoss_client.client import Client as AOSSClient

__all__ = ['client']

_AOSS_CLIENT = None



def _get_aoss_client() -> AOSSClient:
    global _AOSS_CLIENT
    if _AOSS_CLIENT is not None:
        return _AOSS_CLIENT
    default_conf = os.path.expanduser('~/aoss.conf')
    conf_path = os.getenv('AOSS_CONF', default_conf)
    conf_path = os.path.abspath(os.path.expanduser(conf_path))
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f'AOSS 配置文件不存在: {conf_path}. 请创建该文件或通过环境变量 AOSS_CONF 指定配置路径。')
    _AOSS_CLIENT = AOSSClient(conf_path)
    return _AOSS_CLIENT

client = _get_aoss_client()



