import signal


class ShutdownManager:
    """
    用于捕获信号 (e.g. SIGTERM, SIGINT, custom) 并设置 shutdown flag。
    供训练 / 主流程 / callback 查询是否收到停止请求。
    """

    def __init__(self, signals=None):
        # signals: list of signal numbers to catch, e.g. [signal.SIGTERM, signal.SIGINT]
        if signals is None:
            signals = [signal.SIGUSR1, signal.SIGUSR2]
        self._signals = signals

        # 内部 flag + lock
        self._shutdown_requested = False
        # self._lock = threading.Lock()

    def _handler(self, signum, frame):
        # with self._lock:
        self._shutdown_requested = True

    def register(self):
        """ 注册信号处理函数 """
        for s in self._signals:
            signal.signal(s, self._handler)

    def should_shutdown(self) -> bool:
        """ 返回是否已请求 shutdown """
        # with self._lock:
        return self._shutdown_requested

    def reset(self):
        """ 重置 flag (如果你希望重用 manager) """
        # with self._lock:
        self._shutdown_requested = False
