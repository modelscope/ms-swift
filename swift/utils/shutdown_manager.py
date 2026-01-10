import signal


class ShutdownManager:

    def __init__(self, signals=None):
        if signals is None:
            signals = [signal.SIGUSR1, signal.SIGUSR2]
        self._signals = signals

        self._shutdown_requested = False

    def _handler(self, signum, frame):
        self._shutdown_requested = True

    def register(self):
        for s in self._signals:
            signal.signal(s, self._handler)

    def should_shutdown(self) -> bool:
        return self._shutdown_requested

    def reset(self):
        self._shutdown_requested = False
