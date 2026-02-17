import os
import signal


class ShutdownManager:

    def __init__(self, signals=None, stop_file=None):
        if signals is None:
            signals = [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2]
        self._signals = signals
        self._stop_file = stop_file or '/tmp/stop'

        self._shutdown_requested = False
        self._old_handlers = {}

    def _handler(self, signum, frame):
        self._shutdown_requested = True

    def register(self):
        for s in self._signals:
            self._old_handlers[s] = signal.getsignal(s)
            signal.signal(s, self._handler)

    def unregister(self):
        for s, handler in self._old_handlers.items():
            signal.signal(s, handler)
        self._old_handlers = {}

    def should_shutdown(self) -> bool:
        if self._shutdown_requested:
            return True
        return os.path.exists(self._stop_file)

    def reset(self):
        self._shutdown_requested = False
