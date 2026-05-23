import threading

class ThreadLocalTokenTracker:
    """
    线程局部token追踪器。
    每个线程维护独立的计数器，reset/add/get操作互不干扰
    """
    def __init__(self):
        self._local = threading.local()

    def _ensure(self):
        if not hasattr(self._local, 'input_tokens'):
            self._local.input_tokens = 0
            self._local.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int):
        self._ensure()
        self._local.input_tokens += input_tokens
        self._local.output_tokens += output_tokens

    def reset(self):
        self._local.input_tokens = 0
        self._local.output_tokens = 0

    def get(self):
        self._ensure()
        return {
            "input_tokens": self._local.input_tokens,
            "output_tokens": self._local.output_tokens
        }
        
# 全局token追踪器实例
token_tracker = ThreadLocalTokenTracker()