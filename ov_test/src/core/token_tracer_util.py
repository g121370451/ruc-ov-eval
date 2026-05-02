class TokenTracker:
    """追踪ChatAPI调用过程中的输入/输出token消耗（线程安全）"""
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int):
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def reset(self):
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0

    def get(self):
        with self._lock:
            return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
        
# 全局token追踪器实例
token_tracker = TokenTracker()