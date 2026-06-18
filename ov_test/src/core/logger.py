import logging
import os


class _DropHttpxRequestInfo(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "httpx" or record.levelno != logging.INFO:
            return True
        return not record.getMessage().startswith("HTTP Request:")


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 配置 Logger
    logger = logging.getLogger("Benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除旧 handler 避免重复
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    # 文件 Handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.addFilter(_DropHttpxRequestInfo())
    logger.addHandler(fh)

    # 控制台 Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.addFilter(_DropHttpxRequestInfo())
    logger.addHandler(sh)

    # Also capture module loggers from MoDora/OpenAI wrappers. Benchmark logger
    # does not receive those records unless the root logger has handlers.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("Benchmark")
