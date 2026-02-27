import logging
import os


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 配置 Logger
    logger = logging.getLogger("Benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除旧 handler 避免重复

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    # 文件 Handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台 Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("Benchmark")
