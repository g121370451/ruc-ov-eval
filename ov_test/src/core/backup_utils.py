import os
import shutil
from datetime import datetime


def backup_store(store_path, logger=None):
    """备份 store 目录，返回备份路径。目录为空或不存在时跳过。"""
    if not os.path.exists(store_path):
        if logger:
            logger.info(f"Store path does not exist, skipping backup: {store_path}")
        return None

    if not os.listdir(store_path):
        if logger:
            logger.info(f"Store path is empty, skipping backup: {store_path}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{store_path}_backup_{timestamp}"
    shutil.copytree(store_path, backup_path)

    if logger:
        logger.info(f"Store backed up: {store_path} -> {backup_path}")

    return backup_path
