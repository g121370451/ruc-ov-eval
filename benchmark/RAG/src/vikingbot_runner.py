#!/usr/bin/env python3

import os
import sys
import time
import json
import uuid
import random
import string
import hashlib
import subprocess
import atexit
import urllib.request
import urllib.error
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


sys.path.append(str(Path(__file__).parent))

from core.logger import get_logger

logger = get_logger()

_OV_CONF_PATH = str((Path(__file__).parent.parent / "ov.conf").resolve())
_OPENVIKING_SERVER_PROCESS: Optional[subprocess.Popen] = None
_CURRENT_OV_CONF_PATH: Optional[str] = None
_SERVER_LOCK = threading.Lock()


def _generate_temp_ov_conf(original_conf_path: str, vector_store_path: str) -> str:
    """
    Generate a temporary ov.conf file with the specified vector store path.
    
    Args:
        original_conf_path: Path to the original ov.conf file
        vector_store_path: Path to the vector store directory
        
    Returns:
        Path to the temporary ov.conf file
    """
    # Read original config
    with open(original_conf_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Ensure root_api_key is a string (not null) for Pydantic validation
    if 'server' not in config:
        config['server'] = {}
    if config['server'].get('root_api_key') is None:
        config['server']['root_api_key'] = ""
    
    # Update storage workspace to point to vector store
    if 'storage' not in config:
        config['storage'] = {}
    config['storage']['workspace'] = vector_store_path
    
    # Create temporary config file with stable name based on vector_store_path
    temp_dir = Path(__file__).parent.parent / ".temp"
    temp_dir.mkdir(exist_ok=True)
    
    # 使用 vector_store_path 的哈希值作为稳定的文件名
    # 这样相同的 vector_store 会使用同一个临时配置文件
    vector_store_path_bytes = vector_store_path.encode('utf-8')
    path_hash = hashlib.md5(vector_store_path_bytes).hexdigest()
    temp_conf_path = str(temp_dir / f"ov_{path_hash}.conf")
    
    # 只有当文件不存在或者配置内容变化时才重新写入
    need_write = True
    if os.path.exists(temp_conf_path):
        try:
            with open(temp_conf_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            if existing_config == config:
                need_write = False
        except Exception:
            need_write = True
    
    if need_write:
        # Write temporary config
        with open(temp_conf_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    return temp_conf_path


def _healthcheck(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _load_server_url_and_key(ov_conf_path: str) -> tuple[str, str]:
    with open(ov_conf_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    server = data.get("server", {}) if isinstance(data, dict) else {}
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 1933)
    api_key = server.get("root_api_key", "") or ""
    return f"http://{host}:{port}", api_key


def _stop_openviking_server() -> None:
    global _OPENVIKING_SERVER_PROCESS, _CURRENT_OV_CONF_PATH
    proc = _OPENVIKING_SERVER_PROCESS
    # Only consider "killing all servers" if this runner started one in this process.
    started_by_us = bool(proc) or bool(_CURRENT_OV_CONF_PATH)
    _OPENVIKING_SERVER_PROCESS = None
    _CURRENT_OV_CONF_PATH = None
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    
    # Optional safety measure: kill all openviking-server processes.
    #
    # This is intentionally disabled by default to avoid killing a server that was started
    # outside of this benchmark process (e.g. a developer's local OV server).
    # To enable it, set `OPENVIKING_BENCH_KILL_ALL_SERVERS=1`.
    #
    # We also only do this if this process started an OV server at least once.
    kill_all = os.environ.get("OPENVIKING_BENCH_KILL_ALL_SERVERS", "").strip() in ("1", "true", "True")
    if not (kill_all and started_by_us):
        return

    # 额外的安全措施：杀死所有 openviking-server 进程
    try:
        if sys.platform == "darwin" or sys.platform.startswith("linux"):
            # 使用 pgrep 和 pkill 在 macOS 和 Linux 上
            result = subprocess.run(
                ["pgrep", "-f", "openviking-server"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Found openviking-server processes: {result.stdout.strip()}")
                subprocess.run(["pkill", "-f", "openviking-server"], capture_output=True)
                time.sleep(1)
    except Exception as e:
        logger.debug(f"Failed to kill all openviking-server processes: {e}")


atexit.register(_stop_openviking_server)


def _ensure_openviking_server(ov_conf_path: str) -> None:
    global _OPENVIKING_SERVER_PROCESS, _CURRENT_OV_CONF_PATH

    with _SERVER_LOCK:
        server_url, api_key = _load_server_url_and_key(ov_conf_path)
        health_url = f"{server_url}/health"

        # 检查是否已经有正确配置的服务器在运行
        if (_CURRENT_OV_CONF_PATH == ov_conf_path and 
            _OPENVIKING_SERVER_PROCESS and 
            _OPENVIKING_SERVER_PROCESS.poll() is None and 
            _healthcheck(health_url)):
            # 已有正确配置的服务器在运行，直接返回
            return

        # 需要启动新服务器，先停止旧的
        _stop_openviking_server()
        _CURRENT_OV_CONF_PATH = None

        # 确保旧进程真的被终止
        time.sleep(0.5)

        # 启动新服务器
        env = os.environ.copy()
        env["OPENVIKING_CONFIG_FILE"] = ov_conf_path
        _OPENVIKING_SERVER_PROCESS = subprocess.Popen(
            ["openviking-server", "--config", ov_conf_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        _CURRENT_OV_CONF_PATH = ov_conf_path

        # 等待服务器健康
        deadline = time.time() + 20
        while time.time() < deadline:
            if _OPENVIKING_SERVER_PROCESS.poll() is not None:
                raise RuntimeError("openviking-server exited unexpectedly")
            if _healthcheck(health_url):
                return
            time.sleep(0.3)

        raise RuntimeError("openviking-server did not become healthy in time")


def _build_vikingbot_env(ov_conf_path: str, max_iterations: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENVIKING_CONFIG_FILE"] = ov_conf_path
    env["NANOBOT_AGENTS__MAX_TOOL_ITERATIONS"] = str(int(max_iterations))
    
    # 设置 ovcli.conf 的路径，和原始 ov.conf 在同一个目录（不是临时文件的目录）
    original_ov_conf_dir = os.path.dirname(_OV_CONF_PATH)
    ovcli_conf_path = os.path.join(original_ov_conf_dir, "ovcli.conf")
    if os.path.exists(ovcli_conf_path):
        env["OPENVIKING_CLI_CONFIG_FILE"] = ovcli_conf_path
        logger.debug(f"Set OPENVIKING_CLI_CONFIG_FILE to: {ovcli_conf_path}")
    
    # Read API key from ov.conf
    try:
        with open(ov_conf_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if 'vlm' in config and 'api_key' in config['vlm']:
            api_key = config['vlm']['api_key']
            env['OPENAI_API_KEY'] = api_key
            logger.debug("Set OPENAI_API_KEY from ov.conf")
    except Exception as e:
        logger.warning(f"Failed to read API key from ov.conf: {e}")
    
    return env


class VikingBotRunner:
    """
    Wrapper for VikingBot to support Agentic RAG evaluation.
    
    This class provides a simple interface to run VikingBot
    for generating answers using Agentic RAG approach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VikingBotRunner.
        
        Args:
            config: Configuration dictionary containing vikingbot settings
        """
        self.config = config
        self.vikingbot_config = config.get('vikingbot', {})
        self.max_iterations = self.vikingbot_config.get('max_iterations', 50)
        self.log_tool_calls = self.vikingbot_config.get('log_tool_calls', True)
        # Get vector store path from config if available
        self.vector_store_path = config.get('paths', {}).get('vector_store')
    
    def generate_answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        allowed_target_uris: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer using VikingBot via CLI.
        
        Args:
            question: The question to answer
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing answer, tool calls, and timing info
        """
        session_id = session_id or f"eval_{uuid.uuid4().hex}"
        
        start_time = time.time()
        
        try:
            # Use temporary config if vector store path is specified
            ov_conf_path = _OV_CONF_PATH
            temp_conf_path = None
            if self.vector_store_path:
                temp_conf_path = _generate_temp_ov_conf(_OV_CONF_PATH, self.vector_store_path)
                ov_conf_path = temp_conf_path
                logger.info(f"Using vector store: {self.vector_store_path}")
            
            _ensure_openviking_server(ov_conf_path)
            
            allowed_target_uris = None

            if allowed_target_uris:
                allowed_block = "\n".join(f"- {u}" for u in allowed_target_uris)
                scope_line = (
                    "Always search ONLY within the following directories:\n"
                    f"{allowed_block}\n"
                    "Do not access any other URI.\n"
                    "\n"
                    "Efficiency tip:\n"
                    "- Prefer openviking_search in the allowed directory. This is usually more efficient than layer-by-layer grep.\n"
                )
            else:
                scope_line = (
                    "Always search in viking://resources/ path.\n"
                )
            input_msg = (
                "Answer this question as briefly as possible. Use only the information available in the database. "
                "Do not use web search or any external source. "
                + scope_line
                + f"\n\nQuestion: {question}"
            )
            env = _build_vikingbot_env(ov_conf_path, self.max_iterations)
            cmd = [
                "vikingbot",
                "chat",
                "-m",
                input_msg,
                "-s",
                session_id,
                "-e",
                "--no-markdown",
                "-c",
                ov_conf_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
                env=env,
            )
            stdout = (result.stdout or "").strip()
            
            json_start = stdout.rfind('{"text"')
            if json_start == -1:
                raise ValueError(f"No JSON output found in vikingbot stdout (len={len(stdout)})")
            
            import re
            raw_json = stdout[json_start:]
            raw_json = re.sub(r'[\x00-\x1f\x7f]', ' ', raw_json)
            resp_json, _ = json.JSONDecoder().raw_decode(raw_json)
            
            result_dict = {
                "answer": resp_json.get("text", "") or "",
                "total_time_sec": float(resp_json.get("time_cost", time.time() - start_time)),
                "token_usage": resp_json.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "tools_used_names": resp_json.get("tools_used_names") or [],
                "iterations_used": int(resp_json.get("iteration") or 0),
            }
            
            # 不删除临时配置文件，因为其他任务可能还在使用
            # 使用相同 vector_store 的任务会共享同一个临时配置文件
            
            logger.info(f"VikingBot answer generated in {result_dict['total_time_sec']:.2f}s")
            return result_dict
            
        except Exception as e:
            logger.error(f"Error generating answer with VikingBot: {e}")
            return {
                "answer": f"[ERROR] {str(e)}",
                "total_time_sec": time.time() - start_time,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }


def stop_openviking_server() -> None:
    """
    Stop the currently running OpenViking server.
    This function is exposed for pipeline.py to call.
    """
    _stop_openviking_server()


def run_vikingbot_query(
    question: str,
    config: Dict[str, Any],
    session_id: Optional[str] = None,
    allowed_target_uris: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for VikingBotRunner.generate_answer.
    
    Args:
        question: The question to answer
        config: Configuration dictionary
        session_id: Optional session identifier
        
    Returns:
        Dictionary containing answer and metadata
    """
    runner = VikingBotRunner(config)
    return runner.generate_answer(question, session_id, allowed_target_uris=allowed_target_uris)
