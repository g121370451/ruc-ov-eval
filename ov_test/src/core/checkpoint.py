import os
import json
import hashlib
import threading
import copy
from datetime import datetime
from typing import Dict, Any, Optional, Set


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, config: Dict[str, Any]):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, "benchmark_checkpoint.json")
        self.config = config
        self.config_hash = self._compute_config_hash(config)
        self.legacy_config_hashes = self._compute_legacy_config_hashes(config)
        self._lock = threading.Lock()

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        config_copy = self._stable_config_for_hash(config)
        config_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()

    def _compute_legacy_config_hashes(self, config: Dict[str, Any]) -> Set[str]:
        """Return hashes produced by the previous full-config algorithm."""
        variants = [copy.deepcopy(config)]
        variants.extend(self._path_alias_variants(config))

        hashes = set()
        for variant in variants:
            config_copy = copy.deepcopy(variant)
            if 'execution' in config_copy:
                exec_config = config_copy['execution'].copy()
                exec_config.pop('worker_id', None)
                exec_config.pop('num_workers', None)
                config_copy['execution'] = exec_config
            config_str = json.dumps(config_copy, sort_keys=True)
            hashes.add(hashlib.md5(config_str.encode('utf-8')).hexdigest())
        return hashes

    def _path_alias_variants(self, config: Dict[str, Any]) -> list:
        """Handle legacy hashes created through /home vs /data00/home paths."""
        aliases = (
            ("/home/zhanggaoyuan.225", "/data00/home/zhanggaoyuan.225"),
            ("/data00/home/zhanggaoyuan.225", "/home/zhanggaoyuan.225"),
        )
        variants = []
        for old, new in aliases:
            variant = self._replace_string_prefixes(copy.deepcopy(config), old, new)
            if variant != config:
                variants.append(variant)
        return variants

    def _replace_string_prefixes(self, obj: Any, old: str, new: str) -> Any:
        if isinstance(obj, str):
            return obj.replace(old, new)
        if isinstance(obj, dict):
            return {k: self._replace_string_prefixes(v, old, new) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._replace_string_prefixes(item, old, new) for item in obj]
        return obj

    def _stable_config_for_hash(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Keep semantic fields only, excluding runtime and environment noise."""
        config_copy = copy.deepcopy(config)

        if 'paths' in config_copy:
            paths = config_copy['paths'].copy()
            paths.pop('log_file', None)
            paths.pop('output_dir', None)
            for key in ('raw_data', 'vector_store', 'doc_output_dir'):
                if key in paths and isinstance(paths[key], str):
                    paths[key] = os.path.realpath(paths[key])
            config_copy['paths'] = paths

        if 'execution' in config_copy:
            exec_config = config_copy['execution'].copy()
            for key in (
                'worker_id',
                'num_workers',
                'skip_ingestion',
                'query_group_workers',
                'max_workers',
                'save_frequency',
            ):
                exec_config.pop(key, None)
            config_copy['execution'] = exec_config

        store_config = config_copy.get('store')
        if isinstance(store_config, dict):
            lightrag_config = store_config.get('lightrag_config')
            if isinstance(lightrag_config, dict):
                for key in (
                    'llm_base_url',
                    'llm_api_key',
                    'embedding_base_url',
                    'embedding_api_key',
                    'rerank_ak_env',
                    'rerank_sk_env',
                    'max_parallel_insert',
                    'llm_model_max_async',
                    'embedding_func_max_async',
                ):
                    lightrag_config.pop(key, None)

            for nested_name in ('hipporag_config', 'sql_agent_config'):
                nested_config = store_config.get(nested_name)
                if isinstance(nested_config, dict):
                    self._drop_secret_and_runtime_fields(nested_config)

        llm_config = config_copy.get('llm')
        if isinstance(llm_config, dict):
            for key in ('base_url', 'api_key', 'api_key_env_var'):
                llm_config.pop(key, None)

        return config_copy

    def _drop_secret_and_runtime_fields(self, config: Dict[str, Any]) -> None:
        for key in list(config.keys()):
            value = config[key]
            lowered = key.lower()
            if (
                'api_key' in lowered
                or lowered.endswith('_ak')
                or lowered.endswith('_sk')
                or 'base_url' in lowered
                or 'max_async' in lowered
                or 'parallel' in lowered
                or 'worker' in lowered
            ):
                config.pop(key, None)
            elif isinstance(value, dict):
                self._drop_secret_and_runtime_fields(value)

    def _validate_config(self, checkpoint_data: Dict[str, Any]) -> bool:
        saved_hash = checkpoint_data.get("config_hash")
        if saved_hash and saved_hash != self.config_hash and saved_hash not in self.legacy_config_hashes:
            return False
        return True

    def checkpoint_exists(self) -> bool:
        return os.path.exists(self.checkpoint_file)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not self.checkpoint_exists():
            return None
        try:
            with self._lock:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
            if not self._validate_config(checkpoint_data):
                return None
            return checkpoint_data
        except Exception:
            return None

    def save_checkpoint(self, execution_state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        checkpoint_data = {
            "checkpoint_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "config_hash": self.config_hash,
            "execution_state": execution_state,
            "metadata": metadata or {}
        }
        try:
            with self._lock:
                if os.path.exists(self.checkpoint_file):
                    with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                    checkpoint_data["created_at"] = old_data.get("created_at", checkpoint_data["created_at"])
                with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def delete_checkpoint(self):
        with self._lock:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)

    def get_completed_tasks(self, step: str) -> Set[int]:
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return set()
        execution_state = checkpoint.get("execution_state", {})
        if execution_state.get("current_step") != step:
            return set()
        return set(execution_state.get("completed_tasks", []))

    def update_completed_tasks(self, step: str, completed_tasks: Set[int], total_tasks: int,
                               extra: Optional[Dict[str, Any]] = None):
        checkpoint = self.load_checkpoint()
        execution_state = {
            "current_step": step,
            "completed_tasks": sorted(completed_tasks),
            "total_tasks": total_tasks,
        }
        if checkpoint:
            old_state = checkpoint.get("execution_state", {})
            for key in ("ingested_samples", "ingest_stats"):
                if key in old_state:
                    execution_state[key] = old_state[key]
        if extra:
            execution_state.update(extra)
        metadata = {
            "dataset_name": self.config.get("dataset_name", "Unknown"),
            "output_dir": self.config.get("paths", {}).get("output_dir", "")
        }
        self.save_checkpoint(execution_state, metadata)

    def get_ingested_samples(self) -> Set[str]:
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return set()
        execution_state = checkpoint.get("execution_state", {})
        return set(execution_state.get("ingested_samples", []))

    def update_ingested_samples(self, ingested_samples: Set[str], total_samples: int,
                                ingest_stats: Optional[Dict[str, Any]] = None):
        checkpoint = self.load_checkpoint()
        completed_tasks = []
        total_tasks = 0
        current_step = "ingestion"
        if checkpoint:
            old_state = checkpoint.get("execution_state", {})
            completed_tasks = old_state.get("completed_tasks", [])
            total_tasks = old_state.get("total_tasks", 0)
            current_step = old_state.get("current_step", "ingestion")
            if not ingest_stats:
                ingest_stats = old_state.get("ingest_stats", {})
        execution_state = {
            "current_step": current_step,
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "ingested_samples": sorted(ingested_samples),
            "ingest_stats": ingest_stats or {},
        }
        metadata = {
            "dataset_name": self.config.get("dataset_name", "Unknown"),
            "output_dir": self.config.get("paths", {}).get("output_dir", "")
        }
        self.save_checkpoint(execution_state, metadata)

    def get_ingest_stats(self) -> dict:
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return {}
        return checkpoint.get("execution_state", {}).get("ingest_stats", {})
