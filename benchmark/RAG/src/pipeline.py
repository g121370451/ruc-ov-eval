import os
import json
import time
import uuid
import random
import re
import hashlib
import threading
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import sys
from typing import Set

sys.path.append(str(Path(__file__).parent))

from adapters_no_prompt.base import BaseAdapter
from core.logger import get_logger
from core.vector_store import VikingStoreWrapper
from core.monitor import BenchmarkMonitor
from core.metrics import MetricsCalculator
from core.judge_util import llm_grader
from core.checkpoint import CheckpointManager
from vikingbot_runner import run_vikingbot_query, stop_openviking_server


class BenchmarkPipeline:
    def __init__(self, config, adapter: BaseAdapter, vector_db: VikingStoreWrapper = None, llm = None, resume: bool = False):
        self.config = config
        self.adapter = adapter
        self.db = vector_db
        self.llm = llm
        self.logger = get_logger()
        self.monitor = BenchmarkMonitor()
        self.resume = resume
        
        self.output_dir = self.config['paths']['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.generated_file = os.path.join(self.output_dir, "generated_answers.json")
        self.eval_file = os.path.join(self.output_dir, "qa_eval_detailed_results.json")
        self.report_file = os.path.join(self.output_dir, "benchmark_metrics_report.json")
        
        self.checkpoint_manager = CheckpointManager(self.output_dir, self.config)
        self._file_lock = threading.Lock()
        # Save each completed task by default so a single stuck future does not
        # make the previous completed evaluations repeat after resume.
        self.save_frequency = self.config.get("execution", {}).get("save_frequency", 1)
        
        self.metrics_summary = {
            "insertion": {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
            "deletion": {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
        }
        
        # 设置信号处理器，确保 Ctrl+C 时正确停止 ov 服务
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器，确保在测试被中断时正确停止 ov 服务"""
        def handle_signal(signum, frame):
            self.logger.info(f"Received signal {signum}, stopping OpenViking server...")
            stop_openviking_server()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        # 注册 atexit 处理器，确保程序正常退出时也停止 ov 服务
        atexit.register(stop_openviking_server)

    def run_generation(self):
        """Step 1: Data Preparation"""
        self.logger.info(">>> Stage: Ingestion & Generation")
        try:
            doc_dir = self.config['paths'].get('doc_output_dir')
            if not doc_dir:
                doc_dir = os.path.join(self.output_dir, "docs")
            
            try:
                doc_info = self.adapter.data_prepare(doc_dir)
            except Exception as e:
                self.logger.exception(f"Data preparation failed: {e}")
                exit(1)
            
            skip_ingestion = self.config['execution'].get('skip_ingestion', False)

            if skip_ingestion:
                self.logger.info(f"Skipping Ingestion. Using existing docs at: {doc_dir}")
                if not os.path.exists(doc_dir):
                     self.logger.warning(f"Warning: Doc directory {doc_dir} not found, but ingestion is skipped.")
                self.metrics_summary["insertion"] = {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
            else:
                ingest_workers = self.config['execution'].get('ingest_workers', 10)
                ingest_mode = self.config['execution'].get('ingest_mode', 'per_file')
                
                mode_desc = {
                    'directory': 'Unified directory mode',
                    'per_file': 'Per-file mode'
                }
                self.logger.info(f"Ingestion mode: {ingest_mode} ({mode_desc.get(ingest_mode, 'Unknown mode')})")
                self.logger.info(f"Number of documents: {len(doc_info)}")
                
                ingest_stats = self.db.ingest(
                    doc_info, 
                    max_workers=ingest_workers, 
                    monitor=self.monitor,
                    ingest_mode=ingest_mode
                )
                self.metrics_summary["insertion"] = ingest_stats
                self.logger.info(f"Insertion finished. Time: {ingest_stats['time']:.2f}s")

                self._update_report({
                    "Insertion Efficiency (Total Dataset)": {
                        "Total Insertion Time (s)": self.metrics_summary["insertion"]["time"],
                        "Total Input Tokens": self.metrics_summary["insertion"]["input_tokens"],
                        "Total Output Tokens": self.metrics_summary["insertion"]["output_tokens"],
                        "Total Embedding Tokens": self.metrics_summary["insertion"].get("embedding_tokens", 0)
                    }
                })
            
            samples = self.adapter.load_and_transform()    
            tasks = self._prepare_tasks(samples)
            results_map = {}
            max_workers = self.config['execution']['max_workers']
            
            completed_tasks: Set[int] = set()
            if self.resume:
                completed_tasks = self.checkpoint_manager.get_completed_tasks("generation")
                if completed_tasks:
                    self.logger.info(f"Resuming from checkpoint. {len(completed_tasks)} tasks already completed.")
                    if os.path.exists(self.generated_file):
                        with open(self.generated_file, "r", encoding="utf-8") as f:
                            saved_data = json.load(f)
                            for result in saved_data.get("results", []):
                                results_map[result["_global_index"]] = result
            
            remaining_tasks = [task for task in tasks if task["id"] not in completed_tasks]
            self.logger.info(f"Total tasks: {len(tasks)}, Remaining: {len(remaining_tasks)}")
            
            if remaining_tasks:
                initial_completed = len(completed_tasks)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_task = {
                        executor.submit(self._process_generation_task, task): task 
                        for task in remaining_tasks
                    }
                    
                    pbar = tqdm(total=len(tasks), desc="Generating Answers", unit="task", initial=len(completed_tasks))
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            res = future.result()
                            results_map[res['_global_index']] = res
                            completed_tasks.add(res['_global_index'])
                            
                            newly_completed = len(completed_tasks) - initial_completed
                            if newly_completed % self.save_frequency == 0 or len(completed_tasks) == len(tasks):
                                self.checkpoint_manager.update_completed_tasks("generation", completed_tasks, len(tasks))
                                self._save_partial_results(results_map)
                        except Exception as e:
                            self.logger.error(f"Generation failed for task {task['id']}: {e}")
                            self.monitor.worker_end(success=False)
                        pbar.set_postfix(self.monitor.get_status_dict())
                        pbar.update(1)
                    pbar.close()
            else:
                self.logger.info("All tasks already completed!")
            
            sorted_results = [results_map[i] for i in sorted(results_map.keys())]
            dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
            save_data = {
                "summary": {"dataset": dataset_name, "total_queries": len(sorted_results)},
                "results": sorted_results
            }
            query_efficiency = self._compute_query_efficiency(sorted_results)
            if query_efficiency:
                self._update_report({
                        "Query Efficiency (Average Per Query)": query_efficiency
                    }
                )
            with open(self.generated_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            self.checkpoint_manager.delete_checkpoint()
        finally:
            # 确保在 generation 阶段结束后停止 ov 服务
            self.logger.info("Generation stage completed, stopping OpenViking server...")
            stop_openviking_server()
    
    def _save_partial_results(self, results_map: dict):
        with self._file_lock:
            sorted_results = [results_map[i] for i in sorted(results_map.keys())]
            dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
            save_data = {
                "summary": {"dataset": dataset_name, "total_queries": len(sorted_results)},
                "results": sorted_results
            }
            with open(self.generated_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _numeric_values(items, getter):
        values = []
        for item in items:
            value = getter(item)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def _compute_query_efficiency(self, items):
        if not items:
            return {}

        specs = [
            ("Average Retrieval Time (s)", lambda i: (i.get("retrieval", {}) or {}).get("latency_sec")),
            ("Average Retrieval Embedding Time (s)", lambda i: (i.get("retrieval", {}) or {}).get("retrieval_embedding_time_sec")),
            ("Average Retrieval Search Time (s)", lambda i: (i.get("retrieval", {}) or {}).get("retrieval_search_time_sec")),
            ("Average Retrieval Wall Time (s)", lambda i: (i.get("retrieval", {}) or {}).get("retrieval_wall_time_sec")),
            ("Average Retrieval Filter Time (s)", lambda i: (i.get("retrieval", {}) or {}).get("retrieval_filter_time_sec")),
            ("Average Target Resolve Time (s)", lambda i: (i.get("retrieval", {}) or {}).get("target_resolve_time_sec")),
            ("Average Input Tokens", lambda i: (i.get("token_usage", {}) or {}).get("total_input_tokens")),
            ("Average Output Tokens", lambda i: (i.get("token_usage", {}) or {}).get("llm_output_tokens")),
            ("Average Retrieval Embedding Tokens", lambda i: (i.get("token_usage", {}) or {}).get("retrieval_embedding_tokens")),
            ("Average Searched Chunks", lambda i: (i.get("retrieval", {}) or {}).get("searched_chunk_count")),
            ("Average Target Chunks", lambda i: (i.get("retrieval", {}) or {}).get("target_chunk_count")),
        ]

        query_efficiency = {}
        for name, getter in specs:
            values = self._numeric_values(items, getter)
            if values:
                query_efficiency[name] = sum(values) / len(values)
        return query_efficiency

    def run_evaluation(self):
        """Step 4: Evaluation"""
        self.logger.info(">>> Stage: Evaluation")

        if not os.path.exists(self.generated_file):
            self.logger.error("Generated answers file not found.")
            return

        with open(self.generated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("results", [])

        query_efficiency = self._compute_query_efficiency(items)
        if query_efficiency:
            self._update_report(
                {
                    "Query Efficiency (Average Per Query)": query_efficiency
                }
            )

        eval_items = items
        eval_results_map = {}
        
        completed_eval_tasks: Set[int] = set()
        if self.resume:
            completed_eval_tasks = self.checkpoint_manager.get_completed_tasks("evaluation")
            if completed_eval_tasks:
                self.logger.info(f"Resuming from checkpoint. {len(completed_eval_tasks)} evaluations already completed.")
                if os.path.exists(self.eval_file):
                    with open(self.eval_file, "r", encoding="utf-8") as f:
                        saved_eval_data = json.load(f)
                        for result in saved_eval_data.get("results", []):
                            eval_results_map[result["_global_index"]] = result
        
        remaining_eval_items = [item for item in eval_items if item["_global_index"] not in completed_eval_tasks]
        self.logger.info(f"Total evaluations: {len(eval_items)}, Remaining: {len(remaining_eval_items)}")
        
        if remaining_eval_items:
            initial_completed_eval = len(completed_eval_tasks)
            with ThreadPoolExecutor(max_workers=self.config['execution']['max_workers']) as executor:
                future_to_item = {
                    executor.submit(self._process_evaluation_task, item): item 
                    for item in remaining_eval_items
                }
                
                pbar = tqdm(total=len(eval_items), desc="Evaluating", unit="item", initial=len(completed_eval_tasks))
                for future in as_completed(future_to_item):
                    try:
                        res = future.result()
                        eval_results_map[res['_global_index']] = res
                        completed_eval_tasks.add(res['_global_index'])
                        
                        newly_completed_eval = len(completed_eval_tasks) - initial_completed_eval
                        if newly_completed_eval % self.save_frequency == 0 or len(completed_eval_tasks) == len(eval_items):
                            self.checkpoint_manager.update_completed_tasks("evaluation", completed_eval_tasks, len(eval_items))
                            self._save_partial_eval_results(eval_results_map)
                    except Exception as e:
                        self.logger.error(f"Evaluation failed: {e}")
                        raise
                    pbar.update(1)
                pbar.close()
        else:
            self.logger.info("All evaluations already completed!")

        eval_records = list(eval_results_map.values())
        total = len(eval_records)

        with open(self.eval_file, "w", encoding="utf-8") as f:
            json.dump({"results": eval_records}, f, indent=2, ensure_ascii=False)

        if total > 0:
            self._update_report({
                "Dataset": self.config.get('dataset_name', 'Unknown_Dataset'),
                "Total Queries Evaluated": total,
                "Performance Metrics": {
                    "Average F1 Score": sum(r['metrics']['F1'] for r in eval_records) / total,
                    "Average Recall": sum(r['metrics']['Recall'] for r in eval_records) / total,
                    "Average Accuracy (Hit 0-4)": sum(r['metrics']['Accuracy'] for r in eval_records) / total,
                    "Average Accuracy (normalization)": (sum(r['metrics']['Accuracy'] for r in eval_records) / total)/4,
                }
            })
        
        self.checkpoint_manager.delete_checkpoint()
    
    def _save_partial_eval_results(self, eval_results_map: dict):
        with self._file_lock:
            eval_records = list(eval_results_map.values())
            with open(self.eval_file, "w", encoding="utf-8") as f:
                json.dump({"results": eval_records}, f, indent=2, ensure_ascii=False)

    def run_deletion(self):
        """Step 5: Cleanup"""
        self.logger.info(">>> Stage: Deletion")
        start_time = time.time()
        self.db.clear()
        duration = time.time() - start_time
        self.metrics_summary["deletion"] = {"time": duration, "input_tokens": 0, "output_tokens": 0}
        self.logger.info(f"Deletion finished. Time: {duration:.2f}s")

        self._update_report({
            "Deletion Efficiency (Total Dataset)": {
                "Total Deletion Time (s)": duration,
                "Total Input Tokens": 0,
                "Total Output Tokens": 0
            }
        })

    def _prepare_tasks(self, samples):
        tasks = []
        global_idx = 0
        max_queries = self.config['execution'].get('max_queries')
        for sample in samples:
            for qa in sample.qa_pairs:
                if max_queries is not None and global_idx >= max_queries:
                    break
                tasks.append({"id": global_idx, "sample_id": sample.sample_id, "qa": qa})
                global_idx += 1
            if max_queries is not None and global_idx >= max_queries:
                break
        return tasks

    def _process_generation_task(self, task):
        self.monitor.worker_start()
        try:
            qa = task['qa']
            
            use_vikingbot = self.config['execution'].get('use_vikingbot', False)
            
            if use_vikingbot:
                return self._process_vikingbot_task(task, qa)
            else:
                return self._process_standard_rag_task(task, qa)
        except Exception as e:
            self.monitor.worker_end(success=False)
            raise e
    
    def _process_standard_rag_task(self, task, qa):
        t0 = time.time()
        target_resolve_time = 0.0
        retrieval_instruction = self.config['execution'].get('retrieval_instruction', '')
        if retrieval_instruction:
            enhanced_query = f"{retrieval_instruction} {qa.question}"
        else:
            enhanced_query = qa.question
        
        dataset_name = self.config.get('dataset_name', '')
        
        topk = int(self.config['execution']['retrieval_topk'])
        restrict_to_qa_doc = bool(self.config.get("execution", {}).get("restrict_to_qa_doc", False))
        if restrict_to_qa_doc:
            target_t0 = time.perf_counter()
            target_uris = self._resolve_target_uris(task, qa)
            target_resolve_time = time.perf_counter() - target_t0
        else:
            target_uris = []
        candidate_k = max(topk, int(self.config.get("execution", {}).get("candidate_topk", topk)))
        retrieval_embedding_tokens = 0
        retrieval_telemetry = {}
        retrieval_core_times = []

        if target_uris and len(target_uris) > 1:
            merged = {}
            total_target_chunks = 0
            total_searched_chunks = 0
            for uri in target_uris:
                retrieve_res = self.db.retrieve(query=enhanced_query, topk=candidate_k, target_uri=uri)
                if isinstance(retrieve_res, tuple) and len(retrieve_res) == 2:
                    search_res, tokens = retrieve_res
                else:
                    search_res, tokens = retrieve_res, 0
                retrieval_embedding_tokens += int(tokens or 0)
                telemetry = getattr(search_res, "telemetry", {}) or {}
                core_time = telemetry.get("retrieval_core_time_sec")
                if isinstance(core_time, (int, float)) and not isinstance(core_time, bool):
                    retrieval_core_times.append(float(core_time))
                total_target_chunks += int(telemetry.get("target_chunk_count", 0) or 0)
                total_searched_chunks += int(telemetry.get("searched_chunk_count", 0) or 0)
                for r in getattr(search_res, 'resources', []) or []:
                    key = getattr(r, 'uri', None) or id(r)
                    if key not in merged or getattr(r, 'score', 0) > getattr(merged[key], 'score', 0):
                        merged[key] = r
            final_results = sorted(merged.values(), key=lambda r: getattr(r, 'score', 0), reverse=True)[:topk]
            retrieval_telemetry = {
                "search_scope": "multi_target",
                "target_chunk_count": total_target_chunks,
                "searched_chunk_count": total_searched_chunks,
            }
        else:
            target_uri = target_uris[0] if target_uris else "viking://resources"
            retrieve_res = self.db.retrieve(query=enhanced_query, topk=candidate_k, target_uri=target_uri)
            if isinstance(retrieve_res, tuple) and len(retrieve_res) == 2:
                search_res, retrieval_embedding_tokens = retrieve_res
            else:
                search_res = retrieve_res
                retrieval_embedding_tokens = 0
            final_results = (getattr(search_res, 'resources', []) or [])[:topk]
            retrieval_telemetry = getattr(search_res, "telemetry", {}) or {}
        
        retrieval_wall_time = time.time() - t0
        core_time = retrieval_telemetry.get("retrieval_core_time_sec")
        if retrieval_core_times:
            latency = sum(retrieval_core_times)
        elif isinstance(core_time, (int, float)) and not isinstance(core_time, bool):
            latency = float(core_time)
        else:
            latency = retrieval_wall_time
        retrieval_telemetry["retrieval_wall_time_sec"] = retrieval_wall_time
        if target_resolve_time:
            retrieval_telemetry["target_resolve_time_sec"] = target_resolve_time
        
        retrieved_texts = []
        retrieved_uris = []
        context_blocks = []
        
        for r in final_results:
            retrieved_uris.append(r.uri)
            content = self.db.read_resource(r.uri) if getattr(r, 'level', 2) == 2 else f"{getattr(r, 'abstract', '')}\n{getattr(r, 'overview', '')}"
            retrieved_texts.append(content)
            clean = content[:8000]
            context_blocks.append(clean)
        
        recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)
        
        full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
        
        ans_raw = self.llm.generate(full_prompt)
        ans = self.adapter.post_process_answer(qa, ans_raw, meta)

        in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question)
        out_tokens = self.db.count_tokens(ans)
        self.monitor.worker_end(tokens=in_tokens + out_tokens + retrieval_embedding_tokens)
        
        self.logger.info(
            f"[Query-{task['id']}] Q: {qa.question[:30]}... | Recall: {recall:.2f} | "
            f"Latency: {latency:.2f}s | Wall: {retrieval_wall_time:.2f}s"
        )

        return {
            "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
            "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
            "retrieval": {"latency_sec": latency, "uris": retrieved_uris, "target_uris": target_uris, **retrieval_telemetry},
            "llm": {"final_answer": ans},
            "metrics": {"Recall": recall}, "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens, "retrieval_embedding_tokens": retrieval_embedding_tokens}
        }
    
    def _process_vikingbot_task(self, task, qa):
        self.logger.info(f"[Query-{task['id']}] Using VikingBot for agentic RAG")
        
        session_id = f"query_{uuid.uuid4().hex}"
        
        restrict_to_qa_doc = bool(self.config.get("execution", {}).get("restrict_to_qa_doc", False))
        allowed_target_uris = self._resolve_target_uris(task, qa) if restrict_to_qa_doc else None

        vikingbot_result = run_vikingbot_query(
            question=qa.question,
            config=self.config,
            session_id=session_id,
            allowed_target_uris=allowed_target_uris,
        )
        
        ans = vikingbot_result['answer']
        total_time = vikingbot_result['total_time_sec']
        
        recall = 0.0
        vb_usage = vikingbot_result.get("token_usage") or {}
        in_tokens = int(vb_usage.get("prompt_tokens", 0) or 0)
        out_tokens = int(vb_usage.get("completion_tokens", 0) or 0)
        tools_used_names = vikingbot_result.get("tools_used_names") or []
        iterations_used = int(vikingbot_result.get("iterations_used") or 0)
        
        self.monitor.worker_end(tokens=in_tokens + out_tokens)
        
        self.logger.info(f"[Query-{task['id']}] Q: {qa.question[:30]}... | Latency: {total_time:.2f}s | Mode: Agentic RAG")

        return {
            "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
            "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
            "retrieval": {"latency_sec": total_time, "uris": [], "mode": "agentic", "target_uris": allowed_target_uris or []},
            "llm": {"final_answer": ans},
            "metrics": {"Recall": recall}, 
            "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens},
            "vikingbot": {"iterations_used": iterations_used, "tools_used_names": tools_used_names},
        }

    def _sanitize_for_path(self, text: str, max_length: int = 50) -> str:
        safe = re.sub(
            r"[^\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u3400-\u4dbf\U00020000-\U0002a6df\s-]",
            "",
            text,
        )
        safe = re.sub(r"\s+", "_", safe)
        safe = safe.strip("_")
        if not safe:
            return "section"
        if len(safe) > max_length:
            hash_suffix = hashlib.sha256(text.encode()).hexdigest()[:8]
            return f"{safe[: max_length - 9]}_{hash_suffix}"
        return safe

    def _resolve_child_uri(self, parent_uri: str, raw_child: str) -> str:
        """
        Resolve a child directory URI under `parent_uri` in a way that matches OV's on-disk naming.
        This avoids per-query target_uris pointing to non-existent directories (e.g. '_' vs '__').
        """
        def _norm_underscores(s: str) -> str:
            # OV naming can produce multiple consecutive '_' (e.g. from parentheses/comma),
            # while some dataset titles/callers collapse them. Treat them as equivalent.
            return re.sub(r"_+", "_", s).strip("_")

        raw = str(raw_child)
        try:
            from openviking_cli.utils.uri import VikingURI

            child = VikingURI.sanitize_segment(raw)
        except Exception:
            child = raw

        # Fallback candidate; not guaranteed to match OV storage, so we always probe disk.
        child_alt = self._sanitize_for_path(raw)
        child_norm = _norm_underscores(child)
        child_alt_norm = _norm_underscores(child_alt) if child_alt else ""

        store_path = getattr(self.db, "store_path", None) if self.db is not None else None
        if not store_path:
            return f"{parent_uri}/{child_alt or child}"

        prefix = "viking://resources/"
        rel = parent_uri[len(prefix) :] if parent_uri.startswith(prefix) else ""
        base_dir = Path(store_path) / "viking" / "default" / "resources"
        parent_dir = base_dir / rel if rel else base_dir

        if (parent_dir / child).exists():
            return f"{parent_uri}/{child}"
        if child_alt and child_alt != child and (parent_dir / child_alt).exists():
            return f"{parent_uri}/{child_alt}"
        if not parent_dir.exists():
            return f"{parent_uri}/{child_alt or child}"

        candidates = []
        for p in parent_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if name == child:
                return f"{parent_uri}/{name}"
            if child_alt and name == child_alt:
                return f"{parent_uri}/{name}"
            # Handle cases like `Gibson__cocktail__doc` vs `Gibson_cocktail_doc`.
            name_norm = _norm_underscores(name)
            if child_norm and name_norm == child_norm:
                return f"{parent_uri}/{name}"
            if child_alt_norm and name_norm == child_alt_norm:
                return f"{parent_uri}/{name}"
            for base in (child, child_alt):
                if not base:
                    continue
                if name.startswith(base + "_"):
                    suffix = name[len(base) + 1 :]
                    if suffix.isdigit():
                        candidates.append((0, int(suffix), name))
                    elif re.fullmatch(r"[0-9a-f]{8}", suffix, flags=re.IGNORECASE):
                        candidates.append((1, 0, name))
                else:
                    # Also consider suffix matching on normalized forms.
                    base_norm = _norm_underscores(base)
                    if base_norm and name_norm.startswith(base_norm + "_"):
                        suffix = name_norm[len(base_norm) + 1 :]
                        if suffix.isdigit():
                            candidates.append((0, int(suffix), name))
                        elif re.fullmatch(r"[0-9a-f]{8}", suffix, flags=re.IGNORECASE):
                            candidates.append((1, 0, name))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            return f"{parent_uri}/{candidates[0][2]}"

        return f"{parent_uri}/{child_alt or child}"

    def _resolve_target_uris(self, task, qa) -> list[str]:
        dataset_name = self.config.get("dataset_name", "") or ""
        sample_id = str(task.get("sample_id", ""))
        target_doc_paths = (qa.metadata or {}).get("target_doc_paths") or []

        if target_doc_paths and getattr(self.db, "retriever_type", "") == "naive":
            return [f"naive://{os.path.abspath(str(p))}" for p in target_doc_paths if p]

        if dataset_name.startswith("VersionRAG"):
            uris = []
            for p in target_doc_paths:
                if not p:
                    continue
                stem = Path(str(p)).stem
                uris.append(self._resolve_child_uri("viking://resources", stem))
            return uris

        if dataset_name == "FinanceBench":
            return [self._resolve_child_uri("viking://resources/pdfs", sample_id)]

        if dataset_name == "SyllabusQA":
            return [self._resolve_child_uri("viking://resources/SyllabusQA_processed_docs", f"{sample_id}_doc")]

        if dataset_name == "Locomo":
            return [self._resolve_child_uri("viking://resources/Locomo_processed_docs", f"{sample_id}_doc")]

        if dataset_name == "Qasper":
            return [self._resolve_child_uri("viking://resources/Qasper_processed_docs", f"{sample_id}_doc")]

        if dataset_name == "ClapNQ":
            passages = (qa.metadata or {}).get("passages") or []
            titles = []
            for p in passages:
                if isinstance(p, dict):
                    t = p.get("title")
                    if t:
                        titles.append(str(t))
            uniq = []
            seen = set()
            for t in titles:
                raw = str(t)
                if raw and raw not in seen:
                    seen.add(raw)
                    uniq.append(self._resolve_child_uri("viking://resources/ClapNQ_processed_docs", raw))
            return uniq

        if dataset_name == "HotpotQA":
            titles = (qa.metadata or {}).get("supporting_fact_titles") or []
            uniq = []
            seen = set()
            for t in titles:
                if not t:
                    continue
                raw = str(t)
                if raw and raw not in seen:
                    seen.add(raw)
                    uniq.append(self._resolve_child_uri("viking://resources/HotpotQA_processed_docs", f"{raw}_doc"))
            return uniq

        if dataset_name == "LegalBench" or dataset_name.startswith("LegalBench_"):
            parent = f"viking://resources/{dataset_name}_processed_docs"
            file_paths = (qa.metadata or {}).get("file_paths") or [sample_id]
            uniq = []
            seen = set()
            for rel_path in file_paths:
                if not rel_path:
                    continue
                raw = str(rel_path).replace("/", "_").replace(".txt", "")
                if raw and raw not in seen:
                    seen.add(raw)
                    uniq.append(self._resolve_child_uri(parent, raw))
            return uniq

        return []

    def _process_evaluation_task(self, item):
        """
        Process a single evaluation task, computing F1 and Accuracy metrics.
        
        For multi-annotator scenarios (like Qasper dataset), a question may have multiple gold answers.
        Evaluation logic:
        - F1: Compute for each gold answer separately and take the maximum
        - Accuracy: Pass all gold answers to LLM at once for comprehensive judgment
        
        This correctly handles multi-annotator scenarios while maintaining compatibility with single-answer datasets (like Locomo).
        """
        ans, golds = item['llm']['final_answer'], item['gold_answers']
        
        f1 = max((MetricsCalculator.calculate_f1(ans, gt) for gt in golds), default=0.0)
        
        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        
        eval_record = {
            "score": 0.0,
            "reasoning": "",
            "prompt_type": ""
        }
        
        try:
            eval_res = llm_grader(
                self.llm.llm, 
                self.config['llm']['model'], 
                item['question'], 
                golds,
                ans,
                dataset_name=dataset_name
            )
            eval_record = eval_res
                
        except Exception as e:
            self.logger.error(f"Grader error: {e}")
            raise
            
        if MetricsCalculator.check_refusal(ans) and any(MetricsCalculator.check_refusal(gt) for gt in golds):
            f1 = 1.0
            eval_record["score"] = 4.0
            eval_record["reasoning"] = "System successfully identified Unanswerable/Refusal condition."
            eval_record["prompt_type"] = "Heuristic_Refusal_Check"

        acc = eval_record["score"]

        item["metrics"].update({"F1": f1, "Accuracy": acc})
        
        item["llm_evaluation"] = {
            "prompt_used": eval_record["prompt_type"],
            "reasoning": eval_record["reasoning"],
            "normalized_score": acc
        }

        detailed_info = (
            f"\n" + "="*60 +
            f"\n[Query ID]: {item['_global_index']}"
            f"\n[Question]: {item['question']}"
            f"\n[Retrieved URIs]: {item['retrieval'].get('uris', [])}"
            f"\n[LLM Answer]: {ans}"
            f"\n[Gold Answer]: {golds}"
            f"\n[Metrics]: {item['metrics']}"
            f"\n[LLM Judge Reasoning]: {eval_record['reasoning']}"
            f"\n" + "="*60
        )
        self.logger.info(detailed_info)
        return item

    def _update_report(self, data):
        """Read existing report, merge new data, and write back"""
        report = {}
        if os.path.exists(self.report_file):
            with open(self.report_file, "r", encoding="utf-8") as f:
                try:
                    report = json.load(f)
                except json.JSONDecodeError:
                    report = {}
        report.update(data)
        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Report updated -> {self.report_file}")
