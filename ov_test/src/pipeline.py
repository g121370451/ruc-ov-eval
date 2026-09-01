import os
import json
import time
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set
from tqdm import tqdm

from src.adapters.base import BaseAdapter
from src.core.logger import get_logger

from .core.monitor import BenchmarkMonitor
from .core.metrics import MetricsCalculator
from .core.judge_util import llm_grader
from .core.checkpoint import CheckpointManager
from .core.store_contract import store_provides_final_answer


class BenchmarkPipeline:
    def __init__(self, config, adapter: BaseAdapter, vector_db, llm, resume: bool = False):
        self.config = config
        self.adapter = adapter
        self.db = vector_db
        self.llm = llm
        self.logger = get_logger()
        self.monitor = BenchmarkMonitor()
        self.store_type = config.get('store', {}).get('type', 'viking')
        self.resume = resume

        self.output_dir = self.config['paths']['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.generated_file = os.path.join(self.output_dir, "generated_answers.json")
        self.eval_file = os.path.join(self.output_dir, "qa_eval_detailed_results.json")
        self.report_file = os.path.join(self.output_dir, "benchmark_metrics_report.json")

        self.metrics_summary = {
            "insertion": {"time": 0, "input_tokens": 0, "output_tokens": 0},
            "deletion": {"time": 0, "input_tokens": 0, "output_tokens": 0}
        }

        self.checkpoint_manager = CheckpointManager(self.output_dir, self.config)
        self._file_lock = threading.Lock()
        self.save_frequency = 10

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

    def _save_partial_eval_results(self, eval_results_map: dict):
        with self._file_lock:
            eval_records = list(eval_results_map.values())
            with open(self.eval_file, "w", encoding="utf-8") as f:
                json.dump({"results": eval_records}, f, indent=2, ensure_ascii=False)

    def run_ingestion(self):
        self.logger.info(">>> Stage: Ingestion Only")
        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        try:
            doc_info = self.adapter.data_prepare(doc_dir)
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise

        import shutil
        store_path = self.config['paths'].get('vector_store', '')
        if self.resume:
            ingested_samples = self.checkpoint_manager.get_ingested_samples()
            self.logger.info(
                f"Resuming ingestion with {len(ingested_samples)} checkpointed "
                "sample(s); existing store files will be preserved."
            )
            if store_path:
                os.makedirs(store_path, exist_ok=True)
        else:
            self.checkpoint_manager.delete_checkpoint()
            if os.path.isdir(store_path):
                shutil.rmtree(store_path)
            if store_path:
                os.makedirs(store_path, exist_ok=True)
                self.logger.info(f"Store directory initialized cleanly: {store_path}")

        ingest_workers = self.config['execution'].get('ingest_workers', 10)
        ingest_stats = self.db.ingest(
            doc_info,
            max_workers=ingest_workers,
            monitor=self.monitor,
            checkpoint_manager=self.checkpoint_manager
        )
        self.metrics_summary["insertion"] = ingest_stats
        self.logger.info(f"Insertion finished. Time: {ingest_stats['time']:.2f}s")

        self._update_report({
            "Insertion Efficiency (Total Dataset)": {
                "Total Insertion Time (s)": ingest_stats["time"],
                "Total Input Tokens": ingest_stats["input_tokens"],
                "Total Output Tokens": ingest_stats["output_tokens"]
            }
        })

        self.checkpoint_manager.delete_checkpoint()

    def run_generation(self):
        self.logger.info(">>> Stage: Generation (Retrieve + Generate)")

        if self.resume:
            ingested_samples = self.checkpoint_manager.get_ingested_samples()
            if ingested_samples:
                checkpoint_data = self.checkpoint_manager.load_checkpoint()
                ingest_stats = (checkpoint_data or {}).get("execution_state", {}).get(
                    "ingest_stats", {"time": 0, "input_tokens": 0, "output_tokens": 0})
                self.metrics_summary["insertion"] = ingest_stats
            else:
                self.logger.info("No ingestion checkpoint found. Assuming ingestion was done previously.")
        else:
            self.logger.info("Skipping ingestion. Using existing store index.")

        samples = self.adapter.load_and_transform()
        tasks = self._prepare_tasks(samples)

        completed_tasks: Set[int] = set()
        results_map = {}

        if self.resume:
            completed_tasks = self.checkpoint_manager.get_completed_tasks("generation")
            if completed_tasks:
                self.logger.info(f"Resuming generation. {len(completed_tasks)} tasks already completed.")
                if os.path.exists(self.generated_file):
                    try:
                        with open(self.generated_file, "r", encoding="utf-8") as f:
                            saved_data = json.load(f)
                        for result in saved_data.get("results", []):
                            results_map[result["_global_index"]] = result
                    except Exception as e:
                        self.logger.warning(f"Failed to load previous generated results, continuing fresh: {e}")

        remaining_tasks = [task for task in tasks if task["id"] not in completed_tasks]
        self.logger.info(f"Total tasks: {len(tasks)}, Remaining: {len(remaining_tasks)}")

        max_workers = self.config['execution']['max_workers']

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
            self.logger.info("All generation tasks already completed!")

        sorted_results = [results_map[i] for i in sorted(results_map.keys())]
        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        save_data = {
            "summary": {"dataset": dataset_name, "total_queries": len(sorted_results)},
            "results": sorted_results
        }
        total = len(sorted_results)
        if total > 0:
            latency_scopes = sorted({
                r['retrieval'].get('latency_scope', 'retrieval_only')
                for r in sorted_results
            })
            self._update_report({
                    "Query Efficiency (Average Per Query)": {
                        "Average Retrieval Time (s)": sum(r['retrieval']['latency_sec'] for r in sorted_results) / total,
                        "Latency Scope": latency_scopes[0] if len(latency_scopes) == 1 else latency_scopes,
                        "Average Input Tokens": sum(r['token_usage']['total_input_tokens'] for r in sorted_results) / total,
                        "Average Output Tokens": sum(r['token_usage']['llm_output_tokens'] for r in sorted_results) / total,
                    }
                }
            )
        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        self.checkpoint_manager.delete_checkpoint()

        # DeepRead 迭代轮次统计（如有日志则自动解析并追加到报告）
        if self.store_type == 'DeepRead':
            log_path = os.path.join(self.output_dir, "deepread_run.log")
            if os.path.exists(log_path):
                try:
                    from scripts.analyze_deepread_log import analyze
                    iter_stats = analyze(log_path)
                    if iter_stats:
                        self._update_report({
                            "DeepRead Agent Iterations": {
                                "Average Rounds": iter_stats.get("average_rounds", 0),
                                "Median Rounds": iter_stats.get("median_rounds", 0),
                                "Max Rounds": iter_stats.get("max_rounds", 0),
                                "Min Rounds": iter_stats.get("min_rounds", 0),
                                "Distribution": iter_stats.get("distribution", {})
                            }
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to analyze deepread_run.log: {e}")

    def run_evaluation(self):
        self.logger.info(">>> Stage: Evaluation")

        if not os.path.exists(self.generated_file):
            self.logger.error("Generated answers file not found.")
            return

        with open(self.generated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("results", [])

        eval_items = items
        eval_results_map = {}

        completed_eval_tasks: Set[int] = set()
        if self.resume:
            completed_eval_tasks = self.checkpoint_manager.get_completed_tasks("evaluation")
            if completed_eval_tasks:
                self.logger.info(f"Resuming evaluation. {len(completed_eval_tasks)} evaluations already completed.")
                if os.path.exists(self.eval_file):
                    try:
                        with open(self.eval_file, "r", encoding="utf-8") as f:
                            saved_eval_data = json.load(f)
                        for result in saved_eval_data.get("results", []):
                            eval_results_map[result["_global_index"]] = result
                    except Exception as e:
                        self.logger.warning(f"Failed to load previous eval results, continuing fresh: {e}")

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
                    pbar.update(1)
                pbar.close()
        else:
            self.logger.info("All evaluation tasks already completed!")

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
                    "Average Accuracy (Hit  0-4 )": sum(r['metrics']['Accuracy'] for r in eval_records) / total,
                    "Average Accuracy (normalization)": (sum(r['metrics']['Accuracy'] for r in eval_records) / total)/4,
                }
            })
        self.checkpoint_manager.delete_checkpoint()

    def run_deletion(self):
        self.logger.info(">>> Stage: Deletion")
        t0 = time.time()
        self.db.clear()
        elapsed = time.time() - t0
        self.metrics_summary["deletion"] = {"time": elapsed, "input_tokens": 0, "output_tokens": 0}
        self.logger.info(f"Deletion finished. Time: {elapsed:.2f}s")
        self._update_report({
            "Deletion Efficiency": {
                "Total Deletion Time (s)": elapsed,
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

        worker_id = self.config['execution'].get('worker_id')
        num_workers = self.config['execution'].get('num_workers')
        if worker_id is not None and num_workers is not None:
            tasks = tasks[worker_id::num_workers]
            self.logger.info(f"Shard {worker_id}/{num_workers}: {len(tasks)} tasks"
                             f"(indices {[t['id'] for t in tasks[:3]]}...)")
        return tasks

    def _process_generation_task(self, task):
        self.monitor.worker_start()
        try:
            qa = task['qa']

            t0 = time.time()
            if self.store_type == 'sql_agent':
                search_res = self.db.retrieve(
                    query=qa.question, topk=self.config['execution']['retrieval_topk'],
                    sample_id=task['sample_id'], qa_metadata=qa.metadata)
            else:
                search_res = self.db.retrieve(query=qa.question, topk=self.config['execution']['retrieval_topk'])
            latency = time.time() - t0

            retrieved_texts, context_blocks, retrieved_uris = self.db.process_retrieval_results(search_res)

            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)

            retrieve_in = getattr(search_res, 'retrieve_input_tokens', 0)
            retrieve_out = getattr(search_res, 'retrieve_output_tokens', 0)

            provides_final_answer = store_provides_final_answer(self.db)
            if provides_final_answer:
                ans = self.db.get_final_answer(search_res)
                in_tokens = retrieve_in
                out_tokens = retrieve_out
            else:
                full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
                ans_raw = self.llm.generate(full_prompt)
                ans = self.adapter.post_process_answer(qa, ans_raw, meta)
                in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question) + retrieve_in
                out_tokens = self.db.count_tokens(ans) + retrieve_out

            not_mentioned_reason = ""
            if self.config.get('execution', {}).get('explain_not_mentioned', False):
                if MetricsCalculator.check_refusal(ans):
                    not_mentioned_reason = self.llm.explain_not_mentioned(qa.question, context_blocks)

            self.monitor.worker_end(tokens=in_tokens + out_tokens)

            self.logger.info(f"[Query-{task['id']}] Q: {qa.question[:30]}... | Recall: {recall:.2f} | Latency: {latency:.2f}s")

            return {
                "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
                "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
                "retrieval": {"latency_sec": latency, "uris": retrieved_uris,
                              "latency_scope": "end_to_end" if provides_final_answer else "retrieval_only",
                              "query_mode": getattr(search_res, 'query_mode', None),
                              "internal_llm_calls": getattr(search_res, 'llm_calls', 0),
                              "llm_calls_categories": getattr(search_res, 'llm_calls_categories', {}),
                              "input_tokens_categories": getattr(search_res, 'input_tokens_categories', {}),
                              "output_tokens_categories": getattr(search_res, 'output_tokens_categories', {}),
                              "recall_texts": retrieved_texts, "prompt_texts": context_blocks,
                              "sql_queries": getattr(search_res, 'sql_queries', [])},
                "llm": {"final_answer": ans, "not_mentioned_reason": not_mentioned_reason},
                "metrics": {"Recall": recall}, "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens}
            }
        except Exception as e:
            self.monitor.worker_end(success=False)
            raise e

    def _process_evaluation_task(self, item):
        ans, golds = item['llm']['final_answer'], item['gold_answers']

        f1 = max((MetricsCalculator.calculate_f1(ans, gt) for gt in golds), default=0.0)

        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')

        best_eval_record = {
            "score": 0.0,
            "reasoning": "",
            "prompt_type": ""
        }

        try:
            gold_answer_str = json.dumps(golds, ensure_ascii=False)
            eval_res = llm_grader(
                self.llm.llm,
                self.config['llm']['model'],
                item['question'],
                gold_answer_str,
                ans,
                dataset_name=dataset_name
            )
            best_eval_record = eval_res
        except Exception as e:
            self.logger.error(f"Grader error: {e}")

        if MetricsCalculator.check_refusal(ans) and any(MetricsCalculator.check_refusal(gt) for gt in golds):
            f1 = 1.0
            best_eval_record["score"] = 4.0
            best_eval_record["reasoning"] = "System successfully identified Unanswerable/Refusal condition."
            best_eval_record["prompt_type"] = "Heuristic_Refusal_Check"

        acc = best_eval_record["score"]

        item["metrics"].update({"F1": f1, "Accuracy": acc})

        item["llm_evaluation"] = {
            "prompt_used": best_eval_record["prompt_type"],
            "reasoning": best_eval_record["reasoning"],
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
            f"\n[LLM Judge Reasoning]: {best_eval_record['reasoning']}"
            f"\n" + "="*60
        )
        self.logger.info(detailed_info)
        return item

    def _update_report(self, data):
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
