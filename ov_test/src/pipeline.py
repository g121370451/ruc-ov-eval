# src/pipeline.py
import asyncio
import os
import json
import time
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.adapters.base import BaseAdapter
from src.core.logger import get_logger

from .core.monitor import BenchmarkMonitor
from .core.metrics import MetricsCalculator
from .core.judge_util import llm_grader

class BenchmarkPipeline:
    def __init__(self, config, adapter: BaseAdapter, vector_db, llm):
        self.config = config
        self.adapter = adapter
        self.db = vector_db
        self.llm = llm
        self.logger = get_logger()
        self.monitor = BenchmarkMonitor()
        self.store_type = config.get('store', {}).get('type', 'viking')

        # 结果文件路径
        self.output_dir = self.config['paths']['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.generated_file = os.path.join(self.output_dir, "generated_answers.json")
        self.eval_file = os.path.join(self.output_dir, "qa_eval_detailed_results.json")
        self.report_file = os.path.join(self.output_dir, "benchmark_metrics_report.json")

        # 用于存储各阶段汇总指标
        self.metrics_summary = {
            "insertion": {"time": 0, "input_tokens": 0, "output_tokens": 0},
            "deletion": {"time": 0, "input_tokens": 0, "output_tokens": 0}
        }

        # ---- 断点恢复 records ----
        self.records_file = os.path.join(self.output_dir, "_pipeline_records.json")
        self.records = self._load_records()
        self._records_lock = threading.Lock()

    # ---- 记录持久化 ----

    def _load_records(self) -> dict:
        if os.path.exists(self.records_file):
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"ingested": False, "tasks": {}}

    def _save_records(self):
        with self._records_lock:
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, indent=2, ensure_ascii=False)

    def run_generation(self):
        """Step1 数据预处理"""
        self.logger.info(">>> Stage: Ingestion & Generation")
        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")
        # 0. 预处理数据集
        try:
            doc_info = self.adapter.data_prepare(doc_dir)
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
        skip_ingestion = self.config['execution'].get('skip_ingestion', False)

        # 断点恢复：如果 records 标记已入库完成，跳过入库
        if self.records.get("ingested"):
            self.logger.info("Records indicate ingestion already completed, skipping.")
            skip_ingestion = True
            ingest_stats = self.records.get("ingest_stats", {"time": 0, "input_tokens": 0, "output_tokens": 0})
            self.metrics_summary["insertion"] = ingest_stats

        if skip_ingestion:
            self.logger.info(f"Skipping Ingestion. Using existing docs at: {doc_dir}")
            if not os.path.exists(doc_dir):
                 self.logger.warning(f"Warning: Doc directory {doc_dir} not found, but ingestion is skipped.")
            if not self.records.get("ingested"):
                self.metrics_summary["insertion"] = {"time": 0, "input_tokens": 0, "output_tokens": 0}

        else:  # 正常执行入库
            import shutil
            from src.core.backup_utils import backup_store
            store_path = self.config['paths'].get('vector_store', '')
            # 清空 store 目录
            if os.path.isdir(store_path):
                shutil.rmtree(store_path)
                os.makedirs(store_path, exist_ok=True)
                self.logger.info(f"Store directory cleared: {store_path}")
            ingest_workers = self.config['execution'].get('ingest_workers')
            ingest_stats = self.db.ingest(
                doc_info,
                max_workers=ingest_workers,
                monitor=self.monitor
            )
            self.metrics_summary["insertion"] = ingest_stats
            self.logger.info(f"Insertion finished. Time: {ingest_stats['time']:.2f}s")

            # 标记入库完成
            with self._records_lock:
                self.records["ingested"] = True
                self.records["ingest_stats"] = ingest_stats
            self._save_records()
            # 入库完成后备份
            # 将 insertion 效率数据写入报告
            self._update_report({
                "Insertion Efficiency (Total Dataset)": {
                    "Total Insertion Time (s)": self.metrics_summary["insertion"]["time"],
                    "Total Input Tokens": self.metrics_summary["insertion"]["input_tokens"],
                    "Total Output Tokens": self.metrics_summary["insertion"]["output_tokens"]
                }
            })
            # backup_store(store_path, self.logger)
        """Step 2 & 3: 数据入库 + 检索生成"""
        # 1. 始终加载数据
        samples = self.adapter.load_and_transform()
        # 2. 准备 QA 任务
        tasks = self._prepare_tasks(samples)

        # 断点恢复：从 records 中加载已完成的 task 结果
        completed_tasks = self.records.get("tasks", {})
        results_map = {}
        pending_tasks = []
        for task in tasks:
            tid = str(task['id'])
            if tid in completed_tasks:
                results_map[task['id']] = completed_tasks[tid]
            else:
                pending_tasks.append(task)

        if results_map:
            self.logger.info(f"Resumed {len(results_map)} completed tasks, {len(pending_tasks)} remaining")

        if self.store_type == 'lightrag':
            async_results = asyncio.run(self._run_lightrag_generation_group_async(pending_tasks))
            for task_id, res in async_results.items():
                results_map[task_id] = res
                with self._records_lock:
                    self.records.setdefault("tasks", {})[str(task_id)] = res
                self._save_records()
        else:
            max_workers = self.config.get('execution', {}).get('max_workers', 1)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(self._process_generation_task, task): task
                    for task in pending_tasks
                }

                pbar = tqdm(total=len(pending_tasks), desc="Generating Answers", unit="task")
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        res = future.result()
                        results_map[res['_global_index']] = res
                        # 持久化单个 task 结果
                        with self._records_lock:
                            self.records.setdefault("tasks", {})[str(res['_global_index'])] = res
                        self._save_records()
                    except Exception as e:
                        self.logger.error(f"Generation failed for task {task['id']}: {e}")
                        self.monitor.worker_end(success=False)
                    pbar.set_postfix(self.monitor.get_status_dict())
                    pbar.update(1)
                pbar.close()

        # 3. 保存中间回答文件
        sorted_results = [results_map[i] for i in sorted(results_map.keys())]
        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        save_data = {
            "summary": {"dataset": dataset_name, "total_queries": len(sorted_results)},
            "results": sorted_results
        }
        total = len(sorted_results)
        if total > 0:
            self._update_report({
                    "Query Efficiency (Average Per Query)": {
                        "Average Retrieval Time (s)": sum(r['retrieval']['latency_sec'] for r in sorted_results) / total,
                        "Average Input Tokens": sum(r['token_usage']['total_input_tokens'] for r in sorted_results) / total,
                        "Average Output Tokens": sum(r['token_usage']['llm_output_tokens'] for r in sorted_results) / total,
                    }
                }
            )
        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

    def run_evaluation(self):
        """Step 4: 结果评测打分"""
        self.logger.info(">>> Stage: Evaluation")

        if not os.path.exists(self.generated_file):
            self.logger.error("Generated answers file not found.")
            return

        with open(self.generated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("results", [])

        eval_items = items
        eval_results_map = {}
        
        with ThreadPoolExecutor(max_workers=self.config.get('execution', {}).get('max_workers', 1)) as executor:
            future_to_item = {
                executor.submit(self._process_evaluation_task, item): item 
                for item in eval_items
            }
            
            pbar = tqdm(total=len(eval_items), desc="Evaluating", unit="item")
            for future in as_completed(future_to_item):
                try:
                    res = future.result()
                    eval_results_map[res['_global_index']] = res
                except Exception as e:
                    self.logger.error(f"Evaluation failed: {e}")
                pbar.update(1)
            pbar.close()

        # 保存详细评测文件 & 将评测指标写入报告
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

    def run_deletion(self):
        """Step 5: 计时删除"""
        self.logger.info(">>> Stage: Deletion")
        t0 = time.time()
        self.db.clear()
        elapsed = time.time() - t0
        self.metrics_summary["deletion"] = {"time": elapsed, "input_tokens": 0, "output_tokens": 0}
        self.logger.info(f"Deletion finished. Time: {elapsed:.2f}s")
        # 更新 records
        with self._records_lock:
            self.records["deleted"] = True
            self.records["delete_time"] = elapsed
        self._save_records()

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

            # 1. Retrieval
            t0 = time.time()
            retrieval_topk = self.config.get('execution', {}).get('retrieval_topk')
            if self.store_type == 'sql_agent':
                search_res = self.db.retrieve(
                    query=qa.question, topk=retrieval_topk,
                    sample_id=task['sample_id'], qa_metadata=qa.metadata)
            elif self.store_type == 'lightrag' and retrieval_topk is None:
                search_res = self.db.retrieve(query=qa.question)
            else:
                search_res = self.db.retrieve(query=qa.question, topk=retrieval_topk)
            latency = time.time() - t0

            retrieved_texts, context_blocks, retrieved_uris = self.db.process_retrieval_results(search_res)

            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)

            # 2. 构建 prompt → LLM 生成
            retrieve_in = getattr(search_res, 'retrieve_input_tokens', 0)
            retrieve_out = getattr(search_res, 'retrieve_output_tokens', 0)
            native_answer_used = bool(getattr(search_res, 'native_generation_used', False))

            if native_answer_used:
                ans_raw = getattr(search_res, 'native_final_answer', '')
                ans = self.adapter.post_process_answer(qa, ans_raw, {})
                in_tokens = getattr(search_res, 'native_input_tokens', retrieve_in)
                out_tokens = getattr(search_res, 'native_output_tokens', retrieve_out)
            else:
                full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
                ans_raw = self.llm.generate(full_prompt)
                ans = self.adapter.post_process_answer(qa, ans_raw, meta)
                in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question) + retrieve_in
                out_tokens = self.db.count_tokens(ans) + retrieve_out

            # 检查是否需要解释 Not mentioned
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
                              "recall_texts": retrieved_texts, "prompt_texts": context_blocks,
                              "sql_queries": getattr(search_res, 'sql_queries', [])},
                "llm": {"final_answer": ans, "not_mentioned_reason": not_mentioned_reason},
                "metrics": {"Recall": recall}, "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens}
            }
        except Exception as e:
            self.logger.exception(f"[Query-{task['id']}] Failed during generation task")
            self.monitor.worker_end(success=False)
            raise e

    async def _run_lightrag_single_generation_task_async(self, task, semaphore, retrieval_topk):
        qa = task['qa']
        task_id = task['id']
        sample_id = task['sample_id']
        self.monitor.worker_start()

        async with semaphore:
            t0 = time.time()
            try:
                if retrieval_topk is None:
                    search_res = await self.db.aretrieve(query=qa.question)
                else:
                    search_res = await self.db.aretrieve(query=qa.question, topk=retrieval_topk)
                latency = time.time() - t0

                retrieved_texts, context_blocks, retrieved_uris = self.db.process_retrieval_results(search_res)

                recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)

                retrieve_in = getattr(search_res, 'retrieve_input_tokens', 0)
                retrieve_out = getattr(search_res, 'retrieve_output_tokens', 0)
                native_answer_used = bool(getattr(search_res, 'native_generation_used', False))

                if native_answer_used:
                    ans_raw = getattr(search_res, 'native_final_answer', '')
                    ans = self.adapter.post_process_answer(qa, ans_raw, {})
                    in_tokens = getattr(search_res, 'native_input_tokens', retrieve_in)
                    out_tokens = getattr(search_res, 'native_output_tokens', retrieve_out)
                else:
                    full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
                    ans_raw = await self.llm.agenerate(full_prompt)
                    ans = self.adapter.post_process_answer(qa, ans_raw, meta)
                    in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question) + retrieve_in
                    out_tokens = self.db.count_tokens(ans) + retrieve_out

                not_mentioned_reason = ""
                if self.config.get('execution', {}).get('explain_not_mentioned', False):
                    if MetricsCalculator.check_refusal(ans):
                        not_mentioned_reason = await self.llm.aexplain_not_mentioned(qa.question, context_blocks)

                self.monitor.worker_end(tokens=in_tokens + out_tokens)
                self.logger.info(
                    f"[Query-{task_id}] Q: {qa.question[:30]}... | Recall: {recall:.2f} | Latency: {latency:.2f}s"
                )

                return {
                    "_global_index": task_id, "sample_id": sample_id, "question": qa.question,
                    "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
                    "retrieval": {"latency_sec": latency, "uris": retrieved_uris,
                                  "recall_texts": retrieved_texts, "prompt_texts": context_blocks,
                                  "sql_queries": getattr(search_res, 'sql_queries', [])},
                    "llm": {"final_answer": ans, "not_mentioned_reason": not_mentioned_reason},
                    "metrics": {"Recall": recall},
                    "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens}
                }
            except Exception:
                self.logger.exception(f"[Query-{task_id}] Failed during async generation task")
                self.monitor.worker_end(success=False)
                raise

    async def _run_lightrag_generation_group_async(self, pending_tasks):
        query_group_workers = int(self.config.get('execution', {}).get('query_group_workers', 10) or 10)
        retrieval_topk = self.config.get('execution', {}).get('retrieval_topk')
        semaphore = asyncio.Semaphore(query_group_workers)
        results_map = {}
        if hasattr(self.db, "aensure_ready"):
            await self.db.aensure_ready()

        pbar = tqdm(total=len(pending_tasks), desc="Generating Answers", unit="task")
        running = [
            asyncio.create_task(self._run_lightrag_single_generation_task_async(task, semaphore, retrieval_topk))
            for task in pending_tasks
        ]
        try:
            for completed in asyncio.as_completed(running):
                try:
                    res = await completed
                    results_map[res['_global_index']] = res
                except Exception as e:
                    self.logger.error(f"Generation failed during async LightRAG group execution: {e}")
                pbar.set_postfix(self.monitor.get_status_dict())
                pbar.update(1)
        finally:
            pbar.close()
            if hasattr(self.db, "afinalize"):
                await self.db.afinalize()
        return results_map

    def _process_evaluation_task(self, item):
        """
        处理单个评估任务，计算 F1 和 Accuracy 指标。
        
        对于多标注者场景（如 Qasper 数据集），一个问题可能有多个 gold answers。
        评估逻辑：
        - F1: 对每个 gold answer 分别计算，取最大值
        - Accuracy: 对每个 gold answer 分别让 LLM 判断，取最高值
        
        这样可以正确处理多标注者场景，同时保持对单答案数据集（如 Locomo）的兼容性。
        """
        ans, golds = item['llm']['final_answer'], item['gold_answers']
        
        # F1: 对每个 gold answer 分别计算，取最大值
        f1 = max((MetricsCalculator.calculate_f1(ans, gt) for gt in golds), default=0.0)
        
        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        
        # 初始化评测结果
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
                
        # 兜底：处理拒绝回答的情况
        if MetricsCalculator.check_refusal(ans) and any(MetricsCalculator.check_refusal(gt) for gt in golds):
            f1 = 1.0
            # best_eval_record["score"] = 1.0
            best_eval_record["score"] = 4.0
            best_eval_record["reasoning"] = "System successfully identified Unanswerable/Refusal condition."
            best_eval_record["prompt_type"] = "Heuristic_Refusal_Check"

        acc = best_eval_record["score"]

        # 将基础数值指标写入 metrics
        item["metrics"].update({"F1": f1, "Accuracy": acc})
        
        # 将 LLM 裁判的详细打分信息挂载到 item 下，它会被自动导出到 JSON 文件中
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
        """读取已有报告，合并新数据后写回"""
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
