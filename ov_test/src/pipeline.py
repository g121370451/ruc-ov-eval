# src/pipeline.py
import asyncio
import os
import json
import time
import threading
from typing import Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.adapters.base import BaseAdapter
from src.core.logger import get_logger
from src.core.checkpoint import CheckpointManager

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
        """Step1: 数据预处理 + 入库"""
        self.logger.info(">>> Stage: Ingestion")
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

        ingested_samples = self.checkpoint_manager.get_ingested_samples()
        if ingested_samples:
            self.logger.info("Checkpoint indicates ingestion already completed, skipping.")
            skip_ingestion = True
            ingest_stats = self.checkpoint_manager.get_ingest_stats()
            self.metrics_summary["insertion"] = ingest_stats

        if skip_ingestion:
            self.logger.info(f"Skipping Ingestion. Using existing docs at: {doc_dir}")
            if not os.path.exists(doc_dir):
                 self.logger.warning(f"Warning: Doc directory {doc_dir} not found, but ingestion is skipped.")
            if not ingested_samples:
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

            self.checkpoint_manager.update_ingested_samples(
                ingested_samples={"all"}, total_samples=1, ingest_stats=ingest_stats
            )
            # 将 insertion 效率数据写入报告
            self._update_report({
                "Insertion Efficiency (Total Dataset)": {
                    "Total Insertion Time (s)": self.metrics_summary["insertion"]["time"],
                    "Total Input Tokens": self.metrics_summary["insertion"]["input_tokens"],
                    "Total Output Tokens": self.metrics_summary["insertion"]["output_tokens"]
                }
            })
            # backup_store(store_path, self.logger)

    def run_generation(self):
        """Step 2 & 3: 检索生成（默认复用已有入库结果）"""
        self.logger.info(">>> Stage: Generation (Retrieve + Generate)")

        # 1. 始终加载数据
        samples = self.adapter.load_and_transform()
        # 2. 准备 QA 任务
        tasks = self._prepare_tasks(samples)

        # 断点恢复：从 checkpoint 加载已完成的 task ID，从 JSON 文件加载结果
        completed_tasks: Set[int] = self.checkpoint_manager.get_completed_tasks("generation")
        results_map = {}
        if completed_tasks and os.path.exists(self.generated_file):
            try:
                with open(self.generated_file, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                for result in saved_data.get("results", []):
                    results_map[result["_global_index"]] = result
            except Exception as e:
                self.logger.warning(f"Failed to load previous generated results: {e}")

        pending_tasks = [task for task in tasks if task["id"] not in completed_tasks]
        if completed_tasks:
            self.logger.info(f"Resumed {len(completed_tasks)} completed generation tasks, {len(pending_tasks)} remaining")

        if self.store_type == 'lightrag':
            async_results = asyncio.run(self._run_lightrag_generation_group_async(pending_tasks, completed_tasks))
            results_map.update(async_results)
        else:
            max_workers = self.config.get('execution', {}).get('max_workers', 1)

            if pending_tasks:
                initial_completed = len(completed_tasks)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_task = {
                        executor.submit(self._process_generation_task, task): task
                        for task in pending_tasks
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

        # 3. 保存最终回答文件
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
        self.checkpoint_manager.delete_checkpoint()

    def run_evaluation(self):
        """Step 4: 结果评测打分"""
        self.logger.info(">>> Stage: Evaluation")

        if not os.path.exists(self.generated_file):
            self.logger.error("Generated answers file not found.")
            return

        with open(self.generated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("results", [])

        completed_eval_tasks: Set[int] = self.checkpoint_manager.get_completed_tasks("evaluation")
        eval_results_map = {}
        if completed_eval_tasks and os.path.exists(self.eval_file):
            try:
                with open(self.eval_file, "r", encoding="utf-8") as f:
                    saved_eval_data = json.load(f)
                for result in saved_eval_data.get("results", []):
                    eval_results_map[result["_global_index"]] = result
            except Exception as e:
                self.logger.warning(f"Failed to load previous eval results: {e}")

        pending_items = [item for item in items if item["_global_index"] not in completed_eval_tasks]
        if completed_eval_tasks:
            self.logger.info(f"Resumed {len(completed_eval_tasks)} evaluated tasks, {len(pending_items)} remaining")

        if pending_items:
            initial_completed_eval = len(completed_eval_tasks)
            with ThreadPoolExecutor(max_workers=self.config.get('execution', {}).get('max_workers', 1)) as executor:
                future_to_item = {
                    executor.submit(self._process_evaluation_task, item): item
                    for item in pending_items
                }

                pbar = tqdm(total=len(items), desc="Evaluating", unit="item", initial=len(completed_eval_tasks))
                for future in as_completed(future_to_item):
                    try:
                        res = future.result()
                        eval_results_map[res['_global_index']] = res
                        completed_eval_tasks.add(res['_global_index'])

                        newly_completed_eval = len(completed_eval_tasks) - initial_completed_eval
                        if newly_completed_eval % self.save_frequency == 0 or len(completed_eval_tasks) == len(items):
                            self.checkpoint_manager.update_completed_tasks("evaluation", completed_eval_tasks, len(items))
                            self._save_partial_eval_results(eval_results_map)
                    except Exception as e:
                        self.logger.error(f"Evaluation failed: {e}")
                    pbar.update(1)
                pbar.close()
        else:
            self.logger.info("All evaluation tasks already completed!")

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
        self.checkpoint_manager.delete_checkpoint()

    def run_deletion(self):
        """Step 5: 计时删除"""
        self.logger.info(">>> Stage: Deletion")
        t0 = time.time()
        delete_stats = self.db.clear()
        elapsed = time.time() - t0
        delete_input_tokens = 0
        delete_output_tokens = 0
        if isinstance(delete_stats, dict):
            delete_input_tokens = int(delete_stats.get("input_tokens", 0) or 0)
            delete_output_tokens = int(delete_stats.get("output_tokens", 0) or 0)
        self.metrics_summary["deletion"] = {
            "time": elapsed,
            "input_tokens": delete_input_tokens,
            "output_tokens": delete_output_tokens,
        }
        self._update_report({
            "Deletion Efficiency (Total Dataset)": {
                "Total Deletion Time (s)": elapsed,
                "Total Input Tokens": delete_input_tokens,
                "Total Output Tokens": delete_output_tokens,
            }
        })
        self.logger.info(f"Deletion finished. Time: {elapsed:.2f}s")

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
            retrieve_usage_source = getattr(search_res, 'retrieve_token_usage_source', '')
            retrieve_official_calls = getattr(search_res, 'retrieve_official_usage_calls', 0)
            retrieve_estimated_calls = getattr(search_res, 'retrieve_estimated_usage_calls', 0)
            retrieve_missing_official_calls = getattr(search_res, 'retrieve_missing_official_usage_calls', 0)
            native_answer_used = bool(getattr(search_res, 'native_generation_used', False))

            if native_answer_used:
                ans_raw = getattr(search_res, 'native_final_answer', '')
                ans = self.adapter.post_process_answer(qa, ans_raw, {})
                answer_in = getattr(search_res, 'native_input_tokens', 0)
                answer_out = getattr(search_res, 'native_output_tokens', 0)
                answer_usage_source = getattr(search_res, 'native_token_usage_source', 'native')
                in_tokens = retrieve_in + answer_in
                out_tokens = retrieve_out + answer_out
            else:
                full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
                if hasattr(self.llm, "generate_with_usage"):
                    ans_raw, answer_usage = self.llm.generate_with_usage(full_prompt)
                else:
                    ans_raw = self.llm.generate(full_prompt)
                    answer_usage = {"usage_source": "missing_official"}
                ans = self.adapter.post_process_answer(qa, ans_raw, meta)
                answer_usage_source = answer_usage.get("usage_source", "missing_official")
                answer_in = answer_usage.get("input_tokens", 0)
                answer_out = answer_usage.get("output_tokens", 0)
                if answer_usage_source != "official":
                    answer_usage_source = "estimated"
                    answer_in = self.db.count_tokens(full_prompt)
                    answer_out = self.db.count_tokens(ans)
                in_tokens = retrieve_in + answer_in
                out_tokens = retrieve_out + answer_out

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
                "metrics": {"Recall": recall},
                "token_usage": {
                    "total_input_tokens": in_tokens,
                    "llm_output_tokens": out_tokens,
                    "retrieve_token_usage_source": retrieve_usage_source,
                    "retrieve_official_usage_calls": retrieve_official_calls,
                    "retrieve_estimated_usage_calls": retrieve_estimated_calls,
                    "retrieve_missing_official_usage_calls": retrieve_missing_official_calls,
                    "answer_input_tokens": answer_in,
                    "answer_output_tokens": answer_out,
                    "answer_token_usage_source": answer_usage_source,
                }
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
                    answer_in = getattr(search_res, 'native_input_tokens', 0)
                    answer_out = getattr(search_res, 'native_output_tokens', 0)
                    answer_usage_source = getattr(search_res, 'native_token_usage_source', 'native')
                    in_tokens = retrieve_in + answer_in
                    out_tokens = retrieve_out + answer_out
                else:
                    full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
                    if hasattr(self.llm, "agenerate_with_usage"):
                        ans_raw, answer_usage = await self.llm.agenerate_with_usage(full_prompt)
                    else:
                        ans_raw = await self.llm.agenerate(full_prompt)
                        answer_usage = {"usage_source": "missing_official"}
                    ans = self.adapter.post_process_answer(qa, ans_raw, meta)
                    answer_usage_source = answer_usage.get("usage_source", "missing_official")
                    answer_in = answer_usage.get("input_tokens", 0)
                    answer_out = answer_usage.get("output_tokens", 0)
                    if answer_usage_source != "official":
                        answer_usage_source = "estimated"
                        answer_in = self.db.count_tokens(full_prompt)
                        answer_out = self.db.count_tokens(ans)
                    in_tokens = retrieve_in + answer_in
                    out_tokens = retrieve_out + answer_out

                not_mentioned_reason = ""
                if self.config.get('execution', {}).get('explain_not_mentioned', False):
                    if MetricsCalculator.check_refusal(ans):
                        not_mentioned_reason = await self.llm.aexplain_not_mentioned(qa.question, context_blocks)

                self.monitor.worker_end(tokens=in_tokens + out_tokens)
                return {
                    "_global_index": task_id, "sample_id": sample_id, "question": qa.question,
                    "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
                    "retrieval": {"latency_sec": latency, "uris": retrieved_uris,
                                  "recall_texts": retrieved_texts, "prompt_texts": context_blocks,
                                  "sql_queries": getattr(search_res, 'sql_queries', [])},
                    "llm": {"final_answer": ans, "not_mentioned_reason": not_mentioned_reason},
                    "metrics": {"Recall": recall},
                    "token_usage": {
                        "total_input_tokens": in_tokens,
                        "llm_output_tokens": out_tokens,
                        "answer_input_tokens": answer_in,
                        "answer_output_tokens": answer_out,
                        "answer_token_usage_source": answer_usage_source,
                    }
                }
            except Exception:
                self.logger.exception(f"[Query-{task_id}] Failed during async generation task")
                self.monitor.worker_end(success=False)
                raise

    async def _run_lightrag_generation_group_async(self, pending_tasks, completed_tasks: Set[int]):
        query_group_workers = int(self.config.get('execution', {}).get('query_group_workers', 10) or 10)
        retrieval_topk = self.config.get('execution', {}).get('retrieval_topk')
        semaphore = asyncio.Semaphore(query_group_workers)
        results_map = {}
        total_tasks = len(pending_tasks) + len(completed_tasks)
        if hasattr(self.db, "aensure_ready"):
            await self.db.aensure_ready()

        initial_completed = len(completed_tasks)
        if not pending_tasks:
            self.logger.info("All generation tasks already completed!")
            return results_map

        running = [
            asyncio.create_task(self._run_lightrag_single_generation_task_async(task, semaphore, retrieval_topk))
            for task in pending_tasks
        ]
        pbar = tqdm(
            total=total_tasks,
            desc="Generating Queries",
            unit="query",
            initial=initial_completed,
        )
        try:
            for completed in asyncio.as_completed(running):
                try:
                    res = await completed
                    results_map[res['_global_index']] = res
                    completed_tasks.add(res['_global_index'])

                    newly_completed = len(completed_tasks) - initial_completed
                    progress = f"{len(completed_tasks)}/{total_tasks}"
                    recall = res.get("metrics", {}).get("Recall", 0.0)
                    latency = res.get("retrieval", {}).get("latency_sec", 0.0)
                    question = str(res.get("question", ""))
                    self.logger.info(
                        f"[{progress}] [Query-{res['_global_index']}] "
                        f"Q: {question[:30]}... | Recall: {recall:.2f} | Latency: {latency:.2f}s"
                    )
                    if newly_completed % self.save_frequency == 0 or len(completed_tasks) == total_tasks:
                        self.checkpoint_manager.update_completed_tasks("generation", completed_tasks, total_tasks)
                        self._save_partial_results(results_map)
                except Exception as e:
                    self.logger.error(f"Generation failed during async LightRAG group execution: {e}")
                finally:
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
