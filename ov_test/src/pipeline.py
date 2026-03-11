# src/pipeline.py
import os
import json
import time
import random
import re
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
            exit(1)
        skip_ingestion = self.config['execution'].get('skip_ingestion', False)

        if skip_ingestion:
            self.logger.info(f"Skipping Ingestion. Using existing docs at: {doc_dir}")
            if not os.path.exists(doc_dir):
                 self.logger.warning(f"Warning: Doc directory {doc_dir} not found, but ingestion is skipped.")
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
            ingest_workers = self.config['execution'].get('ingest_workers', 10)
            ingest_stats = self.db.ingest(
                doc_info,
                max_workers=ingest_workers,
                monitor=self.monitor
            )
            self.metrics_summary["insertion"] = ingest_stats
            self.logger.info(f"Insertion finished. Time: {ingest_stats['time']:.2f}s")
            # 入库完成后备份
            backup_store(store_path, self.logger)

            # 将 insertion 效率数据写入报告
            self._update_report({
                "Insertion Efficiency (Total Dataset)": {
                    "Total Insertion Time (s)": self.metrics_summary["insertion"]["time"],
                    "Total Input Tokens": self.metrics_summary["insertion"]["input_tokens"],
                    "Total Output Tokens": self.metrics_summary["insertion"]["output_tokens"]
                }
            })
        """Step 2 & 3: 数据入库 + 检索生成"""
        # 1. 始终加载数据
        samples = self.adapter.load_and_transform()    
        # 2. 准备 QA 任务
        tasks = self._prepare_tasks(samples)
        results_map = {}
        max_workers = self.config['execution']['max_workers']
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._process_generation_task, task): task 
                for task in tasks
            }
            
            pbar = tqdm(total=len(tasks), desc="Generating Answers", unit="task")
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    res = future.result()
                    results_map[res['_global_index']] = res
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
        
        with ThreadPoolExecutor(max_workers=self.config['execution']['max_workers']) as executor:
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
            search_res = self.db.retrieve(query=qa.question, topk=self.config['execution']['retrieval_topk'])
            latency = time.time() - t0
            
            retrieved_texts, context_blocks, retrieved_uris = self.db.process_retrieval_results(search_res)
            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)
            
            # 2. Prompting logic (调用 Adapter 动态生成)
            full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
            
            # 3. Generation
            ans_raw = self.llm.generate(full_prompt)

            # 4. Post-processing (调用 Adapter 动态解析)
            ans = self.adapter.post_process_answer(qa, ans_raw, meta)

            # 5. Token stats（含检索阶段 token）
            retrieve_in = getattr(search_res, 'retrieve_input_tokens', 0)
            retrieve_out = getattr(search_res, 'retrieve_output_tokens', 0)
            in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question) + retrieve_in
            out_tokens = self.db.count_tokens(ans) + retrieve_out
            self.monitor.worker_end(tokens=in_tokens + out_tokens)
            
            self.logger.info(f"[Query-{task['id']}] Q: {qa.question[:30]}... | Recall: {recall:.2f} | Latency: {latency:.2f}s")

            return {
                "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
                "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
                "retrieval": {"latency_sec": latency, "uris": retrieved_uris},
                "llm": {"final_answer": ans},
                "metrics": {"Recall": recall}, "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens}
            }
        except Exception as e:
            self.monitor.worker_end(success=False)
            raise e

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
        
        # 初始化最优评测结果存储字典
        best_eval_record = {
            "score": 0.0,
            "reasoning": "",
            "prompt_type": ""
        }
        
        for gt in golds:
            try:
                eval_res = llm_grader(
                    self.llm.llm, 
                    self.config['llm']['model'], 
                    item['question'], 
                    gt,  # 单个 gold answer
                    ans,
                    dataset_name=dataset_name
                )
                
                # 如果有多个答案，保留得分最高的那次评测的理由和分数
                if eval_res["score"] >= best_eval_record["score"]:
                    best_eval_record = eval_res
                    
            except Exception as e:
                self.logger.error(f"Grader error for gold answer '{gt[:50]}...': {e}")
                
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