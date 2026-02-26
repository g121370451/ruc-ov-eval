# src/pipeline.py
import os
import json
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .core.monitor import BenchmarkMonitor
from .core.metrics import MetricsCalculator
from .core.judge_util import locomo_grader

QA_PROMPT = """Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""
QA_PROMPT_CAT_5 = """Based on the above context, answer the following question.

Question: {} Short answer:
"""
MISSING_RULE = "If no information is available to answer the question, write 'Not mentioned'."

class BenchmarkPipeline:
    def __init__(self, config, adapter, vector_db, llm, logger):
        self.config = config
        self.adapter = adapter
        self.db = vector_db
        self.llm = llm
        self.logger = logger
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
        """Step 2 & 3: 数据入库 + 检索生成"""
        self.logger.info(">>> Stage: Ingestion & Generation")
        
        # 1. 始终加载数据
        samples = self.adapter.load_and_transform()
        
        skip_ingestion = self.config['execution'].get('skip_ingestion', False)
        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        if skip_ingestion:
            self.logger.info(f"Skipping Ingestion. Using existing docs at: {doc_dir}")
            if not os.path.exists(doc_dir):
                 self.logger.warning(f"Warning: Doc directory {doc_dir} not found, but ingestion is skipped.")
            self.metrics_summary["insertion"] = {"time": 0, "input_tokens": 0, "output_tokens": 0}
            
        else:  # 正常执行入库
            os.makedirs(doc_dir, exist_ok=True)
            ingest_stats = self.db.ingest(samples, base_dir=doc_dir)
            self.metrics_summary["insertion"] = ingest_stats
            self.logger.info(f"Insertion finished. Time: {ingest_stats['time']:.2f}s")

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
        save_data = {
            "summary": {"dataset": "LocoMo", "total_queries": len(sorted_results)},
            "results": sorted_results
        }
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

        # 过滤 Category 5
        eval_items = [item for item in items if str(item.get('category')) != '5']
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

        # 汇总并保存最终报告 
        self._save_reports(list(eval_results_map.values()))

    def run_deletion(self):
        """Step 5: 数据清理"""
        self.logger.info(">>> Stage: Deletion")
        start_time = time.time()
        self.db.clear()
        duration = time.time() - start_time
        self.metrics_summary["deletion"] = {"time": duration, "input_tokens": 0, "output_tokens": 0}
        self.logger.info(f"Deletion finished. Time: {duration:.2f}s")

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
            category = str(qa.category)
            
            # Retrieval
            t0 = time.time()
            search_res = self.db.retrieve(query=qa.question, topk=self.config['execution']['retrieval_topk'])
            latency = time.time() - t0
            
            retrieved_texts = []
            context_blocks = []
            for r in search_res.resources:
                content = self.db.read_resource(r.uri) if getattr(r, 'is_leaf', False) else f"{getattr(r, 'abstract', '')}\n{getattr(r, 'overview', '')}"
                retrieved_texts.append(content)
                clean = re.sub(r' \[.*?\]', '', content)[:2000]
                context_blocks.append(clean)

            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)
            
            # Prompting logic
            eff_q, tmpl, opts = qa.question, QA_PROMPT, None
            if category == "2": eff_q += " Use DATE of CONVERSATION to answer with an approximate date."
            if category == "5":
                tmpl, gold = QA_PROMPT_CAT_5, qa.gold_answers[0] if qa.gold_answers else None
                if gold and gold.lower() != "not mentioned":
                    if random.random() < 0.5:
                        eff_q += f" Select the correct answer: (a) Not mentioned in the conversation (b) {gold}."
                        opts = {"a": "Not mentioned in the conversation", "b": gold}
                    else:
                        eff_q += f" Select the correct answer: (a) {gold} (b) Not mentioned in the conversation."
                        opts = {"a": gold, "b": "Not mentioned in the conversation"}

            context_text = "\n\n".join(context_blocks)
            full_prompt = f"{context_text}\n\n{MISSING_RULE}\n\n{tmpl.format(eff_q)}"
            
            # Generation
            ans_raw = self.llm.generate(full_prompt)
            ans = ans_raw.strip()
            if category == "5" and opts:
                mp = ans.lower()
                if len(mp) == 1 and mp in opts: ans = opts[mp]
                elif len(mp) == 3 and mp[1] == ')' and mp[0] in opts: ans = opts[mp[0]]

            # Token stats (Consistent with original script)
            in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question)
            out_tokens = self.db.count_tokens(ans)
            self.monitor.worker_end(tokens=in_tokens + out_tokens)

            return {
                "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
                "gold_answers": qa.gold_answers, "category": category, "evidence": qa.evidence,
                "retrieval": {"latency_sec": latency}, "llm": {"final_answer": ans},
                "metrics": {"Recall": recall}, "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens}
            }
        except Exception as e:
            self.monitor.worker_end(success=False)
            raise e

    def _process_evaluation_task(self, item):
        ans, golds = item['llm']['final_answer'], item['gold_answers']
        f1 = max((MetricsCalculator.calculate_f1(ans, gt) for gt in golds), default=0.0)
        
        # Accuracy via LLM Judge (Consistent with judge_util.py)
        try:
            acc = 1.0 if locomo_grader(self.llm.llm, self.config['llm']['model'], item['question'], "\n".join(golds), ans) else 0.0
        except:
            acc = 0.0

        if MetricsCalculator.check_refusal(ans) and any(MetricsCalculator.check_refusal(gt) for gt in golds):
            f1, acc = 1.0, 1.0

        item["metrics"].update({"F1": f1, "Accuracy": acc})
        return item

    def _save_reports(self, eval_records):
        """生成最终报告"""
        total = len(eval_records)
        if total == 0: return

        # 计算平均性能指标
        avg_f1 = sum(r['metrics']['F1'] for r in eval_records) / total
        avg_acc = sum(r['metrics']['Accuracy'] for r in eval_records) / total
        avg_recall = sum(r['metrics']['Recall'] for r in eval_records) / total
        
        # 计算效率指标
        avg_lat = sum(r['retrieval']['latency_sec'] for r in eval_records) / total
        avg_in_tokens = sum(r['token_usage']['total_input_tokens'] for r in eval_records) / total
        avg_out_tokens = sum(r['token_usage']['llm_output_tokens'] for r in eval_records) / total

        report_data = {
            "Dataset": "LocoMo",
            "Total Queries Evaluated": total,
            "Performance Metrics": {
                "Average F1 Score": avg_f1,
                "Average Recall": avg_recall,
                "Average Accuracy (Hit Rate)": avg_acc
            },
            "Query Efficiency (Average Per Query)": {
                "Average Retrieval Time (s)": avg_lat,
                "Average Input Tokens": avg_in_tokens,
                "Average Output Tokens": avg_out_tokens
            },
            "Insertion Efficiency (Total Dataset)": {
                "Total Insertion Time (s)": self.metrics_summary["insertion"]["time"],
                "Total Input Tokens": self.metrics_summary["insertion"]["input_tokens"],
                "Total Output Tokens": self.metrics_summary["insertion"]["output_tokens"]
            },
            "Deletion Efficiency (Total Dataset)": {
                "Total Deletion Time (s)": self.metrics_summary["deletion"]["time"],
                "Total Input Tokens": self.metrics_summary["deletion"]["input_tokens"],
                "Total Output Tokens": self.metrics_summary["deletion"]["output_tokens"]
            }
        }

        # 保存详细评测文件
        with open(self.eval_file, "w", encoding="utf-8") as f:
            json.dump({"summary": report_data, "results": eval_records}, f, indent=2, ensure_ascii=False)

        # 保存汇总报告文件
        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)

        self.logger.info(f"Final Report saved to {self.report_file}")
        print(json.dumps(report_data, indent=4, ensure_ascii=False))