"""
bench_framework 泛化 Pipeline。

- generation 阶段返回 GenerationRecord dataclass，不再计算 Recall
- eval 阶段支持可插拔的 RecallStrategy，fallback 到 MetricsCalculator.check_recall()
- 序列化使用 GenerationRecord.to_dict() 保持 JSON 格式兼容
"""
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, TypedDict, Union

from tqdm import tqdm

from bench_framework.adapters.base import BaseAdapter, StandardQA, StandardSample
from bench_framework.stores.base import VectorStoreBase
from bench_framework.core.logger import get_logger
from bench_framework.core.monitor import BenchmarkMonitor
from bench_framework.core.metrics import MetricsCalculator
from bench_framework.core.judge_util import llm_grader
from bench_framework.core.llm_client import LLMClientWrapper
from bench_framework.types import (
    TokenUsage,
    RetrievalInfo,
    GenerationRecord,
    LLMEvaluation,
)
from bench_framework.recall.base import BaseRecallStrategy


class GenerationTask(TypedDict):
    """单条 QA 生成任务的结构"""
    id: int
    sample_id: str
    qa: StandardQA


class BenchmarkPipeline:
    def __init__(
        self,
        config: dict,
        adapter: BaseAdapter,
        vector_db: VectorStoreBase,
        llm: LLMClientWrapper,
        recall_strategy: Optional[BaseRecallStrategy] = None,
    ):
        self.config = config
        self.adapter = adapter
        self.db = vector_db
        self.llm = llm
        self.logger = get_logger()
        self.monitor = BenchmarkMonitor()
        self.recall_strategy = recall_strategy

        self.output_dir: str = self.config['paths']['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.generated_file = os.path.join(self.output_dir, "generated_answers.json")
        self.eval_file = os.path.join(self.output_dir, "qa_eval_detailed_results.json")
        self.report_file = os.path.join(self.output_dir, "benchmark_metrics_report.json")

        self.metrics_summary: Dict[str, Dict[str, Union[int, float]]] = {
            "insertion": {"time": 0, "input_tokens": 0, "output_tokens": 0},
            "deletion": {"time": 0, "input_tokens": 0, "output_tokens": 0}
        }

    # ==============================================================
    # Generation
    # ==============================================================

    def run_generation(self) -> None:
        """数据预处理 + 入库 + 检索生成"""
        self.logger.info(">>> Stage: Ingestion & Generation")
        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        try:
            doc_info = self.adapter.data_prepare(doc_dir)
        except Exception:
            exit(1)

        skip_ingestion = self.config['execution'].get('skip_ingestion', False)
        if skip_ingestion:
            self.logger.info(f"Skipping Ingestion. Using existing docs at: {doc_dir}")
            if not os.path.exists(doc_dir):
                self.logger.warning(f"Warning: Doc directory {doc_dir} not found, but ingestion is skipped.")
            self.metrics_summary["insertion"] = {"time": 0, "input_tokens": 0, "output_tokens": 0}
        else:
            ingest_workers = self.config['execution'].get('ingest_workers', 10)
            ingest_stats = self.db.ingest(doc_info, max_workers=ingest_workers, monitor=self.monitor)
            self.metrics_summary["insertion"] = {
                "time": ingest_stats.time,
                "input_tokens": ingest_stats.input_tokens,
                "output_tokens": ingest_stats.output_tokens,
            }
            self.logger.info(f"Insertion finished. Time: {ingest_stats.time:.2f}s")
            self._update_report({
                "Insertion Efficiency (Total Dataset)": {
                    "Total Insertion Time (s)": ingest_stats.time,
                    "Total Input Tokens": ingest_stats.input_tokens,
                    "Total Output Tokens": ingest_stats.output_tokens,
                }
            })

        # 加载数据 & 准备 QA 任务
        samples = self.adapter.load_and_transform()
        tasks = self._prepare_tasks(samples)
        results_map: Dict[int, GenerationRecord] = {}
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
                    results_map[res.global_index] = res
                except Exception as e:
                    self.logger.error(f"Generation failed for task {task['id']}: {e}")
                    self.monitor.worker_end(success=False)
                pbar.set_postfix(self.monitor.get_status_dict())
                pbar.update(1)
            pbar.close()

        sorted_results: List[GenerationRecord] = [results_map[i] for i in sorted(results_map.keys())]
        dataset_name: str = self.config.get('dataset_name', 'Unknown_Dataset')
        save_data = {
            "summary": {"dataset": dataset_name, "total_queries": len(sorted_results)},
            "results": [r.to_dict() for r in sorted_results]
        }
        total: int = len(sorted_results)
        if total > 0:
            self._update_report({
                "Query Efficiency (Average Per Query)": {
                    "Average Retrieval Time (s)": sum(r.retrieval.latency_sec for r in sorted_results) / total,
                    "Average Input Tokens": sum(r.token_usage.input_tokens for r in sorted_results) / total,
                    "Average Output Tokens": sum(r.token_usage.output_tokens for r in sorted_results) / total,
                }
            })
        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

    # ==============================================================
    # Evaluation
    # ==============================================================

    def run_evaluation(self) -> None:
        """结果评测打分"""
        self.logger.info(">>> Stage: Evaluation")
        if not os.path.exists(self.generated_file):
            self.logger.error("Generated answers file not found.")
            return

        with open(self.generated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            items: List[GenerationRecord] = [
                GenerationRecord.from_dict(d) for d in data.get("results", [])
            ]

        eval_results_map: Dict[int, GenerationRecord] = {}
        with ThreadPoolExecutor(max_workers=self.config['execution']['max_workers']) as executor:
            future_to_item = {
                executor.submit(self._process_evaluation_task, item): item
                for item in items
            }
            pbar = tqdm(total=len(items), desc="Evaluating", unit="item")
            for future in as_completed(future_to_item):
                try:
                    res: GenerationRecord = future.result()
                    eval_results_map[res.global_index] = res
                except Exception as e:
                    self.logger.error(f"Evaluation failed: {e}")
                pbar.update(1)
            pbar.close()

        eval_records: List[GenerationRecord] = list(eval_results_map.values())
        total: int = len(eval_records)

        with open(self.eval_file, "w", encoding="utf-8") as f:
            json.dump({"results": [r.to_dict() for r in eval_records]}, f, indent=2, ensure_ascii=False)

        if total > 0:
            self._update_report({
                "Dataset": self.config.get('dataset_name', 'Unknown_Dataset'),
                "Total Queries Evaluated": total,
                "Performance Metrics": {
                    "Average F1 Score": sum(r.metrics.get('F1', 0.0) for r in eval_records) / total,
                    "Average Recall": sum(r.metrics.get('Recall', 0.0) for r in eval_records) / total,
                    "Average Accuracy (Hit  0-4 )": sum(r.metrics.get('Accuracy', 0.0) for r in eval_records) / total,
                    "Average Accuracy (normalization)": (sum(r.metrics.get('Accuracy', 0.0) for r in eval_records) / total) / 4,
                }
            })

    # ==============================================================
    # Deletion
    # ==============================================================

    def run_deletion(self) -> None:
        """数据清理"""
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
                "Total Output Tokens": 0,
            }
        })

    # ==============================================================
    # Internal helpers
    # ==============================================================

    def _prepare_tasks(self, samples: List[StandardSample]) -> List[GenerationTask]:
        tasks: List[GenerationTask] = []
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

    def _process_generation_task(self, task: GenerationTask) -> GenerationRecord:
        self.monitor.worker_start()
        try:
            qa = task['qa']

            # 1. Retrieval
            t0 = time.time()
            search_res = self.db.retrieve(
                query=qa.question,
                topk=self.config['execution']['retrieval_topk'],
            )
            latency = time.time() - t0

            retrieved_texts: List[str] = []
            retrieved_uris: List[str] = []
            context_blocks: List[str] = []
            for r in search_res.resources:
                retrieved_uris.append(r.uri)
                content = (
                    self.db.read_resource(r.uri)
                    if r.level == 2
                    else f"{r.abstract}\n{r.overview}"
                )
                retrieved_texts.append(content)
                context_blocks.append(content[:2000])

            # 2. Build prompt
            full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)

            # 3. Generation
            ans_raw = self.llm.generate(full_prompt)

            # 4. Post-processing
            ans = self.adapter.post_process_answer(qa, ans_raw, meta)

            # 5. Token stats
            in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question)
            out_tokens = self.db.count_tokens(ans)
            self.monitor.worker_end(tokens=in_tokens + out_tokens)

            self.logger.info(
                f"[Query-{task['id']}] Q: {qa.question[:30]}... | Latency: {latency:.2f}s"
            )

            return GenerationRecord(
                global_index=task['id'],
                sample_id=task['sample_id'],
                question=qa.question,
                gold_answers=qa.gold_answers,
                category=str(qa.category),
                evidence=qa.evidence,
                retrieval=RetrievalInfo(
                    uris=retrieved_uris,
                    latency_sec=latency,
                    retrieved_texts=retrieved_texts,
                ),
                final_answer=ans,
                token_usage=TokenUsage(
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                ),
            )
        except Exception as e:
            self.monitor.worker_end(success=False)
            raise e

    def _process_evaluation_task(self, item: GenerationRecord) -> GenerationRecord:
        """
        处理单个评估任务，计算 F1 和 Accuracy 指标。

        多标注者场景（如 Qasper）：一个问题可能有多个 gold answers。
        - F1: 对每个 gold answer 分别计算，取最大值
        - Accuracy: 对每个 gold answer 分别让 LLM 判断，取最高分
        - Recall: 如果有 recall_strategy 则使用，否则 fallback 到 MetricsCalculator
        """
        ans = item.final_answer
        golds = item.gold_answers

        # F1
        f1 = max((MetricsCalculator.calculate_f1(ans, gt) for gt in golds), default=0.0)

        # Recall
        recall = 0.0
        if self.recall_strategy and item.evidence:
            try:
                locations = self.recall_strategy.preprocess(item.evidence)
                recall = self.recall_strategy.compute_recall(locations, item.retrieval)
            except Exception as e:
                self.logger.warning(f"RecallStrategy failed, fallback: {e}")
                recall = MetricsCalculator.check_recall(
                    item.retrieval.retrieved_texts, item.evidence,
                )
        elif item.evidence:
            recall = MetricsCalculator.check_recall(
                item.retrieval.retrieved_texts, item.evidence,
            )

        # Accuracy (LLM judge)
        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        best_score = 0.0
        best_reasoning = ""
        best_prompt_type = ""

        for gt in golds:
            try:
                eval_res = llm_grader(
                    self.llm.llm,
                    self.config['llm']['model'],
                    item.question,
                    gt,
                    ans,
                    dataset_name=dataset_name,
                )
                if eval_res["score"] >= best_score:
                    best_score = float(eval_res["score"])
                    best_reasoning = str(eval_res["reasoning"])
                    best_prompt_type = str(eval_res["prompt_type"])
            except Exception as e:
                self.logger.error(f"Grader error for gold answer '{gt[:50]}...': {e}")

        # 拒绝回答兜底
        if MetricsCalculator.check_refusal(ans) and any(
            MetricsCalculator.check_refusal(gt) for gt in golds
        ):
            f1 = 1.0
            best_score = 4.0
            best_reasoning = "System successfully identified Unanswerable/Refusal condition."
            best_prompt_type = "Heuristic_Refusal_Check"

        item.metrics = {"F1": f1, "Recall": recall, "Accuracy": best_score}
        item.llm_evaluation = LLMEvaluation(
            prompt_used=best_prompt_type,
            reasoning=best_reasoning,
            normalized_score=best_score,
        )

        self.logger.info(
            f"\n{'=' * 60}"
            f"\n[Query ID]: {item.global_index}"
            f"\n[Question]: {item.question}"
            f"\n[Retrieved URIs]: {item.retrieval.uris}"
            f"\n[LLM Answer]: {ans}"
            f"\n[Gold Answer]: {golds}"
            f"\n[Metrics]: {item.metrics}"
            f"\n[LLM Judge Reasoning]: {best_reasoning}"
            f"\n{'=' * 60}"
        )
        return item

    def _update_report(self, data: Dict[str, Union[str, int, float, dict]]) -> None:
        """读取已有报告，合并新数据后写回"""
        report: dict = {}
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
