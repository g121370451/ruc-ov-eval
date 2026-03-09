# src/pipeline_per_question.py
"""
逐问题评估策略（基于 target_uri 限定检索范围）：

继承 BenchmarkPipeline，唯一区别是检索时通过 target_uri 限定到当前问题相关的文档。
入库、评估、删除、保存、报告逻辑全部复用父类。
"""
import os
import time

from src.pipeline import BenchmarkPipeline
from src.core.metrics import MetricsCalculator


class PerQuestionPipeline(BenchmarkPipeline):

    def run_generation(self):
        """
        覆写父类 run_generation：
        在入库完成后、检索之前，构建 sample_id -> URI 映射，
        检索时使用 target_uri 限定范围。其余流程与父类完全一致。
        """
        self.logger.info(">>> Stage: Ingestion & Generation (Per-Question)")
        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        # 0. 预处理数据集
        try:
            doc_info = self.adapter.data_prepare(doc_dir)
        except Exception as e:
            exit(1)

        # 1. 入库（与父类逻辑一致）
        skip_ingestion = self.config['execution'].get('skip_ingestion', False)
        if skip_ingestion:
            self.logger.info(f"Skipping Ingestion. Using existing docs")
        else:
            ingest_workers = self.config['execution'].get('ingest_workers', 10)
            ingest_stats = self.db.ingest(
                doc_info,
                max_workers=ingest_workers,
                monitor=self.monitor
            )
            self.metrics_summary["insertion"] = ingest_stats
            self.logger.info(f"Insertion finished. Time: {ingest_stats['time']:.2f}s")

            self._update_report({
                "Insertion Efficiency (Total Dataset)": {
                    "Total Insertion Time (s)": self.metrics_summary["insertion"]["time"],
                    "Total Input Tokens": self.metrics_summary["insertion"]["input_tokens"],
                    "Total Output Tokens": self.metrics_summary["insertion"]["output_tokens"]
                }
            })

        # 2. 构建 URI 映射（委托给 store 实现）
        self._uri_map = self.db.build_uri_map(doc_info)
        self.logger.info(f"URI map built: {len(self._uri_map)} samples mapped")

        # 3. 加载数据 + 检索生成（与父类一致，但 _process_generation_task 被覆写）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        samples = self.adapter.load_and_transform()
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

        # 4. 保存（与父类一致）
        import json
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
            })
        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
    def _process_generation_task(self, task):
        """
        覆写父类：检索时对每个 target_uri 分别检索，合并结果按 score 取 topK。
        其余逻辑（prompt构建、生成、token统计）与父类完全一致。
        """
        self.monitor.worker_start()
        try:
            qa = task['qa']
            topk = self.config['execution']['retrieval_topk']
            target_uris = self._uri_map.get(task['sample_id'], [])

            # 1. 限定路径检索
            t0 = time.time()
            all_resources = []
            retrieve_in_tokens = 0
            retrieve_out_tokens = 0
            if target_uris:
                for uri in target_uris:
                    try:
                        res = self.db.retrieve(query=qa.question, topk=topk, target_uri=uri)
                        all_resources.extend(res.resources)
                        retrieve_in_tokens += getattr(res, 'retrieve_input_tokens', 0)
                        retrieve_out_tokens += getattr(res, 'retrieve_output_tokens', 0)
                    except Exception as e:
                        self.logger.warning(f"Retrieve from {uri} failed: {e}")
            else:
                # 回退到全局检索
                self.logger.warning("No target URIs found for sample_id %s, falling back to global retrieval.", task['sample_id'])
                res = self.db.retrieve(query=qa.question, topk=topk)
                all_resources = list(res.resources)
                retrieve_in_tokens += getattr(res, 'retrieve_input_tokens', 0)
                retrieve_out_tokens += getattr(res, 'retrieve_output_tokens', 0)

            # 按 score 降序取 topK
            all_resources.sort(key=lambda r: getattr(r, 'score', 0), reverse=True)
            top_resources = all_resources[:topk]
            latency = time.time() - t0
            # 2. 构造临时结果对象，复用 process_retrieval_results 接口
            class _TempResult:
                def __init__(self, resources):
                    self.resources = resources
            retrieved_texts, context_blocks, retrieved_uris = self.db.process_retrieval_results(_TempResult(top_resources))
            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)

            # 3. Prompt + 生成（与父类一致）
            full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
            ans_raw = self.llm.generate(full_prompt)
            ans = self.adapter.post_process_answer(qa, ans_raw, meta)

            # 4. Token stats（含检索阶段 token）
            in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question) + retrieve_in_tokens
            out_tokens = self.db.count_tokens(ans) + retrieve_out_tokens
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