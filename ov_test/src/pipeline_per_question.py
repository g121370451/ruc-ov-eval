# src/pipeline_per_question.py
"""
逐问题评估策略（记录模式）：

将每个 sample 的 doc_paths 列表作为整体入库到同一个 store。
doc_paths 内容完全相同的 sample 共享同一个 store（通过 store_key 去重）。
流程按 store_key 分组：入库 → 检索该组所有 QA → 删除。
通过 _ingest_records.json 记录每个 store_key 的入库/删除时间和 tokens，
已有记录的 store_key 跳过入库和删除。
启动时对 store 父目录做一次备份。
"""
import os
import json
import hashlib
import re
import shutil
import time
import threading
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from tqdm import tqdm

from src.pipeline import BenchmarkPipeline
from src.adapters.base import StandardDoc, StandardSample
from src.core.metrics import MetricsCalculator
from src.core.store_contract import store_provides_final_answer


#region debug-point kohaku-coredump-pipeline
def _debug_report(event: str, **payload):
    try:
        url = "http://127.0.0.1:7777/event"
        data = json.dumps({
            "sessionId": "kohaku-coredump",
            "runId": "pre",
            "hypothesisId": payload.pop("hypothesisId", "unknown"),
            "event": event,
            "payload": payload,
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=0.2).close()
    except Exception:
        pass
#endregion debug-point kohaku-coredump-pipeline


class PerQuestionPipeline(BenchmarkPipeline):

    def __init__(self, config, adapter, vector_db, llm, resume: bool = False):
        super().__init__(config, adapter, vector_db, llm, resume=resume)
        self.store_parent_path = config['paths']['vector_store']
        self.store_type = config.get('store', {}).get('type', 'viking')
        self.store_config = config.get('store', {})
        os.makedirs(self.store_parent_path, exist_ok=True)

        if self.store_type == 'KohakuRAG':
            _debug_report("pipeline.kohaku_prewarm_import.before", hypothesisId="H2")
            import src.core.kohaku_store  # noqa: F401
            _debug_report("pipeline.kohaku_prewarm_import.after", hypothesisId="H2")

        # 记录文件路径（按 store_key 索引）
        self.records_file = os.path.join(self.store_parent_path, "_ingest_records.json")
        self.records: Dict[str, dict] = self._load_records()
        self._records_lock = threading.Lock()

    # ---- 记录持久化 ----

    def _load_records(self) -> Dict[str, dict]:
        if os.path.exists(self.records_file):
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                from src.core.logger import get_logger
                get_logger().warning(f"Failed to load records file {self.records_file}, starting fresh: {e}")
        return {}

    def _save_records(self):
        with self._records_lock:
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, indent=2, ensure_ascii=False)

    # ---- store_key 计算 ----

    @staticmethod
    def _make_store_key(doc_paths: List[str]) -> str:
        """从 doc_paths 列表生成确定性的 store 标识。
        单文件时直接用文件名（去扩展名），多文件时取 sha1 前 16 位。"""
        if len(doc_paths) == 1:
            name = os.path.splitext(os.path.basename(doc_paths[0]))[0]
            # 清理非法目录字符，截断过长名称
            name = re.sub(r'[\\/*?:"<>|]', '_', name)[:120]
            return name if name else hashlib.sha1(doc_paths[0].encode('utf-8')).hexdigest()[:16]
        sorted_names = sorted(os.path.basename(p) for p in doc_paths)
        raw = "|".join(sorted_names)
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

    # ---- store 工厂 ----

    def _create_store(self, store_path):
        if self.store_type == 'DeepRead':
            from src.core.deepread_store import DeepReadWrapper
            return DeepReadWrapper.from_config(
                store_path=store_path,
                doc_output_dir=self.config['paths'].get('doc_output_dir', ''),
                output_dir=self.output_dir,
                llm_cfg=self.config.get('llm', {}),
                store_cfg=self.config.get('store', {})
            )
        elif self.store_type == 'KohakuRAG':
            _debug_report("pipeline.kohaku_import.before", hypothesisId="H2")
            from src.core.kohaku_store import KohakuStoreWrapper
            _debug_report("pipeline.kohaku_import.after", hypothesisId="H2")
            _debug_report("pipeline.kohaku_from_config.before", hypothesisId="H2", store_path=store_path)
            return KohakuStoreWrapper.from_config(
                store_path=store_path,
                doc_output_dir=self.config['paths'].get('doc_output_dir', ''),
                llm_cfg=self.config.get('llm', {}),
                store_cfg=self.config.get('store', {})
            )
        elif self.store_type == 'pageindex':
            from src.core.pageindex_store import PageIndexStoreWrapper
            pageindex_conf = self.store_config.get('pageindex_config_path')
            return PageIndexStoreWrapper(
                store_path=store_path,
                doc_output_dir=self.config['paths'].get('doc_output_dir', ''),
                config_path=pageindex_conf
            )
        elif self.store_type == 'hipporag':
            from src.core.hipporag_store import HippoRAGStoreWrapper
            hipporag_conf = self.store_config.get('hipporag_config', {})
            return HippoRAGStoreWrapper(
                store_path=store_path,
                hipporag_config=hipporag_conf
            )
        elif self.store_type == 'sql_agent':
            from src.core.sql_agent_store import SQLAgentStoreWrapper
            sql_agent_conf = self.store_config.get('sql_agent_config', {})
            sql_agent_conf['dataset_name'] = self.config.get('dataset_name', '')
            sql_agent_conf['raw_data_path'] = self.config['paths'].get('raw_data', '')
            return SQLAgentStoreWrapper(
                store_path=store_path,
                sql_agent_config=sql_agent_conf
            )
        elif self.store_type == 'graphrag':
            from src.core.graphrag_store import GraphRAGStoreWrapper
            return GraphRAGStoreWrapper.from_config(
                store_path=store_path,
                llm_cfg=self.config.get('llm', {}),
                store_cfg=self.store_config,
            )
        else:
            from src.core.vector_store import VikingStoreWrapper
            return VikingStoreWrapper(store_path=store_path)

    def _close_store(self, store):
        if hasattr(store, 'client') and hasattr(store.client, 'close'):
            try:
                store.client.close()
            except Exception:
                pass

    # ---- 备份 ----

    def _backup_store_parent(self):
        """对 store 父目录做一次备份（_backup 后缀），用新入库结果替换旧备份"""
        backup_path = self.store_parent_path.rstrip('/\\') + '_backup'
        if not os.path.exists(self.store_parent_path):
            return None
        contents = [f for f in os.listdir(self.store_parent_path) if not f.startswith('_')]
        if not contents:
            return None
        # 先复制到临时路径，成功后再替换旧备份，避免中途失败丢失两份数据
        temp_backup = backup_path + '_tmp'
        if os.path.exists(temp_backup):
            shutil.rmtree(temp_backup)
        shutil.copytree(self.store_parent_path, temp_backup)
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        os.rename(temp_backup, backup_path)
        self.logger.info(f"Store parent backed up to: {backup_path}")
        return backup_path

    # ---- 主流程 ----

    def _compute_group_tasks(self):
        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        try:
            doc_info = self.adapter.data_prepare(doc_dir)
        except Exception:
            self.logger.error("Data preparation failed")
            raise

        sample_doc_paths: Dict[str, List[str]] = {}
        for doc in doc_info:
            sample_doc_paths.setdefault(doc.sample_id, []).extend(doc.doc_paths)

        samples = self.adapter.load_and_transform()
        max_queries = self.config['execution'].get('max_queries')
        use_all_docs_when_no_mapping = self.config['execution'].get('use_all_docs_when_no_mapping', False)
        all_doc_paths = []
        if use_all_docs_when_no_mapping:
            seen_doc_paths = set()
            for doc in doc_info:
                for path in doc.doc_paths:
                    if path not in seen_doc_paths:
                        all_doc_paths.append(path)
                        seen_doc_paths.add(path)

        groups: OrderedDict[str, dict] = OrderedDict()
        fallback_count = 0
        for sample in samples:
            sid = sample.sample_id
            doc_paths = sample_doc_paths.get(sid, [])
            if not doc_paths and use_all_docs_when_no_mapping:
                doc_paths = all_doc_paths
                fallback_count += 1
            if not doc_paths:
                continue
            store_key = self._make_store_key(doc_paths)
            if store_key not in groups:
                groups[store_key] = {'doc_paths': doc_paths, 'samples': []}
            groups[store_key]['samples'].append(sample)
        if fallback_count:
            self.logger.warning(
                f"Applied all-docs fallback for {fallback_count} samples without explicit doc mapping."
            )

        global_idx = 0
        group_tasks = []
        for store_key, group in groups.items():
            start_idx = global_idx
            count = 0
            for sample in group['samples']:
                for _ in sample.qa_pairs:
                    if max_queries is not None and global_idx >= max_queries:
                        break
                    global_idx += 1
                    count += 1
                if max_queries is not None and global_idx >= max_queries:
                    break
            group_tasks.append((store_key, group, start_idx, count))
            if max_queries is not None and global_idx >= max_queries:
                break

        worker_id = self.config['execution'].get('worker_id')
        num_workers = self.config['execution'].get('num_workers')
        if worker_id is not None and num_workers is not None:
            group_tasks = group_tasks[worker_id::num_workers]
            self.logger.info(f"Shard {worker_id}/{num_workers}: {len(group_tasks)} tasks")

        return group_tasks

    def run_ingestion(self):
        self.logger.info(">>> Stage: Ingestion Only (Per-Question, Parallel Groups)")

        if self.resume:
            all_ingested = all(rec.get('ingested') for rec in self.records.values()) if self.records else False
            if all_ingested and self.records:
                self.logger.info("All groups already ingested from records, skipping.")
                return

        group_tasks = self._compute_group_tasks()
        ingest_workers = self.config['execution'].get('ingest_workers', 4)

        failed_keys = set()

        if os.path.isdir(self.store_parent_path):
            for name in os.listdir(self.store_parent_path):
                if name.startswith('_'):
                    continue
                full = os.path.join(self.store_parent_path, name)
                if os.path.isdir(full):
                    if name not in self.records or not self.records[name].get('ingested'):
                        shutil.rmtree(full)
            self.logger.info(f"Store parent cleaned (kept ingested): {self.store_parent_path}")
        self.records = {k: v for k, v in self.records.items() if v.get('ingested')}
        self._save_records()

        if ingest_workers <= 1:
            pbar = tqdm(total=len(group_tasks), desc="Ingesting Groups", unit="group")
            for sk, grp, _, _ in group_tasks:
                try:
                    self._ingest_group(sk, grp)
                except Exception as e:
                    self.logger.error(f"Ingest group {sk} failed: {e}")
                    failed_keys.add(sk)
                pbar.update(1)
            pbar.close()
        else:
            ingest_timeout = self.config['execution'].get('ingest_timeout')
            with ThreadPoolExecutor(max_workers=ingest_workers) as executor:
                future_to_key = {
                    executor.submit(self._ingest_group, sk, grp): sk
                    for sk, grp, _, _ in group_tasks
                }
                pbar = tqdm(total=len(future_to_key), desc="Ingesting Groups", unit="group")
                try:
                    for future in as_completed(future_to_key, timeout=ingest_timeout):
                        sk = future_to_key[future]
                        try:
                            future.result()
                        except Exception as e:
                            self.logger.error(f"Ingest group {sk} failed: {e}")
                            failed_keys.add(sk)
                        pbar.update(1)
                except TimeoutError:
                    for fut, sk in future_to_key.items():
                        if not fut.done():
                            self.logger.error(f"Ingest group {sk} timed out")
                            failed_keys.add(sk)
                            fut.cancel()
                pbar.close()

        if failed_keys:
            self.logger.warning(f"Failed/timed-out groups: {failed_keys}")

        sum_ingest_time = 0.0
        sum_ingest_in_tokens = 0
        sum_ingest_out_tokens = 0
        for rec in self.records.values():
            sum_ingest_time += rec.get('ingest_time', 0)
            sum_ingest_in_tokens += rec.get('ingest_input_tokens', 0)
            sum_ingest_out_tokens += rec.get('ingest_output_tokens', 0)

        self.metrics_summary["insertion"] = {
            "time": sum_ingest_time,
            "input_tokens": sum_ingest_in_tokens,
            "output_tokens": sum_ingest_out_tokens
        }
        self._backup_store_parent()
        self._update_report({
            "Insertion Efficiency (Total Dataset)": {
                "Total Insertion Time (s)": sum_ingest_time,
                "Total Input Tokens": sum_ingest_in_tokens,
                "Total Output Tokens": sum_ingest_out_tokens
            }
        })

    def run_generation(self):
        self.logger.info(">>> Stage: Generation (Per-Question, Parallel Groups)")

        group_tasks = self._compute_group_tasks()

        failed_keys = set()
        for sk, grp, _, _ in group_tasks:
            record = self.records.get(sk)
            if not record or not record.get('ingested'):
                failed_keys.add(sk)
        if failed_keys:
            self.logger.warning(f"Groups not yet ingested (skipping): {failed_keys}")
            group_tasks = [(sk, grp, si, cnt) for sk, grp, si, cnt in group_tasks if sk not in failed_keys]

        max_workers = self.config['execution'].get('max_workers', 1)
        all_results = {}
        if max_workers <= 1:
            pbar = tqdm(total=len(group_tasks), desc="Query Groups", unit="group")
            for sk, grp, si, cnt in group_tasks:
                try:
                    group_result = self._query_group(sk, grp, si, cnt)
                    all_results.update(group_result)
                except Exception as e:
                    self.logger.error(f"Group {sk} failed: {e}")
                pbar.update(1)
            pbar.close()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {
                    executor.submit(self._query_group, sk, grp, si, cnt): sk
                    for sk, grp, si, cnt in group_tasks
                }
                pbar = tqdm(total=len(future_to_key), desc="Query Groups", unit="group")
                for future in as_completed(future_to_key):
                    sk = future_to_key[future]
                    try:
                        group_result = future.result()
                        all_results.update(group_result)
                    except Exception as e:
                        self.logger.error(f"Group {sk} failed: {e}")
                    pbar.update(1)
                pbar.close()

        results_list = [all_results[i] for i in sorted(all_results.keys())]

        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        save_data = {
            "summary": {"dataset": dataset_name, "total_queries": len(results_list)},
            "results": results_list
        }
        if results_list:
            total = len(results_list)
            total_in = sum(r['token_usage']['total_input_tokens'] for r in results_list)
            total_out = sum(r['token_usage']['llm_output_tokens'] for r in results_list)
            latency_scopes = sorted({
                r['retrieval'].get('latency_scope', 'retrieval_only')
                for r in results_list
            })
            self._update_report({
                "Query Efficiency (Average Per Query)": {
                    "Average Retrieval Time (s)": sum(r['retrieval']['latency_sec'] for r in results_list) / total,
                    "Latency Scope": latency_scopes[0] if len(latency_scopes) == 1 else latency_scopes,
                    "Average Input Tokens": total_in / total,
                    "Average Output Tokens": total_out / total,
                    "Total Input Tokens": total_in,
                    "Total Output Tokens": total_out,
                }
            })
        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

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

    # ---- 入库（单组）----

    def _ingest_group(self, store_key, group):
        """单个 group 入库"""
        doc_paths = group['doc_paths']
        record = self.records.get(store_key)
        store_path = os.path.join(self.store_parent_path, store_key)
        _debug_report("pipeline.ingest_group.start", hypothesisId="H2", store_key=store_key, store_path=store_path, doc_paths=doc_paths)

        if record and record.get('ingested'):
            self.logger.info(f"[{store_key}] Already ingested, skipping.")
            _debug_report("pipeline.ingest_group.skip_record", hypothesisId="H5", store_key=store_key)
            return

        t_ingest = time.time()
        _debug_report("pipeline.create_store.before", hypothesisId="H2", store_key=store_key, store_type=self.store_type)
        store = self._create_store(store_path)
        _debug_report("pipeline.create_store.after", hypothesisId="H2", store_key=store_key, store_class=store.__class__.__name__)
        try:
            if self.store_type == 'sql_agent':
                # SQL Agent 需要真实 sample_id 来过滤原始数据
                docs = []
                for s in group['samples']:
                    doc_meta = {}
                    # 收集 QA 元数据中的辅助信息（如 HotpotQA 的 supporting_titles）
                    all_titles = set()
                    for qa in s.qa_pairs:
                        all_titles.update(qa.metadata.get('supporting_fact_titles', []))
                    if all_titles:
                        doc_meta['supporting_titles'] = list(all_titles)
                    docs.append(StandardDoc(sample_id=s.sample_id, doc_paths=doc_paths,
                                            metadata=doc_meta))
            else:
                docs = [StandardDoc(sample_id=store_key, doc_paths=doc_paths)]
            # Per-question already parallelizes across document groups. Keep
            # each group ingestion serial by default to avoid unsafe nested
            # writes in stores backed by native SQLite/vector extensions.
            doc_ingest_workers = self.config['execution'].get('doc_ingest_workers', 1)
            _debug_report("pipeline.store_ingest.before", hypothesisId="H3", store_key=store_key, doc_ingest_workers=doc_ingest_workers, docs=[d.doc_paths for d in docs])
            stats = store.ingest(docs, max_workers=doc_ingest_workers, monitor=self.monitor)
            _debug_report("pipeline.store_ingest.after", hypothesisId="H3", store_key=store_key, stats=stats)
        except Exception as e:
            _debug_report("pipeline.ingest_group.exception", hypothesisId="H3", store_key=store_key, error=repr(e))
            self.logger.error(f"[{store_key}] Ingest error: {e}")
            raise
        finally:
            _debug_report("pipeline.close_store.before", hypothesisId="H2", store_key=store_key)
            self._close_store(store)
            _debug_report("pipeline.close_store.after", hypothesisId="H2", store_key=store_key)
        elapsed_ingest = time.time() - t_ingest
        with self._records_lock:
            self.records[store_key] = {
                'ingested': True,
                'doc_paths': doc_paths,
                'ingest_time': stats.get('time',0),
                'ingest_input_tokens': stats.get('input_tokens', 0),
                'ingest_output_tokens': stats.get('output_tokens', 0),
                'deleted': False,
                'delete_time': 0,
            }
        self._save_records()

    # ---- 检索生成（单组）----

    def _query_group(self, store_key, group, start_idx, task_count):
        """单个 group：串行检索生成。返回 {idx: result}"""
        store_path = os.path.join(self.store_parent_path, store_key)
        store = self._create_store(store_path)

        try:
            qa_tasks = []
            idx = start_idx
            for sample in group['samples']:
                for qa in sample.qa_pairs:
                    if idx >= start_idx + task_count:
                        break
                    qa_tasks.append({'id': idx, 'sample_id': sample.sample_id, 'qa': qa})
                    idx += 1
                if idx >= start_idx + task_count:
                    break

            results_map = {}
            for t in qa_tasks:
                try:
                    res = self._retrieve_and_generate(t['id'], t['sample_id'], t['qa'], store)
                    results_map[res['_global_index']] = res
                except Exception as e:
                    self.logger.error(f"Generation failed for task {t['id']}: {e}")
        finally:
            self._close_store(store)

        return results_map

    # ---- 检索 + 生成 ----

    def _retrieve_and_generate(self, task_id, sample_id, qa, store):
        """单个问题：从单个 store 检索 → 生成答案"""
        self.monitor.worker_start()
        try:
            topk = self.config['execution']['retrieval_topk']
            t0 = time.time()
            if self.store_type == 'sql_agent':
                res = store.retrieve(query=qa.question, topk=topk,
                                     sample_id=sample_id, qa_metadata=qa.metadata)
            else:
                res = store.retrieve(query=qa.question, topk=topk)
            latency = time.time() - t0

            retrieve_in = getattr(res, 'retrieve_input_tokens', 0)
            retrieve_out = getattr(res, 'retrieve_output_tokens', 0)

            retrieved_texts, context_blocks, retrieved_uris = store.process_retrieval_results(res)
            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)

            provides_final_answer = store_provides_final_answer(store)
            if provides_final_answer:
                ans = store.get_final_answer(res)
                in_tok = retrieve_in
                out_tok = retrieve_out
            else:
                full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
                ans_raw = self.llm.generate(full_prompt)
                ans = self.adapter.post_process_answer(qa, ans_raw, meta)

                in_tok = store.count_tokens(full_prompt) + store.count_tokens(qa.question) + retrieve_in
                out_tok = store.count_tokens(ans) + retrieve_out

            # 检查是否需要解释 Not mentioned
            not_mentioned_reason = ""
            if self.config.get('execution', {}).get('explain_not_mentioned', False):
                if MetricsCalculator.check_refusal(ans):
                    not_mentioned_reason = self.llm.explain_not_mentioned(qa.question, context_blocks)

            self.monitor.worker_end(tokens=in_tok + out_tok)

            self.logger.info(f"[Query-{task_id}] Q: {qa.question[:30]}... | Recall: {recall:.2f} | Latency: {latency:.2f}s")
            return {
                "_global_index": task_id, "sample_id": sample_id,
                "question": qa.question, "gold_answers": qa.gold_answers,
                "category": str(qa.category), "evidence": qa.evidence,
                "retrieval": {"latency_sec": latency, "uris": retrieved_uris,
                              "latency_scope": "end_to_end" if provides_final_answer else "retrieval_only",
                              "query_mode": getattr(res, 'query_mode', None),
                              "internal_llm_calls": getattr(res, 'llm_calls', 0),
                              "llm_calls_categories": getattr(res, 'llm_calls_categories', {}),
                              "input_tokens_categories": getattr(res, 'input_tokens_categories', {}),
                              "output_tokens_categories": getattr(res, 'output_tokens_categories', {}),
                              "recall_texts": retrieved_texts, "prompt_texts": context_blocks,
                              "sql_queries": getattr(res, 'sql_queries', [])},
                "llm": {"final_answer": ans, "not_mentioned_reason": not_mentioned_reason},
                "metrics": {"Recall": recall},
                "token_usage": {"total_input_tokens": in_tok, "llm_output_tokens": out_tok}
            }
        except Exception as e:
            self.monitor.worker_end(success=False)
            raise e

    def run_deletion(self):
        """逐个 store 调用 clear 计时删除"""
        self.logger.info(">>> Stage: Deletion (Per-Question)")
        total_del_time = 0.0

        if os.path.isdir(self.store_parent_path):
            for name in os.listdir(self.store_parent_path):
                if name.startswith('_'):
                    continue
                sp = os.path.join(self.store_parent_path, name)
                if not os.path.isdir(sp):
                    continue
                store = self._create_store(sp)
                t0 = time.time()
                store.clear()
                elapsed = time.time() - t0
                total_del_time += elapsed
                self._close_store(store)
                self.logger.info(f"[{name}] Cleared in {elapsed:.2f}s")
                with self._records_lock:
                    if name in self.records:
                        self.records[name]['deleted'] = True
                        self.records[name]['delete_time'] = elapsed
        self._save_records()

        self.metrics_summary["deletion"] = {"time": total_del_time, "input_tokens": 0, "output_tokens": 0}
        self._update_report({
            "Deletion Efficiency (Total Dataset)": {
                "Total Deletion Time (s)": total_del_time
            }
        })
        self.logger.info(f"Deletion finished. Time: {total_del_time:.2f}s")
