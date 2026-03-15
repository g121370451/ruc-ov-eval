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
import shutil
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from tqdm import tqdm

from src.pipeline import BenchmarkPipeline
from src.adapters.base import StandardDoc, StandardSample
from src.core.metrics import MetricsCalculator


class PerQuestionPipeline(BenchmarkPipeline):

    def __init__(self, config, adapter, vector_db, llm):
        super().__init__(config, adapter, vector_db, llm)
        self.store_parent_path = config['paths']['vector_store']
        self.store_type = config.get('store', {}).get('type', 'viking')
        self.store_config = config.get('store', {})
        os.makedirs(self.store_parent_path, exist_ok=True)

        # 记录文件路径（按 store_key 索引）
        self.records_file = os.path.join(self.store_parent_path, "_ingest_records.json")
        self.records: Dict[str, dict] = self._load_records()
        self._records_lock = threading.Lock()

    # ---- 记录持久化 ----

    def _load_records(self) -> Dict[str, dict]:
        if os.path.exists(self.records_file):
            with open(self.records_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_records(self):
        with self._records_lock:
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, indent=2, ensure_ascii=False)

    # ---- store_key 计算 ----

    @staticmethod
    def _make_store_key(doc_paths: List[str]) -> str:
        """从 doc_paths 列表生成确定性的 store 标识。
        排序后对 basename 拼接取 sha1 前 16 位，避免路径过长。"""
        sorted_names = sorted(os.path.basename(p) for p in doc_paths)
        raw = "|".join(sorted_names)
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

    # ---- store 工厂 ----

    def _create_store(self, store_path):
        if self.store_type == 'pageindex':
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
        """对 store 父目录做一次备份（_backup 后缀），已存在则先删除"""
        backup_path = self.store_parent_path.rstrip('/\\') + '_backup'
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        if not os.path.exists(self.store_parent_path):
            return None
        contents = [f for f in os.listdir(self.store_parent_path) if not f.startswith('_')]
        if not contents:
            return None
        shutil.copytree(self.store_parent_path, backup_path)
        self.logger.info(f"Store parent backed up to: {backup_path}")
        return backup_path

    # ---- 主流程 ----

    def run_generation(self):
        """Phase1: 并行入库 → 备份 → Phase2: 并行检索生成"""
        self.logger.info(">>> Stage: Ingestion & Generation (Per-Question, Parallel Groups)")
        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        try:
            doc_info = self.adapter.data_prepare(doc_dir)
        except Exception:
            exit(1)

        skip_ingestion = self.config['execution'].get('skip_ingestion', False)

        # sample_id -> doc_paths（合并同 sample_id 的路径）
        sample_doc_paths: Dict[str, List[str]] = {}
        for doc in doc_info:
            sample_doc_paths.setdefault(doc.sample_id, []).extend(doc.doc_paths)

        samples = self.adapter.load_and_transform()
        max_queries = self.config['execution'].get('max_queries')

        # 按 store_key 分组 sample，保持原始顺序
        groups: OrderedDict[str, dict] = OrderedDict()
        for sample in samples:
            sid = sample.sample_id
            doc_paths = sample_doc_paths.get(sid, [])
            if not doc_paths:
                continue
            store_key = self._make_store_key(doc_paths)
            if store_key not in groups:
                groups[store_key] = {'doc_paths': doc_paths, 'samples': []}
            groups[store_key]['samples'].append(sample)

        # 分配 global_idx（预先分配，保证顺序确定）
        global_idx = 0
        group_tasks = []  # [(store_key, group, start_idx, task_count)]
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

        ingest_workers = self.config['execution'].get('ingest_workers', 4)

        # ---- Phase 1: 并行入库 ----
        failed_keys = set()
        if not skip_ingestion:
            # 断点续传：只清理未成功入库的 store 目录，保留已完成的
            if os.path.isdir(self.store_parent_path):
                for name in os.listdir(self.store_parent_path):
                    if name.startswith('_'):
                        continue
                    full = os.path.join(self.store_parent_path, name)
                    if os.path.isdir(full):
                        if name not in self.records or not self.records[name].get('ingested'):
                            shutil.rmtree(full)
                self.logger.info(f"Store parent cleaned (kept ingested): {self.store_parent_path}")
            # 清理未完成的 record 条目
            self.records = {k: v for k, v in self.records.items() if v.get('ingested')}
            self._save_records()

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

            # 入库全部完成后备份
            self._backup_store_parent()

        # 过滤掉失败的 group
        if failed_keys:
            group_tasks = [(sk, grp, si, cnt) for sk, grp, si, cnt in group_tasks if sk not in failed_keys]

        # ---- Phase 2: max_workers 控制并发度，pageindex 建议设为 1）----
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

        # 按 global_idx 排序汇总
        results_list = [all_results[i] for i in sorted(all_results.keys())]

        # 汇总入库时间（从 records 中读取）
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
        self._update_report({
            "Insertion Efficiency (Total Dataset)": {
                "Total Insertion Time (s)": sum_ingest_time,
                "Total Input Tokens": sum_ingest_in_tokens,
                "Total Output Tokens": sum_ingest_out_tokens
            }
        })

        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        save_data = {
            "summary": {"dataset": dataset_name, "total_queries": len(results_list)},
            "results": results_list
        }
        if results_list:
            total = len(results_list)
            total_in = sum(r['token_usage']['total_input_tokens'] for r in results_list)
            total_out = sum(r['token_usage']['llm_output_tokens'] for r in results_list)
            self._update_report({
                "Query Efficiency (Average Per Query)": {
                    "Average Retrieval Time (s)": sum(r['retrieval']['latency_sec'] for r in results_list) / total,
                    "Average Input Tokens": total_in / total,
                    "Average Output Tokens": total_out / total,
                    "Total Input Tokens": total_in,
                    "Total Output Tokens": total_out,
                }
            })
        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

    # ---- 入库（单组）----

    def _ingest_group(self, store_key, group):
        """单个 group 入库"""
        doc_paths = group['doc_paths']
        record = self.records.get(store_key)
        store_path = os.path.join(self.store_parent_path, store_key)

        if record and record.get('ingested'):
            self.logger.info(f"[{store_key}] Already ingested, skipping.")
            return

        t_ingest = time.time()
        store = self._create_store(store_path)
        try:
            tmp_doc = StandardDoc(sample_id=store_key, doc_paths=doc_paths)
            ingest_workers = self.config['execution'].get('ingest_workers', 10)
            stats = store.ingest([tmp_doc], max_workers=ingest_workers, monitor=self.monitor)
        except Exception as e:
            self.logger.error(f"[{store_key}] Ingest error: {e}")
            raise
        finally:
            self._close_store(store)
        elapsed_ingest = time.time() - t_ingest
        with self._records_lock:
            self.records[store_key] = {
                'ingested': True,
                'doc_paths': doc_paths,
                'ingest_time': elapsed_ingest,
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

        self._close_store(store)
        return results_map

    # ---- 检索 + 生成 ----

    def _retrieve_and_generate(self, task_id, sample_id, qa, store):
        """单个问题：从单个 store 检索 → 生成答案"""
        self.monitor.worker_start()
        try:
            topk = self.config['execution']['retrieval_topk']
            t0 = time.time()
            res = store.retrieve(query=qa.question, topk=topk)
            latency = time.time() - t0

            retrieve_in = getattr(res, 'retrieve_input_tokens', 0)
            retrieve_out = getattr(res, 'retrieve_output_tokens', 0)

            retrieved_texts, context_blocks, retrieved_uris = \
                store.process_retrieval_results(res)
            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)

            if self.store_type == 'sql_agent':
                # SQL Agent 的 retrieve 已经直接生成了答案
                ans = getattr(res, 'agent_answer', '')
                in_tok = retrieve_in
                out_tok = retrieve_out
            else:
                full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
                ans_raw = self.llm.generate(full_prompt)
                ans = self.adapter.post_process_answer(qa, ans_raw, meta)
                in_tok = store.count_tokens(full_prompt) + store.count_tokens(qa.question) + retrieve_in
                out_tok = store.count_tokens(ans) + retrieve_out

            self.monitor.worker_end(tokens=in_tok + out_tok)

            self.logger.info(f"[Query-{task_id}] Q: {qa.question[:30]}... | Recall: {recall:.2f} | Latency: {latency:.2f}s")
            return {
                "_global_index": task_id, "sample_id": sample_id,
                "question": qa.question, "gold_answers": qa.gold_answers,
                "category": str(qa.category), "evidence": qa.evidence,
                "retrieval": {"latency_sec": latency, "uris": retrieved_uris},
                "llm": {"final_answer": ans},
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
