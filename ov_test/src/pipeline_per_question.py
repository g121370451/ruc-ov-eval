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
from collections import OrderedDict
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

    # ---- 记录持久化 ----

    def _load_records(self) -> Dict[str, dict]:
        if os.path.exists(self.records_file):
            with open(self.records_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_records(self):
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
        """对 store 父目录做一次备份（_backup 后缀），已存在则跳过"""
        backup_path = self.store_parent_path.rstrip('/\\') + '_backup'
        if os.path.exists(backup_path):
            self.logger.info(f"Backup already exists: {backup_path}, skipping.")
            return backup_path
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
        """按 store_key 分组：入库 → 检索该组所有 QA → 删除。"""
        self.logger.info(">>> Stage: Ingestion & Generation (Per-Question, Record Mode)")
        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        try:
            doc_info = self.adapter.data_prepare(doc_dir)
        except Exception:
            exit(1)

        # 未跳过入库时：备份 store 父目录，然后清空以便重新入库
        skip_ingestion = self.config['execution'].get('skip_ingestion', False)
        if not skip_ingestion:
            self._backup_store_parent()
            # 清空 store 父目录（保留 _ 开头的元数据文件）
            if os.path.isdir(self.store_parent_path):
                for name in os.listdir(self.store_parent_path):
                    if name.startswith('_'):
                        continue
                    full = os.path.join(self.store_parent_path, name)
                    if os.path.isdir(full):
                        shutil.rmtree(full)
                self.logger.info(f"Store parent cleared: {self.store_parent_path}")
            # 清空记录，重新入库
            self.records.clear()
            self._save_records()

        # sample_id -> doc_paths（合并同 sample_id 的路径）
        sample_doc_paths: Dict[str, List[str]] = {}
        for doc in doc_info:
            sample_doc_paths.setdefault(doc.sample_id, []).extend(doc.doc_paths)

        samples = self.adapter.load_and_transform()
        max_queries = self.config['execution'].get('max_queries')

        # 按 store_key 分组 sample，保持原始顺序
        # store_key -> { 'doc_paths': [...], 'samples': [StandardSample, ...] }
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

        global_idx = 0
        results_list = []
        sum_ingest_time = 0.0
        sum_ingest_in_tokens = 0
        sum_ingest_out_tokens = 0
        sum_delete_time = 0.0

        pbar = tqdm(groups.items(), desc="Processing Store Groups", unit="group")
        for store_key, group in pbar:
            if max_queries is not None and global_idx >= max_queries:
                break

            doc_paths = group['doc_paths']
            record = self.records.get(store_key)
            store_path = os.path.join(self.store_parent_path, store_key)

            # ---- 入库（有记录则跳过）----
            if record and record.get('ingested'):
                self.logger.info(f"[{store_key}] Already ingested, skipping.")
                sum_ingest_time += record.get('ingest_time', 0)
                sum_ingest_in_tokens += record.get('ingest_input_tokens', 0)
                sum_ingest_out_tokens += record.get('ingest_output_tokens', 0)
            else:
                t_ingest = time.time()
                store = self._create_store(store_path)
                tmp_doc = StandardDoc(sample_id=store_key, doc_paths=doc_paths)
                stats = store.ingest([tmp_doc], monitor=self.monitor)
                self._close_store(store)
                elapsed_ingest = time.time() - t_ingest
                ingest_in = stats.get('input_tokens', 0)
                ingest_out = stats.get('output_tokens', 0)
                sum_ingest_time += elapsed_ingest
                sum_ingest_in_tokens += ingest_in
                sum_ingest_out_tokens += ingest_out
                self.records[store_key] = {
                    'ingested': True,
                    'doc_paths': doc_paths,
                    'ingest_time': elapsed_ingest,
                    'ingest_input_tokens': ingest_in,
                    'ingest_output_tokens': ingest_out,
                    'deleted': False,
                    'delete_time': 0,
                }
                self._save_records()

            # ---- 打开 store 做检索 ----
            store = self._create_store(store_path)

            for sample in group['samples']:
                for qa in sample.qa_pairs:
                    if max_queries is not None and global_idx >= max_queries:
                        break
                    result = self._retrieve_and_generate(
                        global_idx, sample.sample_id, qa, store
                    )
                    results_list.append(result)
                    global_idx += 1
                if max_queries is not None and global_idx >= max_queries:
                    break

            # ---- 删除（有记录则跳过）----
            if record and record.get('deleted'):
                self.logger.info(f"[{store_key}] Already deleted, skipping.")
                sum_delete_time += record.get('delete_time', 0)
                self._close_store(store)
            else:
                from src.core.backup_utils import backup_store as do_backup
                backup_path = do_backup(store_path, self.logger)
                # 计时：只包含 clear
                t_del = time.time()
                store.clear()
                elapsed_del = time.time() - t_del
                self._close_store(store)
                # 从备份恢复
                if backup_path and os.path.isdir(backup_path):
                    if os.path.isdir(store_path):
                        shutil.rmtree(store_path)
                    shutil.copytree(backup_path, store_path)
                    self.logger.info(f"[{store_key}] Restored from backup: {backup_path}")
                sum_delete_time += elapsed_del
                if store_key in self.records:
                    self.records[store_key]['deleted'] = True
                    self.records[store_key]['delete_time'] = elapsed_del
                    self._save_records()

        pbar.close()

        # ---- 汇总报告 ----
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
            },
            "Deletion Efficiency (Total Dataset)": {
                "Total Deletion Time (s)": sum_delete_time
            }
        })

        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        save_data = {
            "summary": {"dataset": dataset_name, "total_queries": len(results_list)},
            "results": results_list
        }
        if results_list:
            total = len(results_list)
            self._update_report({
                "Query Efficiency (Average Per Query)": {
                    "Average Retrieval Time (s)": sum(r['retrieval']['latency_sec'] for r in results_list) / total,
                    "Average Input Tokens": sum(r['token_usage']['total_input_tokens'] for r in results_list) / total,
                    "Average Output Tokens": sum(r['token_usage']['llm_output_tokens'] for r in results_list) / total,
                }
            })
        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

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
        """备份 → 逐个 store 调用 clear 计时 → 关闭 → 恢复"""
        from src.core.backup_utils import backup_store as do_backup
        self.logger.info(">>> Stage: Deletion (Per-Question)")
        total_del_time = 0.0

        if os.path.isdir(self.store_parent_path):
            for name in os.listdir(self.store_parent_path):
                if name.startswith('_'):
                    continue
                sp = os.path.join(self.store_parent_path, name)
                if not os.path.isdir(sp):
                    continue
                backup_path = do_backup(sp, self.logger)
                store = self._create_store(sp)
                t0 = time.time()
                store.clear()
                total_del_time += time.time() - t0
                self._close_store(store)
                # 恢复
                if backup_path and os.path.isdir(backup_path):
                    if os.path.isdir(sp):
                        shutil.rmtree(sp)
                    shutil.copytree(backup_path, sp)
                    self.logger.info(f"[{name}] Restored from backup: {backup_path}")

        self.metrics_summary["deletion"] = {"time": total_del_time, "input_tokens": 0, "output_tokens": 0}
        self.logger.info(f"Deletion finished. Time: {total_del_time:.2f}s")
