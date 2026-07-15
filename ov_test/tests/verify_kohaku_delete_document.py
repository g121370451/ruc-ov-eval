"""验证 KohakuStoreWrapper.delete_document / clear 的正确性。

该脚本包含三部分：
1. 合成数据下验证 delete_document
2. 合成数据下验证 clear() 会遍历所有 doc 并保留 DB 文件
3. 在真实 DB 的拷贝上验证 clear()（不删除原 DB）

不调用任何 Embedding / LLM API。
"""

import os
import sys
import gc
import asyncio
import shutil
import sqlite3
import tempfile
import numpy as np

# 让脚本能找到 ov_test/src 下的模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from kohakurag.datastore import KVaultNodeStore
from kohakurag.types import StoredNode, NodeKind
from kohakuvault import TextVault

from src.core.kohaku_store import KohakuStoreWrapper


DIM = 8
REAL_DB_PATH = "/Users/zhangqianyi/Desktop/ruc-ov/Data/FinanceBench_sample20/KohakuRAG/store/kohaku.db"


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _make_doc_nodes(doc_id: str) -> list[StoredNode]:
    """构造一篇文档的层次节点：doc -> sec1 -> p1 -> s1。"""
    base_vec = _normalize(np.arange(DIM, dtype=np.float32))

    root = StoredNode(
        node_id=doc_id,
        parent_id=None,
        kind=NodeKind.DOCUMENT,
        title=doc_id,
        text=f"Document {doc_id}",
        metadata={"document_id": doc_id},
        embedding=base_vec.copy(),
        child_ids=[f"{doc_id}:sec1"],
    )
    sec = StoredNode(
        node_id=f"{doc_id}:sec1",
        parent_id=doc_id,
        kind=NodeKind.SECTION,
        title="Section",
        text=f"Section of {doc_id}",
        metadata={"document_id": doc_id},
        embedding=base_vec.copy(),
        child_ids=[f"{doc_id}:sec1:p1"],
    )
    para = StoredNode(
        node_id=f"{doc_id}:sec1:p1",
        parent_id=f"{doc_id}:sec1",
        kind=NodeKind.PARAGRAPH,
        title="Paragraph",
        text=f"Paragraph of {doc_id}",
        metadata={"document_id": doc_id},
        embedding=base_vec.copy(),
        child_ids=[f"{doc_id}:sec1:p1:s1"],
    )
    sent = StoredNode(
        node_id=f"{doc_id}:sec1:p1:s1",
        parent_id=f"{doc_id}:sec1:p1",
        kind=NodeKind.SENTENCE,
        title="Sentence",
        text=f"Sentence of {doc_id}",
        metadata={"document_id": doc_id},
        embedding=base_vec.copy(),
        child_ids=[],
    )
    return [root, sec, para, sent]


async def _build_test_db(
    db_path: str, doc_ids: list[str], bm25_pairs: list[tuple[str, str]]
):
    """创建测试数据并附带 BM25 索引。"""
    store = KVaultNodeStore(db_path, table_prefix="kohaku", dimensions=DIM)

    all_nodes = []
    for doc_id in doc_ids:
        all_nodes.extend(_make_doc_nodes(doc_id))
    await store.upsert_nodes(all_nodes)

    bm25 = TextVault(db_path, table="kohaku_bm25")
    for text, node_id in bm25_pairs:
        bm25.insert(text, node_id)

    return store


def _kv_keys(store: KVaultNodeStore) -> list[str]:
    return [
        k.decode() if isinstance(k, bytes) else k
        for k in store._kv.keys(limit=10_000_000)
        if (k.decode() if isinstance(k, bytes) else k) != store.META_KEY
    ]


def _root_doc_ids(store: KVaultNodeStore) -> list[str]:
    """返回所有根文档的 node_id（parent_id is None）。"""
    roots = []
    for key in _kv_keys(store):
        try:
            record = store._kv[key]
        except KeyError:
            continue
        if record.get("parent_id") is None:
            node_id = record.get("node_id")
            if node_id:
                roots.append(node_id)
    return roots


def _assert_no_doc1(keys: list[str]):
    """确保 doc1 的节点被删除，但 doc10 这类前缀碰撞文档保留。"""
    for k in keys:
        assert not (k == "doc1" or k.startswith("doc1:")), f"unexpected doc1 node: {k}"


def _make_wrapper(tmpdir: str, embedding_dim: int = DIM) -> KohakuStoreWrapper:
    return KohakuStoreWrapper(
        store_path=tmpdir,
        doc_output_dir=tmpdir,
        api_key="dummy-api-key-for-test-only",
        api_base="http://localhost/dummy",
        embedding_dimension=embedding_dim,
    )


def _format_size(path: str) -> str:
    if not os.path.exists(path):
        return "N/A"
    size = os.path.getsize(path)
    return f"{size / (1024 * 1024):.2f} MB"


def test_delete_single_document() -> None:
    print("\n=== Test 1: delete_document ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "kohaku.db")
        store = asyncio.run(
            _build_test_db(
                db_path,
                doc_ids=["doc1", "doc2", "doc10"],
                bm25_pairs=[
                    ("doc1 sentence", "doc1:sec1:p1:s1"),
                    ("doc2 sentence", "doc2:sec1:p1:s1"),
                ],
            )
        )

        wrapper = _make_wrapper(tmpdir)
        wrapper._kv_store_cache = store

        before_kv = _kv_keys(store)
        before_vec = store._vectors.count()
        before_bm25 = store._bm25.count() if store._bm25 else 0

        print(f"[BEFORE] KV nodes: {len(before_kv)}, vectors: {before_vec}, BM25 rows: {before_bm25}")
        print(f"[BEFORE] KV keys: {sorted(before_kv)}")

        deleted = wrapper.delete_document("doc1")
        print(f"[DELETE] delete_document('doc1') returned: {deleted}")

        store2 = KVaultNodeStore(db_path, table_prefix="kohaku", dimensions=DIM)
        after_kv = _kv_keys(store2)
        after_vec = store2._vectors.count()
        after_bm25 = store2._bm25.count() if store2._bm25 else 0

        print(f"[AFTER ] KV nodes: {len(after_kv)}, vectors: {after_vec}, BM25 rows: {after_bm25}")
        print(f"[AFTER ] KV keys: {sorted(after_kv)}")

        assert deleted is True
        _assert_no_doc1(after_kv)
        assert "doc2" in after_kv
        assert "doc10" in after_kv
        assert after_vec == before_vec - 4
        assert after_bm25 == before_bm25 - 1

        deleted_again = wrapper.delete_document("doc1")
        store3 = KVaultNodeStore(db_path, table_prefix="kohaku", dimensions=DIM)
        assert deleted_again is False
        assert store3._vectors.count() == after_vec

        print("✅ delete_document 测试通过")


def test_clear_keeps_db_and_deletes_all_docs() -> None:
    print("\n=== Test 2: clear() keeps DB file and removes all docs ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "kohaku.db")
        doc_ids = ["docA", "docB", "docC"]
        store = asyncio.run(
            _build_test_db(
                db_path,
                doc_ids=doc_ids,
                bm25_pairs=[],
            )
        )

        wrapper = _make_wrapper(tmpdir)
        wrapper._kv_store_cache = store

        called: list[str] = []
        original_delete = wrapper.delete_document

        def counting_delete(doc_id: str) -> bool:
            called.append(doc_id)
            return original_delete(doc_id)

        wrapper.delete_document = counting_delete

        wrapper.clear()

        print(f"[CLEAR] delete_document called with: {sorted(called)}")
        assert set(called) == set(doc_ids), f"expected {set(doc_ids)}, got {set(called)}"
        assert os.path.exists(db_path), "DB file should be kept after clear()"

        store2 = KVaultNodeStore(db_path, table_prefix="kohaku", dimensions=DIM)
        remaining = _kv_keys(store2)
        assert len(remaining) == 0, f"expected no non-meta nodes, got {remaining}"
        assert store2._vectors.count() == 0, "expected empty vector table"

        print(f"[CLEAR] DB file still exists: {db_path}")
        print("✅ clear() 保留 DB 并清空文档测试通过")


def test_clear_on_real_db_copy() -> None:
    print("\n=== Test 3: clear() on a copy of the real FinanceBench_sample20 DB ===")
    if not os.path.exists(REAL_DB_PATH):
        print(f"⚠️  Real DB not found at {REAL_DB_PATH}, skipping this test")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "kohaku.db")

        # 拷贝 DB 文件及其 WAL/SHM 伴侣文件（如果存在）
        shutil.copy2(REAL_DB_PATH, db_path)
        for suffix in ["-wal", "-shm"]:
            src = REAL_DB_PATH + suffix
            if os.path.exists(src):
                shutil.copy2(src, db_path + suffix)

        print(f"[REAL] Copied DB to {db_path}")
        print(f"[REAL] DB size before clear: {_format_size(db_path)}")

        wrapper = _make_wrapper(tmpdir, embedding_dim=2048)

        # 先打开 store 数一数文档数
        store = KVaultNodeStore(db_path, table_prefix="kohaku")
        root_ids = _root_doc_ids(store)
        print(f"[REAL] Found {len(root_ids)} root documents")
        print(f"[REAL] KV nodes before: {len(_kv_keys(store))}, vectors: {store._vectors.count()}")

        wrapper._kv_store_cache = store

        wrapper.clear()

        store2 = KVaultNodeStore(db_path, table_prefix="kohaku")
        remaining = _kv_keys(store2)
        vec_count = store2._vectors.count()
        bm25_count = store2._bm25.count() if store2._bm25 else 0

        print(f"[REAL] KV nodes after: {len(remaining)}, vectors: {vec_count}, BM25 rows: {bm25_count}")
        print(f"[REAL] DB size after clear: {_format_size(db_path)}")

        assert len(remaining) == 0, f"expected no non-meta nodes after clear, got {len(remaining)}"
        assert vec_count == 0, f"expected empty vector table, got {vec_count}"
        assert bm25_count == 0, f"expected empty BM25 table, got {bm25_count}"

        print("✅ 真实 DB 拷贝验证通过（原 DB 未被修改）")

        # ------------------------------------------------------------------
        # 4. 在拷贝上执行 VACUUM，观察文件大小变化（原 DB 仍不受影响）
        # ------------------------------------------------------------------
        print("[REAL] Running VACUUM on the cleared copy...")
        size_before_vacuum = os.path.getsize(db_path)

        # 释放 KVaultNodeStore 持有的连接，避免 database is locked
        store2 = None
        wrapper = None
        store = None
        gc.collect()

        conn = sqlite3.connect(db_path)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

        size_after_vacuum = os.path.getsize(db_path)
        print(f"[REAL] DB size before VACUUM: {size_before_vacuum / (1024 * 1024):.2f} MB")
        print(f"[REAL] DB size after  VACUUM: {size_after_vacuum / (1024 * 1024):.2f} MB")
        print(f"[REAL] Space reclaimed: {(size_before_vacuum - size_after_vacuum) / (1024 * 1024):.2f} MB")


def main() -> int:
    test_delete_single_document()
    test_clear_keeps_db_and_deletes_all_docs()
    test_clear_on_real_db_copy()
    print("\n🎉 全部验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
