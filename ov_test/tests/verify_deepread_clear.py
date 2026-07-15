"""验证 DeepReadWrapper.clear() 会逐文档删除并保留 store_path 目录。"""

import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ov_test/src 用于 import src.xxx；仓库根目录用于 import DeepRead 等子模块
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))

from src.core.deepread_store import DeepReadWrapper


def _touch(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def _create_doc_files(store_path: str, doc_id: str):
    """模拟 DeepRead ingest 为一篇文档生成的文件。"""
    _touch(os.path.join(store_path, f"{doc_id}.md"))
    _touch(os.path.join(store_path, f"{doc_id}_corpus.json"))
    _touch(os.path.join(store_path, f"{doc_id}_emb.npy"))
    _touch(os.path.join(store_path, f"{doc_id}_idmap.json"))
    _touch(os.path.join(store_path, f"{doc_id}.json"))


def test_clear_keeps_store_path_and_deletes_docs() -> None:
    print("\n=== Test: DeepRead clear() keeps store path and deletes docs ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "deepread_store")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(store_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        doc_ids = ["docA", "docB", "docC"]
        for doc_id in doc_ids:
            _create_doc_files(store_path, doc_id)

        # 模拟一个临时文件，验证不会被误判成文档
        _touch(os.path.join(store_path, "other_log.txt"))

        wrapper = DeepReadWrapper(
            store_path=store_path,
            doc_output_dir=tmpdir,
            output_dir=output_dir,
            model="dummy-model",
            base_url="http://localhost/dummy",
            api_key="dummy-api-key",
            temperature=0.0,
        )

        # 包装 delete_document 以记录调用
        called: list[str] = []
        original_delete = wrapper.delete_document

        def counting_delete(doc_id: str) -> bool:
            called.append(doc_id)
            return original_delete(doc_id)

        wrapper.delete_document = counting_delete

        wrapper.clear()

        print(f"[CLEAR] delete_document called with: {sorted(called)}")
        assert set(called) == set(doc_ids), f"expected {set(doc_ids)}, got {set(called)}"
        assert os.path.exists(store_path), "store_path should be kept after clear()"

        for doc_id in doc_ids:
            for suffix in wrapper._DOC_FILE_SUFFIXES:
                assert not os.path.exists(
                    os.path.join(store_path, f"{doc_id}{suffix}")
                ), f"{doc_id}{suffix} should be deleted"

        # 非文档文件应保留
        assert os.path.exists(os.path.join(store_path, "other_log.txt")), "non-doc files should be kept"

        print(f"[CLEAR] Store path still exists: {store_path}")
        print("✅ DeepRead clear() 测试通过")


def test_delete_single_document() -> None:
    print("\n=== Test: DeepRead delete_document ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "deepread_store")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(store_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        _create_doc_files(store_path, "doc1")
        _create_doc_files(store_path, "doc2")

        wrapper = DeepReadWrapper(
            store_path=store_path,
            doc_output_dir=tmpdir,
            output_dir=output_dir,
            model="dummy-model",
            base_url="http://localhost/dummy",
            api_key="dummy-api-key",
            temperature=0.0,
        )

        result = wrapper.delete_document("doc1")
        assert result is True, "delete_document should return True"

        for suffix in wrapper._DOC_FILE_SUFFIXES:
            assert not os.path.exists(os.path.join(store_path, f"doc1{suffix}"))
            assert os.path.exists(os.path.join(store_path, f"doc2{suffix}"))

        # 再次删除不存在的文档
        result2 = wrapper.delete_document("doc1")
        assert result2 is False, "deleting non-existent doc should return False"

        print("✅ DeepRead delete_document 测试通过")


def main() -> int:
    test_delete_single_document()
    test_clear_keeps_store_path_and_deletes_docs()
    print("\n🎉 DeepRead 全部验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
