# MoDora Benchmark Harness

This branch is trimmed for testing MoDora in the RAG benchmark harness.

## What Is Kept

- `modora/MoDora-backend`: MoDora backend workspace package.
- `ov_test/config_modora`: MoDora experiment configs.
- `ov_test/src/core/modora_store.py`: benchmark wrapper for MoDora ingestion, retrieval, deletion, and token accounting.
- Shared benchmark code under `ov_test/src`: adapters, pipeline, metrics, logger, monitor, and LLM client.

## What Was Removed

Unrelated backend integrations and their configs were removed, including LightRAG, HippoRAG, PageIndex, SQL Agent, and the OpenViking wrapper.

## Basic Usage

```bash
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
cd /home/zhanggaoyuan.225/modora/ruc-ov-eval/ov_test
uv run --python 3.10 python run.py --config config_modora/finance_config.yaml --step all
```

For a quick smoke test:

```bash
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
cd /home/zhanggaoyuan.225/modora/ruc-ov-eval/ov_test
uv run --python 3.10 python run.py --config config_modora/versionrag_config.yaml --step gen+eval --max-queries 1
```
