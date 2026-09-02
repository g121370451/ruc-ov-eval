# OV-Wiki 数据集运行说明

本目录对应从 `ov-llm-wiki/benchmark/wiki` 接入的六个数据集、13组固定实验。
源项目的 Wiki/VikingBot 流程没有复制；本项目继续使用 `ov_test/run.py` 的
统一入库、检索、生成和评测流程。

## 数据目录

下载缓存和最终数据统一放在工作区公共 `Data` 目录：

```text
Data/OVWikiBenchmark/
├── raw/       # 上游下载缓存
├── datasets/  # Adapter 直接读取的数据
└── runtime/   # 处理后文档和向量库
```

## 准备数据

查看可用数据集：

```bash
uv run python ov_test/scripts/prepare_dataset.py --help
```

准备一组数据：

```bash
uv run python ov_test/scripts/prepare_dataset.py --dataset MDAQAFirst100
```

如果 `Data/OVWikiBenchmark/raw` 已有完整下载缓存，可只生成 Adapter 数据：

```bash
uv run python ov_test/scripts/prepare_dataset.py \
  --dataset MDAQAFirst100 \
  --skip-download
```

支持的固定范围：

- PaperScope Summary：57/93篇 PDF，各含 trend、gap、results_comparison。
- MDA-QA：前100条 QA、143篇 arXiv PDF。
- WildGraphBench Summary：全部主题339条/3894篇，Health 55条/509篇。
- ScholarQA-Multi：101条有效 QA、413份合并引用 TXT。
- MuDABench：Simple/Complex 各166条，共用589篇 PDF。
- EnterpriseRAG-Bench：三个类别80条、323份物理 TXT。

## 运行

例如运行 MDA-QA：

```bash
uv run python ov_test/run.py \
  --config ov_test/config_ov_wiki/mdaqa_first_100.yaml
```

只生成和评测、复用已有入库结果：

```bash
uv run python ov_test/run.py \
  --config ov_test/config_ov_wiki/mdaqa_first_100.yaml \
  --step geneval \
  --skip-ingest
```

每份配置中的 `execution.max_queries` 只限制实际问答数量；`null` 表示全部。

## GraphRAG DRIFT Baseline

`ov_test/config_graphrag/` 已覆盖上述 13 组 OV-Wiki 固定实验。所有
配置统一使用 Standard GraphRAG 入库和 DRIFT 查询，并复用同一批
数据集 adapter：

| 实验 | GraphRAG 配置 | Adapter |
|---|---|---|
| EnterpriseRAG-Bench Selected 80 | `enterprise_rag_bench_selected_80_drift.yaml` | `EnterpriseRAGBenchAdapter` |
| MDA-QA First 100 | `mdaqa_first_100_drift.yaml` | `MDAQAAdapter` |
| MuDABench Simple | `mudabench_simple_drift.yaml` | `MuDABenchAdapter` |
| MuDABench Complex | `mudabench_complex_drift.yaml` | `MuDABenchAdapter` |
| PaperScope 57 Gap | `paperscope_summary_57_gap_drift.yaml` | `PaperScopeSummaryAdapter` |
| PaperScope 57 Results Comparison | `paperscope_summary_57_results_comparison_drift.yaml` | `PaperScopeSummaryAdapter` |
| PaperScope 57 Trend | `paperscope_summary_57_trend_drift.yaml` | `PaperScopeSummaryAdapter` |
| PaperScope 93 Gap | `paperscope_summary_93_gap_drift.yaml` | `PaperScopeSummaryAdapter` |
| PaperScope 93 Results Comparison | `paperscope_summary_93_results_comparison_drift.yaml` | `PaperScopeSummaryAdapter` |
| PaperScope 93 Trend | `paperscope_summary_93_trend_drift.yaml` | `PaperScopeSummaryAdapter` |
| ScholarQA-Multi Valid 101 | `scholarqa_multi_valid_101_drift.yaml` | `ScholarQAMultiAdapter` |
| WildGraphBench Summary All | `wildgraphbench_summary_all_drift.yaml` | `WildGraphBenchSummaryAdapter` |
| WildGraphBench Summary Health | `wildgraphbench_summary_health_drift.yaml` | `WildGraphBenchSummaryAdapter` |

入库和查询建议分开执行：

```bash
uv run python ov_test/run.py \
  --config ov_test/config_graphrag/mdaqa_first_100_drift.yaml \
  --step ingest

uv run python ov_test/run.py \
  --config ov_test/config_graphrag/mdaqa_first_100_drift.yaml \
  --step geneval
```

下列问题 scope 共用语料和索引，只需使用其中一份配置执行
`--step ingest`，其他配置直接执行 `--step geneval`：

- `mudabench_simple` / `mudabench_complex`。
- `paperscope_summary_57` 的 gap / results_comparison / trend。
- `paperscope_summary_93` 的 gap / results_comparison / trend。

每份 DRIFT 配置统一固定 `n_depth=3`、`drift_k_followups=20`、
`primer_folds=5`和内部并发度 8。Embedding 单请求批量限制为 10，
且只生成 DRIFT 必需的 `entity_description` 与
`community_full_content` 两类向量。
