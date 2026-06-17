# MC-indexing 后端配置说明

这个目录里的配置把 MC-indexing 作为 `ruc-ov-eval` 的一个新检索后端接入。

## 关键点

- 使用 `store.type: mc_indexing` 切换到 MC-indexing。
- 数据读取、Prompt 构造、答案生成、Recall、F1、Accuracy 和报告汇总仍然走 `ruc-ov-eval` 原有流水线。
- Recall 使用 `src/core/metrics.py` 里的 `MetricsCalculator.check_recall`，也就是 evidence 文本命中口径，不使用旧 MC-indexing 复现里的 section-level recall。
- MC-indexing 的 summary/keyword/raw-text 多视图只影响检索排序；返回给评估器的是原始片段文本，方便 evidence recall 正常命中。

## Locomo

```powershell
python ov_test/run.py --config ov_test/config_mc_indexing/locomo_config.yaml --step all
```

小样本测试时，先把配置里的：

```yaml
execution:
  max_queries: null
```

临时改成：

```yaml
execution:
  max_queries: 2
```

## Qasper

```powershell
python ov_test/run.py --config ov_test/config_mc_indexing/qasper_config.yaml --step all
```

## 切换检索器

默认使用 BM25：

```yaml
retriever: "bm25"
```

可改为：

```yaml
retriever: "e5"
retriever: "bge"
retriever: "doubao"
```

如果本机没有 `sentence-transformers`，但想先验证 E5/BGE 配置链路，可以设置：

```yaml
allow_dense_fallback: true
```

这样 E5/BGE 会降级成 TF-IDF fallback，适合调通流程，但不代表真实 dense retrieval 结果。
