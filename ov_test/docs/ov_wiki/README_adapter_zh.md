# OV-Wiki Adapter 说明

六个 Adapter 均继承本项目现有的 `src.adapters.base.BaseAdapter`，实现：

```python
data_prepare(doc_dir) -> list[StandardDoc]
load_and_transform() -> list[StandardSample]
build_prompt(qa, context_blocks) -> tuple[str, dict]
```

源 Wiki 项目的 `StandardDoc` 使用单个 `doc_path`；接入本项目时已转换成：

```python
StandardDoc(sample_id=..., doc_paths=[...])
```

数据集实现：

| Adapter | 主要处理 |
| --- | --- |
| `PaperScopeSummaryAdapter` | 校验57/93篇 manifest，按论文组合分组 QA |
| `MDAQAAdapter` | 校验100条 QA 和143篇 support PDF |
| `WildGraphBenchSummaryAdapter` | 合并多个 `gold_statements` 为一个完整答案 |
| `ScholarQAMultiAdapter` | 保留 evidence，并在 gold 后附引用编号到标题映射 |
| `MuDABenchAdapter` | 保留166行及官方重复记录，`source_answer` 作为 evidence |
| `EnterpriseRAGBenchAdapter` | 保留80条三类别 QA 和逻辑/物理文档映射 |

Adapter 会严格检查 manifest 数量、ID唯一性、相对路径安全性和源文件大小。
配置通过 `adapter.module` 和 `adapter.class_name` 动态选择实现，pipeline 不依赖
任何具体数据集类。
