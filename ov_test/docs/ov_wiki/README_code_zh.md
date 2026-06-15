# OV-Wiki 数据代码结构

```text
ov_test/
├── scripts/
│   ├── prepare_dataset.py       # 下载 + 准备统一入口
│   ├── download_dataset.py      # 下载分发和通用归档处理
│   ├── sample_dataset.py        # 数据集准备/抽样分发
│   └── dataset_handlers/        # 六个数据集专用下载与校验
├── src/adapters/                # 六个运行时 Adapter
├── config_ov_wiki/              # 13份实验配置
└── docs/ov_wiki/                # 接入说明
```

调用链：

```text
prepare_dataset.py
  -> download_dataset.py
  -> dataset_handlers/<dataset>.py
  -> sample_dataset.py
  -> Data/OVWikiBenchmark/datasets/<dataset>/

ov_test/run.py
  -> YAML 动态加载 Adapter
  -> adapter.data_prepare()
  -> Store.ingest()
  -> adapter.load_and_transform()
  -> 检索、生成、评测
```

源 `ov-llm-wiki` README 中的 `import/build_wiki/vikingbot` 属于另一条 Wiki
pipeline，不是本项目入口；这里映射为当前 `run.py` 的 `ingest/gen/eval` 阶段。
