# RUC-OV-Eval: OpenViking 性能评估系统

## 项目结构

```
ruc-ov-eval/
├── run_benchmark_non_agentic.py  # 主要基准测试运行脚本
├── judge_util.py                 # 评估工具模块
├── locomo_data/                  # LocoMo 数据集目录
├── experiment_data/              # 实验数据输出目录
├── viking_store_locomo/          # OpenViking 存储目录
├── .run_state.json               # 运行状态文件（断点续跑）
├── ov.conf                       # OpenViking 配置文件（必需）
└── README.md                     # 项目说明文档
```

## 项目说明

- **OpenViking** 作为子模块集成在项目中，使用 workspace 管理
- 项目用于评估 OpenViking 在 LocoMo 数据集上的性能表现
- 主要测试检索能力、生成质量和整体性能

## 安装与运行

### 1. 安装依赖

```bash
# 使用 uv 安装所有依赖
uv sync --all-packages
```

### 2. 配置 OpenViking

**必需**：在项目根目录创建 `ov.conf` 文件，配置 OpenViking 相关参数。

### 3. 运行测试

```bash
python run_benchmark_non_agentic.py
```

## 后续补充

详细的使用说明、性能指标解释和扩展方法将在后续更新。