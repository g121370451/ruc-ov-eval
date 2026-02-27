# RUC-OV-Eval: OpenViking 性能评估系统

本项目是用于评估 **OpenViking** 系统在检索增强生成（RAG）场景下性能表现的自动化基准测试框架。

经过重构，测试核心代码已迁移至 `ov_test` 目录，采用“适配器（Adapter）+ 流水线（Pipeline）”架构，解耦了数据处理、向量库操作和大型语言模型（LLM）的调用。

## 1. 目录结构

项目根目录 `ruc-ov-eval` ：

```text
├─ Data/                               # 数据集
│   └─ Locomo/
├─ Output/                             # 输出
│   └─ Locomo/
└─ ruc-ov-eval/
    ├─ pyproject.toml                  # 项目依赖及 workspace 配置
    ├─ OpenViking/                     # [子模块] OpenViking 源码库
    └─ ov_test/                        # 【重构后的评测核心代码】
        ├─ run.py                          # 主入口：调度解析参数、初始化组件并启动评测
        ├─ ov.conf                         # OpenViking 配置文件
        ├─ config/
        │   └─ config.yaml                 # 全局配置：管理路径、并发数、LLM参数及指定适配器
        └─ src/
            ├─ pipeline.py                 # 流水线：编排入库、检索生成、评测的具体步骤
            ├─ adapters/                   # 数据适配层：
            │   ├─ base.py                     # 定义抽象基类与 StandardSample 数据结构
            │   └─ locomo_adapter.py           # LocoMo 数据集专用转换器
            └─ core/                       # 核心功能组件：
                ├─ vector_store.py             # 封装 OV 入库、检索及 Token 统计
                ├─ llm_client.py               # 封装 LLM 调用及重试机制
                ├─ metrics.py                  # F1、Recall 计算及文本标准化
                ├─ judge_util.py               # locomo_grader：调用 LLM 作为裁判打分
                └─ monitor.py                  # 线程安全的终端运行状态与 QPS 监控

```

## 2. 环境创建与运行

**环境准备**
推荐使用 `uv` 来管理依赖，项目已经在 `pyproject.toml` 中配置了 `OpenViking` 为 workspace members。

```bash
git clone https://github.com/g121370451/ruc-ov-eval.git
cd ruc-ov-eval
git submodule update --init --recursive  # 下载OV的git子模块代码
uv sync # 同步项目依赖并创建虚拟环境
source .venv/bin/activate  # macOS/Linux
# Windows: .venv\Scripts\activate
```

**运行测试**
所有测试命令均在 `ov_test` 目录下通过 `run.py` 统筹执行。

```bash
uv run python ov_test/run.py # 运行完整评测（入库 -> 检索生成 -> 结果评估）
```

## 3. 接入新数据 (Adapter 机制)

适配一个全新的数据集时，主要涉及以下三个部分：

#### A. 新增数据集适配器 (Adapter)

需要根据 `BaseAdapter` 的定义，在 `src/adapters/` 目录下创建一个新的类（例如 `my_dataset_adapter.py`）：

* **数据解析**：实现 `load_and_transform` 方法，将原始数据解析为 `StandardSample` 格式。
* **提示词构建**：实现 `build_prompt` 方法，定义如何将检索到的上下文和问题拼接成发给 LLM 的 Prompt。
* **后处理**：如有必要，在 `post_process_answer` 中清洗 LLM 的输出（例如提取选项字母）。

#### B. 更新评测逻辑 (`judge_util.py`)

`llm_grader` 函数目前包含了针对不同数据集的路由逻辑：

* 如果新数据集的评测逻辑比较特殊（例如需要 LLM 关注特定的日期格式或语义匹配规则），应该在 `llm_grader` 中增加一个 `elif` 分支，为其配置专门的 `ACCURACY_PROMPT`。
* 如果使用通用的匹配规则，代码会自动回退到 `else` 分支中的通用 Prompt。

#### C. 修改配置文件 (`config.yaml`)

这是切换数据集的开关：

* **`dataset_name`**：更新为新数据集的名称。
* **`adapter`**：修改 `module`（指向你的新适配器文件路径）和 `class_name`（类名）。
* **`paths`**：更新 `raw_data` 的路径，确保系统能找到原始文件。

#### 总结：适配新数据集的具体流程

1. **编写 Adapter**：参考 `locomo_adapter.py` 编写一个新的 Adapter 类。
2. **配置 Config**：在 `config.yaml` 中指向新编写的 Adapter 类。
3. **调整 Judge（可选）**：如果默认的 LLM 打分逻辑不够准确，在 `judge_util.py` 中为新数据集定制 Prompt。
4. **运行**：直接执行 `python run.py` 即可，无需改动 `pipeline.py`。

## 4. 评估指标

系统在执行完毕后，会在输出目录生成 `benchmark_metrics_report.json` 文件。以下是所有被记录指标的解释与代码来源：

### A. 质量评估指标
质量指标用于衡量 OpenViking 检索能力和大模型回答准确度，在 `src/pipeline.py` 的 `_save_reports` 方法中汇总计算。

##### 1. **Average Recall**
* **含义**：检索到的上下文中，成功覆盖标注“证据”的比例。
* **计算逻辑**：针对每一题，遍历检索出的 Top-K 文本片段。如果片段中包含了某个标准证据 ID，则记为命中。最终单题的 `Recall = 命中的证据数 / 总证据数`。
* **代码位置**：`src/core/metrics.py` 中的 `MetricsCalculator.check_recall`。


##### 2. **Average F1 Score**
* **含义**：生成的答案与标准答案（Gold Answer）的字面重合度。
* **计算逻辑**：对生成答案和标准答案进行文本清理（去标点、转小写、去英文冠词）。将两者视为词袋，求交集词数。基于此分别计算精确率(Precision)和召回率(Recall)，最终公式为 `(2 * P * R) / (P + R)`。
* **代码位置**：`src/core/metrics.py` 中的 `MetricsCalculator.calculate_f1`。


##### 3. **Average Accuracy⭐**
* **含义**：基于大语言模型（LLM-as-a-Judge）给出的语义级宽容判定。
* **计算逻辑**：将问题、标准答案、生成答案组成 Prompt 送入裁判大模型。只要生成答案提及了标准答案的核心主题或相同时间段，裁判即返回 `CORRECT` (记为 1.0)，否则返回 `WRONG` (记为 0.0)。
* **代码位置**：`src/core/judge_util.py` 中的 `llm_grader`。


### B. 查询效率指标

查询效率衡量在问答（Generation）阶段单条 Query 的平均开销。

#### 1. **Average Retrieval Time (s) (平均检索延迟)**
* **含义**：OpenViking 针对单条问题执行 Top-K 检索的平均耗时。
* **计算逻辑**：在 `_process_generation_task` 阶段，围绕 `self.db.retrieve()` 方法记录 `time.time() - t0`。


#### 2. Average Input Tokens (平均输入 Tokens)

* **含义**：衡量单条问答任务在“向量检索”与“大模型生成”两个阶段中，系统所消耗的总输入 Token 成本。
* **详细计算逻辑**：
单题的输入 Token 由两部分严格相加构成，系统底层统一调用 `tiktoken` 库的 `cl100k_base` 编码算法进行精确截断与计数：
1. **检索阶段输入成本**：统计用户原始问题（`qa.question`）的 Token 数量，这代表了系统在向量数据库中执行相似度查询（Embedding）的输入开销。
2. **生成阶段输入成本**：统计向大模型发送的完整提示词（`full_prompt`）的 Token 数量。该提示词是由多部分拼接而成的庞大字符串，包含：被截断至 2000 字符限制的 Top-K 检索上下文、默认的防幻觉规则（`MISSING_RULE`）。
系统将所有有效评测题目的 `原问题 Token 数 + 完整提示词 Token 数` 进行累加后，除以总题数得出该平均值。


* **对应代码位置**：
  * 核心拼接与计算：`src/pipeline.py` 的 `_process_generation_task` 方法 (`in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question)`)。
  * 编码器实现：`src/core/vector_store.py` 的 `count_tokens` 方法。
  * 平均值运算：`src/pipeline.py` 的 `_save_reports` 方法 (`avg_in_tokens = sum(...) / total`)。



#### 3. Average Output Tokens (平均输出 Tokens)

* **含义**：衡量大语言模型针对单条问题最终输出答案的平均文本长度成本。
* **详细计算逻辑**：
在生成阶段，大模型首先输出原始字符串（`ans_raw`）。系统对这个最终的 `ans` 字符串调用基于 `cl100k_base` 的 `count_tokens` 方法计算其 Token 长度。最后将所有评测任务的输出 Token 长度总和除以总题数得出均值。
* **对应代码位置**：
  * 计算生成结果长度：`src/pipeline.py` 的 `_process_generation_task` 方法 (`out_tokens = self.db.count_tokens(ans)`)。
  * 平均值运算：`src/pipeline.py` 的 `_save_reports` 方法 (`avg_out_tokens = sum(...) / total`)。


### C. 入库效率指标

衡量系统在 Step 2 阶段（将 Markdown 文档处理并持久化到 OpenViking 向量库）的总开销。以下指标均由 `src/core/vector_store.py` 中的 `VikingStoreWrapper.ingest` 方法记录并返回。

1. **Total Insertion Time (s) (入库总耗时)**
* **计算逻辑**：从开始提交 `add_resource` 任务，到 `client.wait_processed()` 阻塞结束，后台处理完所有摘要和向量化任务所经过的总物理时间。


2. **Total Input Tokens (入库输入总 Tokens)**
* **计算逻辑**：通过访问 OpenViking 内部的 `Semantic` 队列监控（`get_queue_manager`），累加文档提取摘要（Summary）和总览（Overview）时消耗的输入 Tokens 成本：`summary_tokens_cost + overview_tokens_cost`。


3. **Total Output Tokens (入库输出总 Tokens)**
* **计算逻辑**：同上，累加模型在生成摘要和总览时吐出的输出 Tokens 成本：`summary_output_tokens_cost + overview_output_tokens_cost`。



### D. 清理效率指标

如果调用清理逻辑，会记录删除所有库内数据所用的指标：

1. **Total Deletion Time (s)**：执行 `client.rm("viking://resources", recursive=True)` 的耗时。
2. **Token 消耗**：本地删除文件，不涉及 LLM 调用，因此输入和输出 Tokens 均固定为 0。

### E. 运行监控指标 (实时输出)

在终端运行生成的过程中，`src/core/monitor.py` 会负责在 `tqdm` 进度条上渲染实时状态：

* **QPS (Queries Per Second)**：当前成功完成的 QA 任务数 / 脚本自启动后逝去的总秒数。
* **Active**: 当前并发工作的活跃线程数。
* **Errs**: 执行中捕获异常或失败的任务总数。

