# 让 MoDora 评测兼容非 VLM 模型的实施计划

## Summary

目标是在当前 `ov_test` 的 MoDora benchmark 中支持纯文本 LLM，不再因为模型不支持 `image_url` 而在入库或查询生成答案阶段报 `Model do not support image input`。

方案采用用户确认的策略：

* 新增一个默认关闭的 backend 配置开关，由 benchmark 配置启用。

* 开关放在 `store` 层，例如 `disable_visual_llm: true`。

* 开启后所有 MoDora LLM 调用不再发送图片，只使用已有文本、metadata、schema、OCR/PDF text 结果。

* 对 PDF 中的 `image/chart/table` 视觉增强不再调用 VLM，直接跳过视觉增强，保留已有文本抽取结果。

* 默认行为保持不变；没有显式开启该开关时，MoDora 仍按原多模态逻辑运行。

## Current State Analysis

当前报错来自非 VLM 模型收到图片输入：

```text
Model do not support image input
param: image_url
```

MoDora backend 当前有多处会向 LLM 传 `base64_image`，最终在 `remote.py` / `local.py` 中转换成 OpenAI-compatible `image_url`：

* `modora/MoDora-backend/src/modora/core/infra/llm/remote.py`

* `modora/MoDora-backend/src/modora/core/infra/llm/local.py`

已确认的图片输入路径：

1. 入库组件增强

   * 文件：`modora/MoDora-backend/src/modora/core/services/enrichment.py`

   * 行为：对 `image/chart/table` 组件裁剪 PDF 区域，再调用 `generate_annotation_async(base64_img, co.type)`。

2. 入库 build\_tree 标题层级生成

   * 文件：`modora/MoDora-backend/src/modora/core/services/hierarchy.py`

   * 行为：裁剪标题区域图片，再调用 `generate_levels(title_list, image)`。

3. 查询语义检索相关性判断

   * 文件：`modora/MoDora-backend/src/modora/core/services/retrieve/semantic_retriever.py`

   * 行为：裁剪节点区域图片，再调用 `check_node_mm(titled_data, query, image)`。

4. 查询最终答案生成

   * 文件：`modora/MoDora-backend/src/modora/core/services/qa_service.py`

   * 行为：裁剪检索命中位置图片，再调用 `reason_retrieved(..., images=images)`。

5. 查询 fallback 整页推理

   * 文件：`modora/MoDora-backend/src/modora/core/services/qa_service.py`

   * 行为：调用 `pdf_to_base64(source_path)`，再调用 `reason_whole(..., image=whole_doc)`。

此外，`BaseAsyncLLMClient` 中以下方法本身是多模态接口，会继续被服务层调用：

* `generate_levels`

* `check_node_mm`

* `reason_retrieved`

* `reason_whole`

* `generate_annotation_async`

当前 benchmark 配置会把 `store` 层配置通过 `ModoraStoreWrapper._build_inline_modora_config()` 写入 MoDora config 文件。该函数目前维护了一个允许转发的 key 列表，需要新增 `disable_visual_llm`。

## Proposed Changes

### 1. 在 Settings 中新增 `disable_visual_llm`

文件：

* `modora/MoDora-backend/src/modora/core/settings.py`

改动：

* 在 `Settings` dataclass 中新增：

```python
disable_visual_llm: bool = False
```

* 在 `Settings.load()` / `pick()` 解析区域中读取该配置：

```python
disable_visual_llm = _coerce_bool(
    pick("disable_visual_llm", False), default=False
)
```

* 在构造 `Settings(...)` 时传入该字段。

原因：

* 让 backend 统一知道当前模型是否允许视觉输入。

* 默认值为 `False`，避免影响现有 VLM 场景。

### 2. 让 benchmark store 配置能传递该开关

文件：

* `ov_test/src/core/modora_store.py`

改动：

* 在 `_build_inline_modora_config()` 的白名单中加入：

```python
"disable_visual_llm",
```

原因：

* 用户希望开关放在 benchmark 的 `store` 层。

* `ModoraStoreWrapper` 目前只转发白名单内 key，不加白名单则 YAML 中该字段不会进入 backend `Settings`。

### 3. 在 VersionRAG MoDora 配置中启用 text-only 模式

文件：

* `ov_test/config_modora/versionrag_config.yaml`

改动：

* 在 `store` 下增加：

```yaml
disable_visual_llm: true
```

原因：

* 当前用户使用的模型是非 VLM，需要该实验默认走纯文本兼容模式。

* 仅修改当前 VersionRAG 配置，避免影响其他数据集配置。

可选后续：如果其他 `ov_test/config_modora/*.yaml` 也需要用纯文本 LLM，再逐个加同一配置。

### 4. 入库 enrichment 阶段跳过视觉增强

文件：

* `modora/MoDora-backend/src/modora/core/services/enrichment.py`

改动：

* 在 `EnrichmentService` 增加可选 `settings` 或 `disable_visual_llm` 参数。

* 在 `get_components_async()` 创建 `EnrichmentService` 时传入当前 settings。

* 如果 `disable_visual_llm=True`：

  * 不裁剪图片。

  * 不调用 `generate_annotation_async()`。

  * 直接返回原始 `co_pack`。

  * 打一条 `INFO` 或 `WARNING` 日志，例如：

```text
visual enrichment skipped because disable_visual_llm=true
```

原因：

* 用户明确选择 text-only 下“跳过视觉增强”。

* 这能避免 image/chart/table 在入库阶段触发 `image_url` 400。

### 5. build\_tree 标题层级生成改为文本-only 降级

文件：

* `modora/MoDora-backend/src/modora/core/services/hierarchy.py`

* `modora/MoDora-backend/src/modora/core/preprocess.py`

改动：

* 在 `AsyncLevelGenerator.generate_level()` 中判断 `config.disable_visual_llm`。

* 如果为 `True`：

  * 不调用 `self.media.crop_image()`。

  * 不调用 `generate_levels(..., image)`。

  * 使用文本顺序的保守默认层级：保留现有 `title_level`，如果缺失则默认 `1`。

  * 打日志说明使用 text-only fallback。

原因：

* 标题层级的视觉判断依赖 VLM；非 VLM 场景不能发图。

* 保守默认不会生成高质量层级，但能保证入库不中断。

### 6. 查询语义检索相关性判断改为文本-only

文件：

* `modora/MoDora-backend/src/modora/core/services/retrieve/semantic_retriever.py`

改动：

* 在 `_is_relevant()` 中判断 `self.settings.disable_visual_llm`。

* 如果为 `True`：

  * 不裁剪图片。

  * 不调用 `check_node_mm()`。

  * 改调用文本接口：

```python
result = await self.llm.check_node(titled_data, query)
```

* 记录 trace 日志，例如 `semantic relevance used text-only mode`。

原因：

* 当前 `_is_relevant()` 每个节点都会裁剪并用图片判断相关性。

* text-only 模式下应该仍可用节点文本和 metadata 判断相关性，而不是失败后返回 False。

### 7. 查询最终答案生成不再传图片

文件：

* `modora/MoDora-backend/src/modora/core/services/qa_service.py`

改动：

* 在 `qa()` 的 answer reasoning 阶段判断 `settings.disable_visual_llm` 或 `retriever_settings.disable_visual_llm`。

* 如果为 `True`：

  * 不调用 `self.cropper.crop_image(...)`。

  * 调用 `reason_retrieved()` 时传 `images=None`，或者直接调用 `_call_llm` 的纯文本路径。

  * 日志中标明 `image_count=0` / `text-only mode`。

具体数据仍使用：

```python
query=query
evidence=str(result.text_map)
schema=str(schema)
```

原因：

* 这是当前回答阶段最直接触发 `image_url` 400 的路径。

* text-only 模式仍保留检索到的文本 evidence 和 tree schema。

### 8. 查询 fallback 不再整页传图

文件：

* `modora/MoDora-backend/src/modora/core/services/qa_service.py`

改动：

* 在 `answer_ok=False` 的 fallback 分支里，如果 `disable_visual_llm=True`：

  * 不调用 `self.cropper.pdf_to_base64(source_path)`。

  * 调用 `reason_whole(query=query, data=str(clean_tree_data), image=None)`。

  * 或新增更明确的 text-only helper，比如 `reason_whole_text()`，内部仍复用 `whole_reasoning_prompt`。

原因：

* 当前 fallback 会传整页 PDF 图片，这对非 VLM 一定失败。

* text-only fallback 可以基于 `tree.get_clean_structure()` 尽量生成答案。

### 9. 可选修复：build\_tree 传入 config，减少 local-default fallback 噪音

文件：

* `modora/MoDora-backend/src/modora/lab/commands/build_tree.py`

当前状态：

* `_run_one_job()` 调用：

```python
tree = await build_tree_async(cp, logger, source)
```

* 没有显式传入 `settings` / `ui_settings`。

计划：

* 在 `run_build_tree_pipeline()` 读取：

```python
ui_settings = load_ui_settings_from_config(config_path)
```

* 把 `settings` 和 `ui_settings` 传到 `_run_jobs()` / `_run_one_job()` / `build_tree_async()`。

原因：

* 这不是非 VLM 的唯一必要条件，但它能确保 `levelGenerator` 和 `metadataGenerator` 使用 YAML 里指定的 `JUDGE_LLM`，避免先尝试 `local-default`。

* 属于同一条 MoDora benchmark 兼容性修复。

## Assumptions & Decisions

* 用户已确认使用 `Text-only开关` 策略。

* 用户已确认实现范围以 benchmark 为目标，但接受在 backend 增加默认关闭的配置开关。

* 用户已确认开关放在 `store` 层。

* 用户已确认 text-only 模式下跳过视觉增强。

* 默认行为必须保持兼容：`disable_visual_llm` 缺省为 `False`。

* 本次只默认修改 `versionrag_config.yaml`，不批量修改所有 `config_modora/*.yaml`。

* text-only 模式下回答质量可能弱于 VLM 模式，尤其是问题依赖图像、图表、视觉布局时；目标是“可运行且不因 image\_url 报错”，不是保证多模态能力等价。

* 若缓存中已有带失败/默认内容的 `co.json/tree.json`，启用新逻辑后建议用户清理对应 cache 或设置重新入库，否则可能复用旧缓存。

## Verification Steps

1. 静态检查

```bash
uv run python -m py_compile \
  modora/MoDora-backend/src/modora/core/settings.py \
  modora/MoDora-backend/src/modora/core/services/enrichment.py \
  modora/MoDora-backend/src/modora/core/services/hierarchy.py \
  modora/MoDora-backend/src/modora/core/services/retrieve/semantic_retriever.py \
  modora/MoDora-backend/src/modora/core/services/qa_service.py \
  modora/MoDora-backend/src/modora/lab/commands/build_tree.py \
  ov_test/src/core/modora_store.py
```

1. 配置解析验证

```bash
uv run python - <<'PY'
from modora.core.settings import Settings
s = Settings.load("ov_test/config_modora/versionrag_config.yaml")
print("disable_visual_llm=", getattr(s, "disable_visual_llm", None))
PY
```

期望输出：

```text
disable_visual_llm= True
```

1. 小样本入库 smoke test

```bash
uv run python ov_test/run.py \
  --config ov_test/config_modora/versionrag_config.yaml \
  --step import \
  --max-queries 1
```

期望：

* 不再出现 `Model do not support image input`。

* 日志出现 text-only 跳过视觉增强/视觉推理的提示。

* `preprocess_ocr` 和 `build_tree` 返回码为 0。

1. 小样本查询生成 smoke test

```bash
uv run python ov_test/run.py \
  --config ov_test/config_modora/versionrag_config.yaml \
  --step gen \
  --skip-ingest \
  --max-queries 1
```

期望：

* 不再出现 `param: image_url`。

* `generated_answers.json` 中有结果。

* 如 API 没返回 usage，能看到之前新增的 token usage estimated warning。

1. 回归检查 VLM 默认行为

准备一个不设置 `disable_visual_llm` 的配置，或临时设为 `false`：

```yaml
disable_visual_llm: false
```

期望：

* 仍按原逻辑裁剪并发送图片。

* 不破坏现有 VLM 路径。

