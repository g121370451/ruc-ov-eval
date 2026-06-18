# MoDora 非 VLM 图片拦截最小改造计划

## Summary

目标是把当前 `disable_visual_llm` 的行为改成更符合预期的“LLM 调用边界图片拦截”，而不是在入库、建树、检索、查询服务层提前跳过逻辑。

新的期望行为：

- 不改变 MoDora 原有流程：仍然 OCR、组件分析、裁剪、build_tree、检索、查询。
- 只有在最终调用 LLM、准备把图片塞进 `image_url` 请求体时才拦截图片。
- 但不是所有带图片调用都继续调用 LLM：如果该调用去掉图片后没有有效文本上下文，则直接跳过 LLM。
- 对有文本上下文的多模态调用，拦截后仍调用同一个 LLM 方法，只是请求里只保留文本 prompt，不带图片。
- 参数改名新增，不继续用 `disable_visual_llm` 作为主语义。
- 还原之前服务层为了 `disable_visual_llm` 做的跳过逻辑，避免改变上游行为。
- 诊断日志写入 `benchmark.log`，但不在终端输出。

建议新增参数名：

```yaml
strip_llm_images: true
```

语义：当 LLM 请求携带图片时，按调用语义处理：

- 纯视觉任务：跳过 LLM，不发送无意义请求。
- 文本+图片任务：在 LLM client 组装消息前移除图片，仅发送文本内容。

## Current State Analysis

当前代码中图片最终进入 LLM 的统一入口只有两处：

1. 远程 LLM：
   - 文件：`modora/MoDora-backend/src/modora/core/infra/llm/remote.py`
   - 方法：`AsyncRemoteLLMClient._call_llm(prompt, base64_image=None)`
   - 现状：如果 `base64_image` 有值，会追加 OpenAI-compatible 的 `image_url`。

2. 本地 LLM：
   - 文件：`modora/MoDora-backend/src/modora/core/infra/llm/local.py`
   - 方法：`AsyncLocalLLMClient._create_messages(prompt, base64_image=None)`
   - 现状：如果 `base64_image is not None`，会追加 `image_url`。

上层调用点包括：

- `BaseAsyncLLMClient.generate_levels(...)`
- `BaseAsyncLLMClient.check_node_mm(...)`
- `BaseAsyncLLMClient.reason_retrieved(...)`
- `BaseAsyncLLMClient.reason_whole(...)`
- `BaseAsyncLLMClient.generate_annotation_async(...)`

这些调用点都最终落到 `remote.py` 或 `local.py` 的消息构造逻辑。

但这些调用点去掉图片后的语义不同，需要分类处理：

| 调用 | 输入文本内容 | 去掉图片后是否仍有意义 | 处理方式 |
|---|---|---:|---|
| `generate_annotation_async(base64_image, cp_type)` | 只有“描述这张 image/chart/table”的固定 prompt，没有组件文本或上下文 | 否 | 直接跳过 LLM |
| `generate_levels(title_list, base64_image)` | 有 `title_list` | 是 | 拦截图片，继续文本调用 |
| `check_node_mm(data, query, base64_image)` | 有 `query + data` | 是 | 拦截图片，继续文本调用 |
| `reason_retrieved(query, schema, evidence, images)` | 有 `query + schema + evidence` | 是 | 拦截图片，继续文本调用 |
| `reason_whole(query, data, image)` | 有 `query + document tree` | 是 | 拦截图片，继续文本调用 |

因此不能做“所有图片统一去掉后继续调 LLM”的无差别处理。必须在纯视觉 enrichment 任务上直接跳过 LLM，否则会让普通 LLM 在没有图片的情况下凭空生成图片/图表/表格描述。

之前已经添加的服务层跳过逻辑分布在：

- `modora/MoDora-backend/src/modora/core/services/enrichment.py`
  - 现在会在 `disable_visual_llm=True` 时直接跳过视觉增强。

- `modora/MoDora-backend/src/modora/core/services/hierarchy.py`
  - 现在会在 `disable_visual_llm=True` 时直接跳过 `crop_image` 和 `generate_levels`。

- `modora/MoDora-backend/src/modora/core/services/retrieve/semantic_retriever.py`
  - 现在会在 `disable_visual_llm=True` 时绕过 `check_node_mm`，改调 `check_node`。

- `modora/MoDora-backend/src/modora/core/services/qa_service.py`
  - 现在会在 `disable_visual_llm=True` 时跳过 `crop_images`，并给 `reason_retrieved` / `reason_whole` 传 `None` 图片。

这些服务层改动不符合当前期望，因为它们改变了上游流程，而不是只在 LLM 请求边界拦截图片。

日志方面：

- `ov_test/src/core/logger.py` 同时把 root logger 绑定到 FileHandler 和 StreamHandler。
- 所以 MoDora backend 的 `logger.info(...)` 既会写入 `benchmark.log`，也会打印到终端。
- 如果希望某些诊断信息“只进文件不进终端”，需要给 console `StreamHandler` 增加过滤器，过滤掉指定 logger/message；FileHandler 不加该过滤器。

## Proposed Changes

### 1. 新增配置字段 `strip_llm_images`

文件：

- `modora/MoDora-backend/src/modora/core/settings.py`

改动：

- 在 `Settings` dataclass 中新增：

```python
strip_llm_images: bool = False
```

- 在 `Settings.load()` 中解析：

```python
strip_llm_images = _coerce_bool(
    pick("strip_llm_images", False), default=False
)
```

- 构造 `Settings(...)` 时传入。

兼容策略：

- 保留现有 `disable_visual_llm` 字段一段时间，但不再作为服务层跳过逻辑使用。
- 可选兼容：如果 `strip_llm_images` 未显式配置，但 `disable_visual_llm=True`，则把 `strip_llm_images` 视为 `True`，避免旧配置完全失效。计划实现时采用这个兼容策略。

### 2. benchmark store 配置转发新增字段

文件：

- `ov_test/src/core/modora_store.py`

改动：

- 在 `_build_inline_modora_config()` 白名单中加入：

```python
"strip_llm_images",
```

- 保留 `"disable_visual_llm"` 转发以兼容旧配置，但新版配置使用 `strip_llm_images`。

### 3. 更新 VersionRAG 配置

文件：

- `ov_test/config_modora/versionrag_config.yaml`

改动：

- 使用新参数：

```yaml
strip_llm_images: true
```

- 不再依赖：

```yaml
disable_visual_llm: true
```

如果当前文件里有 `disable_visual_llm`，计划将其删除或设为 `false`，避免语义混乱。推荐删除。

### 4. 在纯视觉 annotation 调用中跳过 LLM

文件：

- `modora/MoDora-backend/src/modora/core/infra/llm/base.py`

改动：

- 在 `BaseAsyncLLMClient.generate_annotation_async()` 中优先判断：

```python
if getattr(self.settings, "strip_llm_images", False):
    logger.info(
        "visual annotation LLM skipped because strip_llm_images=true "
        f"(type={cp_type})"
    )
    return title, metadata, content
```

- 返回现有默认值：

```python
title = "Default Title"
metadata = "Default Metadata"
content = "Default Content"
```

- 不进入 3 次 retry 循环，不调用 `_call_llm()`。

原因：

- `generate_annotation_async()` 的 prompt 只有“描述这张图/图表/表格”的指令，没有组件文本、OCR 内容或上下文。
- 去掉图片后继续调用 LLM 是无意义请求，会产生幻觉式 metadata/content。
- 该调用不同于 `generate_levels/check_node_mm/reason_*`，后者即使去图仍有文本证据。

### 5. 在远程 LLM client 拦截有文本上下文的图片

文件：

- `modora/MoDora-backend/src/modora/core/infra/llm/remote.py`

改动：

- 在 `_call_llm()` 开始组装 `messages` 前判断：

```python
strip_images = bool(getattr(self.settings, "strip_llm_images", False))
```

- 如果 `strip_images=True` 且 `base64_image` 有有效图片：
  - 记录日志：

```text
LLM image input stripped before remote request (images=N, model=..., instance_id=...)
```

  - 将 `base64_image = None` 或不追加 `image_url`。
  - 仍然继续发送文本 prompt。

- `image_count` 统计应使用实际发送的图片数量，而不是原始传入数量。

原因：

- 这是远程模型真正出现 `image_url` 的位置。
- 在这里拦截能覆盖 build_tree、retrieval、qaService 等去图后仍有文本上下文的远程 LLM 调用。
- enrichment 的纯视觉 annotation 不依赖这里兜底，已在第 4 步提前跳过。

### 6. 在本地 LLM client 拦截有文本上下文的图片

文件：

- `modora/MoDora-backend/src/modora/core/infra/llm/local.py`

改动：

- 在 `_create_messages()` 中追加 `image_url` 前判断 `settings.strip_llm_images`。
- 如果开启：
  - 不追加图片内容。
  - 记录同类日志：

```text
LLM image input stripped before local request (images=N, model=..., instance_id=...)
```

原因：

- 虽然当前 VersionRAG 用的是远程 LLM，但本地 client 也有相同 `image_url` 组装逻辑。
- 同一参数应同时覆盖 local/remote。

### 7. 还原服务层跳过逻辑

文件：

- `modora/MoDora-backend/src/modora/core/services/enrichment.py`
- `modora/MoDora-backend/src/modora/core/services/hierarchy.py`
- `modora/MoDora-backend/src/modora/core/services/retrieve/semantic_retriever.py`
- `modora/MoDora-backend/src/modora/core/services/qa_service.py`
- `modora/MoDora-backend/src/modora/core/preprocess.py`

改动：

1. `enrichment.py`
   - 移除 `disable_visual_llm=True` 时直接 `return co_pack` 的逻辑。
   - 是否保留 `settings` 参数由实际需要决定；若仅为旧跳过逻辑服务，则撤掉。
   - 视觉组件仍然会 crop，仍然会调用 `generate_annotation_async`。
   - `generate_annotation_async` 会在 `strip_llm_images=True` 时自行跳过 LLM，避免纯视觉无意义请求。

2. `hierarchy.py`
   - 移除 `disable_visual_llm=True` 时提前返回的逻辑。
   - 保留 crop 和 `generate_levels` 原流程。
   - 保留当前新增的 build_tree 诊断日志，但处理终端输出，见第 7 步。

3. `semantic_retriever.py`
   - 移除 `disable_visual_llm=True` 时改调 `check_node` 的逻辑。
   - 恢复原来的 `crop_image` + `check_node_mm`。
   - 图片是否进入请求由 LLM client 拦截。

4. `qa_service.py`
   - 移除 `text_only` 分支。
   - 恢复原来的 `crop_images`、`reason_retrieved(..., images=images)`、`reason_whole(..., image=whole_doc)`。
   - 图片是否进入请求由 LLM client 拦截。

5. `preprocess.py`
   - 如果 `EnrichmentService(settings=...)` 仅用于旧跳过逻辑，则恢复为 `EnrichmentService(llm, cropper)`。

结果：

- 上游逻辑保持原样。
- 只有真正构造 LLM 请求时会移除图片。

### 8. build_tree 诊断日志只写文件，不刷终端

文件：

- `ov_test/src/core/logger.py`

改动：

- 新增一个 console-only filter，用于过滤掉 build_tree 视觉诊断日志。
- 只加到 `StreamHandler`，不加到 `FileHandler`。
- 匹配消息前缀：

```text
build_tree visual LLM crop started
build_tree visual LLM crop finished
build_tree visual LLM generate_levels started
build_tree visual LLM generate_levels returned
build_tree visual LLM generate_levels succeeded
build_tree visual LLM generate_levels failed
```

示意：

```python
class _DropVisualDiagnosticConsole(logging.Filter):
    PREFIXES = (...)
    def filter(self, record):
        return not record.getMessage().startswith(self.PREFIXES)
```

- `fh` 不加这个 filter，因此 `benchmark.log` 仍保留这些信息。
- `sh` 加这个 filter，因此终端不显示这些信息。

说明：

- 这只影响 `ov_test` benchmark 运行时的终端输出。
- MoDora backend 单独 CLI 运行时是否显示，取决于 backend 自己的 logging setup；本计划只满足当前 benchmark 场景。

### 9. 保留 build_tree 诊断日志，但不改变行为

文件：

- `modora/MoDora-backend/src/modora/core/services/hierarchy.py`

保留日志：

- crop started
- crop finished
- generate_levels started
- generate_levels returned
- generate_levels succeeded
- generate_levels failed

原因：

- 用户需要确定结果，不要靠“可能”判断。
- 日志能直接证明：
  - 是否尝试 crop
  - crop 得到几张图片
  - 是否调用 `generate_levels`
  - LLM 是否返回
  - 是否成功解析层级

## Assumptions & Decisions

- 用户确认当前服务层跳过图片逻辑不符合预期。
- 用户期望最小修改：只在尝试调用 LLM 且请求中带图片时进行图片拦截。
- 用户选择新增改名参数，不继续把 `disable_visual_llm` 作为主参数名。
- 本计划使用 `strip_llm_images` 作为新参数名。
- `generate_annotation_async` 属于纯视觉任务，去图后没有有效输入，必须直接跳过 LLM。
- `generate_levels`、`check_node_mm`、`reason_retrieved`、`reason_whole` 去图后仍有文本输入，可以继续调用 LLM。
- 之前为了 `disable_visual_llm` 添加的服务层提前返回/替换调用逻辑应撤回。
- build_tree 诊断日志保留到文件，但不在终端输出。
- 不改变图片裁剪、检索、建树、fallback 等上游流程。
- 不改变非图片 LLM 调用。
- 不改变 VLM 默认行为：`strip_llm_images=False` 时仍会正常发送图片。

## Verification Steps

1. 静态语法检查

```bash
uv run python -m py_compile \
  modora/MoDora-backend/src/modora/core/settings.py \
  modora/MoDora-backend/src/modora/core/infra/llm/base.py \
  modora/MoDora-backend/src/modora/core/infra/llm/remote.py \
  modora/MoDora-backend/src/modora/core/infra/llm/local.py \
  modora/MoDora-backend/src/modora/core/services/enrichment.py \
  modora/MoDora-backend/src/modora/core/services/hierarchy.py \
  modora/MoDora-backend/src/modora/core/services/retrieve/semantic_retriever.py \
  modora/MoDora-backend/src/modora/core/services/qa_service.py \
  modora/MoDora-backend/src/modora/core/preprocess.py \
  ov_test/src/core/logger.py \
  ov_test/src/core/modora_store.py
```

2. 配置解析验证

```bash
uv run python -c "from modora.core.settings import Settings; s=Settings.load('ov_test/config_modora/versionrag_config.yaml'); print(getattr(s, 'strip_llm_images', None))"
```

期望：

```text
True
```

3. benchmark wrapper 转发验证

```bash
PYTHONPATH=ov_test uv run python -c "from pathlib import Path; import yaml; from src.core.modora_store import ModoraStoreWrapper; cfg=yaml.safe_load(Path('ov_test/config_modora/versionrag_config.yaml').read_text())['store']; print(ModoraStoreWrapper.__new__(ModoraStoreWrapper)._build_inline_modora_config(cfg).get('strip_llm_images'))"
```

期望：

```text
True
```

4. build_tree 小样本日志验证

运行小样本 import 或 build_tree 后检查 `benchmark.log`：

```bash
rg "build_tree visual LLM|LLM image input stripped" /home/zhanggaoyuan.225/modora/Output/VersionRAG/modora_global/experiment_0001/benchmark.log
```

期望：

- `benchmark.log` 中能看到 build_tree crop/generate_levels 诊断日志。
- 终端不显示这些 build_tree 诊断日志。
- 如果裁剪得到图片且 `strip_llm_images=True`，能看到：

```text
LLM image input stripped before remote request
```

5. enrichment 纯视觉跳过验证

使用含 `image/chart/table` 组件的样本，或构造一次 `generate_annotation_async()` 调用：

期望：

- 能看到：

```text
visual annotation LLM skipped because strip_llm_images=true
```

- 不出现对应的 remote/local LLM 请求。
- 不产生基于空图片的幻觉式图像描述。

6. 非 VLM 保护验证

在 `strip_llm_images=True` 下运行会触发图片的查询或 build_tree：

期望：

- 不再出现：

```text
Model do not support image input
param: image_url
```

- 上游日志仍显示 crop/generate_levels/reasoning 流程正常执行。

7. VLM 默认行为验证

将配置设为：

```yaml
strip_llm_images: false
```

期望：

- LLM client 不拦截图片。
- 如果 `base64_image` 有有效图片，仍构造 `image_url`。
- 行为恢复为 MoDora 原多模态路径。
