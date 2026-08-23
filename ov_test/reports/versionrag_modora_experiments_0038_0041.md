# VersionRAG MoDora 消融实验简报

实验范围：`Output/VersionRAG/modora_global/experiment_0038` - `experiment_0041`

说明：四组实验用于对比 **OCR 开关** 与 **embedding 检索开关** 对 VersionRAG 结果的影响。所有实验均评测 100 条问题。

> 注：`experiment_0040` 的 report 名称显示为 `VersionRAG-ocr-embedding`，但日志中没有 `Starting vector retrieval`，因此按实际运行行为判定为 **OCR 开 + embedding 关**。

## 实验结果

| 实验 | 实际配置 | 平均检索耗时 | 平均输入 tokens | 平均输出 tokens | 平均 F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| `0038` | OCR 关 + embedding 开 | 123.12s | 92.7k | 14.4k | 0.393 | **0.780** |
| `0039` | OCR 关 + embedding 关 | 114.51s | 79.2k | 14.1k | 0.309 | 0.655 |
| `0040` | OCR 开 + embedding 关 | **96.55s** | **53.6k** | **10.8k** | 0.306 | 0.600 |
| `0041` | OCR 开 + embedding 开 | 111.02s | 98.1k | 11.1k | **0.424** | 0.745 |

## 关键观察

1. **embedding 对效果有明显帮助**

   在无 OCR 条件下，开启 embedding 后：

   - Accuracy：`0.655 -> 0.780`
   - F1：`0.309 -> 0.393`

   在 OCR 条件下，开启 embedding 后：

   - Accuracy：`0.600 -> 0.745`
   - F1：`0.306 -> 0.424`

2. **OCR 不一定提升 VersionRAG 当前任务效果**

   当前结果中，`OCR 关 + embedding 开` 的 Accuracy 最高，为 `0.780`。这说明对 VersionRAG 这批文档，直接使用 PDF 文本解析可能比 OCR 更保留问答所需文本结构。

3. **OCR 可以降低 token 和耗时**

   `0040` 的平均输入 tokens 最低，为 `53.6k`，平均检索耗时也最低，为 `96.55s`。但它的 F1 和 Accuracy 都最低，说明更短上下文不一定带来更好答案。

4. **embedding 首次运行会引入额外成本**

   `0041` 日志中出现了 `Found 414 documents missing embeddings` 和 `Upserted 414 new embeddings to Chroma`，说明这组实验包含首次补 embedding 索引的成本。后续复用 Chroma 索引时，耗时可能下降。

## 当前结论

综合 Accuracy、F1 和运行成本，当前更推荐：

```text
OCR 关 + embedding 开
```

对应实验：

```text
experiment_0038
```

理由：

- Accuracy 最高：`0.780`
- F1 明显高于无 embedding baseline
- 相比 OCR 路径，PDF 文本解析更适合当前 VersionRAG 文档

如果只看 F1，`experiment_0041` 最好，但它的 Accuracy 低于 `0038`，并且首次 embedding 构建带来了额外耗时。

## 非 VLM 模型支持方式

MoDora 原始流程中有多处会使用视觉输入，例如截图、页面裁剪、非文本组件 enrichment、视觉相关性判断和视觉推理。我们的实验模型是非 VLM，因此需要把 MoDora 降级为 **text-only RAG**。

当前支持方式如下：

```yaml
ocr_model: pdf_text
text_only_mode: true
enrich_non_text_components: false
visual_level_generation: false
visual_relevance_check: false
visual_reasoning: false
```

含义：

- `ocr_model: pdf_text`：不调用 Paddle OCR/VLM，直接从文本 PDF 中抽取文字块。
- `text_only_mode: true`：告诉 MoDora 后端当前模型不能接收图片输入。
- `enrich_non_text_components: false`：跳过图片、表格、图表等非文本组件的视觉 enrichment。
- `visual_level_generation: false`：不使用页面截图生成标题层级。
- `visual_relevance_check: false`：检索相关性判断只使用文本，不裁剪图片送模型。
- `visual_reasoning: false`：最终回答阶段只给文本 evidence，不传截图或图片。

这样做后，MoDora 仍保留核心思想：

```text
PDF 文本解析 -> component extraction -> build_tree -> 树检索/向量检索 -> 文本 QA
```

也就是说，我们没有绕过 MoDora 的树结构，只是把依赖 VLM 的视觉增强步骤改成文本路径。

### embedding 与 rerank

当前实验采用：

```yaml
enable_vector_search: true
enable_rerank: false
```

含义：

- 开启 embedding 后，MoDora 会把 tree node 的文本向量化并写入 Chroma。
- embedding 不是入库阶段生成，而是在首次检索时懒生成。
- rerank 暂时关闭，避免引入额外模型和接口变量。

当前 embedding 使用：

```env
EMBEDDING_MODEL=doubao-embedding-vision-250615
```

该模型走 Ark 的 `embeddings/multimodal` endpoint。虽然它是 vision embedding 模型，但在当前实验中只输入文本，不输入图片，因此仍符合非 VLM QA 的 text-only 约束。

## 后续建议

1. 固定 `OCR 关 + embedding 开` 作为当前主实验配置。
2. 对 embedding 建索引做 warmup，避免首次 gen 的耗时混入检索评测。
