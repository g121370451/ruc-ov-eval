import json

def analyze_qa_metrics(file_path, output_file='/home/fengxiaoyao/Viking/Output/Qasper/experiment_v21/Qasper_filtered_anomalies.log', high=0.8, low=0.2):
    """
    1. 筛选并打印 JSON 数据集中 Recall 与 Accuracy 存在显著不一致的异常样本。
    
    2. 思路说明：
       - 加载 JSON 文件并遍历 `results` 列表中的每个样本。
       - 获取每个样本 `metrics` 字段中的 `Recall` 和 `Accuracy`。
       - 分别应用两个过滤条件：(Recall >= 0.8 且 Accuracy <= 0.2) 和 (Accuracy >= 0.8 且 Recall <= 0.2)。
       - 将符合条件的样本详细信息（ID、问题、指标、回答）格式化输出至文件。

    3. 输入参数：
       - file_path: str, 输入的 JSON 文件路径（必填）
       - output_file: str, 结果保存的文件路径（可选，默认 'filtered_anomalies.txt'）
       - high: float, 高分判定的阈值（可选，默认 0.8）
       - low: float, 低分判定的阈值（可选，默认 0.2）

    4. 返回值类型：None (直接生成并保存文件)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('results', [])
    case1_hr_la = [] # High Recall, Low Accuracy
    case2_ha_lr = [] # High Accuracy, Low Recall

    for item in results:
        metrics = item.get('metrics', {})
        recall = metrics.get('Recall', 0)
        accuracy = metrics.get('Accuracy', 0)

        # 情况 1: Recall 很高，但是 Accuracy 很低
        if recall >= high and accuracy <= low:
            case1_hr_la.append(item)

        # 情况 2: Accuracy 很高，但是 Recall 很低
        if accuracy >= high and recall <= low:
            case2_ha_lr.append(item)

    # 打印/输出到文件
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"=== 情况 1: Recall 高 (>= {high}) 但 Accuracy 低 (<= {low}) ===\n")
        out.write(f"共发现 {len(case1_hr_la)} 例\n\n")
        for item in case1_hr_la:
            out.write(f"Query ID: {item.get('_global_index')}\n")
            out.write(f"Question: {item.get('question')}\n")
            out.write(f"Metrics: {item.get('metrics')}\n")
            out.write(f"LLM Answer: {item.get('llm', {}).get('final_answer')}\n")
            out.write(f"Gold Answer: {item.get('gold_answers')}\n")
            out.write("-" * 50 + "\n")

        out.write(f"\n\n=== 情况 2: Accuracy 高 (>= {high}) 但 Recall 低 (<= {low}) ===\n")
        out.write(f"共发现 {len(case2_ha_lr)} 例\n\n")
        for item in case2_ha_lr:
            out.write(f"Query ID: {item.get('_global_index')}\n")
            out.write(f"Question: {item.get('question')}\n")
            out.write(f"Metrics: {item.get('metrics')}\n")
            out.write(f"LLM Answer: {item.get('llm', {}).get('final_answer')}\n")
            out.write(f"Gold Answer: {item.get('gold_answers')}\n")
            out.write("-" * 50 + "\n")

# 调用执行
analyze_qa_metrics('/home/fengxiaoyao/Viking/Output/Qasper/experiment_v21/prompt-v1/qa_eval_detailed_results.json')