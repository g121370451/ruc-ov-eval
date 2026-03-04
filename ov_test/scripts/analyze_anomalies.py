"""
分析评测结果中的异常 QA 对

异常类型：
1. Recall=0 但 Accuracy 高（检索失败但 LLM 答对了）
2. Recall 高 但 Accuracy=0（检索成功但 LLM 答错了）
3. F1 高 但 Accuracy 低（指标不一致）
4. 拒绝回答但 Accuracy=0（可能误判）
"""

import json
import argparse
import os
from typing import List, Dict, Any


def load_results(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"结果文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"无法解析结果文件格式: {file_path}")


def analyze_anomalies(results: List[Dict[str, Any]], thresholds: dict) -> Dict[str, List]:
    anomalies = {
        "recall_zero_acc_high": [],
        "recall_high_acc_zero": [],
        "f1_high_acc_low": [],
        "refusal_acc_zero": [],
        "all_zero": [],
        "all_perfect": [],
    }
    
    for item in results:
        metrics = item.get('metrics', {})
        recall = metrics.get('Recall', 0)
        f1 = metrics.get('F1', 0)
        acc = metrics.get('Accuracy', 0)
        
        llm_answer = item.get('llm', {}).get('final_answer', '')
        
        if recall == 0 and acc >= thresholds['acc_high']:
            anomalies["recall_zero_acc_high"].append(item)
        
        elif recall >= thresholds['recall_high'] and acc == 0:
            anomalies["recall_high_acc_zero"].append(item)
        
        elif f1 >= thresholds['f1_high'] and acc < thresholds['acc_low']:
            anomalies["f1_high_acc_low"].append(item)
        
        if llm_answer and any(kw in llm_answer.lower() for kw in ['not mentioned', 'cannot', 'unable', 'don\'t know', 'no information']):
            if acc == 0:
                anomalies["refusal_acc_zero"].append(item)
        
        if recall == 0 and f1 == 0 and acc == 0:
            anomalies["all_zero"].append(item)
        
        if recall == 1.0 and f1 == 1.0 and acc == 4:
            anomalies["all_perfect"].append(item)
    
    return anomalies


def save_anomalies_json(anomalies: Dict[str, List], output_dir: str, input_filename: str):
    base_name = os.path.splitext(input_filename)[0]
    
    for anomaly_type, items in anomalies.items():
        if items:
            output_file = os.path.join(output_dir, f"{base_name}_{anomaly_type}.json")
            output_data = {
                "anomaly_type": anomaly_type,
                "count": len(items),
                "results": items
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"  保存: {output_file} ({len(items)} 条)")


def main():
    parser = argparse.ArgumentParser(description="分析评测结果中的异常 QA")
    parser.add_argument("--input", "-i", required=True, help="输入结果文件路径 (JSON)")
    parser.add_argument("--output-dir", "-o", default=None, help="输出目录 (默认: 与输入文件同目录)")
    parser.add_argument("--acc-high", type=float, default=3, help="Accuracy 高阈值 (默认: 3, 范围0-4)")
    parser.add_argument("--acc-low", type=float, default=1, help="Accuracy 低阈值 (默认: 1, 范围0-4)")
    parser.add_argument("--recall-high", type=float, default=0.5, help="Recall 高阈值 (默认: 0.5)")
    parser.add_argument("--f1-high", type=float, default=0.5, help="F1 高阈值 (默认: 0.5)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input)
    
    print(f"读取结果文件: {args.input}")
    results = load_results(args.input)
    print(f"共加载 {len(results)} 条结果")
    
    thresholds = {
        'acc_high': args.acc_high,
        'acc_low': args.acc_low,
        'recall_high': args.recall_high,
        'f1_high': args.f1_high,
    }
    
    print("分析异常 QA...")
    anomalies = analyze_anomalies(results, thresholds)
    
    input_filename = os.path.basename(args.input)
    print(f"\n保存异常结果到: {args.output_dir}")
    save_anomalies_json(anomalies, args.output_dir, input_filename)
    
    print("\n=== 统计摘要 ===")
    for key, items in anomalies.items():
        print(f"  {key}: {len(items)} 条")


if __name__ == "__main__":
    main()
