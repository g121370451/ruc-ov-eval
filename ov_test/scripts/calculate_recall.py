#!/usr/bin/env python3
"""
独立脚本：根据 qa_eval_detailed_results.json 中的 LLM final_answer 计算 average recall。

用法:
    python scripts/calculate_recall.py --result_file path/to/qa_eval_detailed_results.json

Recall 计算逻辑：
    1. 从 results.json 中读取每个样本的:
        - llm.final_answer (生成的答案)
        - evidence (证据片段列表，    2. 对 final_answer 进行标准化处理
    3. 棏个 evidence 片段在 final_answer 中进行软匹配
    4. 计算 recall = 彽中 evidence 数 / 总 evidence 数
"""

import argparse
import json
import re
import string
from pathlib import Path
from typing import List


def normalize_answer(s: str) -> str:
    """标准化答案文本"""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', '', text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def soft_match(evidence: str, answer: str) -> bool:
    """检查 evidence 片段是否在 answer 中出现（软匹配）"""
    evidence_lower = evidence.lower()
    answer_lower = answer.lower()
    
    evidence_normalized = normalize_answer(evidence)
    answer_normalized = normalize_answer(answer)
    
    if evidence_lower in answer_lower:
        return True
    
    if evidence_normalized in answer_normalized:
        return True
    
    evidence_tokens = set(evidence_normalized.split())
    answer_tokens = set(answer_normalized.split())
    
    if evidence_tokens and evidence_tokens.issubset(answer_tokens):
        return True
    
    return False


def calculate_recall(final_answer: str, evidence_list: List[str]) -> float:
    """
    计算 recall
    
    Args:
        final_answer: LLM 生成的答案
        evidence_list: 证据片段列表
    
    Returns:
        recall: 0.0 ~ 1.0
    """
    if not evidence_list:
        return 0.0
    
    hits = 0
    for evidence in evidence_list:
        if soft_match(evidence, final_answer):
            hits += 1
    
    return hits / len(evidence_list)


def main():
    parser = argparse.ArgumentParser(description='Calculate average recall from qa_eval_detailed_results.json')
    parser.add_argument('--result-file', '-r', type=str, required=True,
                        help='Path to qa_eval_detailed_results.json')
    args = parser.parse_args()
    
    result_file = Path(args.result_file)
    
    if not result_file.exists():
        print(f"Error: File not found: {result_file}")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    if not results:
        print(f"Error: No 'results' field found in {result_file}")
        return
    
    all_recalls = []
    category_recalls = {}
    
    for item in results:
        metrics = item.get('metrics', {})
        recall = metrics.get('Recall', 0.0)
        category = item.get('category', 'unknown')
        
        if recall is None or recall < 0:
            continue
        
        all_recalls.append(recall)
        
        if category not in category_recalls:
            category_recalls[category] = []
        category_recalls[category].append(recall)
    
    if not all_recalls:
        print("No valid results found.")
        return
    
    avg_recall = sum(all_recalls) / len(all_recalls)
    
    print(f"\n{'='*60}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Total samples: {len(all_recalls)}")
    print(f"{'='*60}\n")
    
    print("\nRecall by Category:")
    print("-" * 40)
    for cat in sorted(category_recalls.keys()):
        cat_avg = sum(category_recalls[cat]) / len(category_recalls[cat])
        print(f"  Category {cat}: {cat_avg:.4f} ({len(category_recalls[cat])} samples)")
    
    print("\n" + "-" * 40)
    print(f"{'='*60}\n")
    
    output_data = {
        "average_recall": avg_recall,
        "total_samples": len(all_recalls),
        "category_breakdown": {
            cat: {
                "recall": sum(recalls) / len(recalls),
                "count": len(recalls)
            }
            for cat, recalls in category_recalls.items()
        }
    }
    
    output_file = result_file.parent / "recall_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
