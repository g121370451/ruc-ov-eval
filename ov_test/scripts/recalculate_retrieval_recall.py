#!/usr/bin/env python3
"""
独立脚本：重新计算检索Recall，使用改进的匹配算法

用法:
    python scripts/recalculate_retrieval_recall.py --result_file path/to/qa_eval_detailed_results.json --data_root path/to/data/root

功能：
    1. 读取 qa_eval_detailed_results.json
    2. 根据检索到的文档URI，重新读取文档内容
    3. 使用改进的匹配算法重新计算检索recall
    4. 对比新旧recall值，输出改进效果报告

改进算法：
    - Level 1: 原始文本精确匹配
    - Level 2: 标准化后匹配（解决换行符、多空格问题）
    - Level 3: 词汇子集匹配（最宽松）
"""

import argparse
import json
import re
import string
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class ImprovedRecallCalculator:
    """改进的检索Recall计算器"""
    
    @staticmethod
    def normalize_text(text: str, case_sensitive: bool = False) -> str:
        """
        标准化文本：统一换行符、空白字符
        
        Args:
            text: 原始文本
            case_sensitive: 是否区分大小写
        
        Returns:
            标准化后的文本
        """
        if not case_sensitive:
            text = text.lower()
        
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def match_evidence(evidence: str, retrieved_text: str, mode: str = 'normal') -> bool:
        """
        检查evidence是否在检索文本中匹配
        
        Args:
            evidence: 证据片段
            retrieved_text: 检索到的文本
            mode: 匹配模式
                - 'strict': 仅精确匹配
                - 'normal': 标准化匹配（默认）
                - 'lenient': 包含词汇匹配
        
        Returns:
            是否匹配成功
        """
        if mode == 'strict':
            return evidence in retrieved_text
        
        if mode == 'normal':
            if evidence in retrieved_text:
                return True
            
            ev_normalized = ImprovedRecallCalculator.normalize_text(evidence)
            text_normalized = ImprovedRecallCalculator.normalize_text(retrieved_text)
            
            return ev_normalized in text_normalized
        
        if mode == 'lenient':
            if evidence in retrieved_text:
                return True
            
            ev_normalized = ImprovedRecallCalculator.normalize_text(evidence)
            text_normalized = ImprovedRecallCalculator.normalize_text(retrieved_text)
            
            if ev_normalized in text_normalized:
                return True
            
            ev_tokens = set(ev_normalized.split())
            text_tokens = set(text_normalized.split())
            
            if ev_tokens and ev_tokens.issubset(text_tokens):
                return True
        
        return False
    
    @staticmethod
    def check_recall_improved(
        retrieved_texts: List[str], 
        evidence_ids: List[str],
        mode: str = 'normal'
    ) -> Tuple[float, Dict[str, bool]]:
        """
        计算改进的检索recall
        
        Args:
            retrieved_texts: 检索到的文本列表
            evidence_ids: 证据片段列表
            mode: 匹配模式
        
        Returns:
            (recall值, 每个evidence的匹配详情)
        """
        if not evidence_ids:
            return 0.0, {}
        
        match_details = {}
        hits = 0
        
        for ev_id in evidence_ids:
            matched = any(
                ImprovedRecallCalculator.match_evidence(ev_id, text, mode)
                for text in retrieved_texts
            )
            match_details[ev_id] = matched
            if matched:
                hits += 1
        
        return hits / len(evidence_ids), match_details


def load_retrieved_content(uri: str, data_root: Path) -> str:
    """
    根据URI加载检索到的文档内容
    
    Args:
        uri: 文档URI (格式: viking://resources/...)
        data_root: 数据根目录
    
    Returns:
        文档内容
    """
    if uri.startswith("viking://resources/"):
        relative_path = uri.replace("viking://resources/", "")
        file_path = data_root / relative_path
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
                return ""
    
    return ""


def calculate_old_recall(retrieved_texts: List[str], evidence_ids: List[str]) -> float:
    """计算旧的recall（原始严格匹配）"""
    if not evidence_ids:
        return 0.0
    hits = sum(1 for ev_id in evidence_ids if any(ev_id in text for text in retrieved_texts))
    return hits / len(evidence_ids)


def main():
    parser = argparse.ArgumentParser(
        description='Recalculate retrieval recall with improved matching algorithm'
    )
    parser.add_argument(
        '--result-file', '-r', 
        type=str, 
        required=True,
        help='Path to qa_eval_detailed_results.json'
    )
    parser.add_argument(
        '--data-root', '-d',
        type=str,
        required=True,
        help='Root directory of the data (e.g., /path/to/viking_store_index)'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['strict', 'normal', 'lenient'],
        default='normal',
        help='Matching mode: strict, normal (default), or lenient'
    )
    parser.add_argument(
        '--output-detail',
        action='store_true',
        help='Output detailed match information for each sample'
    )
    
    args = parser.parse_args()
    
    result_file = Path(args.result_file)
    data_root = Path(args.data_root)
    
    if not result_file.exists():
        print(f"Error: Result file not found: {result_file}")
        return
    
    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        return
    
    print(f"\n{'='*80}")
    print(f"Recalculating Retrieval Recall")
    print(f"{'='*80}")
    print(f"Result file: {result_file}")
    print(f"Data root: {data_root}")
    print(f"Matching mode: {args.mode}")
    print(f"{'='*80}\n")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    if not results:
        print(f"Error: No 'results' field found in {result_file}")
        return
    
    old_recalls = []
    new_recalls = []
    improvements = []
    category_stats = defaultdict(lambda: {'old': [], 'new': [], 'improved': 0})
    
    detailed_results = []
    
    for idx, item in enumerate(results):
        old_recall = item.get('metrics', {}).get('Recall', 0.0)
        evidence_list = item.get('evidence', [])
        retrieval_uris = item.get('retrieval', {}).get('uris', [])
        category = item.get('category', 'unknown')
        question = item.get('question', '')
        
        retrieved_texts = []
        for uri in retrieval_uris:
            content = load_retrieved_content(uri, data_root)
            if content:
                retrieved_texts.append(content)
        
        new_recall, match_details = ImprovedRecallCalculator.check_recall_improved(
            retrieved_texts, 
            evidence_list,
            mode=args.mode
        )
        
        old_recalls.append(old_recall)
        new_recalls.append(new_recall)
        
        improvement = new_recall - old_recall
        if improvement > 0:
            improvements.append(improvement)
        
        category_stats[category]['old'].append(old_recall)
        category_stats[category]['new'].append(new_recall)
        if improvement > 0:
            category_stats[category]['improved'] += 1
        
        if args.output_detail or improvement > 0:
            detailed_results.append({
                'index': idx,
                'sample_id': item.get('sample_id', ''),
                'question': question[:100] + '...' if len(question) > 100 else question,
                'category': category,
                'old_recall': old_recall,
                'new_recall': new_recall,
                'improvement': improvement,
                'evidence_count': len(evidence_list),
                'matched_evidence': sum(1 for matched in match_details.values() if matched),
                'match_details': match_details if args.output_detail else None
            })
    
    avg_old_recall = sum(old_recalls) / len(old_recalls) if old_recalls else 0.0
    avg_new_recall = sum(new_recalls) / len(new_recalls) if new_recalls else 0.0
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
    
    print(f"\n{'='*80}")
    print(f"Overall Statistics")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Improved samples: {len(improvements)} ({len(improvements)/len(results)*100:.1f}%)")
    print(f"\nOld Average Recall: {avg_old_recall:.4f}")
    print(f"New Average Recall: {avg_new_recall:.4f}")
    print(f"Average Improvement: {avg_improvement:.4f}")
    print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"Statistics by Category")
    print(f"{'='*80}")
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        cat_old_avg = sum(stats['old']) / len(stats['old']) if stats['old'] else 0.0
        cat_new_avg = sum(stats['new']) / len(stats['new']) if stats['new'] else 0.0
        cat_improvement = cat_new_avg - cat_old_avg
        
        print(f"\nCategory: {cat}")
        print(f"  Samples: {len(stats['old'])}")
        print(f"  Improved: {stats['improved']} ({stats['improved']/len(stats['old'])*100:.1f}%)")
        print(f"  Old Recall: {cat_old_avg:.4f}")
        print(f"  New Recall: {cat_new_avg:.4f}")
        print(f"  Improvement: {cat_improvement:.4f}")
    
    print(f"\n{'='*80}\n")
    
    if detailed_results:
        print(f"\n{'='*80}")
        print(f"Improved Samples Detail (showing top 10)")
        print(f"{'='*80}")
        
        sorted_details = sorted(detailed_results, key=lambda x: x['improvement'], reverse=True)
        for detail in sorted_details[:10]:
            print(f"\nSample {detail['index']}: {detail['sample_id']}")
            print(f"  Question: {detail['question']}")
            print(f"  Category: {detail['category']}")
            print(f"  Old Recall: {detail['old_recall']:.4f}")
            print(f"  New Recall: {detail['new_recall']:.4f}")
            print(f"  Improvement: {detail['improvement']:.4f}")
            print(f"  Evidence: {detail['matched_evidence']}/{detail['evidence_count']} matched")
        
        print(f"\n{'='*80}\n")
    
    output_data = {
        "summary": {
            "total_samples": len(results),
            "improved_samples": len(improvements),
            "improvement_rate": len(improvements) / len(results) if results else 0.0,
            "old_avg_recall": avg_old_recall,
            "new_avg_recall": avg_new_recall,
            "avg_improvement": avg_improvement,
            "matching_mode": args.mode
        },
        "category_breakdown": {
            cat: {
                "total_samples": len(stats['old']),
                "improved_samples": stats['improved'],
                "old_avg_recall": sum(stats['old']) / len(stats['old']) if stats['old'] else 0.0,
                "new_avg_recall": sum(stats['new']) / len(stats['new']) if stats['new'] else 0.0
            }
            for cat, stats in category_stats.items()
        },
        "improved_samples": detailed_results
    }
    
    output_file = result_file.parent / f"recall_improvement_report_{args.mode}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
