#!/usr/bin/env python3
"""
分析recall提升样本的accuracy分布

用法:
    python scripts/analyze_recall_accuracy_correlation.py --result_file path/to/qa_eval_detailed_results.json --report_file path/to/recall_improvement_report_normal.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description='Analyze accuracy distribution for recall-improved samples'
    )
    parser.add_argument(
        '--result-file', '-r', 
        type=str, 
        required=True,
        help='Path to qa_eval_detailed_results.json'
    )
    parser.add_argument(
        '--report-file', '-rf',
        type=str,
        required=True,
        help='Path to recall_improvement_report_normal.json'
    )
    
    args = parser.parse_args()
    
    result_file = Path(args.result_file)
    report_file = Path(args.report_file)
    
    if not result_file.exists():
        print(f"Error: Result file not found: {result_file}")
        return
    
    if not report_file.exists():
        print(f"Error: Report file not found: {report_file}")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    with open(report_file, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    results = results_data.get('results', [])
    improved_samples = report_data.get('improved_samples', [])
    
    improved_indices = {s['index'] for s in improved_samples if s['improvement'] > 0}
    
    print(f"\n{'='*80}")
    print(f"Recall Improvement vs Accuracy Analysis")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Recall-improved samples: {len(improved_indices)}")
    print(f"{'='*80}\n")
    
    accuracy_distribution = defaultdict(int)
    accuracy_by_category = defaultdict(lambda: {'total': 0, 'sum': 0.0, 'count_by_level': defaultdict(int)})
    
    high_accuracy_count = 0
    low_accuracy_count = 0
    
    for idx in improved_indices:
        if idx < len(results):
            item = results[idx]
            accuracy = item.get('metrics', {}).get('Accuracy', 0)
            category = item.get('category', 'unknown')
            
            accuracy_distribution[accuracy] += 1
            
            accuracy_by_category[category]['total'] += 1
            accuracy_by_category[category]['sum'] += accuracy
            accuracy_by_category[category]['count_by_level'][accuracy] += 1
            
            if accuracy >= 3:
                high_accuracy_count += 1
            elif accuracy <= 1:
                low_accuracy_count += 1
    
    print(f"\n{'='*80}")
    print(f"Accuracy Distribution for Recall-Improved Samples")
    print(f"{'='*80}")
    
    for acc in sorted(accuracy_distribution.keys(), reverse=True):
        count = accuracy_distribution[acc]
        percentage = count / len(improved_indices) * 100
        bar = '█' * int(percentage / 2)
        print(f"Accuracy {acc}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}")
    print(f"High Accuracy (≥3): {high_accuracy_count} ({high_accuracy_count/len(improved_indices)*100:.1f}%)")
    print(f"Medium Accuracy (2): {accuracy_distribution.get(2, 0)} ({accuracy_distribution.get(2, 0)/len(improved_indices)*100:.1f}%)")
    print(f"Low Accuracy (≤1): {low_accuracy_count} ({low_accuracy_count/len(improved_indices)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"Average Accuracy by Category")
    print(f"{'='*80}")
    for cat in sorted(accuracy_by_category.keys()):
        stats = accuracy_by_category[cat]
        avg_acc = stats['sum'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"\n{cat}:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Average accuracy: {avg_acc:.2f}")
        print(f"  Distribution:")
        for acc in sorted(stats['count_by_level'].keys(), reverse=True):
            count = stats['count_by_level'][acc]
            pct = count / stats['total'] * 100
            print(f"    Accuracy {acc}: {count} ({pct:.1f}%)")
    
    print(f"\n{'='*80}\n")
    
    correlation_data = {
        "total_improved_samples": len(improved_indices),
        "accuracy_distribution": dict(accuracy_distribution),
        "high_accuracy_count": high_accuracy_count,
        "low_accuracy_count": low_accuracy_count,
        "high_accuracy_percentage": high_accuracy_count / len(improved_indices) * 100,
        "low_accuracy_percentage": low_accuracy_count / len(improved_indices) * 100,
        "accuracy_by_category": {
            cat: {
                "total": stats['total'],
                "average_accuracy": stats['sum'] / stats['total'] if stats['total'] > 0 else 0.0,
                "distribution": dict(stats['count_by_level'])
            }
            for cat, stats in accuracy_by_category.items()
        }
    }
    
    output_file = result_file.parent / "recall_accuracy_correlation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(correlation_data, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed correlation data saved to: {output_file}")


if __name__ == "__main__":
    main()
