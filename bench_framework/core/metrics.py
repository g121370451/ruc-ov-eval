"""
bench_framework 指标计算模块。
"""
import re
import string
import collections
from typing import List


class MetricsCalculator:
    @staticmethod
    def normalize_answer(s):
        """标准化答案文本：去标点、转小写、去冠词"""
        s = str(s).replace(',', "")
        def remove_articles(text): return re.sub(r'\b(a|an|the|and)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        return white_space_fix(remove_articles(remove_punc(s.lower())))

    @staticmethod
    def calculate_f1(prediction: str, ground_truth: str) -> float:
        pred_tokens = MetricsCalculator.normalize_answer(prediction).split()
        truth_tokens = MetricsCalculator.normalize_answer(ground_truth).split()
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def check_refusal(text: str) -> bool:
        refusals = ["not mentioned", "no information", "cannot be answered", "none", "unknown", "don't know"]
        return any(r in text.lower() for r in refusals)

    @staticmethod
    def check_recall(retrieved_texts: List[str], evidence_list: List[str],
                     soft_threshold: float = 0.8, min_soft_match_tokens: int = 4) -> float:
        if not evidence_list:
            return 0.0

        combined_retrieved = " ".join(retrieved_texts)
        normalized_retrieved = MetricsCalculator.normalize_answer(combined_retrieved)
        ret_tokens = set(normalized_retrieved.split())

        hit_count = 0
        for evidence in evidence_list:
            if evidence in combined_retrieved:
                hit_count += 1
                continue

            normalized_ev = MetricsCalculator.normalize_answer(evidence)
            ev_tokens = set(normalized_ev.split())

            if not ev_tokens:
                continue

            if len(ev_tokens) < min_soft_match_tokens:
                continue

            overlap_count = len(ev_tokens & ret_tokens)
            coverage = overlap_count / len(ev_tokens)

            if coverage >= soft_threshold:
                hit_count += 1

        return hit_count / len(evidence_list)
