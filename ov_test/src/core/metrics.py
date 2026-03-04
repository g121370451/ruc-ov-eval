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

    # @staticmethod
    # def check_recall(retrieved_texts: List[str], evidence_ids: List[str]) -> float:
    #     if not evidence_ids: return 0.0 
    #     hits = sum(1 for ev_id in evidence_ids if any(ev_id in text for text in retrieved_texts))
    #     return hits / len(evidence_ids)
    
    @staticmethod
    def check_recall(retrieved_texts: List[str], evidence_list: List[str], soft_threshold: float = 0.8, min_soft_match_tokens: int = 4) -> float:
        """
        1. 核心作用：计算检索召回率（Recall），结合严格子串匹配与动态长度判断的 Token 软匹配。
        
        2. 思路说明：
           - 拼接与预处理：将检索到的多个url文本块拼接为单一字符串。
           - 严格匹配优先（步骤 1）：优先判断 evidence 是否作为完整子串存在于拼接后的检索文本中。
           - 长度阻断机制（防止 ID 误判）：计算 evidence 的有效 Token 数量。如果数量低于设定阈值（如短 ID 或短实体），严格匹配失败后直接判定未命中，**禁止**进入软匹配。
           - 软匹配兜底（步骤 2）：对于长文本 evidence，计算其 Token 在检索文本中的覆盖率，达到阈值即视为命中。
           - 均等计分：每条 evidence 权重相同，最终得分为命中条数/总条数。
           
        3. 输入参数：
           - retrieved_texts: List[str]，检索模块返回的文本块列表（必填）
           - evidence_list: List[str]，真实的证据列表，包含 ID 或长文本证据（必填）
           - soft_threshold: float，软匹配判定为命中的覆盖率阈值（可选，取值范围 0.0~1.0，默认 0.8）
           - min_soft_match_tokens: int，允许降级使用软匹配的最小有效 Token 数量阈值（可选，默认 4。低于此长度的短文本必须严格匹配）
           
        4. 返回值类型和具体含义：
           - float 类型，表示检索召回得分，取值范围 0.0 到 1.0。
        """
        if not evidence_list: 
            return 0.0 
            
        combined_retrieved = " ".join(retrieved_texts)
        
        normalized_retrieved = MetricsCalculator.normalize_answer(combined_retrieved)
        ret_tokens = set(normalized_retrieved.split())
        
        hit_count = 0
        
        for evidence in evidence_list:
            # 【匹配步骤 1】：严格子串匹配
            if evidence in combined_retrieved:
                hit_count += 1
                continue
                
            # 预处理 evidence 获取有效 Tokens
            normalized_ev = MetricsCalculator.normalize_answer(evidence)
            ev_tokens = set(normalized_ev.split())
            
            if not ev_tokens:
                continue
                
            # 【长度阻断机制】
            # 如果 evidence 的有效词汇数量太少（比如 "ID 5" 只有 2 个 token），严格匹配失败后直接判定未命中。
            if len(ev_tokens) < min_soft_match_tokens:
                continue
                
            # 【匹配步骤 2】：软匹配（仅限长文本）
            overlap_count = len(ev_tokens & ret_tokens)
            coverage = overlap_count / len(ev_tokens)
            
            if coverage >= soft_threshold:
                hit_count += 1
                
        return hit_count / len(evidence_list)
