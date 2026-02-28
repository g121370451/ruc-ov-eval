import asyncio
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import json
from langchain_core.messages import HumanMessage, SystemMessage

def llm_grader(
    llm_client, model: str, question: str, gold_answer: str, response: str, dataset_name: str = "Locomo"
) -> float:
    """
    1.使用大模型裁判对生成的答案进行细粒度的 5 档评分（1-5分），并归一化为 0.0~1.0 的最终得分。
    2.思路说明：
      - 依据 dataset_name 路由选择特定数据集的评分标准（Rubric）。
      - 引入业界标准的 5 档计分制（1到5分），并在 Prompt 中为每个档位明确具体的判定依据（如完美、微小遗漏、部分正确、严重错误、完全无关/幻觉）。
      - 强制模型输出包含 `score` (1-5的整数) 和 `reasoning` 的 JSON 格式。
      - 解析结果后，将 1-5 分线性映射/归一化到 0.0~1.0区间。
    3.输入参数：
      - llm_client: Any，大模型客户端实例（必填）
      - model: str，使用的模型名称标识（必填）
      - question: str，原始问题文本（必填）
      - gold_answer: str，标准答案文本（必填）
      - response: str，待评测的生成答案文本（必填）
      - dataset_name: str，数据集名称（可选，默认 "Locomo"）
    4.返回值类型和具体含义：float 类型，表示生成的答案与标准答案的语义匹配归一化得分，取值范围通常为 0.0 到 1.0。
    """
    
    if "Locomo" in dataset_name.lower():
        system_prompt = """
        You are an expert evaluator scoring an AI's ability to remember and retrieve personal facts or past conversational context.
        """
        ACCURACY_PROMPT = f"""
        Please score the Generated Answer against the Gold Answer on a scale of 1 to 5.
        The question tests if the AI remembers specific details about the user based on prior interactions. 

        [Evaluation Rubric]
        - Score 5 (Perfect): The answer correctly and precisely identifies the core fact/time/item required by the gold answer.
        - Score 4 (Good): The answer is correct but includes slightly extraneous conversational filler, or formats a date/time slightly differently but means the exact same thing.
        - Score 3 (Acceptable): The answer captures the general idea but misses a specific detail (e.g., gets the month right but misses the exact day).
        - Score 2 (Poor): The answer touches on the topic but gets the core fact wrong.
        - Score 1 (Wrong): Completely wrong, hallucinates a different fact, or falsely claims no memory of the event.

        Question: {question}
        Gold Answer: {gold_answer}
        Generated Answer: {response}

        First, write a 1-sentence reasoning. Then output the integer score.
        Respond ONLY with a JSON object: {{"score": integer, "reasoning": "string"}}
        """

    elif "Qasper" in dataset_name.lower():
        system_prompt = """
        You are a strict academic peer reviewer scoring an AI's answer against a gold standard derived from a scientific paper.
        """
        ACCURACY_PROMPT = f"""
        Please score the Generated Answer against the Gold Answer on a scale of 1 to 5.
        
        [Evaluation Rubric]
        - Score 5 (Perfect): Contains all exact entities, metrics, or factual claims. For lists, ALL key items are present. If the Gold answer is "Unanswerable", the AI correctly states it is missing.
        - Score 4 (Minor Omissions): Correct core facts, but misses 1 or 2 minor items in a long list, or uses a slightly different but scientifically equivalent synonym.
        - Score 3 (Partial): Captures about half of the required information (e.g., gets 2 out of 4 list items right), but mixes in some irrelevant (though not contradictory) context.
        - Score 2 (Severe Errors): Misses the vast majority of required facts, or provides a heavily distorted interpretation of the metric/finding.
        - Score 1 (Wrong/Hallucinated): Explicitly contradicts the Gold Answer, completely hallucinates facts, or makes up an answer when the Gold Answer indicates "Unanswerable".

        Question: {question}
        Gold Answer: {gold_answer}
        Generated Answer: {response}

        First, write a 1-sentence reasoning. Then output the integer score.
        Respond ONLY with a JSON object: {{"score": integer, "reasoning": "string"}}
        """

    else:
        system_prompt = """
        You are an expert evaluator scoring how well an AI-generated answer matches a gold standard (ground truth).
        """
        ACCURACY_PROMPT = f"""
        Please score the Generated Answer against the Gold Answer on a scale of 1 to 5.

        [Evaluation Rubric]
        - Score 5 (Perfect): Completely and accurately captures the core meaning and facts of the gold answer.
        - Score 4 (Good): Captures the main facts but includes unnecessary verbosity or minor non-contradictory details.
        - Score 3 (Partial): Missing some key factual information but touches on the correct topic.
        - Score 2 (Poor): Mostly incorrect or severely incomplete.
        - Score 1 (Wrong): Completely wrong, contradicts the gold answer, or hallucinates.

        Question: {question}
        Gold Answer: {gold_answer}
        Generated Answer: {response}

        First, write a 1-sentence reasoning. Then output the integer score.
        Respond ONLY with a JSON object: {{"score": integer, "reasoning": "string"}}
        """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=ACCURACY_PROMPT)
    ]
    
    try:
        resp = llm_client.invoke(messages)
        content = resp.content
        
        # 尝试解析 JSON
        result = json.loads(content)
        raw_score = int(result.get("score", 1))
        
    except Exception:
        # 容错：如果 JSON 解析失败，尝试从文本中直接提取 1-5 的数字，兜底给 1 分 (最差)
        raw_score = 1
        if "score\": 5" in content or "\"score\":5" in content: raw_score = 5
        elif "score\": 4" in content or "\"score\":4" in content: raw_score = 4
        elif "score\": 3" in content or "\"score\":3" in content: raw_score = 3
        elif "score\": 2" in content or "\"score\":2" in content: raw_score = 2

    # 限制范围在 1-5 之间
    raw_score = max(1, min(5, raw_score))
    
    # 归一化到 0.0 - 1.0 区间 (1分->0.0, 2分->0.25, 3分->0.5, 4分->0.75, 5分->1.0)
    normalized_score = (raw_score - 1) / 4.0
    
    return normalized_score



# def llm_grader(
#     llm_client, model: str, question: str, gold_answer: str, response: str, dataset_name: str = "Locomo"
# ) -> bool:
    
#     # 1. 根据 dataset_name 路由选择 Prompt
#     if "Locomo" in dataset_name.lower():
#         system_prompt = """
#         You are an expert grader that determines if answers to questions match a gold standard answer
#         """
#         ACCURACY_PROMPT = f"""
#     Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
#         (1) a question (posed by one user to another user),
#         (2) a 'gold' (ground truth) answer,
#         (3) a generated answer
#     which you will score as CORRECT/WRONG.

#     The point of the question is to ask about something one user should know about the other user based on their prior conversations.
#     The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
#     Question: Do you remember what I got the last time I went to Hawaii?
#     Gold answer: A shell necklace
#     The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

#     For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

#     Now it's time for the real question:
#     Question: {question}
#     Gold answer: {gold_answer}
#     Generated answer: {response}

#     First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
#     Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

#     Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
#     """
    
#     elif "Qasper" in dataset_name.lower():
#         system_prompt = """
#         You are an expert academic peer reviewer evaluating the accuracy of an AI's answer against a gold standard derived from a scientific paper.
#         """
#         ACCURACY_PROMPT = f"""Your task is to evaluate if a Generated Answer is correct based on the Gold Answer.
        
# Grading Rules:
# 1. Exact/Semantic Match: If the Generated Answer contains the exact entities, metrics, or factual claims present in the Gold Answer, grade it as CORRECT.
# 2. Unanswerable Cases: If the Gold Answer indicates the info is missing (e.g., "Not mentioned", "Unanswerable") AND the Generated Answer correctly identifies that the context lacks this information, grade it as CORRECT. If the Generated Answer hallucinates a fact when it should be unanswerable, grade it as WRONG.
# 3. Lists/Multiple Items: If the Gold Answer is a list of items, the Generated Answer should identify the core items to be CORRECT. Minor variations in terminology (e.g., "F1-score" vs "F1 measure") are CORRECT.
# 4. Contradiction: If the Generated Answer explicitly contradicts the core facts of the Gold Answer, it is WRONG.

# Question: {question}
# Gold Answer: {gold_answer}
# Generated Answer: {response}

# First, provide a brief reasoning (1 sentence), then finish with CORRECT or WRONG.
# Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
# """
    
#     else:
#         # 通用 Prompt 或其他数据集的 Prompt
#         system_prompt = """
#         You are an expert grader that determines if an AI-generated answer matches the gold standard (ground truth) answer for a given question.
#         """
#         ACCURACY_PROMPT = f"""
#         Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
#             (1) A question
#             (2) A 'gold' (ground truth) answer
#             (3) A generated answer

#         Grading rules:
#         - If the generated answer correctly encompasses the core semantic meaning or facts of the gold answer, grade it as CORRECT.
#         - If the generated answer contradicts the gold answer or misses the key factual information, it is WRONG.

#         Question: {question}
#         Gold answer: {gold_answer}
#         Generated answer: {response}

#         First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
#         Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
#         """

#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=ACCURACY_PROMPT)
#     ]
#     resp = llm_client.invoke(messages)
#     content = resp.content
    
#     try:
#         result = json.loads(content)
#         label = result.get("is_correct", result.get("label", "WRONG"))
#         return label.strip().lower() == "correct"
#     except json.JSONDecodeError:
#         # 容错：防止 LLM 没按格式输出 JSON
#         return "CORRECT" in content.upper()