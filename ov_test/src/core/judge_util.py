import asyncio
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import json
from langchain_core.messages import HumanMessage, SystemMessage

import json
from langchain_core.messages import HumanMessage, SystemMessage

def llm_grader(
    llm_client, model: str, question: str, gold_answer: str, response: str, dataset_name: str = "Locomo"
) -> dict:
    """
    1.使用大模型裁判对生成的答案进行细粒度的 5 档评分，并返回包含得分、理由和所用 Prompt 类型的完整评估字典。
    2.思路说明：
      - 依据 dataset_name 路由选择特定数据集的评分标准，并记录对应的 prompt_type 标签。
      - 强制模型输出包含 `score` 和 `reasoning` 的 JSON 格式。
      - 异常处理：若 JSON 解析失败，从文本提取数字并保留原始输出作为 reasoning。
      - 最终将归一化得分（0.0~1.0）、理由和 Prompt 类型打包为字典返回，供上层写入评测报告。
    3.输入参数：
      - llm_client: Any，大模型客户端实例（必填）
      - model: str，使用的模型名称标识（必填）
      - question: str，原始问题文本（必填）
      - gold_answer: str，标准答案文本（必填）
      - response: str，待评测的生成答案文本（必填）
      - dataset_name: str，数据集名称（可选，默认 "Locomo"）
    4.返回值类型和具体含义：dict，包含以下键值对：
      - 'score' (float): 归一化后的匹配得分 (0.0~1.0)
      - 'reasoning' (str): 大模型给出的评分理由
      - 'prompt_type' (str): 评测所使用的 Prompt 模板类型
    """
    
    prompt_type = "Generic_5-Point"
    
    if "locomo" in dataset_name.lower():
        prompt_type = "Locomo_5-Point"
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

    elif "qasper" in dataset_name.lower():
        prompt_type = "Qasper_5-Point"
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
    
    reasoning = "Parsing failed. Defaulting to 1."
    raw_score = 1
    
    try:
        resp = llm_client.invoke(messages)
        content = resp.content
        
        # 尝试解析 JSON
        result = json.loads(content)
        raw_score = int(result.get("score", 1))
        reasoning = result.get("reasoning", "No reasoning provided.")
        
    except Exception:
        # 容错：兜底给 1 分，并将完整文本存入 reasoning
        if content:
            reasoning = f"Parse Error. Raw Output: {content.strip()}"
            if "score\": 5" in content or "\"score\":5" in content: raw_score = 5
            elif "score\": 4" in content or "\"score\":4" in content: raw_score = 4
            elif "score\": 3" in content or "\"score\":3" in content: raw_score = 3
            elif "score\": 2" in content or "\"score\":2" in content: raw_score = 2

    raw_score = max(1, min(5, raw_score))
    normalized_score = (raw_score - 1) / 4.0
    
    return {
        "score": normalized_score,
        "reasoning": reasoning,
        "prompt_type": prompt_type
    }
