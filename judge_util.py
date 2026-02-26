import asyncio
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def locomo_grader(
    llm_client:ChatOpenAI , model: str, question: str, gold_answer: str, response: str
) -> bool:
    system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """

    ACCURACY_PROMPT = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=ACCURACY_PROMPT)
    ]
    resp = llm_client.invoke(messages)
    content = resp.content
    result = json.loads(content)

    label = result.get("is_correct", result.get("label", "WRONG"))
    return label.strip().lower() == "correct"