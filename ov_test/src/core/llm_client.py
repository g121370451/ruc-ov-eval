import asyncio
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI


@dataclass
class _LLMResponse:
    content: str


class LLMClientWrapper:
    def __init__(self, config: dict, api_key: str):
        self.model = config['model']
        self.temperature = config.get('temperature', 0)
        self.client = OpenAI(api_key=api_key, base_url=config['base_url'])
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=config['base_url'])
        self.retry_count = 3

    @staticmethod
    def _normalize_messages(messages: list[Any]) -> list[dict[str, str]]:
        normalized = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized.append({
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", "")),
                })
            else:
                normalized.append({
                    "role": getattr(msg, "role", "user"),
                    "content": str(getattr(msg, "content", msg)),
                })
        return normalized

    def invoke(self, messages: list[Any]) -> _LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self._normalize_messages(messages),
        )
        return _LLMResponse(content=resp.choices[0].message.content or "")

    async def ainvoke(self, messages: list[Any]) -> _LLMResponse:
        resp = await self.async_client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self._normalize_messages(messages),
        )
        return _LLMResponse(content=resp.choices[0].message.content or "")

    def generate(self, prompt: str) -> str:
        """调用 LLM 生成回答，包含简单的指数退避重试"""
        last_err = None
        for attempt in range(self.retry_count):
            try:
                resp = self.invoke([{"role": "user", "content": prompt}])
                return resp.content
            except Exception as e:
                last_err = e
                # 简单指数退避: 1.5s, 3.0s, 4.5s
                time.sleep(1.5 * (attempt + 1))
        
        return f"ERROR: {str(last_err)}"

    async def agenerate(self, prompt: str) -> str:
        """异步调用 LLM 生成回答，优先使用原生 ainvoke，退化为 to_thread。"""
        last_err = None
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(self.retry_count):
            try:
                resp = await self.ainvoke(messages)
                return resp.content
            except Exception as e:
                last_err = e
                await asyncio.sleep(1.5 * (attempt + 1))

        return f"ERROR: {str(last_err)}"
    
    
    def explain_not_mentioned(
        self,
        question: str,
        context_texts: list,
    ) -> str:
        """
        当生成答案为 'Not mentioned' 时，让 LLM 解释为什么提供的上下文无法回答该问题。
        """
        context_str = "\n\n".join(context_texts[:10])
        prompt = f"""The following context was retrieved to answer a question, but the system concluded "Not mentioned".
    Explain briefly why the context is insufficient to answer the question.

    Context:
    {context_str}

    Question: {question}

    Respond with a short explanation (2-3 sentences).
    """
        try:
            resp = self.invoke([
                {"role": "system", "content": "You are a helpful assistant that analyzes retrieval quality."},
                {"role": "user", "content": prompt},
            ])
            return resp.content.strip() if resp and hasattr(resp, "content") else ""
        except Exception:
            return ""

    async def aexplain_not_mentioned(
        self,
        question: str,
        context_texts: list,
    ) -> str:
        """
        异步版本的 Not mentioned 原因解释。
        """
        context_str = "\n\n".join(context_texts[:10])
        prompt = f"""The following context was retrieved to answer a question, but the system concluded "Not mentioned".
    Explain briefly why the context is insufficient to answer the question.

    Context:
    {context_str}

    Question: {question}

    Respond with a short explanation (2-3 sentences).
    """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes retrieval quality."},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = await self.ainvoke(messages)
            return resp.content.strip() if resp and hasattr(resp, "content") else ""
        except Exception:
            return ""
