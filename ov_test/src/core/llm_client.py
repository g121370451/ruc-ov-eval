import asyncio
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class LLMClientWrapper:
    def __init__(self, config: dict, api_key: str):
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=config['temperature'],
            api_key=api_key,
            base_url=config['base_url']
        )
        self.retry_count = 3

    def generate(self, prompt: str) -> str:
        """调用 LLM 生成回答，包含简单的指数退避重试"""
        last_err = None
        for attempt in range(self.retry_count):
            try:
                # invoke 返回的是 AIMessage，需要取 .content
                resp = self.llm.invoke([HumanMessage(content=prompt)])
                return resp.content
            except Exception as e:
                last_err = e
                # 简单指数退避: 1.5s, 3.0s, 4.5s
                time.sleep(1.5 * (attempt + 1))
        
        return f"ERROR: {str(last_err)}"

    async def agenerate(self, prompt: str) -> str:
        """异步调用 LLM 生成回答，优先使用原生 ainvoke，退化为 to_thread。"""
        last_err = None
        messages = [HumanMessage(content=prompt)]
        for attempt in range(self.retry_count):
            try:
                if hasattr(self.llm, "ainvoke"):
                    resp = await self.llm.ainvoke(messages)
                else:
                    resp = await asyncio.to_thread(self.llm.invoke, messages)
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
            resp = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant that analyzes retrieval quality."),
                HumanMessage(content=prompt),
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
            SystemMessage(content="You are a helpful assistant that analyzes retrieval quality."),
            HumanMessage(content=prompt),
        ]
        try:
            if hasattr(self.llm, "ainvoke"):
                resp = await self.llm.ainvoke(messages)
            else:
                resp = await asyncio.to_thread(self.llm.invoke, messages)
            return resp.content.strip() if resp and hasattr(resp, "content") else ""
        except Exception:
            return ""
