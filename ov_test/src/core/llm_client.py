import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "rate" in msg or "throttl" in msg or "tpm" in msg or "rpm" in msg or "too many requests" in msg

class LLMClientWrapper:
    def __init__(self, config: dict, api_key: str):
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=config['temperature'],
            api_key=api_key,
            base_url=config['base_url']
        )
        self.retry_count = 5
        self.base_delay = 2.0
        self.max_delay = 60.0

    def _backoff_delay(self, attempt: int, is_rate_limit: bool) -> float:
        delay = self.base_delay * (2 ** attempt)
        if is_rate_limit:
            delay *= 2
        return min(delay, self.max_delay)

    def generate(self, prompt: str) -> str:
        last_err = None
        for attempt in range(self.retry_count):
            try:
                resp = self.llm.invoke([HumanMessage(content=prompt)])
                return resp.content
            except Exception as e:
                last_err = e
                is_rl = _is_rate_limit_error(e)
                delay = self._backoff_delay(attempt, is_rl)
                time.sleep(delay)
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