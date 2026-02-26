import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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