
import os
import json
from pathlib import Path
from langchain_openai import ChatOpenAI

class PageIndexConfig:
    """PageIndex配置管理类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = "pageindex.conf"
        
        if not os.path.exists(config_path):
            # 尝试在脚本目录中查找
            script_dir = Path(__file__).parent.parent
            candidate_path = os.path.join(script_dir, "ov_test", "pageindex.conf")
            print(f"[Info] Trying to find config in: {candidate_path}")
            if os.path.exists(candidate_path):
                config_path = candidate_path
            else:
                raise FileNotFoundError(f"PageIndex config file not found: {config_path}")
        
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def get_pageindex_config(self):
        """获取pageindex配置"""
        return {k: v for k, v in self.config.items() if k != 'vlm'}
    
    def get_vlm_config(self):
        """获取VLM配置"""
        return self.config.get('vlm', {})
    
    def get_model_name(self):
        """获取模型名称"""
        return self.config.get('model', 'doubao-seed-2-0-pro-260215')
    
    def create_api_client(self):
        """
        创建并返回LangChain OpenAI客户端
        
        Returns:
            ChatOpenAI: 配置好的LangChain OpenAI客户端
        """
        vlm_config = self.get_vlm_config()
        model_name = self.get_model_name()
        
        api_key = vlm_config.get('api_key', '')
        api_base = vlm_config.get('api_base', 'https://ark.cn-beijing.volces.com/api/v3')
        
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
            base_url=api_base
        )

# 全局配置实例
_config_instance = None

def get_pageindex_config(config_path: str = None) -> PageIndexConfig:
    """
    获取PageIndex配置实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        PageIndexConfig: 配置实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = PageIndexConfig(config_path)
    return _config_instance

def get_api_client(config_path: str = None) -> ChatOpenAI:
    """
    获取API客户端
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ChatOpenAI: API客户端
    """
    config = get_pageindex_config(config_path)
    return config.create_api_client()

