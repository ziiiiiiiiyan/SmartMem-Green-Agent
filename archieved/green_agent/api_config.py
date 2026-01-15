"""
Green Agent API 配置模块

支持多种 LLM API 服务:
1. Ollama 本地 (默认)
2. OpenAI API
3. Azure OpenAI
4. Anthropic Claude
5. 其他兼容 OpenAI 格式的 API

使用方法:
    # 使用 Ollama 本地
    agent = GreenAgent.from_ollama(model="qwen2.5-coder:7b")
    
    # 使用 OpenAI API
    agent = GreenAgent.from_openai(model="gpt-4o", api_key="sk-...")
    
    # 使用 Claude API
    agent = GreenAgent.from_anthropic(model="claude-3-5-sonnet-20241022", api_key="...")
    
    # 使用自定义 API
    agent = GreenAgent.from_custom(
        base_url="https://my-api.com/v1",
        api_key="...",
        model="my-model"
    )
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API 配置"""
    provider: str  # "ollama", "openai", "anthropic", "azure", "custom"
    base_url: str
    api_key: str
    model: str
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class APIConfigFactory:
    """API 配置工厂"""
    
    @staticmethod
    def ollama(
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434/v1"
    ) -> APIConfig:
        """Ollama 本地配置"""
        return APIConfig(
            provider="ollama",
            base_url=base_url,
            api_key="ollama",
            model=model
        )
    
    @staticmethod
    def openai(
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1"
    ) -> APIConfig:
        """OpenAI API 配置"""
        return APIConfig(
            provider="openai",
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            model=model
        )
    
    @staticmethod
    def azure_openai(
        deployment_name: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview"
    ) -> APIConfig:
        """Azure OpenAI 配置"""
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        
        return APIConfig(
            provider="azure",
            base_url=f"{endpoint}/openai/deployments/{deployment_name}",
            api_key=api_key or os.getenv("AZURE_OPENAI_KEY", ""),
            model=deployment_name,
            extra_params={"api_version": api_version}
        )
    
    @staticmethod
    def anthropic(
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None
    ) -> APIConfig:
        """Anthropic Claude 配置"""
        return APIConfig(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            model=model
        )
    
    @staticmethod
    def deepseek(
        model: str = "deepseek-chat",
        api_key: Optional[str] = None
    ) -> APIConfig:
        """DeepSeek API 配置"""
        return APIConfig(
            provider="deepseek",
            base_url="https://api.deepseek.com/v1",
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
            model=model
        )
    
    @staticmethod
    def openrouter(
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None
    ) -> APIConfig:
        """OpenRouter 配置 (多模型网关)"""
        return APIConfig(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
            model=model
        )
    
    @staticmethod
    def together(
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key: Optional[str] = None
    ) -> APIConfig:
        """Together AI 配置"""
        return APIConfig(
            provider="together",
            base_url="https://api.together.xyz/v1",
            api_key=api_key or os.getenv("TOGETHER_API_KEY", ""),
            model=model
        )
    
    @staticmethod
    def custom(
        base_url: str,
        api_key: str,
        model: str,
        **extra_params
    ) -> APIConfig:
        """自定义 API 配置"""
        return APIConfig(
            provider="custom",
            base_url=base_url,
            api_key=api_key,
            model=model,
            extra_params=extra_params
        )


# 预定义的常用配置
PRESET_CONFIGS = {
    # Ollama 本地模型
    "ollama-qwen": APIConfigFactory.ollama("qwen2.5-coder:7b"),
    "ollama-llama": APIConfigFactory.ollama("llama3.2:latest"),
    "ollama-codellama": APIConfigFactory.ollama("codellama:7b"),
    
    # OpenAI
    "gpt-4o": APIConfigFactory.openai("gpt-4o"),
    "gpt-4o-mini": APIConfigFactory.openai("gpt-4o-mini"),
    "gpt-4-turbo": APIConfigFactory.openai("gpt-4-turbo"),
    
    # Anthropic
    "claude-3.5-sonnet": APIConfigFactory.anthropic("claude-3-5-sonnet-20241022"),
    "claude-3-opus": APIConfigFactory.anthropic("claude-3-opus-20240229"),
    
    # DeepSeek
    "deepseek-chat": APIConfigFactory.deepseek("deepseek-chat"),
    "deepseek-coder": APIConfigFactory.deepseek("deepseek-coder"),
}


def get_api_config(preset_or_provider: str, **kwargs) -> APIConfig:
    """
    获取 API 配置
    
    Args:
        preset_or_provider: 预设名称 或 provider 类型
        **kwargs: 额外参数
    
    Examples:
        # 使用预设
        config = get_api_config("gpt-4o")
        config = get_api_config("ollama-qwen")
        
        # 使用 provider + 参数
        config = get_api_config("openai", model="gpt-4o-mini", api_key="sk-...")
        config = get_api_config("ollama", model="llama3.2")
    """
    # 检查是否是预设
    if preset_or_provider in PRESET_CONFIGS:
        config = PRESET_CONFIGS[preset_or_provider]
        # 允许覆盖 API key
        if "api_key" in kwargs:
            config.api_key = kwargs["api_key"]
        return config
    
    # 否则作为 provider 处理
    provider = preset_or_provider.lower()
    
    if provider == "ollama":
        return APIConfigFactory.ollama(**kwargs)
    elif provider == "openai":
        return APIConfigFactory.openai(**kwargs)
    elif provider == "anthropic" or provider == "claude":
        return APIConfigFactory.anthropic(**kwargs)
    elif provider == "azure":
        return APIConfigFactory.azure_openai(**kwargs)
    elif provider == "deepseek":
        return APIConfigFactory.deepseek(**kwargs)
    elif provider == "openrouter":
        return APIConfigFactory.openrouter(**kwargs)
    elif provider == "together":
        return APIConfigFactory.together(**kwargs)
    elif provider == "custom":
        return APIConfigFactory.custom(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {preset_or_provider}")


def list_available_configs() -> Dict[str, str]:
    """列出所有可用的预设配置"""
    return {
        name: f"{config.provider} - {config.model}"
        for name, config in PRESET_CONFIGS.items()
    }


# ============== 测试 ==============

if __name__ == "__main__":
    print("Available API presets:")
    for name, desc in list_available_configs().items():
        print(f"  {name}: {desc}")
    
    print("\nExample usage:")
    print("  from api_config import get_api_config")
    print('  config = get_api_config("gpt-4o", api_key="sk-...")')
    print('  config = get_api_config("ollama", model="qwen2.5-coder:7b")')
