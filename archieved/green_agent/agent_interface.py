"""
Agent 黑盒接口定义

设计原则：
1. Agent 是黑盒，Eval 只通过文本 I/O 交互
2. Eval 提供环境和工具函数，Agent 自行决定如何使用
3. 支持任意 Agent 实现（Purple Agent、LLM API、自定义等）

使用方式：
    # 方式1: 直接继承实现
    class MyAgent(AgentInterface):
        def chat(self, message: str) -> str:
            return my_llm_call(message)
    
    # 方式2: 使用预置适配器
    agent = OpenAIAgent(model="gpt-4o", api_key="...")
    agent = OllamaAgent(model="qwen2.5-coder:7b")
    agent = PurpleAgentAdapter(purple_agent_instance)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import json
import re


# ============== 抽象接口 ==============

class AgentInterface(ABC):
    """
    Agent 黑盒接口
    
    Eval 框架通过此接口与任意 Agent 交互：
    - chat(): 发送指令，获取响应
    - reset(): 重置 Agent 状态
    - get_tool_calls(): 可选，解析响应中的工具调用
    """
    
    @abstractmethod
    def chat(self, message: str) -> str:
        """
        与 Agent 对话
        
        Args:
            message: 用户指令（自然语言）
        
        Returns:
            Agent 的文本响应
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置 Agent 状态（清空历史、记忆等）"""
        pass
    
    def get_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        从响应中解析工具调用（可选实现）
        
        默认实现：尝试从 JSON 格式解析
        Agent 可以覆盖此方法实现自定义解析
        
        Returns:
            工具调用列表，格式: [{"action": "update", "key": "...", "value": ...}]
        """
        return self._default_parse_tool_calls(response)
    
    def _default_parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """默认的工具调用解析"""
        tool_calls = []
        
        # 尝试解析 JSON 格式
        json_pattern = r'\{[^{}]*"action"[^{}]*\}'
        matches = re.findall(json_pattern, response)
        
        for match in matches:
            try:
                call = json.loads(match)
                if 'action' in call:
                    tool_calls.append(call)
            except json.JSONDecodeError:
                continue
        
        # 尝试解析函数调用格式 (如 manage_ac_temperature(25))
        func_pattern = r'manage_(\w+)\(([^)]+)\)'
        for match in re.finditer(func_pattern, response):
            func_name = match.group(1)
            args_str = match.group(2)
            
            # 简单解析参数
            try:
                # 尝试作为 JSON 值解析
                value = json.loads(args_str)
            except:
                value = args_str.strip('"\'')
            
            tool_calls.append({
                "action": "update",
                "key": func_name,
                "value": value
            })
        
        return tool_calls
    
    @property
    def name(self) -> str:
        """Agent 名称（用于报告）"""
        return self.__class__.__name__


@dataclass
class AgentResponse:
    """Agent 响应的结构化表示"""
    raw_text: str
    tool_calls: List[Dict[str, Any]]
    thinking: Optional[str] = None
    error: Optional[str] = None


# ============== OpenAI API 适配器 ==============

class OpenAIAgent(AgentInterface):
    """
    OpenAI API Agent 适配器
    
    支持任何兼容 OpenAI API 格式的服务：
    - OpenAI (gpt-4o, gpt-4o-mini, etc.)
    - Azure OpenAI
    - 本地 Ollama (http://localhost:11434/v1)
    - 其他兼容服务
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools_schema: Optional[List[dict]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024
    ):
        """
        初始化 OpenAI Agent
        
        Args:
            model: 模型名称
            api_key: API Key (可从环境变量 OPENAI_API_KEY 获取)
            base_url: API 地址 (可从环境变量 OPENAI_API_BASE 获取)
            system_prompt: 系统提示词
            tools_schema: OpenAI 格式的工具定义 (可选)
            temperature: 生成温度
            max_tokens: 最大 token 数
        """
        import os
        from openai import OpenAI
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools_schema = tools_schema
        
        # 默认系统提示词
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # 初始化客户端
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_API_BASE")
        )
        
        # 对话历史
        self.history: List[dict] = []
        self._init_history()
    
    def _default_system_prompt(self) -> str:
        return """You are a smart home assistant. You can control various devices in the house.

Available devices and their valid values:
- living_room_light: "on" or "off"
- living_room_color: "warm" or "cool"
- bedroom_light: "on" or "off"
- bedroom_color: "warm" or "cool"
- ac_power: "on" or "off"
- ac_temperature: integer from 16 to 30
- fan_speed: "low", "medium", "high", or "off"
- music_volume: integer from 0 to 100
- front_door_lock: "locked" or "unlocked"
- kitchen_light: "on" or "off"

When you need to control a device, respond with a JSON action:
{"action": "update", "key": "<device_name>", "value": <new_value>}

You can include multiple actions in your response.
Always explain what you're doing in natural language as well."""
    
    def _init_history(self):
        """初始化对话历史"""
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def chat(self, message: str) -> str:
        """发送消息并获取响应"""
        self.history.append({"role": "user", "content": message})
        
        try:
            kwargs = {
                "model": self.model,
                "messages": self.history,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            if self.tools_schema:
                kwargs["tools"] = self.tools_schema
            
            response = self.client.chat.completions.create(**kwargs)
            
            assistant_message = response.choices[0].message
            content = assistant_message.content or ""
            
            # 处理工具调用
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    tool_results.append({
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    })
                content += f"\n[Tool Calls: {json.dumps(tool_results)}]"
            
            self.history.append({"role": "assistant", "content": content})
            return content
            
        except Exception as e:
            error_msg = f"[Error: {str(e)}]"
            self.history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def reset(self):
        """重置对话历史"""
        self._init_history()
    
    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"


# ============== Anthropic API 适配器 ==============

class AnthropicAgent(AgentInterface):
    """
    Anthropic (Claude) API Agent 适配器
    """
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024
    ):
        import os
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.history: List[dict] = []
    
    def _default_system_prompt(self) -> str:
        return """You are a smart home assistant. You can control various devices.

Available devices:
- living_room_light: "on"/"off", living_room_color: "warm"/"cool"
- bedroom_light: "on"/"off", bedroom_color: "warm"/"cool"
- ac_power: "on"/"off", ac_temperature: 16-30
- fan_speed: "low"/"medium"/"high"/"off"
- music_volume: 0-100
- front_door_lock: "locked"/"unlocked"
- kitchen_light: "on"/"off"

When controlling a device, include JSON: {"action": "update", "key": "<device>", "value": <value>}"""
    
    def chat(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=self.history
            )
            
            content = response.content[0].text
            self.history.append({"role": "assistant", "content": content})
            return content
            
        except Exception as e:
            error_msg = f"[Error: {str(e)}]"
            return error_msg
    
    def reset(self):
        self.history = []
    
    @property
    def name(self) -> str:
        return f"Claude ({self.model})"


# ============== Ollama 本地适配器 ==============

class OllamaAgent(OpenAIAgent):
    """
    Ollama 本地模型适配器
    
    继承自 OpenAIAgent，因为 Ollama 兼容 OpenAI API 格式
    """
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434/v1",
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024
    ):
        super().__init__(
            model=model,
            api_key="ollama",  # Ollama 不需要真正的 key
            base_url=base_url,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    @property
    def name(self) -> str:
        return f"Ollama ({self.model})"


# ============== Purple Agent 适配器 ==============

class PurpleAgentAdapter(AgentInterface):
    """
    Purple Agent 黑盒适配器
    
    将现有的 Purple Agent 包装为标准接口
    不直接访问内部状态，只通过 step() 方法交互
    """
    
    def __init__(self, env=None, model: str = "gpt-4o", **kwargs):
        """
        初始化 Purple Agent 适配器
        
        Args:
            env: SmartHomeEnv 实例（可选，用于工具绑定）
            model: 使用的模型
            **kwargs: 传递给 Purple Agent 的其他参数
        """
        self.env = env
        self.model = model
        self.kwargs = kwargs
        self.agent = None
        self._init_agent()
    
    def _init_agent(self):
        """初始化 Purple Agent"""
        try:
            from purple_agent.purple_agent import Agent, functions_map, tools_schema
            from purple_agent.prompts import SYSTEM_PROMPT
            
            self.agent = Agent(
                function_map=functions_map,
                system_prompt=SYSTEM_PROMPT,
                backbone_model=self.model,
                tools_schema=tools_schema,
                **self.kwargs
            )
        except ImportError as e:
            raise ImportError(f"无法导入 Purple Agent: {e}")
    
    def chat(self, message: str) -> str:
        """通过 step() 方法与 Purple Agent 交互"""
        if not self.agent:
            return "[Error: Purple Agent not initialized]"
        
        try:
            # 调用 Purple Agent 的 step 方法
            success = self.agent.step(
                user_input=message,
                generation_args={
                    "temperature": 0.2,
                    "max_completion_tokens": 1024,
                    "top_p": 0.95
                },
                limit_iters=5
            )
            
            # 获取最后的响应
            if self.agent.memory and len(self.agent.memory.items) > 0:
                last_item = self.agent.memory.items[-1]
                return last_item.content or "[No response]"
            
            return "[No response from agent]"
            
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def reset(self):
        """重新初始化 Agent"""
        self._init_agent()
    
    def get_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """从 Purple Agent 的记忆中提取工具调用"""
        tool_calls = []
        
        if self.agent and self.agent.memory:
            # 获取最后一条消息的工具链
            for item in reversed(self.agent.memory.items):
                if hasattr(item, 'tool_chain') and item.tool_chain:
                    for interaction in item.tool_chain:
                        # 解析工具调用
                        tool_name = interaction.tool_name
                        tool_input = interaction.tool_input
                        
                        # 转换为标准格式
                        if tool_name.startswith("manage_"):
                            device = tool_name.replace("manage_", "")
                            if tool_input:
                                # 获取第一个参数值
                                value = list(tool_input.values())[0] if tool_input else None
                                tool_calls.append({
                                    "action": "update",
                                    "key": device,
                                    "value": value
                                })
                    break
        
        # 同时使用默认解析作为后备
        tool_calls.extend(self._default_parse_tool_calls(response))
        
        return tool_calls
    
    @property
    def name(self) -> str:
        return f"Purple Agent ({self.model})"


# ============== 模拟测试 Agent ==============

class MockAgent(AgentInterface):
    """
    模拟 Agent，用于测试 Eval 框架
    
    可配置错误率来模拟不完美的 Agent
    """
    
    def __init__(self, error_rate: float = 0.0, expected_actions: Optional[List[dict]] = None):
        """
        Args:
            error_rate: 错误率 (0-1)，随机跳过动作的概率
            expected_actions: 预期动作列表（用于测试时注入）
        """
        import random
        self.error_rate = error_rate
        self.expected_actions = expected_actions or []
        self.random = random
    
    def set_expected_actions(self, actions: List[dict]):
        """设置下一轮的预期动作"""
        self.expected_actions = actions
    
    def chat(self, message: str) -> str:
        """根据预期动作生成响应"""
        response_parts = [f"I received: {message[:50]}..."]
        executed_actions = []
        
        for action in self.expected_actions:
            # 随机引入错误
            if self.random.random() < self.error_rate:
                continue  # 跳过这个动作
            
            executed_actions.append(action)
            response_parts.append(json.dumps(action))
        
        return "\n".join(response_parts)
    
    def reset(self):
        self.expected_actions = []
    
    @property
    def name(self) -> str:
        return f"Mock Agent (error_rate={self.error_rate})"


# ============== A2A 协议适配器 ==============

class A2AAgentAdapter(AgentInterface):
    """
    A2A (Agent-to-Agent) 协议适配器
    
    支持通过 Google A2A 协议连接远程 Agent
    
    A2A 协议参考: https://github.com/google/a2a-spec
    
    Example:
        agent = A2AAgentAdapter(
            agent_url="https://my-agent.example.com",
            agent_id="smart-home-agent"
        )
        response = agent.chat("打开客厅灯")
    """
    
    def __init__(
        self,
        agent_url: str,
        agent_id: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60
    ):
        """
        Args:
            agent_url: A2A Agent 的基础 URL
            agent_id: 目标 Agent ID（可选）
            api_key: 认证 API Key（可选）
            timeout: 请求超时时间（秒）
        """
        self.agent_url = agent_url.rstrip('/')
        self.agent_id = agent_id
        self.api_key = api_key
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.task_id: Optional[str] = None
    
    def chat(self, message: str) -> str:
        """
        通过 A2A 协议发送消息
        
        使用 tasks/send 端点发送任务，轮询直到完成
        """
        import requests
        import time
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 构建 A2A Task
        task_request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"text": message}]
                }
            },
            "id": str(int(time.time() * 1000))
        }
        
        # 如果有 session_id，继续之前的会话
        if self.session_id:
            task_request["params"]["sessionId"] = self.session_id
        
        try:
            # 发送任务
            response = requests.post(
                f"{self.agent_url}/a2a",
                json=task_request,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                return f"A2A Error: {result['error']}"
            
            task_result = result.get("result", {})
            
            # 保存 session 和 task ID
            self.session_id = task_result.get("sessionId")
            self.task_id = task_result.get("id")
            
            # 检查任务状态
            status = task_result.get("status", {}).get("state", "unknown")
            
            if status == "completed":
                # 提取响应文本
                artifacts = task_result.get("artifacts", [])
                response_texts = []
                for artifact in artifacts:
                    for part in artifact.get("parts", []):
                        if "text" in part:
                            response_texts.append(part["text"])
                return "\n".join(response_texts) if response_texts else ""
            
            elif status in ("working", "submitted"):
                # 需要轮询等待完成
                return self._poll_task_result()
            
            else:
                return f"Task status: {status}"
                
        except requests.exceptions.RequestException as e:
            return f"A2A Request Error: {e}"
    
    def _poll_task_result(self, max_attempts: int = 30, interval: float = 1.0) -> str:
        """轮询任务结果"""
        import requests
        import time
        
        if not self.task_id:
            return "No task to poll"
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        for _ in range(max_attempts):
            time.sleep(interval)
            
            poll_request = {
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": self.task_id},
                "id": str(int(time.time() * 1000))
            }
            
            try:
                response = requests.post(
                    f"{self.agent_url}/a2a",
                    json=poll_request,
                    headers=headers,
                    timeout=self.timeout
                )
                result = response.json()
                task = result.get("result", {})
                status = task.get("status", {}).get("state", "unknown")
                
                if status == "completed":
                    artifacts = task.get("artifacts", [])
                    response_texts = []
                    for artifact in artifacts:
                        for part in artifact.get("parts", []):
                            if "text" in part:
                                response_texts.append(part["text"])
                    return "\n".join(response_texts) if response_texts else ""
                
                elif status in ("failed", "canceled"):
                    return f"Task {status}: {task.get('status', {}).get('message', '')}"
                    
            except requests.exceptions.RequestException:
                continue
        
        return "Task polling timeout"
    
    def reset(self):
        """重置会话"""
        self.session_id = None
        self.task_id = None
    
    @property
    def name(self) -> str:
        return f"A2A Agent ({self.agent_url})"


# ============== MCP 协议适配器 ==============

class MCPAgentAdapter(AgentInterface):
    """
    MCP (Model Context Protocol) Agent 适配器
    
    支持通过 MCP 协议连接 Agent 服务
    
    MCP 协议参考: https://modelcontextprotocol.io/
    
    Example:
        agent = MCPAgentAdapter(
            server_url="http://localhost:3000",
            transport="http"  # or "stdio", "websocket"
        )
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        transport: str = "http",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        timeout: int = 60
    ):
        """
        Args:
            server_url: MCP 服务器 URL（HTTP 传输）
            transport: 传输方式 ("http", "stdio", "websocket")
            command: stdio 传输时的命令
            args: stdio 传输时的参数
            timeout: 请求超时时间
        """
        self.server_url = server_url
        self.transport = transport
        self.command = command
        self.args = args or []
        self.timeout = timeout
        self.conversation_history: List[Dict[str, str]] = []
        self._process = None
    
    def chat(self, message: str) -> str:
        """发送消息到 MCP Agent"""
        
        if self.transport == "http":
            return self._chat_http(message)
        elif self.transport == "stdio":
            return self._chat_stdio(message)
        else:
            return f"Unsupported transport: {self.transport}"
    
    def _chat_http(self, message: str) -> str:
        """通过 HTTP 传输发送消息"""
        import requests
        
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            # MCP 使用 JSON-RPC 2.0
            request = {
                "jsonrpc": "2.0",
                "method": "sampling/createMessage",
                "params": {
                    "messages": self.conversation_history,
                    "maxTokens": 4096
                },
                "id": len(self.conversation_history)
            }
            
            response = requests.post(
                f"{self.server_url}/mcp",
                json=request,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                return f"MCP Error: {result['error']}"
            
            # 提取响应
            content = result.get("result", {}).get("content", {})
            if isinstance(content, dict) and "text" in content:
                assistant_message = content["text"]
            elif isinstance(content, str):
                assistant_message = content
            else:
                assistant_message = str(content)
            
            self.conversation_history.append({
                "role": "assistant", 
                "content": assistant_message
            })
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"MCP HTTP Error: {e}"
    
    def _chat_stdio(self, message: str) -> str:
        """通过 stdio 传输发送消息"""
        import subprocess
        import json as json_module
        
        if not self.command:
            return "No command specified for stdio transport"
        
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            # 创建子进程
            request = {
                "jsonrpc": "2.0",
                "method": "sampling/createMessage",
                "params": {
                    "messages": self.conversation_history,
                    "maxTokens": 4096
                },
                "id": len(self.conversation_history)
            }
            
            process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(
                input=json_module.dumps(request) + "\n",
                timeout=self.timeout
            )
            
            if stderr:
                return f"MCP stdio Error: {stderr}"
            
            result = json_module.loads(stdout)
            content = result.get("result", {}).get("content", {})
            assistant_message = content.get("text", str(content))
            
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            return f"MCP stdio Error: {e}"
    
    def reset(self):
        """重置会话"""
        self.conversation_history = []
        if self._process:
            self._process.terminate()
            self._process = None
    
    @property
    def name(self) -> str:
        return f"MCP Agent ({self.transport})"


# ============== HTTP Agent 适配器 ==============

class HTTPAgentAdapter(AgentInterface):
    """
    通用 HTTP Agent 适配器
    
    支持任意提供 HTTP API 的 Agent 服务
    
    Example:
        agent = HTTPAgentAdapter(
            url="https://api.example.com/agent/chat",
            method="POST",
            message_field="message",
            response_field="response"
        )
    """
    
    def __init__(
        self,
        url: str,
        method: str = "POST",
        message_field: str = "message",
        response_field: str = "response",
        headers: Optional[Dict[str, str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer ",
        timeout: int = 60
    ):
        """
        Args:
            url: Agent API URL
            method: HTTP 方法 (POST/GET)
            message_field: 请求中消息字段名
            response_field: 响应中回复字段名（支持点号路径如 "data.response"）
            headers: 额外的请求头
            extra_params: 额外的请求参数
            api_key: API 密钥
            api_key_header: API 密钥的 header 名
            api_key_prefix: API 密钥前缀
            timeout: 请求超时时间
        """
        self.url = url
        self.method = method.upper()
        self.message_field = message_field
        self.response_field = response_field
        self.headers = headers or {}
        self.extra_params = extra_params or {}
        self.timeout = timeout
        
        if api_key:
            self.headers[api_key_header] = f"{api_key_prefix}{api_key}"
        
        self.session_data: Dict[str, Any] = {}
    
    def chat(self, message: str) -> str:
        """发送 HTTP 请求"""
        import requests
        
        # 构建请求数据
        data = {self.message_field: message}
        data.update(self.extra_params)
        data.update(self.session_data)
        
        headers = {"Content-Type": "application/json"}
        headers.update(self.headers)
        
        try:
            if self.method == "POST":
                response = requests.post(
                    self.url,
                    json=data,
                    headers=headers,
                    timeout=self.timeout
                )
            else:
                response = requests.get(
                    self.url,
                    params=data,
                    headers=headers,
                    timeout=self.timeout
                )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取响应（支持嵌套路径）
            value = result
            for key in self.response_field.split('.'):
                if isinstance(value, dict):
                    value = value.get(key, "")
                else:
                    break
            
            # 保存会话数据（如果有）
            if "session_id" in result:
                self.session_data["session_id"] = result["session_id"]
            if "conversation_id" in result:
                self.session_data["conversation_id"] = result["conversation_id"]
            
            return str(value) if value else ""
            
        except requests.exceptions.RequestException as e:
            return f"HTTP Error: {e}"
    
    def reset(self):
        """重置会话"""
        self.session_data = {}
    
    @property
    def name(self) -> str:
        return f"HTTP Agent ({self.url})"


# ============== LangChain Agent 适配器 ==============

class LangChainAgentAdapter(AgentInterface):
    """
    LangChain Agent 适配器
    
    支持包装任意 LangChain Agent
    
    Example:
        from langchain.agents import create_openai_functions_agent
        
        lc_agent = create_openai_functions_agent(...)
        agent = LangChainAgentAdapter(lc_agent)
    """
    
    def __init__(self, langchain_agent: Any, agent_executor: Any = None):
        """
        Args:
            langchain_agent: LangChain Agent 实例
            agent_executor: LangChain AgentExecutor（可选）
        """
        self.langchain_agent = langchain_agent
        self.agent_executor = agent_executor
        self.chat_history: List[Any] = []
    
    def chat(self, message: str) -> str:
        """调用 LangChain Agent"""
        try:
            # 如果有 AgentExecutor，使用它
            if self.agent_executor:
                result = self.agent_executor.invoke({
                    "input": message,
                    "chat_history": self.chat_history
                })
                output = result.get("output", str(result))
                
                # 更新历史
                try:
                    from langchain_core.messages import HumanMessage, AIMessage
                    self.chat_history.append(HumanMessage(content=message))
                    self.chat_history.append(AIMessage(content=output))
                except ImportError:
                    pass
                
                return output
            
            # 直接调用 agent
            elif hasattr(self.langchain_agent, 'invoke'):
                result = self.langchain_agent.invoke(message)
                return str(result)
            elif hasattr(self.langchain_agent, 'run'):
                return self.langchain_agent.run(message)
            else:
                return str(self.langchain_agent(message))
                
        except Exception as e:
            return f"LangChain Error: {e}"
    
    def reset(self):
        """重置会话历史"""
        self.chat_history = []
    
    @property
    def name(self) -> str:
        agent_type = type(self.langchain_agent).__name__
        return f"LangChain Agent ({agent_type})"


# ============== AutoGen Agent 适配器 ==============

class AutoGenAgentAdapter(AgentInterface):
    """
    AutoGen Agent 适配器
    
    支持包装 Microsoft AutoGen Agent
    
    Example:
        from autogen import AssistantAgent
        
        autogen_agent = AssistantAgent(name="assistant", ...)
        agent = AutoGenAgentAdapter(autogen_agent)
    """
    
    def __init__(self, autogen_agent: Any, user_proxy: Any = None):
        """
        Args:
            autogen_agent: AutoGen Agent 实例
            user_proxy: UserProxyAgent 用于发起对话（可选）
        """
        self.autogen_agent = autogen_agent
        self.user_proxy = user_proxy
        self.last_message: str = ""
    
    def chat(self, message: str) -> str:
        """调用 AutoGen Agent"""
        try:
            if self.user_proxy:
                # 通过 user_proxy 发起对话
                self.user_proxy.initiate_chat(
                    self.autogen_agent,
                    message=message,
                    silent=True
                )
                
                # 获取最后的回复
                chat_history = self.user_proxy.chat_messages.get(self.autogen_agent, [])
                if chat_history:
                    last_msg = chat_history[-1]
                    return last_msg.get("content", str(last_msg))
                return ""
            
            # 直接生成回复
            elif hasattr(self.autogen_agent, 'generate_reply'):
                messages = [{"role": "user", "content": message}]
                reply = self.autogen_agent.generate_reply(messages=messages)
                return reply if isinstance(reply, str) else str(reply)
            
            else:
                return "AutoGen agent does not support direct chat"
                
        except Exception as e:
            return f"AutoGen Error: {e}"
    
    def reset(self):
        """重置 Agent 状态"""
        if self.user_proxy and hasattr(self.user_proxy, 'reset'):
            self.user_proxy.reset()
        if hasattr(self.autogen_agent, 'reset'):
            self.autogen_agent.reset()
    
    @property
    def name(self) -> str:
        agent_name = getattr(self.autogen_agent, 'name', type(self.autogen_agent).__name__)
        return f"AutoGen Agent ({agent_name})"


# ============== 工厂函数 ==============

def create_agent(
    agent_type: str,
    **kwargs
) -> AgentInterface:
    """
    创建 Agent 实例的工厂函数
    
    Args:
        agent_type: Agent 类型
            - "openai": OpenAI API
            - "anthropic" / "claude": Anthropic Claude
            - "ollama": 本地 Ollama
            - "purple": Purple Agent
            - "mock": 模拟测试 Agent
            - "a2a": A2A 协议 Agent
            - "mcp": MCP 协议 Agent
            - "http": 通用 HTTP Agent
            - "langchain": LangChain Agent
            - "autogen": AutoGen Agent
        **kwargs: Agent 特定参数
    
    Returns:
        AgentInterface 实例
    
    Examples:
        # 模型 API
        agent = create_agent("openai", model="gpt-4o", api_key="...")
        agent = create_agent("ollama", model="qwen2.5-coder:7b")
        
        # Agent 协议
        agent = create_agent("a2a", agent_url="https://agent.example.com")
        agent = create_agent("mcp", server_url="http://localhost:3000")
        agent = create_agent("http", url="https://api.example.com/chat")
        
        # Agent 框架
        agent = create_agent("langchain", langchain_agent=lc_agent)
        agent = create_agent("autogen", autogen_agent=ag_agent)
    """
    agent_type = agent_type.lower()
    
    agents = {
        # 模型 API
        "openai": OpenAIAgent,
        "anthropic": AnthropicAgent,
        "claude": AnthropicAgent,
        "ollama": OllamaAgent,
        
        # Purple Agent
        "purple": PurpleAgentAdapter,
        
        # 测试
        "mock": MockAgent,
        
        # Agent 协议
        "a2a": A2AAgentAdapter,
        "mcp": MCPAgentAdapter,
        "http": HTTPAgentAdapter,
        
        # Agent 框架
        "langchain": LangChainAgentAdapter,
        "autogen": AutoGenAgentAdapter,
    }
    
    if agent_type not in agents:
        available = ", ".join(agents.keys())
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {available}")
    
    return agents[agent_type](**kwargs)


# ============== 测试 ==============

if __name__ == "__main__":
    # 测试 Mock Agent
    print("Testing Mock Agent...")
    agent = MockAgent(error_rate=0.3)
    agent.set_expected_actions([
        {"action": "update", "key": "living_room_light", "value": "on"},
        {"action": "update", "key": "ac_temperature", "value": 24}
    ])
    
    response = agent.chat("Turn on the light and set AC to 24")
    print(f"Response: {response}")
    print(f"Parsed tool calls: {agent.get_tool_calls(response)}")
    
    # 测试 Ollama Agent (如果可用)
    try:
        print("\nTesting Ollama Agent...")
        ollama = OllamaAgent(model="qwen2.5-coder:7b")
        response = ollama.chat("Turn on the living room light")
        print(f"Response: {response[:200]}...")
    except Exception as e:
        print(f"Ollama not available: {e}")
