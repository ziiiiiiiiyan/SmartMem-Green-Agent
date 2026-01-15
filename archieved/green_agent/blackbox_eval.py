"""
Eval 框架 - 黑盒评估器

设计原则：
1. Agent 是黑盒，只通过文本 I/O 交互
2. Eval 提供环境和工具函数
3. Agent 自行决定如何注册和使用工具
4. 评分基于最终状态和动作序列

接口对齐：
- Eval 端: 起环境 + 提供工具函数 + 解析响应 + 评分
- Agent 端: 接收指令 -> 返回文本响应 (可包含工具调用)
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.environment import SmartHomeEnv
from app.evaluator import TurnEvaluator
from green_agent.agent_interface import AgentInterface, create_agent


# ============== 工具函数定义 ==============

TOOL_FUNCTIONS = {
    "manage_living_room_light": {
        "description": "Control the living room light",
        "parameters": {"state": {"type": "string", "enum": ["on", "off"]}}
    },
    "manage_living_room_color": {
        "description": "Set the living room light color",
        "parameters": {"color": {"type": "string", "enum": ["warm", "cool"]}}
    },
    "manage_bedroom_light": {
        "description": "Control the bedroom light",
        "parameters": {"state": {"type": "string", "enum": ["on", "off"]}}
    },
    "manage_bedroom_color": {
        "description": "Set the bedroom light color",
        "parameters": {"color": {"type": "string", "enum": ["warm", "cool"]}}
    },
    "manage_ac_power": {
        "description": "Control the air conditioner power",
        "parameters": {"state": {"type": "string", "enum": ["on", "off"]}}
    },
    "manage_ac_temperature": {
        "description": "Set the AC temperature (16-30)",
        "parameters": {"temperature": {"type": "integer", "minimum": 16, "maximum": 30}}
    },
    "manage_fan_speed": {
        "description": "Set the fan speed",
        "parameters": {"speed": {"type": "string", "enum": ["low", "medium", "high", "off"]}}
    },
    "manage_music_volume": {
        "description": "Set the music volume (0-100)",
        "parameters": {"volume": {"type": "integer", "minimum": 0, "maximum": 100}}
    },
    "manage_front_door_lock": {
        "description": "Control the front door lock",
        "parameters": {"state": {"type": "string", "enum": ["locked", "unlocked"]}}
    },
    "manage_kitchen_light": {
        "description": "Control the kitchen light",
        "parameters": {"state": {"type": "string", "enum": ["on", "off"]}}
    },
    "read_all_states": {
        "description": "Read the current state of all devices",
        "parameters": {}
    }
}

# 工具名到设备键的映射
TOOL_TO_DEVICE = {
    "manage_living_room_light": "living_room_light",
    "manage_living_room_color": "living_room_color",
    "manage_bedroom_light": "bedroom_light",
    "manage_bedroom_color": "bedroom_color",
    "manage_ac_power": "ac_power",
    "manage_ac_temperature": "ac_temperature",
    "manage_fan_speed": "fan_speed",
    "manage_music_volume": "music_volume",
    "manage_front_door_lock": "front_door_lock",
    "manage_kitchen_light": "kitchen_light"
}


# ============== 响应解析器 ==============

class ResponseParser:
    """
    从 Agent 响应中解析工具调用
    
    支持多种格式：
    1. JSON 格式: {"action": "update", "key": "...", "value": ...}
    2. 函数调用格式: manage_ac_temperature(25)
    3. OpenAI Tool Calls 格式
    """
    
    @staticmethod
    def parse(response: str) -> List[Dict[str, Any]]:
        """解析响应，返回标准化的动作列表"""
        actions = []
        
        # 1. 解析 JSON 动作格式
        actions.extend(ResponseParser._parse_json_actions(response))
        
        # 2. 解析函数调用格式
        actions.extend(ResponseParser._parse_function_calls(response))
        
        # 3. 解析 OpenAI Tool Calls 格式
        actions.extend(ResponseParser._parse_openai_tool_calls(response))
        
        # 去重
        seen = set()
        unique_actions = []
        for action in actions:
            key = (action.get('key'), str(action.get('value')))
            if key not in seen:
                seen.add(key)
                unique_actions.append(action)
        
        return unique_actions
    
    @staticmethod
    def _parse_json_actions(response: str) -> List[Dict[str, Any]]:
        """解析 JSON 格式的动作"""
        actions = []
        
        # 匹配 {"action": "update", ...} 格式
        json_pattern = r'\{[^{}]*"action"\s*:\s*"update"[^{}]*\}'
        matches = re.findall(json_pattern, response, re.IGNORECASE)
        
        for match in matches:
            try:
                action = json.loads(match)
                if 'key' in action and 'value' in action:
                    actions.append({
                        "action": "update",
                        "key": action['key'],
                        "value": action['value']
                    })
            except json.JSONDecodeError:
                continue
        
        return actions
    
    @staticmethod
    def _parse_function_calls(response: str) -> List[Dict[str, Any]]:
        """解析函数调用格式"""
        actions = []
        
        # 匹配 manage_xxx(value) 或 manage_xxx(key=value) 格式
        func_pattern = r'manage_(\w+)\s*\(\s*([^)]+)\s*\)'
        
        for match in re.finditer(func_pattern, response):
            device = match.group(1)
            args_str = match.group(2).strip()
            
            # 处理设备名映射
            full_device = None
            for tool_name, dev_key in TOOL_TO_DEVICE.items():
                if tool_name == f"manage_{device}":
                    full_device = dev_key
                    break
            
            if not full_device:
                # 尝试直接使用
                full_device = device
            
            # 解析参数值
            value = ResponseParser._parse_arg_value(args_str)
            
            if value is not None:
                actions.append({
                    "action": "update",
                    "key": full_device,
                    "value": value
                })
        
        return actions
    
    @staticmethod
    def _parse_openai_tool_calls(response: str) -> List[Dict[str, Any]]:
        """解析 OpenAI Tool Calls 格式"""
        actions = []
        
        # 匹配 [Tool Calls: ...] 格式
        tool_calls_pattern = r'\[Tool Calls:\s*(\[.*?\])\]'
        match = re.search(tool_calls_pattern, response, re.DOTALL)
        
        if match:
            try:
                tool_calls = json.loads(match.group(1))
                for call in tool_calls:
                    func_name = call.get('name', '')
                    args_str = call.get('arguments', '{}')
                    
                    try:
                        args = json.loads(args_str)
                    except:
                        args = {}
                    
                    # 转换为标准格式
                    if func_name in TOOL_TO_DEVICE:
                        device = TOOL_TO_DEVICE[func_name]
                        # 获取第一个参数值
                        value = list(args.values())[0] if args else None
                        if value is not None:
                            actions.append({
                                "action": "update",
                                "key": device,
                                "value": value
                            })
            except json.JSONDecodeError:
                pass
        
        return actions
    
    @staticmethod
    def _parse_arg_value(args_str: str) -> Any:
        """解析参数值"""
        args_str = args_str.strip()
        
        # 处理 key=value 格式
        if '=' in args_str:
            args_str = args_str.split('=')[-1].strip()
        
        # 去除引号
        if (args_str.startswith('"') and args_str.endswith('"')) or \
           (args_str.startswith("'") and args_str.endswith("'")):
            return args_str[1:-1]
        
        # 尝试解析为数字
        try:
            if '.' in args_str:
                return float(args_str)
            return int(args_str)
        except ValueError:
            return args_str


# ============== 黑盒评估器 ==============

@dataclass
class EvalResult:
    """单轮评估结果"""
    turn_id: int
    instruction: str
    agent_response: str
    parsed_actions: List[Dict[str, Any]]
    expected_actions: List[Dict[str, Any]]
    expected_state: Dict[str, Any]
    actual_state: Dict[str, Any]
    score: float
    max_score: float
    passed: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class TestCaseResult:
    """测试用例评估结果"""
    scenario_id: str
    dimension: str
    difficulty: str
    total_score: float
    max_score: float
    passed: bool
    turn_results: List[EvalResult] = field(default_factory=list)


class BlackBoxEvaluator:
    """
    黑盒评估器
    
    与 Agent 只通过文本 I/O 交互，不访问 Agent 内部状态
    """
    
    def __init__(self, env: Optional[SmartHomeEnv] = None):
        """
        初始化评估器
        
        Args:
            env: SmartHome 环境实例（可选，会自动创建）
        """
        self.env = env or SmartHomeEnv()
        self.parser = ResponseParser()
    
    def evaluate_turn(
        self,
        agent: AgentInterface,
        instruction: str,
        expected_actions: List[Dict[str, Any]],
        expected_state: Dict[str, Any]
    ) -> EvalResult:
        """
        评估单轮对话
        
        Args:
            agent: Agent 实例
            instruction: 用户指令
            expected_actions: 预期动作列表
            expected_state: 预期最终状态
        
        Returns:
            EvalResult 评估结果
        """
        # 1. 发送指令给 Agent
        # If agent supports injecting expected actions (mock), set them to make smoke tests deterministic
        if hasattr(agent, 'set_expected_actions'):
            try:
                agent.set_expected_actions(expected_actions)
            except Exception:
                pass
        response = agent.chat(instruction)
        
        # 2. 解析 Agent 响应中的动作
        parsed_actions = self.parser.parse(response)
        
        # 也尝试使用 Agent 自己的解析
        agent_parsed = agent.get_tool_calls(response)
        for action in agent_parsed:
            if action not in parsed_actions:
                parsed_actions.append(action)
        
        # 3. 执行解析出的动作
        for action in parsed_actions:
            if action.get('action') == 'update':
                key = action.get('key')
                value = action.get('value')
                if key and value is not None:
                    self.env.update_state(key, value)
                    self.env.record_action(action)
        
        # 4. 获取实际状态
        actual_state = self.env.get_state()['state']
        
        # 5. 评分
        evaluator = TurnEvaluator(expected_actions, expected_state)
        actual_actions = self.env.get_action_history()
        result = evaluator.evaluate(actual_actions, actual_state)
        
        return EvalResult(
            turn_id=0,
            instruction=instruction,
            agent_response=response,
            parsed_actions=parsed_actions,
            expected_actions=expected_actions,
            expected_state=expected_state,
            actual_state=actual_state,
            score=result['score'],
            max_score=1.0,
            passed=result['score'] == 1.0,
            errors=result.get('details', {}).get('errors', [])
        )
    
    def evaluate_test_case(
        self,
        agent: AgentInterface,
        test_case: Dict[str, Any]
    ) -> TestCaseResult:
        """
        评估完整测试用例
        
        Args:
            agent: Agent 实例
            test_case: 测试用例数据
        
        Returns:
            TestCaseResult 评估结果
        """
        # 重置
        agent.reset()
        initial_state = test_case.get('initial_state', {})
        self.env.reset(initial_state=initial_state)
        
        turn_results = []
        total_score = 0.0
        max_score = 0.0
        
        for turn in test_case.get('turns', []):
            # 重置 turn 历史
            self.env.reset_turn_history()
            
            turn_id = turn.get('turn_id', len(turn_results) + 1)
            instruction = turn.get('gm_instruction', '')
            expected_actions = turn.get('expected_agent_action', [])
            expected_state = turn.get('expected_final_state', {})
            
            result = self.evaluate_turn(
                agent=agent,
                instruction=instruction,
                expected_actions=expected_actions,
                expected_state=expected_state
            )
            result.turn_id = turn_id
            
            turn_results.append(result)
            total_score += result.score
            max_score += result.max_score
        
        return TestCaseResult(
            scenario_id=test_case.get('scenario_id', 'unknown'),
            dimension=test_case.get('dimension', 'unknown'),
            difficulty=test_case.get('difficulty', 'unknown'),
            total_score=total_score,
            max_score=max_score,
            passed=total_score == max_score,
            turn_results=turn_results
        )
    
    def evaluate_batch(
        self,
        agent: AgentInterface,
        test_cases: List[Dict[str, Any]],
        verbose: bool = False
    ) -> List[TestCaseResult]:
        """
        批量评估测试用例
        
        Args:
            agent: Agent 实例
            test_cases: 测试用例列表
            verbose: 是否打印详细信息
        
        Returns:
            TestCaseResult 列表
        """
        results = []
        
        for i, case in enumerate(test_cases, 1):
            if verbose:
                print(f"  [{i}/{len(test_cases)}] {case.get('scenario_id', 'unknown')}...", end=" ")
            
            result = self.evaluate_test_case(agent, case)
            results.append(result)
            
            if verbose:
                status = "✓" if result.passed else "✗"
                print(f"{status} ({result.total_score}/{result.max_score})")
        
        return results
    
    def get_tools_schema_openai(self) -> List[dict]:
        """
        获取 OpenAI 格式的工具定义
        
        Agent 可以使用此定义来注册工具
        """
        tools = []
        
        for func_name, func_info in TOOL_FUNCTIONS.items():
            tool = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": func_info["parameters"],
                        "required": list(func_info["parameters"].keys())
                    }
                }
            }
            tools.append(tool)
        
        return tools
    
    def get_tools_description_text(self) -> str:
        """
        获取工具的文本描述
        
        用于在 prompt 中告诉 Agent 可用的工具
        """
        lines = ["Available tools:\n"]
        
        for func_name, func_info in TOOL_FUNCTIONS.items():
            params = func_info["parameters"]
            params_str = ", ".join(
                f"{k}: {v.get('type', 'any')}" 
                for k, v in params.items()
            )
            lines.append(f"- {func_name}({params_str}): {func_info['description']}")
        
        return "\n".join(lines)


# ============== 便捷函数 ==============

def quick_evaluate(
    agent_type: str,
    test_cases: List[Dict[str, Any]],
    verbose: bool = True,
    **agent_kwargs
) -> Tuple[List[TestCaseResult], Dict[str, Any]]:
    """
    快速评估函数
    
    Args:
        agent_type: Agent 类型 ("openai", "ollama", "purple", "mock", etc.)
        test_cases: 测试用例列表
        verbose: 是否打印详细信息
        **agent_kwargs: Agent 参数
    
    Returns:
        (results, summary) 元组
    
    Example:
        results, summary = quick_evaluate(
            "openai", 
            test_cases,
            model="gpt-4o",
            api_key="sk-..."
        )
    """
    # 创建 Agent
    agent = create_agent(agent_type, **agent_kwargs)
    
    # 创建评估器
    evaluator = BlackBoxEvaluator()
    
    if verbose:
        print(f"🔍 Evaluating {agent.name} on {len(test_cases)} test cases...")
    
    # 评估
    results = evaluator.evaluate_batch(agent, test_cases, verbose=verbose)
    
    # 统计
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    total_score = sum(r.total_score for r in results)
    max_score = sum(r.max_score for r in results)
    
    summary = {
        "agent_name": agent.name,
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / max(1, total),
        "total_score": total_score,
        "max_score": max_score,
        "score_rate": total_score / max(1, max_score)
    }
    
    if verbose:
        print(f"\n📊 Summary:")
        print(f"   Pass Rate: {summary['pass_rate']*100:.1f}% ({passed}/{total})")
        print(f"   Score Rate: {summary['score_rate']*100:.1f}%")
    
    return results, summary


# ============== 测试 ==============

if __name__ == "__main__":
    # 测试解析器
    print("Testing ResponseParser...")
    
    test_responses = [
        '{"action": "update", "key": "living_room_light", "value": "on"}',
        'I will turn on the light. manage_living_room_light("on")',
        'Setting temperature to 24. {"action": "update", "key": "ac_temperature", "value": 24}',
        '[Tool Calls: [{"name": "manage_ac_temperature", "arguments": "{\"temperature\": 25}"}]]'
    ]
    
    for resp in test_responses:
        actions = ResponseParser.parse(resp)
        print(f"  Input: {resp[:50]}...")
        print(f"  Parsed: {actions}\n")
    
    # 测试评估器
    print("\nTesting BlackBoxEvaluator with MockAgent...")
    from green_agent.agent_interface import MockAgent
    
    agent = MockAgent(error_rate=0.0)
    evaluator = BlackBoxEvaluator()
    
    # 简单测试用例
    test_case = {
        "scenario_id": "test_001",
        "dimension": "precision",
        "difficulty": "easy",
        "initial_state": {"living_room_light": "off"},
        "turns": [
            {
                "turn_id": 1,
                "gm_instruction": "Turn on the living room light",
                "expected_agent_action": [
                    {"action": "update", "key": "living_room_light", "value": "on"}
                ],
                "expected_final_state": {"living_room_light": "on"}
            }
        ]
    }
    
    # 设置 Mock Agent 的预期动作
    agent.set_expected_actions([
        {"action": "update", "key": "living_room_light", "value": "on"}
    ])
    
    result = evaluator.evaluate_test_case(agent, test_case)
    print(f"  Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"  Score: {result.total_score}/{result.max_score}")
