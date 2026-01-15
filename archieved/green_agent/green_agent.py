"""
Green Agent v2.0 - 测试用例自动生成器（带验证层）

用于为 SmartMem 智能家居 Agent 生成测试用例数据库。
基于本地 Ollama + Qwen2.5-Coder 模型。

改进点：
1. 更精确的 Prompt 设计（基于 test_case_spec.md）
2. 严格的设备约束验证
3. 自动过滤无效测试用例
4. 重试机制

使用方法:
    python green_agent.py --level easy --count 5
    python green_agent.py --level all --count 10 --retry 3
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, Tuple
from openai import OpenAI

# ============== 配置部分 ==============

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
MODEL_NAME = "qwen2.5-coder:7b"

# 严格的设备约束定义（与 interface_spec.md 完全一致）
DEVICE_CONSTRAINTS = {
    # Living Room
    "living_room_light": {"type": "enum", "values": ["on", "off"]},
    "living_room_color": {"type": "enum", "values": ["white", "red", "blue", "warm"]},
    # Bedroom
    "bedroom_light": {"type": "enum", "values": ["on", "off"]},
    "bedroom_color": {"type": "enum", "values": ["white", "warm", "blue", "red"]},
    # Climate Control
    "ac": {"type": "enum", "values": ["on", "off"]},
    "ac_temperature": {"type": "int", "min": 16, "max": 30},
    "fan_speed": {"type": "enum", "values": ["off", "low", "medium", "high"]},
    # Entertainment & Security
    "music_volume": {"type": "int", "min": 0, "max": 10},
    "front_door_lock": {"type": "enum", "values": ["locked", "unlocked"]},
    "kitchen_light": {"type": "enum", "values": ["on", "off"]},
}

VALID_DEVICE_KEYS = list(DEVICE_CONSTRAINTS.keys())

# 测试维度
DIMENSIONS = ["precision", "ambiguous", "conflict", "memory", "noise"]


# ============== 验证器 ==============

class TestCaseValidator:
    """测试用例验证器 - 确保生成的用例符合规范"""
    
    @staticmethod
    def validate_device_key(key: str) -> Tuple[bool, str]:
        """验证设备 key 是否有效"""
        if key not in VALID_DEVICE_KEYS:
            return False, f"无效的设备 key: '{key}'，有效值: {VALID_DEVICE_KEYS}"
        return True, ""
    
    @staticmethod
    def validate_device_value(key: str, value: Any) -> Tuple[bool, str]:
        """验证设备值是否符合约束"""
        if key not in DEVICE_CONSTRAINTS:
            return False, f"未知设备: {key}"
        
        constraint = DEVICE_CONSTRAINTS[key]
        
        if constraint["type"] == "enum":
            if value not in constraint["values"]:
                return False, f"设备 '{key}' 的值 '{value}' 无效，允许值: {constraint['values']}"
        elif constraint["type"] == "int":
            if not isinstance(value, int):
                return False, f"设备 '{key}' 的值必须是整数，得到: {type(value).__name__}"
            if not (constraint["min"] <= value <= constraint["max"]):
                return False, f"设备 '{key}' 的值 {value} 超出范围 [{constraint['min']}, {constraint['max']}]"
        
        return True, ""
    
    @classmethod
    def validate_test_case(cls, test_case: dict) -> Tuple[bool, List[str]]:
        """完整验证测试用例"""
        errors = []
        
        # 1. 验证必需字段
        required_fields = ['scenario_id', 'difficulty', 'dimension', 'description', 'turns']
        for field in required_fields:
            if field not in test_case:
                errors.append(f"缺少必需字段: {field}")
        
        if errors:
            return False, errors
        
        # 2. 验证 initial_state
        if 'initial_state' in test_case and test_case['initial_state']:
            for key, value in test_case['initial_state'].items():
                valid, msg = cls.validate_device_key(key)
                if not valid:
                    errors.append(f"initial_state: {msg}")
                    continue
                valid, msg = cls.validate_device_value(key, value)
                if not valid:
                    errors.append(f"initial_state: {msg}")
        
        # 3. 验证每个 turn
        for i, turn in enumerate(test_case.get('turns', [])):
            turn_id = turn.get('turn_id', i + 1)
            
            # 验证 turn 结构
            if 'gm_instruction' not in turn:
                errors.append(f"Turn {turn_id}: 缺少 gm_instruction")
            if 'expected_agent_action' not in turn:
                errors.append(f"Turn {turn_id}: 缺少 expected_agent_action")
            if 'expected_final_state' not in turn:
                errors.append(f"Turn {turn_id}: 缺少 expected_final_state")
            
            # 验证 actions
            for j, action in enumerate(turn.get('expected_agent_action', [])):
                if 'key' not in action:
                    errors.append(f"Turn {turn_id} Action {j+1}: 缺少 key")
                    continue
                if 'value' not in action:
                    errors.append(f"Turn {turn_id} Action {j+1}: 缺少 value")
                    continue
                
                key = action['key']
                value = action['value']
                
                valid, msg = cls.validate_device_key(key)
                if not valid:
                    errors.append(f"Turn {turn_id} Action {j+1}: {msg}")
                    continue
                
                valid, msg = cls.validate_device_value(key, value)
                if not valid:
                    errors.append(f"Turn {turn_id} Action {j+1}: {msg}")
            
            # 验证 expected_final_state
            for key, value in turn.get('expected_final_state', {}).items():
                valid, msg = cls.validate_device_key(key)
                if not valid:
                    errors.append(f"Turn {turn_id} expected_final_state: {msg}")
                    continue
                valid, msg = cls.validate_device_value(key, value)
                if not valid:
                    errors.append(f"Turn {turn_id} expected_final_state: {msg}")
        
        # 4. 验证状态一致性（expected_final_state 应该与 actions 一致）
        if not errors:
            state_errors = cls._validate_state_consistency(test_case)
            errors.extend(state_errors)
        
        return len(errors) == 0, errors
    
    @classmethod
    def _validate_state_consistency(cls, test_case: dict) -> List[str]:
        """验证状态一致性：actions 执行后的状态应该与 expected_final_state 一致"""
        errors = []
        
        # 模拟状态
        current_state = dict(test_case.get('initial_state', {}))
        
        for turn in test_case.get('turns', []):
            turn_id = turn.get('turn_id', 0)
            
            # 执行 actions
            for action in turn.get('expected_agent_action', []):
                if action.get('action') == 'update':
                    key = action.get('key')
                    value = action.get('value')
                    if key:
                        current_state[key] = value
            
            # 检查 expected_final_state
            expected_state = turn.get('expected_final_state', {})
            for key, expected_value in expected_state.items():
                actual_value = current_state.get(key)
                if actual_value != expected_value:
                    # 如果在 initial_state 中也没有，可能是遗漏
                    if key not in current_state:
                        errors.append(
                            f"Turn {turn_id}: expected_final_state 中的 '{key}={expected_value}' "
                            f"既不在 initial_state 中，也没有被任何 action 设置"
                        )
        
        return errors


# ============== 数据结构 ==============

class ExpectedAction(BaseModel):
    action: Literal["update"] = "update"
    key: str
    value: Any


class Turn(BaseModel):
    turn_id: int
    gm_instruction: str
    expected_agent_action: List[ExpectedAction] = Field(default_factory=list)
    expected_final_state: Dict[str, Any]


class TestCase(BaseModel):
    scenario_id: str
    difficulty: Literal["easy", "medium", "difficult"]
    dimension: str
    description: str
    initial_state: Dict[str, Any] = Field(default_factory=dict)
    turns: List[Turn]


class TestCaseDatabase(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    test_cases: List[TestCase] = Field(default_factory=list)


# ============== Green Agent 核心 ==============

class GreenAgent:
    """Green Agent v2.0 - 带验证层的测试用例生成器"""
    
    def __init__(
        self, 
        base_url: str = OLLAMA_BASE_URL, 
        api_key: str = OLLAMA_API_KEY, 
        model: str = MODEL_NAME,
        max_retries: int = 3,
        provider: str = "ollama"  # 新增: API provider 类型
    ):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.validator = TestCaseValidator()
        
        # 根据 provider 初始化客户端
        if provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key)
                self._call_llm = self._call_anthropic
            except ImportError:
                raise ImportError("需要安装 anthropic: pip install anthropic")
        else:
            # OpenAI 兼容 API (包括 Ollama, OpenAI, DeepSeek 等)
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self._call_llm = self._call_openai
        
        # 统计信息
        self.stats = {
            "total_attempts": 0,
            "successful": 0,
            "failed_json": 0,
            "failed_validation": 0,
        }
    
    @classmethod
    def from_config(cls, config, max_retries: int = 3):
        """从 APIConfig 创建实例"""
        from green_agent.api_config import APIConfig
        
        if isinstance(config, str):
            from green_agent.api_config import get_api_config
            config = get_api_config(config)
        
        return cls(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            max_retries=max_retries,
            provider=config.provider
        )
    
    @classmethod
    def from_ollama(cls, model: str = "qwen2.5-coder:7b", **kwargs):
        """从 Ollama 本地创建"""
        return cls(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=model,
            provider="ollama",
            **kwargs
        )
    
    @classmethod
    def from_openai(cls, model: str = "gpt-4o", api_key: Optional[str] = None, **kwargs):
        """从 OpenAI API 创建"""
        import os
        return cls(
            base_url="https://api.openai.com/v1",
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            model=model,
            provider="openai",
            **kwargs
        )
    
    @classmethod
    def from_anthropic(cls, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, **kwargs):
        """从 Anthropic Claude API 创建"""
        import os
        return cls(
            base_url="https://api.anthropic.com",
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            model=model,
            provider="anthropic",
            **kwargs
        )
    
    @classmethod
    def from_deepseek(cls, model: str = "deepseek-chat", api_key: Optional[str] = None, **kwargs):
        """从 DeepSeek API 创建"""
        import os
        return cls(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
            model=model,
            provider="deepseek",
            **kwargs
        )
    
    @classmethod
    def from_openrouter(cls, model: str = "anthropic/claude-3.5-sonnet", api_key: Optional[str] = None, **kwargs):
        """从 OpenRouter 创建 (多模型网关)"""
        import os
        return cls(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
            model=model,
            provider="openrouter",
            **kwargs
        )
    
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """调用 OpenAI 兼容 API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4096
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """调用 Anthropic Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text
    
    def _build_system_prompt(self) -> str:
        """构建精确的系统提示词"""
        
        # 构建设备约束字符串
        device_specs = []
        for key, constraint in DEVICE_CONSTRAINTS.items():
            if constraint["type"] == "enum":
                device_specs.append(f'- {key}: {constraint["values"]}')
            else:
                device_specs.append(f'- {key}: integer {constraint["min"]}-{constraint["max"]}')
        
        device_spec_str = "\n".join(device_specs)
        
        return f"""You are a QA Engineer Agent for SmartMem, a smart home AI testing system.
Generate test cases in STRICTLY VALID JSON format.

## CRITICAL RULES - MUST FOLLOW:
1. Output ONLY raw JSON - NO markdown, NO code blocks, NO explanations
2. Device keys MUST be exactly from the allowed list (case-sensitive)
3. Device values MUST match the exact allowed values or ranges
4. expected_final_state MUST be consistent with initial_state + actions

## DEVICE SPECIFICATIONS (EXACT VALUES ONLY):
{device_spec_str}

## IMPORTANT CONSTRAINTS:
- "living_room_light" and "bedroom_light" and "kitchen_light" can ONLY be "on" or "off"
- Colors ("living_room_color", "bedroom_color") can ONLY be "white", "red", "blue", or "warm"
- "fan_speed" can ONLY be "off", "low", "medium", or "high" (NOT integers!)
- "ac" can ONLY be "on" or "off"
- "front_door_lock" can ONLY be "locked" or "unlocked"
- Integer values: ac_temperature (16-30), music_volume (0-10)

## TEST CASE ENCODING SYSTEM:
- A1: Precise command (e.g., "Set AC to 26")
- A2: Ambiguous command (e.g., "Make it cozy")
- A3: Conflicting commands (e.g., "Turn on... wait, turn off")
- A4: State query (e.g., "Is the light on?")
- B0: No action (distractor turn)
- B1: Single device action
- B2: Multiple independent device actions
- B3: Sequential dependent actions
- N0: No noise
- N1: Light chitchat noise
- N2: Logic puzzle noise
- N3: Heavy text noise

## OUTPUT FORMAT:
{{
  "scenario_id": "scenario_A1_B1_C0_N0",
  "difficulty": "easy|medium|difficult",
  "dimension": "precision|ambiguous|conflict|memory|noise",
  "description": "Brief English description",
  "initial_state": {{"device_key": "valid_value"}},
  "turns": [
    {{
      "turn_id": 1,
      "gm_instruction": "User instruction",
      "expected_agent_action": [
        {{"action": "update", "key": "device_key", "value": "valid_value"}}
      ],
      "expected_final_state": {{"device_key": "valid_value"}}
    }}
  ]
}}
"""

    def _build_user_prompt(self, difficulty: str, dimension: str, scenario_number: int) -> str:
        """构建用户提示词"""
        
        difficulty_specs = {
            "easy": """
## EASY Level Specifications:
- Turns: 1-2 maximum
- Intent: A1 (precise commands only)
- Actions: B1 (single device per turn)
- Noise: N0 (none)
- Memory: C0 (immediate)

Example scenarios:
1. "Turn on the living room light" -> living_room_light: "on"
2. "Set the AC temperature to 24 degrees" -> ac_temperature: 24
3. "Lock the front door" -> front_door_lock: "locked"
""",
            "medium": """
## MEDIUM Level Specifications:
- Turns: 2-4
- Intent: A2 (may need reasoning) or A3 (simple conflicts)
- Actions: B1-B2 (single or multiple devices)
- Noise: N0-N1 (0-1 distractor turns allowed)
- Memory: C0-C1 (0-2 turns gap)

Example scenarios:
1. "Make the living room cozy for reading" -> light: on, color: warm, maybe adjust volume
2. "It's too hot" then "Actually it's fine" -> temperature change then revert
3. Distractor: "What's the weather like?" with expected_agent_action: []
""",
            "difficult": """
## DIFFICULT Level Specifications:
- Turns: 4-8
- Intent: A3 (conflicts) or A4 (state queries) mixed with A1/A2
- Actions: B2-B3 (multiple devices, may have order dependency)
- Noise: N1-N3 (2-4 distractor turns between key commands)
- Memory: C2-C3 (recall state from 3+ turns ago)

Example scenario structure:
Turn 1: Set music_volume to 5
Turn 2-4: Distractor turns (chitchat, unrelated topics) with expected_agent_action: []
Turn 5: "Turn it up by 2" -> Agent must remember volume was 5, set to 7
"""
        }
        
        dimension_specs = {
            "precision": "Test EXACT command following. User gives precise values. No ambiguity.",
            "ambiguous": "Test INFERENCE ability. Commands like 'make it comfortable' require reasoning about multiple devices.",
            "conflict": "Test CONFLICT resolution. Include contradictory commands. Later command wins.",
            "memory": "Test STATE RECALL. Set a value, add distractors, then ask to modify based on the old value.",
            "noise": "Test NOISE resistance. Many distractor turns with expected_agent_action: [] between real commands.",
        }
        
        return f"""Generate ONE test case:

DIFFICULTY: {difficulty.upper()}
{difficulty_specs.get(difficulty, difficulty_specs['easy'])}

DIMENSION: {dimension}
{dimension_specs.get(dimension, 'Standard test scenario.')}

SCENARIO NUMBER: {scenario_number}

REMINDERS:
1. Use EXACT device keys: living_room_light, bedroom_light, ac, ac_temperature, etc.
2. Use EXACT values: "on"/"off" for lights, "locked"/"unlocked" for door, integers for temperature/volume
3. fan_speed uses strings: "off", "low", "medium", "high" (NOT numbers!)
4. expected_final_state must include ALL devices that have been modified
5. Distractor turns MUST have: expected_agent_action: []

Output ONLY the JSON object, starting with {{ and ending with }}
"""

    def generate_single_case(
        self,
        difficulty: str = "easy",
        dimension: str = "precision",
        scenario_number: int = 1
    ) -> Optional[TestCase]:
        """生成单个测试用例，带验证和重试"""
        
        for attempt in range(self.max_retries):
            self.stats["total_attempts"] += 1
            
            attempt_str = f" (尝试 {attempt + 1}/{self.max_retries})" if attempt > 0 else ""
            print(f"🟢 Green Agent 生成中{attempt_str}... [难度: {difficulty}, 维度: {dimension}]")
            
            try:
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(difficulty, dimension, scenario_number)
                
                # 使用统一的 LLM 调用接口
                raw_content = self._call_llm(system_prompt, user_prompt)
                
                # 1. JSON 解析 - 尝试提取 JSON
                try:
                    data = json.loads(raw_content)
                except json.JSONDecodeError:
                    # 尝试从响应中提取 JSON
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', raw_content)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                        except json.JSONDecodeError as e:
                            print(f"  ⚠️  JSON 解析失败: {e}")
                            self.stats["failed_json"] += 1
                            continue
                    else:
                        print(f"  ⚠️  未找到有效 JSON")
                        self.stats["failed_json"] += 1
                        continue
                
                # 2. 语义验证
                is_valid, errors = self.validator.validate_test_case(data)
                
                if not is_valid:
                    print(f"  ⚠️  验证失败:")
                    for err in errors[:3]:  # 只显示前3个错误
                        print(f"      - {err}")
                    if len(errors) > 3:
                        print(f"      ... 还有 {len(errors) - 3} 个错误")
                    self.stats["failed_validation"] += 1
                    continue
                
                # 3. 构建 Pydantic 模型
                test_case = TestCase(**data)
                
                print(f"  ✅ 生成成功: {test_case.scenario_id}")
                self.stats["successful"] += 1
                return test_case
                
            except Exception as e:
                print(f"  🔴 生成异常: {e}")
                continue
        
        print(f"  ❌ 达到最大重试次数，跳过此用例")
        return None

    def generate_batch(
        self,
        difficulty: str = "all",
        dimension: str = "all",
        count_per_combo: int = 2
    ) -> TestCaseDatabase:
        """批量生成测试用例"""
        
        difficulties = ["easy", "medium", "difficult"] if difficulty == "all" else [difficulty]
        dimensions = DIMENSIONS if dimension == "all" else [dimension]
        
        database = TestCaseDatabase(
            metadata={
                "version": "2.1-green-validated",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generator": "Green Agent v2.0",
                "model": self.model,
                "validation": "strict",
                "notes": f"Difficulties: {difficulties}, Dimensions: {dimensions}"
            },
            test_cases=[]
        )
        
        scenario_counter = 1
        for diff in difficulties:
            for dim in dimensions:
                for i in range(count_per_combo):
                    case = self.generate_single_case(diff, dim, scenario_counter)
                    if case:
                        database.test_cases.append(case)
                        scenario_counter += 1
        
        # 打印统计
        print(f"\n{'='*60}")
        print(f"📊 生成统计:")
        print(f"   总尝试次数: {self.stats['total_attempts']}")
        print(f"   成功: {self.stats['successful']}")
        print(f"   JSON 解析失败: {self.stats['failed_json']}")
        print(f"   验证失败: {self.stats['failed_validation']}")
        print(f"   成功率: {self.stats['successful'] / max(1, self.stats['total_attempts']) * 100:.1f}%")
        print(f"{'='*60}")
        
        return database

    def save_database(self, database: TestCaseDatabase, output_path: str) -> None:
        """保存数据库到 JSON 文件"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(database.model_dump_json(indent=2))
        
        print(f"💾 已保存到: {output_file}")


# ============== 命令行接口 ==============

def main():
    parser = argparse.ArgumentParser(
        description="Green Agent v2.0 - SmartMem 测试用例生成器（带验证）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python green_agent.py --level easy --count 3
  python green_agent.py --level medium --dimension memory --count 5
  python green_agent.py --level all --dimension all --count 2
  python green_agent.py --single --level difficult --dimension conflict --retry 5
        """
    )
    
    parser.add_argument("--level", "-l", choices=["easy", "medium", "difficult", "all"], default="easy")
    parser.add_argument("--dimension", "-d", choices=DIMENSIONS + ["all"], default="precision")
    parser.add_argument("--count", "-c", type=int, default=3)
    parser.add_argument("--output", "-o", default="test_cases/green_generated.json")
    parser.add_argument("--single", "-s", action="store_true", help="仅生成单个用例")
    parser.add_argument("--retry", "-r", type=int, default=3, help="验证失败时的最大重试次数")
    parser.add_argument("--model", "-m", default=MODEL_NAME)
    parser.add_argument("--base-url", "-u", default=OLLAMA_BASE_URL)
    
    args = parser.parse_args()
    
    agent = GreenAgent(
        base_url=args.base_url,
        model=args.model,
        max_retries=args.retry
    )
    
    print("=" * 60)
    print("🟢 Green Agent v2.0 - 测试用例生成器（带验证）")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"难度: {args.level}")
    print(f"维度: {args.dimension}")
    print(f"最大重试: {args.retry}")
    print("=" * 60 + "\n")
    
    if args.single:
        case = agent.generate_single_case(
            difficulty=args.level if args.level != "all" else "easy",
            dimension=args.dimension if args.dimension != "all" else "precision",
            scenario_number=1
        )
        if case:
            print("\n✅ 生成成功！\n")
            print(case.model_dump_json(indent=2))
    else:
        database = agent.generate_batch(
            difficulty=args.level,
            dimension=args.dimension,
            count_per_combo=args.count
        )
        
        if database.test_cases:
            agent.save_database(database, args.output)
            print(f"\n🎉 完成！共生成 {len(database.test_cases)} 个有效测试用例")
        else:
            print("\n❌ 没有成功生成任何有效测试用例")


if __name__ == "__main__":
    main()
