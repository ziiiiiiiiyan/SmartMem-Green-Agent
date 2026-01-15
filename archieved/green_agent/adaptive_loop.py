"""
Adaptive Adversarial Testing Loop (自适应对抗测试循环)

核心功能：
1. 从简单到难逐步测试 Agent
2. 分析失败模式，识别弱点
3. 针对弱点生成更多类似/更难的问题
4. 循环直到摸清能力边界
5. 输出量化的弱点报告（含雷达图）

架构说明:
- Agent 是黑盒，只通过 chat(instruction) -> response 交互
- 支持任意实现 AgentInterface 的 Agent
- Eval 负责解析响应、执行动作、评分

使用方法:
    # 使用 Mock Agent 测试
    python adaptive_loop.py --agent-type mock --error-rate 0.3
    
    # 使用 OpenAI API
    python adaptive_loop.py --agent-type openai --model gpt-4o
    
    # 使用 Purple Agent
    python adaptive_loop.py --agent-type purple
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from green_agent import GreenAgent, TestCase, DEVICE_CONSTRAINTS, DIMENSIONS
from green_agent.agent_interface import AgentInterface, create_agent, MockAgent
from green_agent.blackbox_eval import BlackBoxEvaluator, ResponseParser
from app.environment import SmartHomeEnv
from app.evaluator import TurnEvaluator

# ============== 数据结构 ==============

@dataclass
class TestResult:
    """单个测试结果"""
    test_case: dict
    score: float  # 0.0 - 1.0
    max_score: float
    passed: bool
    errors: List[str] = field(default_factory=list)
    turn_details: List[dict] = field(default_factory=list)


@dataclass 
class DimensionStats:
    """维度统计"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    total_score: float = 0.0
    max_possible_score: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        return self.passed / max(1, self.total)
    
    @property
    def avg_score(self) -> float:
        return self.total_score / max(1, self.max_possible_score)
    
    @property
    def weakness_score(self) -> float:
        """弱点分数：越高越弱 (0-1)"""
        return 1.0 - self.avg_score


@dataclass
class WeaknessProfile:
    """弱点画像"""
    # 按维度统计
    by_dimension: Dict[str, DimensionStats] = field(default_factory=dict)
    # 按难度统计
    by_difficulty: Dict[str, DimensionStats] = field(default_factory=dict)
    # 按设备统计
    by_device: Dict[str, DimensionStats] = field(default_factory=dict)
    # 失败用例列表
    failed_cases: List[TestResult] = field(default_factory=list)
    # 边界发现
    boundary_found: Dict[str, str] = field(default_factory=dict)  # dimension -> difficulty


# ============== 黑盒 Agent 包装器 ==============

class BlackBoxAgentWrapper:
    """
    黑盒 Agent 包装器
    
    将任意 Agent（通过 AgentInterface）包装成黑盒形式，
    只通过文本 I/O 进行交互，由 Eval 框架负责解析响应并执行动作。
    
    这实现了:
    - eval 框架起环境 + 提供工具函数
    - agent 只负责文本输出
    - eval 框架解析 agent 输出并执行
    """
    
    def __init__(
        self, 
        env: SmartHomeEnv, 
        agent: Optional[AgentInterface] = None,
        agent_type: str = "mock",
        **kwargs
    ):
        """
        初始化黑盒 Agent 包装器
        
        Args:
            env: SmartHome 环境
            agent: 实现 AgentInterface 的 Agent 实例
            agent_type: 如果未提供 agent，则使用此类型创建
                - "mock": 模拟 Agent (按预期输出)
                - "openai": OpenAI API Agent
                - "anthropic": Anthropic Claude Agent
                - "ollama": Ollama 本地 Agent
                - "purple": Purple Agent 包装器
            **kwargs: Agent 创建参数
        """
        self.env = env
        self.kwargs = kwargs
        self.response_parser = ResponseParser()
        
        # 初始化 Agent
        if agent is not None:
            self.agent = agent
        else:
            self.agent = self._create_agent(agent_type, **kwargs)
        
        self.agent_type = agent_type
    
    def _create_agent(self, agent_type: str, **kwargs) -> AgentInterface:
        """创建 Agent 实例"""
        return create_agent(agent_type, **kwargs)
    
    def execute_turn(self, instruction: str, expected_actions: Optional[List[dict]] = None) -> Tuple[List[dict], dict]:
        """
        执行一轮对话（黑盒模式）
        
        流程:
        1. 发送指令给 Agent
        2. 获取 Agent 的文本响应
        3. 解析响应中的动作
        4. 在环境中执行动作
        5. 返回执行结果
        
        Args:
            instruction: 用户指令
            expected_actions: 预期动作（仅用于 mock 模式）
        
        Returns:
            (actual_actions, final_state)
        """
        # 重置环境的 action history
        self.env.reset_turn_history()
        
        # 构建带环境状态的提示
        current_state = self.env.get_state()['state']
        prompt = self._build_prompt(instruction, current_state)
        
        # 获取 Agent 响应
        if isinstance(self.agent, MockAgent):
            # Mock 模式：直接返回预期动作
            response = self._mock_response(expected_actions or [])
        elif isinstance(self.agent, ImperfectMockAgent):
            # Imperfect Mock 模式：设置预期动作后返回带错误的响应
            self.agent.set_expected_actions(expected_actions or [])
            response = self.agent.chat(prompt)
        else:
            # 真实 Agent：获取文本响应
            response = self.agent.chat(prompt)
        
        # 解析响应中的动作
        parsed_actions = self.response_parser.parse(response)
        
        # 在环境中执行动作
        actual_actions = []
        for action in parsed_actions:
            if action.get('action') == 'update':
                key = action.get('key')
                value = action.get('value')
                
                result = self.env.update_state(key, value)
                if result['status'] == 'success':
                    actual_actions.append(action)
                    self.env.record_action(action)
        
        final_state = self.env.get_state()['state']
        return actual_actions, final_state
    
    def _build_prompt(self, instruction: str, current_state: dict) -> str:
        """构建发送给 Agent 的完整提示"""
        state_str = json.dumps(current_state, ensure_ascii=False, indent=2)
        
        prompt = f"""你是一个智能家居控制助手。请根据用户指令控制设备。

当前设备状态:
```json
{state_str}
```

用户指令: {instruction}

请输出你要执行的动作。使用以下 JSON 格式:
```json
{{"actions": [{{"action": "update", "key": "设备名", "value": "新状态"}}]}}
```

如果需要执行多个动作，在 actions 数组中添加多个对象。
如果不需要执行任何动作，返回空数组: {{"actions": []}}
"""
        return prompt
    
    def _mock_response(self, expected_actions: List[dict]) -> str:
        """生成 Mock 响应"""
        return json.dumps({"actions": expected_actions}, ensure_ascii=False)
    
    def reset(self, initial_state: Optional[dict] = None):
        """重置环境和 Agent"""
        self.env.reset(initial_state=initial_state)
        self.agent.reset()
    
    def get_agent_info(self) -> dict:
        """获取 Agent 信息"""
        return {
            "type": self.agent_type,
            "interface": type(self.agent).__name__,
            "kwargs": {k: v for k, v in self.kwargs.items() if k != 'api_key'}
        }


class ImperfectMockAgent(AgentInterface):
    """带随机错误的 Mock Agent（用于测试框架本身）"""
    
    def __init__(self, error_rate: float = 0.2):
        self.error_rate = error_rate
        self.expected_actions = []
    
    def set_expected_actions(self, actions: List[dict]):
        """设置预期动作"""
        self.expected_actions = actions
    
    def chat(self, message: str) -> str:
        """返回带随机错误的响应"""
        import random
        
        # 过滤掉一些动作（模拟遗漏）
        filtered_actions = []
        for action in self.expected_actions:
            if random.random() >= self.error_rate:
                filtered_actions.append(action)
        
        return json.dumps({"actions": filtered_actions}, ensure_ascii=False)
    
    def reset(self):
        self.expected_actions = []
    
    def get_tool_calls(self, response: str = "") -> List[dict]:
        return self.expected_actions


# ============== 向后兼容的 BaselineAgent ==============

class BaselineAgent(BlackBoxAgentWrapper):
    """
    向后兼容的 BaselineAgent 类
    
    这是 BlackBoxAgentWrapper 的别名，保持 API 兼容性。
    新代码建议直接使用 BlackBoxAgentWrapper。
    """
    
    def __init__(self, env: SmartHomeEnv, agent_type: str = "mock", **kwargs):
        """
        初始化 Baseline Agent
        
        Args:
            env: SmartHome 环境
            agent_type: Agent 类型
                - "mock" / "simulated": 模拟 Agent (按预期动作执行)
                - "imperfect": 带随机错误的模拟 Agent
                - "purple_agent" / "purple": Purple Agent
                - "openai": OpenAI API Agent
                - "anthropic": Anthropic Claude Agent
                - "ollama": Ollama 本地 Agent
            **kwargs: 其他参数
        """
        # 映射旧的类型名称
        type_mapping = {
            "simulated": "mock",
            "purple_agent": "purple"
        }
        mapped_type = type_mapping.get(agent_type, agent_type)
        
        # imperfect 模式使用特殊的 Mock Agent
        if agent_type == "imperfect":
            error_rate = kwargs.get('error_rate', 0.2)
            agent = ImperfectMockAgent(error_rate=error_rate)
            super().__init__(env, agent=agent, agent_type="imperfect", **kwargs)
        else:
            super().__init__(env, agent_type=mapped_type, **kwargs)


# ============== 评估器 ==============

class AdaptiveEvaluator:
    """自适应评估器"""
    
    def __init__(self, agent: BaselineAgent):
        self.agent = agent
        self.env = agent.env
    
    def evaluate_test_case(self, test_case: dict) -> TestResult:
        """评估单个测试用例"""
        
        # 重置环境
        initial_state = test_case.get('initial_state', {})
        self.agent.reset(initial_state)
        
        total_score = 0.0
        max_score = 0.0
        turn_details = []
        all_errors = []
        
        for turn in test_case.get('turns', []):
            turn_id = turn.get('turn_id', 0)
            instruction = turn.get('gm_instruction', '')
            expected_actions = turn.get('expected_agent_action', [])
            expected_state = turn.get('expected_final_state', {})
            
            # 重置 turn 历史
            self.env.reset_turn_history()
            
            # Agent 执行
            actual_actions, actual_state = self.agent.execute_turn(instruction, expected_actions)
            
            # 评分
            evaluator = TurnEvaluator(expected_actions, expected_state)
            result = evaluator.evaluate(actual_actions, actual_state)
            
            turn_score = result['score']
            turn_max = 1.0
            
            total_score += turn_score
            max_score += turn_max
            
            turn_details.append({
                'turn_id': turn_id,
                'instruction': instruction,
                'score': turn_score,
                'max_score': turn_max,
                'passed': turn_score == turn_max,
                'errors': result.get('details', {}).get('errors', [])
            })
            
            if result.get('details', {}).get('errors'):
                all_errors.extend(result['details']['errors'])
        
        # 计算总分
        final_score = total_score / max(1, max_score)
        passed = final_score >= 1.0
        
        return TestResult(
            test_case=test_case,
            score=total_score,
            max_score=max_score,
            passed=passed,
            errors=all_errors,
            turn_details=turn_details
        )
    
    def evaluate_batch(self, test_cases: List[dict]) -> List[TestResult]:
        """批量评估"""
        results = []
        for case in test_cases:
            result = self.evaluate_test_case(case)
            results.append(result)
        return results


# ============== 弱点分析器 ==============

class WeaknessAnalyzer:
    """弱点分析器"""
    
    def __init__(self):
        self.profile = WeaknessProfile()
        # 初始化各维度统计
        for dim in DIMENSIONS:
            self.profile.by_dimension[dim] = DimensionStats()
        for diff in ['easy', 'medium', 'difficult']:
            self.profile.by_difficulty[diff] = DimensionStats()
        for device in DEVICE_CONSTRAINTS.keys():
            self.profile.by_device[device] = DimensionStats()
    
    def analyze(self, results: List[TestResult]) -> WeaknessProfile:
        """分析测试结果，更新弱点画像"""
        
        for result in results:
            case = result.test_case
            dimension = case.get('dimension', 'unknown')
            difficulty = case.get('difficulty', 'unknown')
            
            # 更新维度统计
            if dimension in self.profile.by_dimension:
                self._update_stats(self.profile.by_dimension[dimension], result)
            
            # 更新难度统计
            if difficulty in self.profile.by_difficulty:
                self._update_stats(self.profile.by_difficulty[difficulty], result)
            
            # 更新设备统计
            devices_involved = self._extract_devices(case)
            for device in devices_involved:
                if device in self.profile.by_device:
                    self._update_stats(self.profile.by_device[device], result)
            
            # 记录失败用例
            if not result.passed:
                self.profile.failed_cases.append(result)
        
        # 检测能力边界
        self._detect_boundaries()
        
        return self.profile
    
    def _update_stats(self, stats: DimensionStats, result: TestResult):
        """更新统计数据"""
        stats.total += 1
        stats.total_score += result.score
        stats.max_possible_score += result.max_score
        if result.passed:
            stats.passed += 1
        else:
            stats.failed += 1
    
    def _extract_devices(self, case: dict) -> set:
        """提取涉及的设备"""
        devices = set()
        
        # 从 initial_state
        for key in case.get('initial_state', {}).keys():
            devices.add(key)
        
        # 从 turns
        for turn in case.get('turns', []):
            for action in turn.get('expected_agent_action', []):
                if 'key' in action:
                    devices.add(action['key'])
            for key in turn.get('expected_final_state', {}).keys():
                devices.add(key)
        
        return devices
    
    def _detect_boundaries(self):
        """检测能力边界"""
        
        # 对每个维度，找到开始失败的难度
        for dim in DIMENSIONS:
            dim_stats = self.profile.by_dimension.get(dim, DimensionStats())
            
            if dim_stats.total == 0:
                continue
            
            # 简单判断：如果通过率低于 50%，认为达到边界
            if dim_stats.pass_rate < 0.5:
                # 尝试找到具体是哪个难度开始失败
                # 这里简化处理，实际需要更细致的分析
                if self.profile.by_difficulty['easy'].pass_rate < 0.5:
                    self.profile.boundary_found[dim] = 'easy'
                elif self.profile.by_difficulty['medium'].pass_rate < 0.5:
                    self.profile.boundary_found[dim] = 'medium'
                else:
                    self.profile.boundary_found[dim] = 'difficult'
    
    def get_top_weaknesses(self, n: int = 5) -> List[Tuple[str, str, float]]:
        """获取最弱的 N 个维度/设备组合"""
        weaknesses = []
        
        # 维度弱点
        for dim, stats in self.profile.by_dimension.items():
            if stats.total > 0:
                weaknesses.append(('dimension', dim, stats.weakness_score))
        
        # 设备弱点
        for device, stats in self.profile.by_device.items():
            if stats.total > 0:
                weaknesses.append(('device', device, stats.weakness_score))
        
        # 按弱点分数排序
        weaknesses.sort(key=lambda x: x[2], reverse=True)
        return weaknesses[:n]


# ============== 自适应生成策略 ==============

class AdaptiveGenerator:
    """自适应生成策略"""
    
    def __init__(self, green_agent: GreenAgent):
        self.green_agent = green_agent
    
    def generate_targeted(
        self, 
        weaknesses: List[Tuple[str, str, float]], 
        count_per_weakness: int = 5,
        difficulty_boost: bool = True
    ) -> List[TestCase]:
        """针对弱点生成测试用例"""
        
        generated = []
        
        for weakness_type, weakness_name, weakness_score in weaknesses:
            # 确定生成参数
            if weakness_type == 'dimension':
                dimension = weakness_name
                # 根据弱点分数决定难度
                if weakness_score > 0.7:
                    difficulty = 'easy'  # 弱点很明显，用简单题确认
                elif weakness_score > 0.4:
                    difficulty = 'medium'
                else:
                    difficulty = 'difficult'  # 弱点不明显，用难题探测
            else:
                # 设备弱点，用 precision 维度测试
                dimension = 'precision'
                difficulty = 'medium'
            
            if difficulty_boost:
                # 逐步提升难度
                difficulties = ['easy', 'medium', 'difficult']
                current_idx = difficulties.index(difficulty)
                if current_idx < len(difficulties) - 1:
                    difficulty = difficulties[current_idx + 1]
            
            print(f"  🎯 针对弱点 [{weakness_type}: {weakness_name}] 生成 {count_per_weakness} 个 {difficulty} 用例")
            
            for i in range(count_per_weakness):
                case = self.green_agent.generate_single_case(
                    difficulty=difficulty,
                    dimension=dimension,
                    scenario_number=len(generated) + 1
                )
                if case:
                    generated.append(case)
        
        return generated


# ============== 报告生成器 ==============

class ReportGenerator:
    """弱点报告生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_report_path: Path | None = None
        self.last_data_path: Path | None = None
    
    def generate_report(
        self, 
        profile: WeaknessProfile, 
        round_history: List[dict],
        agent_name: str = "Purple Agent"
    ) -> str:
        """生成完整的弱点报告"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"weakness_report_{timestamp}.md"
        
        # 计算雷达图数据
        radar_data = self._compute_radar_data(profile)
        
        report = []
        report.append(f"# 🎯 Agent 能力评估报告")
        report.append(f"\n**评估对象**: {agent_name}")
        report.append(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**测试轮数**: {len(round_history)}")
        
        # 总体统计
        total_cases = sum(r.get('total_cases', 0) for r in round_history)
        total_passed = sum(r.get('passed', 0) for r in round_history)
        report.append(f"\n## 📊 总体统计\n")
        report.append(f"| 指标 | 数值 |")
        report.append(f"|------|------|")
        report.append(f"| 总测试用例 | {total_cases} |")
        report.append(f"| 通过 | {total_passed} |")
        report.append(f"| 失败 | {total_cases - total_passed} |")
        report.append(f"| 通过率 | {total_passed/max(1,total_cases)*100:.1f}% |")
        
        # 维度能力雷达图数据
        report.append(f"\n## 🕸️ 维度能力分析\n")
        report.append(f"### 能力雷达图数据\n")
        report.append(f"```")
        report.append(f"维度能力值 (0-100, 越高越强):")
        for dim, score in radar_data['dimensions'].items():
            bar = '█' * int(score / 5) + '░' * (20 - int(score / 5))
            report.append(f"  {dim:12} [{bar}] {score:.1f}")
        report.append(f"```\n")
        
        # 维度详细统计
        report.append(f"### 维度详细统计\n")
        report.append(f"| 维度 | 总数 | 通过 | 失败 | 通过率 | 平均得分 | 弱点评分 |")
        report.append(f"|------|------|------|------|--------|----------|----------|")
        for dim, stats in profile.by_dimension.items():
            if stats.total > 0:
                report.append(
                    f"| {dim} | {stats.total} | {stats.passed} | {stats.failed} | "
                    f"{stats.pass_rate*100:.1f}% | {stats.avg_score*100:.1f}% | "
                    f"{'🔴' if stats.weakness_score > 0.5 else '🟡' if stats.weakness_score > 0.3 else '🟢'} {stats.weakness_score:.2f} |"
                )
        
        # 难度能力分析
        report.append(f"\n## 📈 难度能力分析\n")
        report.append(f"```")
        report.append(f"各难度通过率:")
        for diff in ['easy', 'medium', 'difficult']:
            stats = profile.by_difficulty.get(diff, DimensionStats())
            if stats.total > 0:
                bar = '█' * int(stats.pass_rate * 20) + '░' * (20 - int(stats.pass_rate * 20))
                report.append(f"  {diff:10} [{bar}] {stats.pass_rate*100:.1f}%")
        report.append(f"```\n")
        
        # 设备能力分析
        report.append(f"\n## 🏠 设备能力分析\n")
        report.append(f"| 设备 | 总数 | 通过率 | 弱点评分 |")
        report.append(f"|------|------|--------|----------|")
        sorted_devices = sorted(
            profile.by_device.items(), 
            key=lambda x: x[1].weakness_score, 
            reverse=True
        )
        for device, stats in sorted_devices:
            if stats.total > 0:
                icon = '🔴' if stats.weakness_score > 0.5 else '🟡' if stats.weakness_score > 0.3 else '🟢'
                report.append(f"| {device} | {stats.total} | {stats.pass_rate*100:.1f}% | {icon} {stats.weakness_score:.2f} |")
        
        # 能力边界
        report.append(f"\n## 🚧 能力边界\n")
        if profile.boundary_found:
            report.append(f"检测到以下能力边界：\n")
            for dim, boundary_diff in profile.boundary_found.items():
                report.append(f"- **{dim}**: 在 `{boundary_diff}` 难度开始显著下降")
        else:
            report.append(f"未检测到明显的能力边界（可能需要更多测试）")
        
        # 主要弱点
        report.append(f"\n## ⚠️ 主要弱点 (Top 5)\n")
        analyzer = WeaknessAnalyzer()
        analyzer.profile = profile
        top_weaknesses = analyzer.get_top_weaknesses(5)
        for i, (w_type, w_name, w_score) in enumerate(top_weaknesses, 1):
            severity = '🔴 严重' if w_score > 0.7 else '🟡 中等' if w_score > 0.4 else '🟢 轻微'
            report.append(f"{i}. **{w_type}: {w_name}** - 弱点分数: {w_score:.2f} ({severity})")
        
        # 测试轮次历史
        report.append(f"\n## 📝 测试轮次历史\n")
        report.append(f"| 轮次 | 用例数 | 通过 | 失败 | 通过率 | 聚焦领域 |")
        report.append(f"|------|--------|------|------|--------|----------|")
        for i, r in enumerate(round_history, 1):
            focus = r.get('focus', 'initial')
            report.append(
                f"| {i} | {r.get('total_cases', 0)} | {r.get('passed', 0)} | "
                f"{r.get('failed', 0)} | {r.get('pass_rate', 0)*100:.1f}% | {focus} |"
            )
        
        # 失败用例示例
        report.append(f"\n## 📋 失败用例示例 (最多显示 5 个)\n")
        for i, result in enumerate(profile.failed_cases[:5], 1):
            case = result.test_case
            report.append(f"### 失败用例 {i}: {case.get('scenario_id', 'unknown')}")
            report.append(f"- **难度**: {case.get('difficulty')}")
            report.append(f"- **维度**: {case.get('dimension')}")
            report.append(f"- **描述**: {case.get('description')}")
            report.append(f"- **得分**: {result.score}/{result.max_score}")
            if result.errors:
                report.append(f"- **错误**: {result.errors[:3]}")
            report.append("")
        
        # 建议
        report.append(f"\n## 💡 改进建议\n")
        if top_weaknesses:
            report.append(f"基于弱点分析，建议重点改进以下领域：\n")
            for w_type, w_name, w_score in top_weaknesses[:3]:
                if w_type == 'dimension':
                    report.append(f"1. **{w_name} 维度**: 加强 {self._get_dimension_advice(w_name)}")
                else:
                    report.append(f"1. **{w_name} 设备**: 加强对该设备的理解和操作能力")
        
        # 写入文件
        report_content = '\n'.join(report)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 同时生成 JSON 数据（供程序读取）
        json_file = self.output_dir / f"weakness_data_{timestamp}.json"
        json_data = {
            "agent_name": agent_name,
            "timestamp": timestamp,
            "radar_data": radar_data,
            "round_history": round_history,
            "top_weaknesses": [
                {"type": t, "name": n, "score": s} 
                for t, n, s in top_weaknesses
            ],
            "dimension_stats": {
                dim: {
                    "total": stats.total,
                    "passed": stats.passed,
                    "pass_rate": stats.pass_rate,
                    "avg_score": stats.avg_score,
                    "weakness_score": stats.weakness_score
                }
                for dim, stats in profile.by_dimension.items()
                if stats.total > 0
            },
            "difficulty_stats": {
                diff: {
                    "total": stats.total,
                    "passed": stats.passed,
                    "pass_rate": stats.pass_rate
                }
                for diff, stats in profile.by_difficulty.items()
                if stats.total > 0
            }
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 报告已生成: {report_file}")
        print(f"📊 数据已生成: {json_file}")
        self.last_report_path = report_file
        self.last_data_path = json_file
        
        return str(report_file)
    
    def _compute_radar_data(self, profile: WeaknessProfile) -> dict:
        """计算雷达图数据"""
        dimensions = {}
        for dim, stats in profile.by_dimension.items():
            if stats.total > 0:
                # 能力分数 = 1 - 弱点分数，转换为 0-100
                dimensions[dim] = (1 - stats.weakness_score) * 100
            else:
                dimensions[dim] = 50  # 未测试的给中间值
        
        return {"dimensions": dimensions}
    
    def _get_dimension_advice(self, dimension: str) -> str:
        """获取维度改进建议"""
        advice = {
            "precision": "精确指令理解和执行能力",
            "ambiguous": "模糊指令的推理和理解能力",
            "conflict": "冲突指令的检测和处理能力",
            "memory": "上下文记忆和状态追踪能力",
            "noise": "噪声过滤和关键信息提取能力"
        }
        return advice.get(dimension, "相关能力")


# ============== 主循环 ==============

class AdaptiveTestLoop:
    """自适应对抗测试主循环"""
    
    def __init__(
        self,
        green_agent: GreenAgent,
        baseline_agent: BaselineAgent,
        output_dir: Path = Path("test_results")
    ):
        self.green_agent = green_agent
        self.baseline_agent = baseline_agent
        self.evaluator = AdaptiveEvaluator(baseline_agent)
        self.analyzer = WeaknessAnalyzer()
        self.generator = AdaptiveGenerator(green_agent)
        self.reporter = ReportGenerator(output_dir)
        
        self.round_history = []
        self.all_results: List[TestResult] = []
    
    def run(
        self,
        max_rounds: int = 5,
        initial_per_dim: int = 10,
        targeted_per_weakness: int = 5,
        convergence_threshold: float = 0.05,
        agent_name: str = "Purple Agent"
    ) -> str:
        """
        运行自适应测试循环
        
        Args:
            max_rounds: 最大测试轮数
            initial_per_dim: 初始每个维度生成的用例数
            targeted_per_weakness: 每个弱点针对性生成的用例数
            convergence_threshold: 收敛阈值（连续两轮通过率变化小于此值时停止）
            agent_name: Agent 名称（用于报告）
        
        Returns:
            报告文件路径
        """
        
        print("=" * 70)
        print("🔄 Adaptive Adversarial Testing Loop")
        print("=" * 70)
        print(f"目标 Agent: {agent_name}")
        print(f"最大轮数: {max_rounds}")
        print(f"初始用例/维度: {initial_per_dim}")
        print(f"收敛阈值: {convergence_threshold}")
        print("=" * 70)
        
        last_pass_rate = None
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n{'='*70}")
            print(f"📍 第 {round_num} 轮测试")
            print(f"{'='*70}")
            
            # 生成测试用例
            if round_num == 1:
                # 第一轮：均匀生成各维度用例
                test_cases = self._generate_initial_cases(initial_per_dim)
                focus = "initial_balanced"
            else:
                # 后续轮：针对弱点生成
                top_weaknesses = self.analyzer.get_top_weaknesses(3)
                test_cases = self.generator.generate_targeted(
                    top_weaknesses, 
                    targeted_per_weakness,
                    difficulty_boost=True
                )
                focus = f"targeted_{top_weaknesses[0][1] if top_weaknesses else 'unknown'}"
            
            if not test_cases:
                print("⚠️ 未能生成测试用例，结束循环")
                break
            
            # 评估
            print(f"\n📝 评估 {len(test_cases)} 个测试用例...")
            results = self.evaluator.evaluate_batch(
                [tc.model_dump() if hasattr(tc, 'model_dump') else tc for tc in test_cases]
            )
            
            # 统计本轮结果
            passed = sum(1 for r in results if r.passed)
            failed = len(results) - passed
            pass_rate = passed / max(1, len(results))
            
            round_info = {
                "round": round_num,
                "total_cases": len(results),
                "passed": passed,
                "failed": failed,
                "pass_rate": pass_rate,
                "focus": focus
            }
            self.round_history.append(round_info)
            self.all_results.extend(results)
            
            print(f"\n📊 本轮结果: {passed}/{len(results)} 通过 ({pass_rate*100:.1f}%)")
            
            # 更新弱点分析
            self.analyzer.analyze(results)
            
            # 显示当前弱点
            top_weaknesses = self.analyzer.get_top_weaknesses(3)
            if top_weaknesses:
                print(f"\n⚠️ 当前主要弱点:")
                for w_type, w_name, w_score in top_weaknesses:
                    print(f"   - {w_type}: {w_name} (弱点分数: {w_score:.2f})")
            
            # 检查收敛
            if last_pass_rate is not None:
                rate_change = abs(pass_rate - last_pass_rate)
                if rate_change < convergence_threshold:
                    print(f"\n✅ 通过率变化 ({rate_change:.3f}) 小于阈值 ({convergence_threshold})，能力边界已稳定")
                    break
            
            last_pass_rate = pass_rate
            
            # 检查是否所有维度都已探测到边界
            if len(self.analyzer.profile.boundary_found) >= len(DIMENSIONS):
                print(f"\n✅ 所有维度能力边界已探测完成")
                break
        
        # 生成报告
        print(f"\n{'='*70}")
        print("📄 生成评估报告...")
        print(f"{'='*70}")
        
        report_path = self.reporter.generate_report(
            self.analyzer.profile,
            self.round_history,
            agent_name
        )
        
        return report_path
    
    def _generate_initial_cases(self, per_dim: int) -> List[TestCase]:
        """生成初始测试用例（各维度均匀分布，从 easy 开始）"""
        cases = []
        
        for dim in DIMENSIONS:
            # easy 少一些，medium/difficult 多一些
            easy_count = per_dim // 3
            medium_count = per_dim // 3
            difficult_count = per_dim - easy_count - medium_count
            
            print(f"\n🟢 生成 {dim} 维度用例 (easy:{easy_count}, medium:{medium_count}, difficult:{difficult_count})")
            
            for diff, count in [('easy', easy_count), ('medium', medium_count), ('difficult', difficult_count)]:
                for i in range(count):
                    case = self.green_agent.generate_single_case(
                        difficulty=diff,
                        dimension=dim,
                        scenario_number=len(cases) + 1
                    )
                    if case:
                        cases.append(case)
        
        return cases


# ============== 命令行接口 ==============

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Adaptive Adversarial Testing Loop - 自适应对抗测试循环",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 模拟模式（测试框架）
  python adaptive_loop.py --rounds 5 --initial-per-dim 10
  
  # 使用本地 Ollama 模型作为被测 Agent
  python adaptive_loop.py --agent-type ollama --agent-model qwen2.5:7b
  
  # 使用 OpenAI API 作为被测 Agent
  python adaptive_loop.py --agent-type openai --agent-model gpt-4o
  
  # 使用 Anthropic Claude 作为被测 Agent
  python adaptive_loop.py --agent-type anthropic --agent-model claude-3-5-sonnet-20241022
  
  # 使用 A2A 协议连接远程 Agent
  python adaptive_loop.py --agent-type a2a --agent-url https://agent.example.com
  
  # 使用 MCP 协议连接 Agent
  python adaptive_loop.py --agent-type mcp --agent-url http://localhost:3000
  
  # 使用通用 HTTP API Agent
  python adaptive_loop.py --agent-type http --agent-url https://api.example.com/chat
  
  # 带错误率的模拟（测试框架鲁棒性）
  python adaptive_loop.py --agent-type imperfect --error-rate 0.3
  
  # 指定 Green Agent 使用远程 API
  python adaptive_loop.py --green-provider openai --green-model gpt-4o-mini
        """
    )
    
    # 测试循环参数
    parser.add_argument("--rounds", "-r", type=int, default=5, help="最大测试轮数")
    parser.add_argument("--initial-per-dim", "-i", type=int, default=10, help="初始每维度用例数")
    parser.add_argument("--targeted-per-weakness", "-t", type=int, default=5, help="每弱点针对性用例数")
    parser.add_argument("--convergence", "-c", type=float, default=0.05, help="收敛阈值")
    parser.add_argument("--agent-name", "-n", default="", help="Agent 名称（用于报告，默认自动生成）")
    parser.add_argument("--output-dir", "-o", default="test_results", help="输出目录")
    
    # Green Agent 参数
    parser.add_argument("--green-provider", default="ollama", 
                        choices=["ollama", "openai", "anthropic", "deepseek", "openrouter"],
                        help="Green Agent API 提供者")
    parser.add_argument("--green-model", default="qwen2.5-coder:7b", help="Green Agent 模型")
    parser.add_argument("--green-base-url", default=None, help="Green Agent API 基础 URL")
    parser.add_argument("--green-api-key", default=None, help="Green Agent API 密钥")
    
    # Baseline Agent 参数
    parser.add_argument("--agent-type", 
                        choices=["mock", "simulated", "imperfect", "ollama", "openai", "anthropic", 
                                 "purple", "a2a", "mcp", "http"],
                        default="mock", help="被测 Agent 类型")
    parser.add_argument("--agent-model", default="qwen2.5:7b", help="被测 Agent 模型（用于 ollama/openai/anthropic）")
    parser.add_argument("--agent-base-url", default=None, help="被测 Agent API 基础 URL")
    parser.add_argument("--agent-url", default=None, help="被测 Agent URL（用于 a2a/mcp/http）")
    parser.add_argument("--agent-api-key", default=None, help="被测 Agent API 密钥")
    parser.add_argument("--error-rate", type=float, default=0.2, help="imperfect 模式的错误率")
    
    args = parser.parse_args()
    
    # 初始化 Green Agent
    print("🟢 初始化 Green Agent...")
    green_agent = GreenAgent(
        model=args.green_model,
        provider=args.green_provider,
        base_url=args.green_base_url,
        api_key=args.green_api_key,
        max_retries=3
    )
    print(f"   Provider: {args.green_provider}, Model: {args.green_model}")
    
    # 初始化环境
    env = SmartHomeEnv()
    
    # 初始化 Baseline Agent
    print("🟣 初始化被测 Agent...")
    
    # 确定 Agent 名称
    agent_name = args.agent_name
    if not agent_name:
        if args.agent_type in ["mock", "simulated"]:
            agent_name = "Simulated Perfect Agent"
        elif args.agent_type == "imperfect":
            agent_name = f"Simulated Agent (error_rate={args.error_rate})"
        elif args.agent_type in ["a2a", "mcp", "http"]:
            agent_name = f"{args.agent_type.upper()} Agent ({args.agent_url})"
        else:
            agent_name = f"{args.agent_type.title()} Agent ({args.agent_model})"
    
    # 构建 Agent 参数
    agent_kwargs = {}
    if args.agent_type in ["ollama", "openai", "anthropic"]:
        agent_kwargs['model'] = args.agent_model
        if args.agent_base_url:
            agent_kwargs['base_url'] = args.agent_base_url
        if args.agent_api_key:
            agent_kwargs['api_key'] = args.agent_api_key
    elif args.agent_type == "a2a":
        if not args.agent_url:
            print("❌ 错误: --agent-url 参数是 A2A 类型必需的")
            return
        agent_kwargs['agent_url'] = args.agent_url
        if args.agent_api_key:
            agent_kwargs['api_key'] = args.agent_api_key
    elif args.agent_type == "mcp":
        if not args.agent_url:
            print("❌ 错误: --agent-url 参数是 MCP 类型必需的")
            return
        agent_kwargs['server_url'] = args.agent_url
    elif args.agent_type == "http":
        if not args.agent_url:
            print("❌ 错误: --agent-url 参数是 HTTP 类型必需的")
            return
        agent_kwargs['url'] = args.agent_url
        if args.agent_api_key:
            agent_kwargs['api_key'] = args.agent_api_key
    elif args.agent_type == "imperfect":
        agent_kwargs['error_rate'] = args.error_rate
    
    baseline_agent = BaselineAgent(env, agent_type=args.agent_type, **agent_kwargs)
    print(f"   Type: {args.agent_type}")
    print(f"   Name: {agent_name}")
    
    # 运行循环
    loop = AdaptiveTestLoop(
        green_agent=green_agent,
        baseline_agent=baseline_agent,
        output_dir=Path(args.output_dir)
    )
    
    report_path = loop.run(
        max_rounds=args.rounds,
        initial_per_dim=args.initial_per_dim,
        targeted_per_weakness=args.targeted_per_weakness,
        convergence_threshold=args.convergence,
        agent_name=agent_name
    )
    
    print(f"\n🎉 测试完成！报告: {report_path}")


if __name__ == "__main__":
    main()
