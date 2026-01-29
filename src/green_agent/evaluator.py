from typing import List, Dict, Any, Tuple, Optional, Set
from .base import TestResult, WeaknessProfile, DimensionStats

# --- Constants ---

DIMENSIONS = ["precision", "ambiguous", "conflict", "memory", "noise"]

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

# --- Evaluators ---

class TurnEvaluator:
    """
    Evaluates a single turn (instruction -> action -> state change).
    """
    
    def __init__(self, expected_actions: List[Dict[str, Any]], expected_final_state: Dict[str, Any]):
        """
        Args:
            expected_actions: List of expected API calls (e.g., [{"action": "update", ...}]).
            expected_final_state: Dict of expected state values (e.g., {"light": "on"}).
        """
        self.expected_actions = expected_actions
        self.expected_final_state = expected_final_state
    
    def evaluate(self, actual_actions: List[Dict[str, Any]], actual_final_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares actual execution against expectations.
        
        Returns:
            Dict containing score (1 or 0), status boolean, and error list.
        """
        errors = [] #TODO: 相比于score - 一个模糊分数 - errors应该更适合用来定位弱点
        
        # 1. Sequence Verification
        # FIXME: 这里只检查了操作数量是否相同, 粒度太粗了? 可以定位到具体漏了什么的
        if len(actual_actions) != len(self.expected_actions):
            errors.append(f"Action count mismatch: expected {len(self.expected_actions)}, got {len(actual_actions)}")
        else:
            for i, (exp, act) in enumerate(zip(self.expected_actions, actual_actions)):
                if exp != act:
                    errors.append(f"Action #{i} mismatch: expected {exp}, got {act}")

        # If sequence failed, we can return early or continue based on strictness. 
        # Here we treat sequence failure as score 0.
        sequence_match = len(errors) == 0
        if not sequence_match: #TODO: 统一返回结果
             return self._build_result(0, sequence_match, False, errors, "Sequence mismatch")

        # 2. State Verification
        # Only verify keys present in expected_final_state
        # FIXME: 只检查期望修改的key会遗漏一些错误, 比如误操作了别的设备
        state_match = True
        for key, exp_val in self.expected_final_state.items():
            act_val = actual_final_state.get(key)
            if act_val != exp_val:
                errors.append(f"State mismatch [{key}]: expected '{exp_val}', got '{act_val}'")
                state_match = False
        
        score = 1 if state_match else 0
        message = "Perfect" if score == 1 else "State mismatch"
        
        return self._build_result(score, sequence_match, state_match, errors, message)

    def _build_result(self, score: int, seq_match: bool, state_match: bool, errors: List[str], msg: str) -> Dict[str, Any]:
        """Helper to construct the return dictionary."""
        # TODO: 参考src/agent.py: L225 直接做出全部需要的结果作为result返回
        return {
            "score": score,
            "details": { #TODO: 把expected_actions, expected_final_state以及actual_actions放进去
                "sequence_match": seq_match,
                "state_match": state_match,
                "errors": errors
            },
            "message": msg
        }


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

# class AdaptiveEvaluator:
#     """
#     Acts as a dynamic score tracker.
#     It receives turn results from the Examiner, evaluates them, 
#     and synchronizes global performance metrics in real-time.
#     """
    
#     def __init__(self):
#         # Internal analyzer to track global stats
#         self.analyzer = WeaknessAnalyzer()
#         self.history: List[Dict[str, Any]] = []
    
#     def evaluate_test_case(
#         self, 
#         actual_actions: List[Dict[str, Any]], 
#         actual_state: Dict[str, Any],
#         expected_actions: List[Dict[str, Any]],
#         expected_state: List[Dict[str, Any]],
#     ) -> Dict[str, Any]:
#         """
#         Evaluates a single turn and updates global stats.
        
#         Args:
#             actual_actions: Actions performed by the agent.
#             actual_state: Final state after execution.
#             expected_actions: Expected actions (ground truth).
#             expected_state: Expected final state (ground truth).
#             metadata: Context dict, must include 'dimension', 'difficulty'. 
#                       Can optionally include 'involved_devices'.
        
#         Returns:
#             The evaluation result for this turn.
#         """
#         # 1. Evaluate the specific turn
#         evaluator = TurnEvaluator(expected_actions, expected_final_state)
#         result = evaluator.evaluate(actual_actions, actual_state)
#         # result ds:
#         # {
#         #     "score": score,
#         #     "details": {
#         #         "sequence_match": seq_match,
#         #         "state_match": state_match,
#         #         "errors": errors
#         #     },
#         #     "message": msg
#         # }
        
#         # 2. Enrich result with metadata for the record
#         full_record = {
#             **result,
#             "index": len(self.history)
#         }
#         self.history.append(full_record)
        
#         # 3. Synchronize global performance (Update Weakness Analyzer)
#         # Ensure metadata has involved devices if not provided
#         if 'involved_devices' not in metadata:
#             metadata['involved_devices'] = self._extract_devices(expected_actions, expected_final_state)
            
#         self.analyzer.update(result, metadata)
        
#         return result

#     def get_global_profile(self) -> WeaknessProfile:
#         """Returns the current global weakness profile."""
#         return self.analyzer.get_profile()

#     def _extract_devices(self, actions: List[Dict], state: Dict) -> Set[str]:
#         """Helper to identify devices involved in this turn."""
#         devices = set()
#         for action in actions:
#             if 'key' in action:
#                 devices.add(action['key'])
#         for key in state.keys():
#             devices.add(key)
#         return devices