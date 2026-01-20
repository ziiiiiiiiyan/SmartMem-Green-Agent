"""
SmartMem Environment - v1.2
管理智能家居模拟环境的状态和约束
"""
from typing import Optional, Dict, Any


class SmartHomeEnv:
    def __init__(self):
        # 定义初始/默认状态 (v1.2 - 10 devices)
        self.default_state = {
            "living_room_light": "off",
            "living_room_color": "white",
            "bedroom_light": "off",
            "bedroom_color": "white",
            "ac": "off",
            "ac_temperature": 24,
            "fan_speed": "off",
            "music_volume": 5,
            "front_door_lock": "locked",
            "kitchen_light": "off"
        }
        self.state = self.default_state.copy()

        # 定义合法值约束 (Schema Validation)
        self.constraints = {
            "living_room_light": ["on", "off"],
            "living_room_color": ["white", "red", "blue", "warm"],
            "bedroom_light": ["on", "off"],
            "bedroom_color": ["white", "warm", "blue", "red"],
            "ac": ["on", "off"],
            "fan_speed": ["off", "low", "medium", "high"],
            "front_door_lock": ["locked", "unlocked"],
            "kitchen_light": ["on", "off"]
            # int 类型的约束 (ac_temperature, music_volume) 在逻辑里判断
        }
        
        # 记录当前Turn的所有API请求
        self.action_history = []

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """
        重置回默认状态。
        如果提供了 initial_state，则覆盖指定的键值对。
        
        Args:
            initial_state: 可选，包含要覆盖的初始状态的字典
        
        Returns:
            包含status、message和current_state的响应字典
        """
        # 重置为默认状态
        self.state = self.default_state.copy()
        self.action_history = []
        
        # 如果提供了初始状态，则合并到当前状态中
        if initial_state:
            for key, value in initial_state.items():
                # 只更新存在的 Key
                if key in self.state:
                    self.state[key] = value
        
        return {
            "status": "success", 
            "message": "Environment reset.",
            "current_state": self.state.copy()
        }
    
    def reset_turn_history(self):
        """重置Turn历史（用于开始新的Turn）"""
        self.action_history = []

    def get_state(self, key=None):
        """
        读取状态
        """
        
        # 1. 处理读取全量状态的情况 (key 为 None)
        if key is None:
            return {
                "status": "success",
                "state": self.state.copy(),
                "message": "Full state retrieved",
                "metadata": {
                    "operation_object": "environment" 
                }
            }

        # 2. 处理指定 Key 读取的情况
        # 准备 metadata
        metadata = {
            "operation_object": key
        }

        if key in self.state:
            return {
                "status": "success",
                "value": self.state[key],
                "message": f"Current value of {key} is {self.state[key]}",
                "metadata": metadata
            }
        else:
            return {
                "status": "error",
                "message": f"Key '{key}' not found",
                "metadata": metadata
            }

    def update_state(self, key, value):
        """根据输入操作环境并返回结果"""
        
        metadata = {
            "operation_object": key
        }

        # 1. 校验 Key 是否存在
        if key not in self.state:
            return {
                "status": "error",
                "message": f"Device '{key}' does not exist",
                "metadata": metadata
            }

        # 2. 校验 Enum 类型
        if key in self.constraints:
            if value not in self.constraints[key]:
                return {
                    "status": "error",
                    "message": f"Invalid value '{value}' for {key}. Allowed: {self.constraints[key]}",
                    "metadata": metadata
                }
        
        # 3. 校验 Int 类型 (温度、音量等)
        if key in ["ac_temperature", "music_volume"]:
            try:
                value = int(value) 
            except (ValueError, TypeError):
                return {
                    "status": "error",
                    "message": f"Value for {key} must be a valid integer (received: {value})",
                    "metadata": metadata
                }

            if key == "ac_temperature" and not (16 <= value <= 30):
                return {
                    "status": "error",
                    "message": "Temperature must be between 16 and 30",
                    "metadata": metadata
                }
                
            if key == "music_volume" and not (0 <= value <= 10):
                return {
                    "status": "error",
                    "message": "Volume must be between 0 and 10",
                    "metadata": metadata
                }

        # 4. 执行更新 (成功情况)
        self.state[key] = value
        return {
            "status": "success",
            "current_value": value,
            "message": f"{key} updated to {value}",
            "metadata": metadata
        }
    
    def record_action(self, action_dict: dict):
        """
        记录一个API请求到历史中（用于方案1的序列验证）
        
        Args:
            action_dict: API请求，例如 {"action": "update", "key": "light", "value": "on"}
        """
        self.action_history.append(action_dict)
    
    def get_action_history(self):
        """返回当前Turn的所有API请求历史"""
        return self.action_history.copy()
