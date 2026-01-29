import random
import json
import copy

class ACSynthesizer:
    def __init__(self):
        # 1. 初始状态
        self.default_state = {
            "power": "off",
            "mode": "cooling",
            "fan_speed": "auto",
            "sleep_mode": "off",
            "temperature": 26,
            "timer": 0.0
        }
        self.current_state = copy.deepcopy(self.default_state)
        
        # 2. 合法值定义
        self.schema = {
            "power": ["on", "off"],
            "mode": ["cooling", "heating", "dehumidify"],
            "fan_speed": ["auto", "1", "2", "3"],
            "sleep_mode": ["on", "off"],
            "temperature": list(range(16, 31)),
            "timer": [x * 0.5 for x in range(0, 11)]
        }
        self.all_attribute_keys = list(self.schema.keys())
        
        # 3. 模式转换逻辑约束 (避免逻辑冲突)
        # Key: 当前模式 -> Value: 允许切换到的目标模式列表
        self.mode_transitions = {
            "cooling": ["cooling", "dehumidify"],      # 制冷时不切制热
            "dehumidify": ["dehumidify", "cooling"],   # 除湿可视作弱制冷，可切回制冷
            "heating": ["heating"]                     # 制热时不切制冷/除湿 (防止热胀冷缩/逻辑冲突)
        }

    def _get_random_value(self, key):
        return random.choice(self.schema[key])

    def _get_valid_next_mode(self, current_mode):
        """根据当前模式，返回符合逻辑的下一模式建议"""
        allowed = self.mode_transitions.get(current_mode, self.schema["mode"])
        return random.choice(allowed)

    def _apply_rules(self, command_params, historical_state):
        """隐式开机、参数覆盖"""
        new_state = copy.deepcopy(historical_state)
        target_params = {k: v for k, v in command_params.items() if k != "action"}
        
        # 规则: 任何模式/风速/温度/定时的设置都会触发开机
        trigger_on_keys = ["mode", "fan_speed", "temperature", "timer"]
        implicit_power_on = any(k in target_params for k in trigger_on_keys)
        
        new_state.update(target_params)
        
        if implicit_power_on:
            new_state["power"] = "on"
            
        return new_state

    def generate_batch(self, distribution_config):
        """
        :param distribution_config: Dict, e.g. {"toggle_sleep": 2, "change_mode": 3...}
        """
        dataset = []
        task_queue = []
        for task_type, count in distribution_config.items():
            task_queue.extend([task_type] * count)
        random.shuffle(task_queue)
        
        for task in task_queue:
            data_point = self._generate_single_step(task)
            dataset.append(data_point)
            
        return dataset

    def _generate_single_step(self, task_type):
        command = {}
        
        # --- A. 构建指令 (Input) ---
        if task_type == "read":
            command["action"] = "read"
            num_targets = random.randint(1, len(self.all_attribute_keys))
            targets = random.sample(self.all_attribute_keys, num_targets)
            command["targets"] = targets
        else:
            command["action"] = "update"
            
            # --- 具体的参数生成逻辑 ---
            if task_type == "switch_power":
                # 反转电源状态
                next_val = "off" if self.current_state["power"] == "on" else "on"
                command["power"] = next_val
                
            elif task_type == "change_mode":
                current_mode = self.current_state["mode"]
                command["mode"] = self._get_valid_next_mode(current_mode)
                
            elif task_type == "toggle_sleep":
                next_sleep = "off" if self.current_state["sleep_mode"] == "on" else "on"
                command["sleep_mode"] = next_sleep
                
            elif task_type == "change_temp":
                command["temperature"] = self._get_random_value("temperature")
                
            elif task_type == "change_fan":
                command["fan_speed"] = self._get_random_value("fan_speed")
                
            elif task_type == "set_timer":
                command["timer"] = self._get_random_value("timer")
                
            elif task_type == "mixed_complex":
                if random.random() < 0.5:
                    command["power"] = "off"
                else:                    
                    # 1. 模式 (带防冲突)
                    current_mode = self.current_state["mode"]
                    command["mode"] = self._get_valid_next_mode(current_mode)
                    
                    # 2. 温度
                    command["temperature"] = self._get_random_value("temperature")
                    
                    # 3. 风速 
                    command["fan_speed"] = self._get_random_value("fan_speed")
                    
                    # 4. 定时 
                    command["timer"] = self._get_random_value("timer")
                    
                    # 5. 睡眠 (随机携带)
                    if random.random() > 0.5:
                        command["sleep_mode"] = self._get_random_value("sleep_mode")

        # --- B. 计算期望 (Expected Output) ---
        valid_outcomes = []

        if command["action"] == "read":
            target_keys = command["targets"]
            report_data = {k: self.current_state[k] for k in target_keys}

            valid_outcomes = [[{"action": "read", "state": report_data}]]
        else:
            predicted_state = self._apply_rules(command, self.current_state)
            is_redundant = (predicted_state == self.current_state)

            action_update = {"action": "update", "state": predicted_state}
            action_read = {"action": "read", "state": self.current_state}

            if is_redundant:
                # 冗余操作：允许 空、读、强制更新
                valid_outcomes = [[], [action_read], [action_update]]
            else:
                # 有效操作
                valid_outcomes = [[action_update]]
                self.current_state = predicted_state # 更新内部状态

        return {
            "input_command": command,
            "expected_choices": valid_outcomes,
            "tag": "ac"
        }


def test():
    synthesizer = ACSynthesizer()

    distribution_config = {
        "read": 5,
        "switch_power": 5,      # 电源
        "change_mode": 5,       # 模式
        "toggle_sleep": 5,      # 睡眠模式
        "change_temp": 5,       # 温度
        "change_fan": 5,        # 风速
        "set_timer": 5,         # 定时
        "mixed_complex": 3      # 混合
    }

    # 生成测试数据
    dataset = synthesizer.generate_batch(distribution_config)

    # 保存到 test_data/ac_dict.json
    output_path = __file__.rsplit("/", 1)[0] + "/test_data/ac_dict.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"测试数据已生成，共 {len(dataset)} 条，保存到: {output_path}")
    return dataset

if __name__ == "__main__":
    test()