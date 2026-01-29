import random
import json
import copy

class LightSystemSynthesizer:
    def __init__(self):
        # 1. 硬件状态
        self.default_state = { "power": "off", "color": "white" }
        self.current_hardware_states = {
            "Bedroom Light": copy.deepcopy(self.default_state),
            "Living Room Light": copy.deepcopy(self.default_state),
            "Kitchen Light": { "power": "off" }
        }

        # 2. 动态偏好 (初始为空，由指令序列填充)
        self.user_preferences = {}

        # 3. 颜色轮转序列（用于自动切换）
        self.color_cycle = ["white", "warm", "blue", "red"]

        # 4. 每个设备的当前颜色索引（用于轮转）
        self.device_color_indices = {
            "Bedroom Light": 0,
            "Living Room Light": 0,
            "Kitchen Light": 0
        }

        # 5. 能力定义
        self.devices = ["Bedroom Light", "Living Room Light", "Kitchen Light"]
        self.capabilities = {
            "Bedroom Light": { "power": ["on", "off"], "color": ["white", "red", "blue", "warm"] },
            "Living Room Light": { "power": ["on", "off"], "color": ["white", "red", "blue", "warm"] },
            "Kitchen Light": { "power": ["on", "off"], "color": None }
        }

        # 6. 知识图谱 (理由 -> 颜色)
        self.reasons_map = {
            "white": ["reading a book", "working from home", "cleaning", "finding items"],
            "warm": ["relaxing", "preparing for bed", "dinner time", "cozy vibes"],
            "blue": ["watching movie", "gaming", "cooling down"],
            "red": ["party mode", "emergency drill", "festival"]
        }

    def _get_random_value(self, device, key):
        values = self.capabilities[device].get(key)
        if values: return random.choice(values)
        return None

    def _infer_color_from_note(self, note):
        """反向查找：理由 -> 颜色"""
        for color, scenarios in self.reasons_map.items():
            if note in scenarios:
                return color
        return None

    def _apply_rules(self, device_name, command_params, historical_state):
        """
        终极规则引擎: Explicit > Fuzzy > Preference > Auto Color Cycle > History
        """
        new_state = copy.deepcopy(historical_state)

        # 提取参数
        target_params = {k: v for k, v in command_params.items()
                         if k not in ["action", "device", "note", "type"]}

        # 1. 识别激活意图
        explicit_color = "color" in target_params
        explicit_power_on = (target_params.get("power") == "on")
        fuzzy_note = command_params.get("note")

        # 只要涉及 显式开灯、显式变色、或者 模糊理由，都视为激活
        is_activating = explicit_power_on or explicit_color or (fuzzy_note is not None)

        # 2. 应用显式参数 (Level 1)
        new_state.update(target_params)

        # 3. 推断颜色 (如果显式参数里没有颜色)
        if is_activating and not explicit_color:
            final_color = None

            # Level 2: 模糊推断
            if fuzzy_note:
                final_color = self._infer_color_from_note(fuzzy_note)

            # Level 3: 偏好 (只有在模糊推断也没结果时才用)
            if not final_color and (device_name in self.user_preferences):
                final_color = self.user_preferences[device_name]

            # Level 4: 自动颜色轮转 (Auto Color Cycle)
            # 开灯 + 无显式颜色 + 无模糊推断 + 无偏好 + 支持颜色
            if not final_color and (device_name not in self.user_preferences):
                if self.capabilities[device_name]["color"] is not None:
                    # 获取轮转的下一个颜色
                    current_idx = self.device_color_indices[device_name]
                    next_color = self.color_cycle[current_idx]
                    final_color = next_color
                    # 更新索引到下一个位置
                    self.device_color_indices[device_name] = (current_idx + 1) % len(self.color_cycle)

            # 应用推断出的颜色 (需检查设备能力)
            if final_color and self.capabilities[device_name]["color"] is not None:
                new_state["color"] = final_color

        # 4. 隐式开机联动
        # 如果颜色变了，或者有模糊理由，强制 Power On
        if "color" in target_params or (fuzzy_note and is_activating):
            new_state["power"] = "on"

        # 5. 如果显式指定了颜色，更新轮转索引到该颜色的下一个位置
        if explicit_color:
            target_color = target_params["color"]
            if target_color in self.color_cycle:
                # 找到目标颜色在轮转序列中的位置，并设置索引为其下一个
                color_idx = self.color_cycle.index(target_color)
                self.device_color_indices[device_name] = (color_idx + 1) % len(self.color_cycle)

        return new_state

    def generate_batch(self, distribution_config):
        """
        Generate a batch of light control data points.

        Args:
            distribution_config: Dict mapping task types to counts

        Possible keys:
            "set_preference": Set user color preference (followed by verify_pref)
                - Generates: set_preference action + verify_pref (power:on only, tests memory)
                - System should infer color from saved preference (tests memory capability)
                - If device is already on, inserts off command first (counts as switch_power)
            "fuzzy_command": Commands with natural language notes (e.g., "reading", "relaxing")
                - System infers color from note using knowledge graph
            "switch_power": Toggle power on/off with auto color cycling
            "read": Query device state (power, color, etc.)
            "change_color": Explicitly change light color

        Example:
            {
                "set_preference": 1,   # 1 preference setup + verify (tests memory inference)
                "fuzzy_command": 2,    # 2 fuzzy commands with notes
                "switch_power": 4,     # 4 power switches with auto cycling
                "read": 1,             # 1 state query
                "change_color": 2      # 2 explicit color changes
            }
        """
        final_sequence = []

        # Build preference pairs (set_preference + verify_pref)
        pref_count = distribution_config.get("set_preference", 0)

        pref_pairs = []
        for _ in range(pref_count):
            color_devices = [d for d in self.devices if self.capabilities[d]["color"] is not None]
            target_device = random.choice(color_devices)
            target_color = self._get_random_value(target_device, "color")

            setup_cmd = {
                "type": "set_preference",
                "device": target_device,
                "color": target_color,
                "tags": ["memory", "context_understanding"]
            }
            # Follow with verify_pref (power:on the same device, color should be inferred from preference)
            verify_cmd = {
                "type": "verify_pref",
                "device": target_device,
                "tags": ["memory", "context_understanding"]
            }
            pref_pairs.append((setup_cmd, verify_cmd))

        # Build other task queue
        other_tasks = []

        for task_type, count in distribution_config.items():
            if task_type == "set_preference":
                continue  # Already handled as pairs
            elif task_type == "fuzzy_command":
                other_tasks.extend([{"type": "fuzzy_command"}] * count)
            elif task_type == "switch_power":
                other_tasks.extend([{"type": "random_switch"}] * count)
            elif task_type in ["read", "change_color"]:
                other_tasks.extend([{"type": task_type}] * count)

        random.shuffle(other_tasks)

        # Insert preference pairs into timeline
        timeline = other_tasks[:]

        for setup, verify in pref_pairs:
            if len(timeline) == 0:
                timeline.append(setup)
                timeline.append(verify)
            else:
                idx = random.randint(0, len(timeline))
                timeline.insert(idx, setup)

                # Check if we need to insert an off command before verify
                target_device = verify["device"]
                # We'll check state during generation, so just insert verify after
                timeline.insert(idx + 1, verify)  # Immediately after

        # Execute generation
        temp_sequence = []
        for task in timeline:
            # Special handling for verify_pref: if device is already on, insert off first
            if isinstance(task, dict) and task.get("type") == "verify_pref":
                target_device = task["device"]
                current_power = self.current_hardware_states[target_device]["power"]

                if current_power == "on":
                    # Insert off command first (consumes from random_switch quota)
                    off_cmd = {"type": "random_switch", "device": target_device, "power": "off"}
                    off_data = self._generate_single_step(off_cmd)
                    if off_data:
                        temp_sequence.append(off_data)

            data_point = self._generate_single_step(task)
            if data_point:
                temp_sequence.append(data_point)

        final_sequence = temp_sequence
                
        return final_sequence

    def _generate_single_step(self, task):
        command = {}
        target_device = ""
        
        # 兼容简单 dict 或纯字符串 key
        task_type = task.get("type", task) if isinstance(task, dict) else task
        
        # --- A. Input Construction ---
        
        if task_type == "set_preference":
            target_device = task["device"]
            pref_color = task["color"]
            command = {
                "action": "set_preference",
                "device": target_device,
                "color": pref_color
            }

        elif task_type == "verify_pref":
            target_device = task["device"]
            # Only specify power, let system infer color from preference
            command = {
                "action": "update",
                "device": target_device,
                "power": "on"
            }

        elif task_type == "fuzzy_command":
            # 1. 选支持颜色的设备
            color_devices = [d for d in self.devices if self.capabilities[d]["color"] is not None]
            target_device = random.choice(color_devices)
            # 2. 随机选一个意图颜色
            intent_color = self._get_random_value(target_device, "color")
            if not intent_color: intent_color = "white"
            # 3. 反查理由
            reason = random.choice(self.reasons_map.get(intent_color, ["unknown"]))
            
            # 构造: 只有 device 和 note
            command = {
                "action": "update",
                "device": target_device,
                "note": reason
            }
        
        elif task_type == "random_switch":
            # If task has pre-defined device and power (e.g., from set_preference), use them
            if isinstance(task, dict) and "device" in task and "power" in task:
                target_device = task["device"]
                target_power = task["power"]
                command = {"action": "update", "device": target_device, "power": target_power}
            else:
                # Otherwise, random toggle
                target_device = random.choice(self.devices)
                curr_pow = self.current_hardware_states[target_device]["power"]
                command = {"action": "update", "device": target_device, "power": "off" if curr_pow=="on" else "on"}

        elif task_type == "read":
            target_device = random.choice(self.devices)
            command = { "action": "read", "device": target_device }
            valid_keys = [k for k in self.capabilities[target_device].keys() if self.capabilities[target_device][k] is not None]
            command["targets"] = random.sample(valid_keys, random.randint(1, len(valid_keys)))
            
        elif task_type == "change_color":
            color_devices = [d for d in self.devices if self.capabilities[d]["color"] is not None]
            target_device = random.choice(color_devices)
            command = {
                "action": "update",
                "device": target_device,
                "color": self._get_random_value(target_device, "color")
            }
            
        else:
            return None # Skip unknown

        # --- B. Expected Outcome ---

        valid_outcomes = []

        if command["action"] == "set_preference":
            # Update Internal Prefs
            self.user_preferences[command["device"]] = command["color"]
            # Silent Action
            valid_outcomes = []

        elif command["action"] == "read":
            state = self.current_hardware_states[command["device"]]
            data = {k: state.get(k) for k in command["targets"]}
            valid_outcomes = [[{"action": "read", "device": command["device"], "state": data}]]

        elif command["action"] == "update":
            current_state = self.current_hardware_states[command["device"]]

            # 【关键】调用包含完整优先级逻辑的规则引擎
            predicted_state = self._apply_rules(command["device"], command, current_state)

            is_redundant = (predicted_state == current_state)

            # Determine which parameters to include in state based on command
            if "note" in command and task_type == "fuzzy_command":
                # Fuzzy command: return all inferred parameters
                state_to_return = predicted_state
            elif task_type == "verify_pref":
                # verify_pref: return all parameters including inferred color from preference
                state_to_return = predicted_state
            elif command.get("power") == "on" and "color" not in command and command["device"] in self.user_preferences:
                # power:on without explicit color but device has preference: return all including color
                state_to_return = predicted_state
            else:
                # Only return parameters that were explicitly in command
                state_to_return = {}
                if "power" in command:
                    state_to_return["power"] = predicted_state.get("power")
                if "color" in command:
                    state_to_return["color"] = predicted_state.get("color")

            action_update = {"action": "update", "device": command["device"], "state": state_to_return}

            if is_redundant:
                valid_outcomes = [[], [{"action": "read", "device": command["device"], "state": state_to_return}], [action_update]]
            else:
                valid_outcomes = [[action_update]]
                self.current_hardware_states[command["device"]] = predicted_state

        # Determine tag based on device type and task type
        device_tag = "color_light" if self.capabilities[target_device]["color"] is not None else "simple_light"

        # Use pre-defined tags if available, otherwise generate based on task type
        if isinstance(task, dict) and "tags" in task:
            tags = [device_tag] + task["tags"]
        else:
            tags = [device_tag]
            # Add context understanding tag for fuzzy commands
            if task_type == "fuzzy_command":
                tags.append("context_understanding")

        return {
            "input_command": command,
            "expected_choices": valid_outcomes,
            "tag": tags
        }


def test():
    """Test function to generate light control data samples"""
    synthesizer = LightSystemSynthesizer()

    # Configure: 3 samples for each key type
    distribution_config = {
        "set_preference": 2,  # Set preferences + verify (tests memory: color inferred from pref)
        "fuzzy_command": 3,   # Commands with natural language notes
        "switch_power": 5,    # Power toggle (may be consumed by verify_pref off commands)
        "read": 3,            # State queries
        "change_color": 3     # Explicit color changes
    }

    # Generate test data
    dataset = synthesizer.generate_batch(distribution_config)

    # Save to test_data/lights_dict.json
    import os
    output_path = os.path.join(os.path.dirname(__file__), "test_data/lights_dict.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Test data generated: {len(dataset)} entries")
    print(f"Saved to: {output_path}")
    return dataset


if __name__ == "__main__":
    test()
