import random
import json
import copy

class SpeakerSynthesizer:
    def __init__(self):
        # 1. 硬件状态初始化
        self.default_state = {"volume": 5}
        self.current_state = copy.deepcopy(self.default_state)
        self.device_name = "Speaker"
        
        # 2. 硬件能力约束
        self.valid_range = list(range(11)) # 0-10

    def generate_batch(self, distribution_config):
        """
        :param distribution_config: e.g. {"update": 5, "read": 2}
        """
        dataset = []
        task_queue = []
        
        # 展开任务队列
        for task, count in distribution_config.items():
            task_queue.extend([task] * count)
        random.shuffle(task_queue)
        
        # 逐个生成
        for task in task_queue:
            data_point = self._generate_single_step(task)
            if data_point:
                dataset.append(data_point)
                
        return dataset

    def _generate_single_step(self, task_type):
        command = {}
        
        # --- A. 构建输入指令 (Input) ---
        
        if task_type == "read":
            command = {
                "action": "read",
                "device": self.device_name,
                "targets": ["volume"]
            }
            
        elif task_type == "update":
            target_vol = random.choice(self.valid_range)
            
            command = {
                "action": "update",
                "device": self.device_name,
                "volume": target_vol
            }
            
        else:
            return None

        # --- B. 计算预期输出 (Output) ---

        valid_outcomes = []

        if command["action"] == "read":
            # 读取当前状态
            valid_outcomes = [[{
                "action": "read",
                "device": self.device_name,
                "state": self.current_state
            }]]

        elif command["action"] == "update":
            # 1. 预测新状态
            predicted_state = copy.deepcopy(self.current_state)
            predicted_state["volume"] = command["volume"]

            # 2. 冗余检测 (Target == Current ?)
            is_redundant = (predicted_state["volume"] == self.current_state["volume"])

            # 3. 标准成功回包
            action_success = {
                "action": "update",
                "device": self.device_name,
                "state": predicted_state
            }

            if is_redundant:
                # 冗余时的三种合理表现：
                # 1. 无动作 (空列表)
                # 2. 仅上报状态 (告诉用户本来就是这个值)
                # 3. 强制上报成功 (Ack)
                valid_outcomes = [
                    [],
                    [{"action": "read", "device": self.device_name, "state": predicted_state}],
                    [action_success]
                ]
            else:
                # 非冗余，必须成功
                valid_outcomes = [[action_success]]
                # 更新内部状态
                self.current_state = predicted_state

        return {
            "input_command": command,
            "expected_choices": valid_outcomes,
            "tag": "speaker"
        }


def test():
    """Test function to generate speaker control data samples"""
    synthesizer = SpeakerSynthesizer()

    # Configure: samples for each key type
    distribution_config = {
        "update": 5,   # Volume adjustments
        "read": 3      # State queries
    }

    # Generate test data
    dataset = synthesizer.generate_batch(distribution_config)

    # Save to test_data/speaker_dict.json
    import os
    output_path = os.path.join(os.path.dirname(__file__), "test_data/speaker_dict.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Test data generated: {len(dataset)} entries")
    print(f"Saved to: {output_path}")
    return dataset


if __name__ == "__main__":
    test()