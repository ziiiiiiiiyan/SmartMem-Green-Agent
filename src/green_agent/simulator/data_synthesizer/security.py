import json
import random
import copy

class SecuritySynthesizer:
    def __init__(self):
        """
        Initialize security system synthesizer.

        State:
            door_lock: "closed" by default
        """
        self.default_state = {
            "door_lock": "closed"
        }
        self.current_state = copy.deepcopy(self.default_state)

        self.schema = {
            "door_lock": ["open", "closed"]
        }

        # Owner leave messages: reason+return time (for NLP downstream hint)
        self.owner_leave_messages = [
            "supermarket+30 minutes",
            "walking the dog+1 hour",
            "gym+2 hours",
            "meeting+3 hours",
            "work+evening",
            "dinner+7 PM",
            "shopping+8 PM",
            "friend's house+tomorrow morning"
        ]

        # Visitor scenarios
        self.visitor_scenarios = [
            "neighbor borrowing something",
            "courier delivery signature required",
            "community survey"
        ]
        self.fixed_visitor_response = "I understand, thank you. I will visit again then."
        self.fixed_owner_query = "Check visitor message"

    def _apply_rules(self, command_params, historical_state):
        """
        Calculate state change.

        Args:
            command_params: Command parameters
            historical_state: Current state

        Returns:
            dict: New state
        """
        new_state = copy.deepcopy(historical_state)
        action = command_params.get("action")

        if action == "update":
            if "door_lock" in command_params:
                new_state["door_lock"] = command_params["door_lock"]

        return new_state

    def generate_batch(self, distribution_config):
        """
        Generate batch of security control data.

        Args:
            distribution_config: Dict with task type counts
                - "toggle_door": Toggle door lock
                - "read": Read door lock state
                - "interaction": Full visitor interaction flow

        Returns:
            list: Generated data points
        """
        dataset = []
        task_queue = []

        for task_type, count in distribution_config.items():
            task_queue.extend([task_type] * count)
        random.shuffle(task_queue)

        for task in task_queue:
            if task == "interaction":
                dataset.extend(self._generate_interaction_flow())
            else:
                dataset.append(self._generate_single_step(task))

        return dataset

    def _generate_single_step(self, task_type):
        """
        Generate single step command.

        Args:
            task_type: "toggle_door" or "read"

        Returns:
            dict: input_command and expected_choices
        """
        command = {}

        # Construct command
        if task_type == "read":
            command["action"] = "read"
            command["targets"] = ["door_lock"]
        elif task_type == "toggle_door":
            target_val = random.choice(self.schema["door_lock"])
            command["action"] = "update"
            command["door_lock"] = target_val

        # Calculate expected outcomes
        valid_outcomes = []

        if command["action"] == "read":
            report_data = {"door_lock": self.current_state["door_lock"]}
            valid_outcomes = [[{"action": "read", "state": report_data}]]

        elif command["action"] == "update":
            if "door_lock" in command:
                predicted_state = self._apply_rules(command, self.current_state)
                is_redundant = (predicted_state["door_lock"] == self.current_state["door_lock"])

                action_update = {"action": "update", "state": predicted_state}
                action_read = {"action": "read", "state": self.current_state}

                if is_redundant:
                    valid_outcomes = [[], [action_read], [action_update]]
                else:
                    valid_outcomes = [[action_update]]
                    self.current_state = predicted_state

        return {
            "input_command": command,
            "expected_choices": valid_outcomes,
            "tag": "door_lock"
        }

    def _generate_interaction_flow(self):
        """
        Generate visitor interaction flow.

        Returns:
            list: 5 steps - owner message, visitor message (triggers relay), visitor response, door open, owner query
        """
        flow_data = []
        owner_msg = random.choice(self.owner_leave_messages)
        visitor_msg = random.choice(self.visitor_scenarios)

        # Step 1: Owner leaves message (NL) - System auto-locks door
        flow_data.append({
            "input_command": {"action": "send_user_msg", "source": "owner", "msg_content": owner_msg},
            "system_action": {"action": "update", "door_lock": "closed"},
            "expected_choices": [],
            "tag": ["door_lock", "memory"]
        })
        # Update internal state to locked
        self.current_state["door_lock"] = "closed"

        # Step 2: Visitor leaves message (NL) - System relays owner message as response
        flow_data.append({
            "input_command": {"action": "send_visitor_msg", "source": "visitor", "msg_content": visitor_msg},
            "expected_choices": [[{
                "action": "update",
                "state": {"intercom_reply": owner_msg}
            }]],
            "tag": ["intercom", "state_tracking"]
        })

        # Step 3: Visitor responds (after hearing owner message)
        flow_data.append({
            "input_command": {"action": "send_visitor_msg", "source": "visitor", "msg_content": self.fixed_visitor_response},
            "expected_choices": [],
            "tag": "intercom"
        })

        # Step 4: Open door
        open_door_cmd = {"action": "update", "door_lock": "open", "note": "The owner comes back."}
        open_state = self._apply_rules(open_door_cmd, self.current_state)
        self.current_state = open_state

        flow_data.append({
            "input_command": open_door_cmd,
            "expected_choices": [[{"action": "update", "state": open_state}]],
            "tag": ["door_lock", "state_tracking"]
        })

        # Step 5: Owner queries (agent replies with visitor message)
        flow_data.append({
            "input_command": {"action": "send_user_msg", "source": "owner", "msg_content": self.fixed_owner_query},
            "expected_choices": [[{
                "action": "agent_reply",
                "state": {"msg_content": visitor_msg}
            }]],
            "tag": ["intercom", "memory"]
        })

        return flow_data

def test():
    """Test function to generate security system data samples"""
    synthesizer = SecuritySynthesizer()

    distribution_config = {
        "toggle_door": 3,     # Toggle door lock
        "read": 2,            # Read door lock state
        "interaction": 1      # Full visitor interaction flow
    }

    # Generate test data
    dataset = synthesizer.generate_batch(distribution_config)

    # Save to test_data/security_dict.json
    import os
    output_path = os.path.join(os.path.dirname(__file__), "test_data/security_dict.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Test data generated: {len(dataset)} entries")
    print(f"Saved to: {output_path}")
    return dataset


if __name__ == "__main__":
    test()