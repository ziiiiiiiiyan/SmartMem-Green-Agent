"""
Unified Instruction Synthesizer

This module integrates all individual synthesizers to create a complete virtual environment
instruction synthesizer.

Main Features:
1. Accepts a dictionary of synthesis tasks and their counts as input
2. Calls individual synthesizers to generate instruction sets
3. Maintains relative order within each synthesizer's output
4. Randomly merges instruction sets to create more natural sequences

Special Handling:
- security interaction_flow tasks: During this period the user is away, so no other
  instructions can be inserted within this sequence
- time_advance actions: System actions that are inserted after AC timer commands
"""

import random
import json
import os
from typing import Dict, List, Any
from .ac import ACSynthesizer
from .lights import LightSystemSynthesizer
from .speaker import SpeakerSynthesizer
from .security import SecuritySynthesizer


class UnifiedSynthesizer:
    """
    Unified Instruction Synthesizer

    Integrates all device instruction synthesizers to generate complete virtual
    environment instruction sequences.
    """

    # Daily conversation topics for casual chat
    DAILY_CHAT_TOPICS = [
        "books",
        "life tips",
        "AI knowledge concepts",
        "sports and health"
    ]

    def __init__(self):
        """
        Initialize Unified Instruction Synthesizer.

        Initializes all sub-synthesizers (AC, lights, speaker, security).
        """
        self.ac_synthesizer = ACSynthesizer()
        self.lights_synthesizer = LightSystemSynthesizer()
        self.speaker_synthesizer = SpeakerSynthesizer()
        self.security_synthesizer = SecuritySynthesizer()

        # Task configuration template with detailed descriptions
        self.task_config_template = {
            "ac": {
                "tasks": {
                    "read": "Query current AC state (power, mode, temperature, fan speed, sleep mode, timer)",
                    "switch_power": "Toggle AC power on/off",
                    "change_mode": "Change AC mode (cooling, heating, dehumidify) with logic constraints",
                    "toggle_sleep": "Toggle sleep mode on/off",
                    "change_temp": "Change temperature (16-30°C)",
                    "change_fan": "Change fan speed (auto, 1, 2, 3)",
                    "set_timer": "Set timer (0-10 hours, 0.5 hour increments)",
                    "mixed_complex": "Complex multi-parameter changes (random mix of above operations)"
                },
                "description": "Air Conditioner Control Tasks"
            },
            "lights": {
                "tasks": {
                    "set_preference": "Set user color preference for a light device (followed by verify_pref)",
                    "fuzzy_command": "Commands with natural language context (e.g., 'reading', 'relaxing') "
                                    "that infer appropriate light color",
                    "switch_power": "Toggle light power on/off with auto color cycling",
                    "read": "Query light state (power, color)",
                    "change_color": "Explicitly change light color"
                },
                "description": "Smart Light Control Tasks - Supports memory, context understanding, "
                             "and multi-device management"
            },
            "speaker": {
                "tasks": {
                    "update": "Adjust speaker volume (0-10)",
                    "read": "Query current speaker volume"
                },
                "description": "Speaker Control Tasks"
            },
            "security": {
                "tasks": {
                    "toggle_door": "Toggle door lock state (open/closed)",
                    "read": "Query current door lock state",
                    "interaction": "Full visitor interaction flow (5 continuous steps): "
                                 "1. Owner leave message -> auto-lock, "
                                 "2. Visitor message -> relay owner message, "
                                 "3. Visitor response, "
                                 "4. Owner returns -> open door, "
                                 "5. Owner queries visitor message. "
                                 "NOTE: This sequence cannot be interrupted by other instructions!"
                },
                "description": "Security System Control Tasks - Supports door lock control and "
                             "visitor interaction workflows"
            },
            "daily_chat": {
                "tasks": {
                    "chat": "Daily casual conversation about various topics (books, concepts, hobbies)"
                },
                "description": "Daily Conversation Tasks - Non-control commands for natural conversation",
                "parameters": {
                    "concise": "Number of concise (brief) chat instances",
                    "detailed_multi_round": "Number of detailed multi-round chat instances",
                    "detailed_single_turn": "Number of detailed single-turn chat instances"
                }
            }
        }

    def print_available_tasks(self):
        """Print all available task types with detailed descriptions"""
        print("=" * 80)
        print("AVAILABLE TASK TYPES")
        print("=" * 80)

        for device, config in self.task_config_template.items():
            print(f"\n[{device.upper()}]")
            print(f"Description: {config['description']}")
            print(f"Available Tasks:")
            for task, desc in config.get("tasks", {}).items():
                print(f"  - {task}: {desc}")

            # Print additional parameters if available
            if "parameters" in config:
                print(f"Parameters:")
                for param, desc in config["parameters"].items():
                    print(f"  - {param}: {desc}")

        print("\n" + "=" * 80)
        print("CONFIGURATION EXAMPLE")
        print("=" * 80)
        print("""
task_config = {
    "ac": {
        "read": 5,              # Query AC state 5 times
        "switch_power": 5,      # Toggle power 5 times
        "change_temp": 3        # Change temperature 3 times
    },
    "lights": {
        "set_preference": 2,    # Set color preference 2 times
        "fuzzy_command": 3,     # Fuzzy context commands 3 times
        "switch_power": 4       # Toggle lights 4 times
    },
    "speaker": {
        "update": 5,            # Adjust volume 5 times
        "read": 2               # Query volume 2 times
    },
    "security": {
        "toggle_door": 3,       # Toggle door 3 times
        "read": 2,              # Query door state 2 times
        "interaction": 1        # 1 visitor interaction flow (5 continuous steps)
    },
    "daily_chat": {
        "concise": 3,               # 3 concise chat instances
        "detailed_multi_round": 2,  # 2 detailed multi-round chat instances
        "detailed_single_turn": 1   # 1 detailed single-turn chat instance
    }
}
        """)

    def _generate_daily_chat(self, concise: int, detailed_multi_round: int, detailed_single_turn: int) -> List[Dict[str, Any]]:
        """
        Generate daily conversation instructions

        Args:
            concise: Number of concise (brief) chat instances
            detailed_multi_round: Number of detailed multi-round chat instances
            detailed_single_turn: Number of detailed single-turn chat instances

        Returns:
            List of chat instruction dictionaries
        """
        chat_data = []

        # Generate concise chats
        for _ in range(concise):
            topic = random.choice(self.DAILY_CHAT_TOPICS)
            command = {
                "action": "chat",
                "topic": topic,
                "is_concise": True,
                "multi_round": False
            }
            chat_data.append({
                "input_command": command,
                "expected_choices": [],
                "tag": "daily_chat"
            })

        # Generate detailed multi-round chats
        for _ in range(detailed_multi_round):
            topic = random.choice(self.DAILY_CHAT_TOPICS)
            command = {
                "action": "chat",
                "topic": topic,
                "is_concise": False,
                "multi_round": True
            }
            chat_data.append({
                "input_command": command,
                "expected_choices": [[{"action": "agent_reply", "msg_content": "answer related to the provided topic." }]],
                "tag": "daily_chat"
            })

        # Generate detailed single-turn chats
        for _ in range(detailed_single_turn):
            topic = random.choice(self.DAILY_CHAT_TOPICS)
            command = {
                "action": "chat",
                "topic": topic,
                "is_concise": False,
                "multi_round": False
            }
            chat_data.append({
                "input_command": command,
                "expected_choices": [],
                "tag": "daily_chat"
            })

        return chat_data

    def generate_batch(self, task_config: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate complete instruction sequence

        Args:
            task_config: Task configuration dictionary with the following structure:
                {
                    "ac": {
                        "read": 5,                    # Number of times to execute this task
                        "switch_power": 3,
                        ...
                    },
                    "lights": {
                        "set_preference": 2,
                        "fuzzy_command": 3,
                        ...
                    },
                    "speaker": {
                        "update": 5,
                        "read": 2,
                        ...
                    },
                    "security": {
                        "toggle_door": 3,
                        "read": 2,
                        "interaction": 1,            # Generates 5 continuous steps
                        ...
                    },
                    "daily_chat": {
                        "concise": 3,               # Number of concise chat instances
                        "detailed_multi_round": 2,  # Number of detailed multi-round chat instances
                        "detailed_single_turn": 1   # Number of detailed single-turn chat instances
                    }
                }

        Available task types for each device:

                AC (Air Conditioner):
                    - read: Query current state
                    - switch_power: Toggle power on/off
                    - change_mode: Change mode (cooling/heating/dehumidify)
                    - toggle_sleep: Toggle sleep mode
                    - change_temp: Change temperature
                    - change_fan: Change fan speed
                    - set_timer: Set timer
                    - mixed_complex: Complex multi-parameter changes

                Lights:
                    - set_preference: Set color preference
                    - fuzzy_command: Natural language context commands
                    - switch_power: Toggle power
                    - read: Query state
                    - change_color: Change color

                Speaker:
                    - update: Adjust volume
                    - read: Query volume

                Security:
                    - toggle_door: Toggle door lock
                    - read: Query door state
                    - interaction: Visitor interaction flow (5 continuous steps)

                Daily Chat:
                    - concise: Number of concise (brief) chat instances
                    - detailed_multi_round: Number of detailed multi-round chat instances
                    - detailed_single_turn: Number of detailed single-turn chat instances

        Returns:
            List[Dict]: Merged instruction sequence, each containing:
                - input_command: Input instruction
                - expected_choices: Expected output options
                - tag: Tags for categorization
        """
        # 1. Collect all synthesizer outputs
        all_sequences = []
        interaction_blocks = []  # Store interaction flows separately (non-interruptible)

        for device_type, device_tasks in task_config.items():
            if not device_tasks:  # Skip empty config
                continue

            if device_type == "ac":
                sequence = self.ac_synthesizer.generate_batch(device_tasks)
                all_sequences.append({"type": "ac", "data": sequence})

            elif device_type == "lights":
                sequence = self.lights_synthesizer.generate_batch(device_tasks)
                all_sequences.append({"type": "lights", "data": sequence})

            elif device_type == "speaker":
                sequence = self.speaker_synthesizer.generate_batch(device_tasks)
                all_sequences.append({"type": "speaker", "data": sequence})

            elif device_type == "security":
                # Special handling: interaction tasks must remain continuous
                interaction_count = device_tasks.get("interaction", 0)
                other_tasks = {k: v for k, v in device_tasks.items()
                              if k != "interaction" and v > 0}

                # Generate interaction flows
                if interaction_count > 0:
                    for _ in range(interaction_count):
                        interaction_flow = self.security_synthesizer._generate_interaction_flow()
                        interaction_blocks.append({
                            "type": "security_interaction",
                            "data": interaction_flow
                        })

                # Generate other security tasks
                if other_tasks:
                    sequence = self.security_synthesizer.generate_batch(other_tasks)
                    if sequence:
                        all_sequences.append({"type": "security", "data": sequence})

            elif device_type == "daily_chat":
                # Generate daily chat instructions
                concise = device_tasks.get("concise", 0)
                detailed_multi_round = device_tasks.get("detailed_multi_round", 0)
                detailed_single_turn = device_tasks.get("detailed_single_turn", 0)

                total = concise + detailed_multi_round + detailed_single_turn
                if total > 0:
                    chat_sequence = self._generate_daily_chat(concise, detailed_multi_round, detailed_single_turn)
                    all_sequences.append({"type": "daily_chat", "data": chat_sequence})

        # 2. Merge sequences (maintain relative order within each synthesizer)
        merged_sequence = self._merge_sequences(all_sequences, interaction_blocks)

        # 3. Insert time advance actions after AC timer instructions
        merged_sequence = self._insert_time_advances(merged_sequence)

        return merged_sequence

    def _merge_sequences(self,
                        all_sequences: List[Dict[str, Any]],
                        interaction_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge multiple sequences while maintaining relative order within each sequence

        Algorithm:
        1. Treat each sequence as a unit, maintaining internal order
        2. Use shuffle-like algorithm to randomly select elements from sequences
        3. Insert interaction_blocks as non-divisible units

        Args:
            all_sequences: List of all regular sequences
            interaction_blocks: List of interaction flow blocks

        Returns:
            List[Dict]: Merged sequence
        """
        result = []

        # 1. Store all sequence data with current index
        sequences_with_index = []
        for seq_info in all_sequences:
            if seq_info["data"]:  # Non-empty sequences only
                sequences_with_index.append({
                    "type": seq_info["type"],
                    "data": seq_info["data"],
                    "index": 0,
                    "total": len(seq_info["data"])
                })

        # 2. Randomly merge sequences (maintaining internal order)
        # Shuffle-like algorithm: randomly select from remaining sequences, take current element
        remaining_indices = list(range(len(sequences_with_index)))

        while remaining_indices:
            # Randomly select a sequence
            seq_idx = random.choice(remaining_indices)
            seq = sequences_with_index[seq_idx]

            # Take current element
            result.append(seq["data"][seq["index"]])
            seq["index"] += 1

            # Remove from remaining list if sequence exhausted
            if seq["index"] >= seq["total"]:
                remaining_indices.remove(seq_idx)

        # 3. Insert interaction blocks at random positions (maintaining block integrity)
        if interaction_blocks:
            for block in interaction_blocks:
                # Random insertion position
                insert_pos = random.randint(0, len(result))
                # Insert entire block
                for item in block["data"]:
                    result.insert(insert_pos, item)
                    insert_pos += 1

        # 4. Redistribute daily_chat items to ensure they're spread throughout the sequence
        result = self._redistribute_daily_chat(result)

        return result

    def _redistribute_daily_chat(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Redistribute daily_chat items throughout the sequence to ensure even distribution.

        Extracts all daily_chat items and re-inserts them at random positions,
        preventing them from clustering together.

        Args:
            sequence: Current instruction sequence

        Returns:
            List[Dict]: Sequence with daily_chat items redistributed
        """
        # Extract all daily_chat items
        daily_chat_items = []
        non_daily_items = []
        tag_list = []

        for item in sequence:
            tag = item.get("tag")
            if isinstance(tag, list):
                tag = tag[0] if tag else "unknown"

            if tag == "daily_chat":
                daily_chat_items.append(item)
            else:
                non_daily_items.append(item)
                tag_list.append(tag)

        if not daily_chat_items:
            return sequence

        # Randomly insert daily_chat items into the sequence
        # Strategy: divide sequence into segments and insert items in each segment
        segment_size = max(len(non_daily_items) // (len(daily_chat_items) + 1), 1)

        for i, chat_item in enumerate(daily_chat_items):
            # Calculate insertion range (after current position or at beginning)
            min_pos = i * segment_size
            max_pos = min((i + 1) * segment_size, len(non_daily_items))
            insert_pos = random.randint(min_pos, max_pos) if max_pos > min_pos else min_pos

            # Insert at calculated position
            non_daily_items.insert(insert_pos, chat_item)

        return non_daily_items

    def _insert_time_advances(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Insert time advance actions after AC timer instructions.

        For each AC instruction that sets a timer, insert a time advance action
        at a random position after it (not necessarily immediately after).

        Args:
            sequence: Original instruction sequence

        Returns:
            List[Dict]: Sequence with time advance actions inserted
        """
        result = sequence[:]
        insertions = []  # List of (position, time_advance_data) tuples

        # Find all AC timer instructions
        for i, item in enumerate(sequence):
            input_cmd = item.get("input_command", {})
            # Check if this is an AC instruction with timer
            if isinstance(input_cmd, dict) and input_cmd.get("action") == "update":
                if "timer" in input_cmd and input_cmd["timer"] > 0:
                    timer_duration = input_cmd["timer"]
                    # Create time advance action
                    time_advance = {
                        "input_command": {
                            "action": "time_advance",
                            "duration": timer_duration
                        },
                        "expected_choices": [[]],  # Time advance has no visible effect
                        "tag": "time_advance"
                    }
                    insertions.append((i, time_advance))

        # Insert time advances at random positions after their timers
        # Process in reverse order to maintain correct positions
        for timer_idx, time_advance in reversed(insertions):
            # Find valid insertion range (after the timer, before end)
            min_pos = timer_idx + 1
            max_pos = len(result)

            if min_pos < max_pos:
                # Random position between (timer_idx + 1) and end
                insert_pos = random.randint(min_pos, max_pos)
                result.insert(insert_pos, time_advance)

        return result

    def save_to_file(self, sequence: List[Dict[str, Any]], output_path: str):
        """
        Save sequence to file

        Args:
            sequence: Instruction sequence
            output_path: Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sequence, f, ensure_ascii=False, indent=2)

        print(f"Sequence saved to: {output_path}")
        print(f"Total instructions generated: {len(sequence)}")


def test():
    """Test function"""
    synthesizer = UnifiedSynthesizer()

    # Print available task types
    synthesizer.print_available_tasks()

    # Configure tasks
    task_config = {
        "ac": {
            "read": 2,
            "switch_power": 2,
            "change_mode": 1,
            "toggle_sleep": 1,
            "change_temp": 1,
            "change_fan": 1,
            "set_timer": 1,
            "mixed_complex": 1
        },
        "lights": {
            "set_preference": 1,
            "fuzzy_command": 2,
            "switch_power": 2,
            "read": 1,
            "change_color": 1
        },
        "speaker": {
            "update": 2,
            "read": 1
        },
        "security": {
            "toggle_door": 1,
            "read": 1,
            "interaction": 1
        },
        "daily_chat": {
            "concise": 2,
            "detailed_multi_round": 1,
            "detailed_single_turn": 1
        }
    }

    print("\n" + "=" * 80)
    print("GENERATING INSTRUCTION SEQUENCE...")
    print("=" * 80)

    # Generate sequence
    sequence = synthesizer.generate_batch(task_config)

    print(f"\nGeneration complete! Total instructions: {len(sequence)}")

    # Print first 5 as examples
    print("\n" + "=" * 80)
    print("FIRST 5 INSTRUCTIONS:")
    print("=" * 80)
    for i, item in enumerate(sequence[:5], 1):
        print(f"\n[Instruction {i}]")
        print(f"  Input: {json.dumps(item['input_command'], ensure_ascii=False)}")
        print(f"  Tag: {item['tag']}")

    # Statistics by device type
    print("\n" + "=" * 80)
    print("INSTRUCTION STATISTICS:")
    print("=" * 80)
    tag_count = {}
    for item in sequence:
        tag = item.get("tag", "unknown")
        if isinstance(tag, list):
            tag = tag[0] if tag else "unknown"
        tag_count[tag] = tag_count.get(tag, 0) + 1

    for tag, count in sorted(tag_count.items()):
        print(f"  {tag}: {count} instructions")

    # Save to file
    import os
    output_path = os.path.join(os.path.dirname(__file__), "test_data/unified_sequence.json")
    synthesizer.save_to_file(sequence, output_path)

    return sequence


if __name__ == "__main__":
    test()
