"""
Dataset Generator for Smart Home Test Cases

Generates 15 test cases across 3 difficulty levels by calling UnifiedSynthesizer.
Each test case includes metadata about turn counts, tag distribution, task composition,
and model information.
"""

import os
import json
import random
import copy
import asyncio
import argparse
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .unified_synthesizer import UnifiedSynthesizer
from .prompts import get_unified_prompt


class DatasetGenerator:
    """
    Generate test case datasets with varying difficulty levels.

    Test cases are organized into 3 levels:
    - Level 1 (Easy): Base tasks + simple tasks only
    - Level 2 (Medium): Level 1 + 3-5 non-simple tasks per test case
    - Level 3 (Hard): Level 2 + 5-10 additional non-simple tasks per test case

    Simple tasks (marked in requirements):
        - AC: all tasks (entire category marked as simple)
        - Lights: read, change_color
        - Speaker: update, read (entire category marked as simple)
        - Security: toggle_door, read
    Non-simple tasks:
        - Lights: set_preference, fuzzy_command, switch_power
        - Security: interaction
        - Daily Chat: all chat tasks (concise, detailed_multi_round, detailed_single_turn)
    """

    # Simple task sets based on requirements
    # Note: lights.switch_power is simple ONLY when there's no set_preference in the same test case
    SIMPLE_TASKS = {
        "ac": {"read", "switch_power", "change_mode", "toggle_sleep", "change_temp", "change_fan", "set_timer", "mixed_complex"},
        "lights": {"read", "change_color", "switch_power"},  # switch_power is simple by default
        "speaker": {"update", "read"},
        "security": {"toggle_door", "read"}
    }

    # Non-simple tasks
    NON_SIMPLE_TASKS = {
        "lights": {"set_preference", "fuzzy_command"},
        "security": {"interaction"},
        "daily_chat": {"concise", "detailed_multi_round", "detailed_single_turn"}
    }

    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize dataset generator.

        Args:
            model_name: Model used for NLP conversion (for metadata)
        """
        self.synthesizer = UnifiedSynthesizer()
        self.model_name = model_name

        # Initialize async LLM client for NLP conversion
        self.llm_client = self._get_llm_client()
        self.model_gen_args = json.loads(os.getenv("MODEL_GEN_ARGS", "{}"))

        # Store conversion history for context
        self.conversion_history = []  # List of (input_command, natural_language) tuples

    def _get_llm_client(self):
        """Get async OpenAI client configured from environment variables."""
        load_dotenv()

        return AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    async def generate_dataset(self, output_path: str) -> Dict[str, Any]:
        """
        Generate complete dataset with 15 test cases across 3 difficulty levels.

        Args:
            output_path: Path to save the generated dataset JSON file

        Returns:
            Dict with dataset metadata including:
                - total_test_cases: Total number of test cases
                - total_turns: Total number of turns across all test cases
                - difficulty_levels: Info about each level
                - test_cases: List of test case data with metadata
        """
        dataset = {
            "metadata": {
                "model_name": self.model_name,
                "total_test_cases": 15,
                "difficulty_levels": {
                    "level_1_easy": {
                        "description": "Base tasks + simple tasks only",
                        "test_case_range": [0, 4]  # 5 test cases
                    },
                    "level_2_medium": {
                        "description": "Level 1 + 3-5 non-simple tasks per test case",
                        "test_case_range": [5, 9]  # 5 test cases
                    },
                    "level_3_hard": {
                        "description": "Level 2 + 5-10 additional non-simple tasks per test case",
                        "test_case_range": [10, 14]  # 5 test cases
                    }
                }
            },
            "test_cases": []
        }

        # Generate test cases for each level
        # Level 1: Easy (test cases 0-4)
        level_1_test_cases = self._generate_level_1()
        dataset["test_cases"].extend(level_1_test_cases)

        # Level 2: Medium (test cases 5-9) - based on Level 1
        level_2_test_cases = self._generate_level_2(level_1_test_cases)
        dataset["test_cases"].extend(level_2_test_cases)

        # Level 3: Hard (test cases 10-14) - based on Level 2
        level_3_test_cases = self._generate_level_3(level_2_test_cases)
        dataset["test_cases"].extend(level_3_test_cases)

        # Batch convert all test cases to natural language (parallel processing)
        print("\n" + "=" * 80)
        print("CONVERTING COMMANDS TO NATURAL LANGUAGE (async batch processing)...")
        print("=" * 80)
        dataset["test_cases"] = await asyncio.gather(
            *[self._add_command_text_to_test_case(tc) for tc in dataset["test_cases"]]
        )

        # Calculate overall statistics
        total_turns = sum(tc["metadata"]["turn_count"] for tc in dataset["test_cases"])
        dataset["metadata"]["total_turns"] = total_turns

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"Dataset generated: {len(dataset['test_cases'])} test cases, {total_turns} total turns")
        print(f"Saved to: {output_path}")

        return dataset

    def _generate_level_1(self) -> List[Dict[str, Any]]:
        """
        Generate Level 1 (Easy) test cases (5 test cases).

        Base tasks (one of each) + exactly 3 random simple tasks.
        """
        test_cases = []

        # Base config: all tasks once each
        base_config = {
            "ac": {
                "read": 1,
                "switch_power": 1,
                "change_mode": 1,
                "toggle_sleep": 1,
                "change_temp": 1,
                "change_fan": 1,
                "set_timer": 1,
                "mixed_complex": 1
            },
            "lights": {
                "set_preference": 1,
                "fuzzy_command": 1,
                "switch_power": 1,
                "read": 1,
                "change_color": 1
            },
            "speaker": {
                "update": 1,
                "read": 1
            },
            "security": {
                "toggle_door": 1,
                "read": 1,
                "interaction": 1
            },
            "daily_chat": {
                "concise": 1,
                "detailed_multi_round": 0,
                "detailed_single_turn": 0
            }
        }

        # Generate 5 Level 1 test cases: base + 3 simple tasks each
        for i in range(5):
            config = self._deep_copy_config(base_config)
            config = self._add_random_simple_tasks(config, min_add=3, max_add=3)
            test_cases.append(self._create_test_case(config, i, "level_1_easy"))

        return test_cases

    def _generate_level_2(self, level_1_test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate Level 2 (Medium) test cases (5 test cases).

        Based on Level 1 test cases, with 3-5 random non-simple tasks added.

        Args:
            level_1_test_cases: List of Level 1 test case task configs

        Returns:
            List of Level 2 test cases
        """
        test_cases = []

        for i, level_1_tc in enumerate(level_1_test_cases):
            # Get task config from Level 1 test case metadata
            config = self._deep_copy_config(level_1_tc["metadata"]["task_composition"])
            # Add 3-5 non-simple tasks
            config = self._add_random_non_simple_tasks(config, min_add=3, max_add=5)
            test_cases.append(self._create_test_case(config, i + 5, "level_2_medium"))

        return test_cases

    def _generate_level_3(self, level_2_test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate Level 3 (Hard) test cases (5 test cases).

        Based on Level 2 test cases, with 5-10 additional random non-simple tasks.

        Args:
            level_2_test_cases: List of Level 2 test case task configs

        Returns:
            List of Level 3 test cases
        """
        test_cases = []

        for i, level_2_tc in enumerate(level_2_test_cases):
            # Get task config from Level 2 test case metadata
            config = self._deep_copy_config(level_2_tc["metadata"]["task_composition"])
            # Add 5-10 non-simple tasks
            config = self._add_random_non_simple_tasks(config, min_add=5, max_add=10)
            test_cases.append(self._create_test_case(config, i + 10, "level_3_hard"))

        return test_cases

    def _create_test_case(self, task_config: Dict[str, Dict[str, int]],
                        index: int, difficulty_level: str) -> Dict[str, Any]:
        """
        Create a single test case with metadata.

        Args:
            task_config: Task configuration dictionary
            index: Test case index
            difficulty_level: Difficulty level identifier

        Returns:
            Dict with:
                - metadata: Test case metadata
                - turns: Generated instruction sequence
        """
        print(f"\nGenerating Test Case {index}: {difficulty_level}")

        # Generate instruction sequence using UnifiedSynthesizer
        turns = self.synthesizer.generate_batch(task_config)

        # Calculate metadata
        tag_counts = self._calculate_tag_counts(turns)
        task_breakdown = self._get_task_breakdown(task_config)

        # Create test case object
        test_case = {
            "metadata": {
                "test_case_id": index,
                "difficulty_level": difficulty_level,
                "turn_count": len(turns),
                "tag_distribution": tag_counts,
                "task_composition": task_breakdown,
                "model_info": {
                    "name": self.model_name,
                    "role": "nlp_conversion"
                }
            },
            "turns": turns
        }

        print(f"  Turns: {len(turns)}")
        print(f"  Tags: {tag_counts}")

        return test_case

    def _merge_task_config(self, base_config: Dict[str, Dict[str, int]],
                       additional_tasks: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        """
        Merge base task config with additional tasks.

        Args:
            base_config: Base task configuration
            additional_tasks: Additional tasks to add

        Returns:
            Dict: Merged task configuration
        """
        merged = {}

        for device_type in base_config:
            merged[device_type] = {}
            base_tasks = base_config[device_type]
            add_tasks = additional_tasks.get(device_type, {})

            # Merge tasks
            for task, count in base_tasks.items():
                merged[device_type][task] = merged[device_type].get(task, 0) + count

            # Add additional tasks
            for task, count in add_tasks.items():
                merged[device_type][task] = merged[device_type].get(task, 0) + count

        return merged

    def _deep_copy_config(self, config: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        """Deep copy task configuration."""
        import copy
        return copy.deepcopy(config)

    def _add_random_simple_tasks(self, config: Dict[str, Dict[str, int]],
                                min_add: int, max_add: int) -> Dict[str, Dict[str, int]]:
        """
        Add random simple tasks to configuration.

        Args:
            config: Base task configuration
            min_add: Minimum number of simple tasks to add
            max_add: Maximum number of simple tasks to add

        Returns:
            Updated configuration with added simple tasks
        """
        num_to_add = random.randint(min_add, max_add)

        # Collect all simple tasks (flatten the dict)
        simple_task_options = []
        for device_type, tasks in self.SIMPLE_TASKS.items():
            for task in tasks:
                simple_task_options.append((device_type, task))

        # Randomly select and add tasks
        for _ in range(num_to_add):
            device_type, task = random.choice(simple_task_options)
            count = random.randint(1, 2)
            config[device_type][task] = config[device_type].get(task, 0) + count

        return config

    def _add_random_non_simple_tasks(self, config: Dict[str, Dict[str, int]],
                                   min_add: int, max_add: int) -> Dict[str, Dict[str, int]]:
        """
        Add random non-simple tasks to configuration.

        Args:
            config: Base task configuration
            min_add: Minimum number of non-simple tasks to add
            max_add: Maximum number of non-simple tasks to add

        Returns:
            Updated configuration with added non-simple tasks
        """
        num_to_add = random.randint(min_add, max_add)

        # Collect all non-simple tasks (flatten the dict)
        non_simple_task_options = []
        for device_type, tasks in self.NON_SIMPLE_TASKS.items():
            for task in tasks:
                non_simple_task_options.append((device_type, task))

        # Randomly select and add tasks
        for _ in range(num_to_add):
            device_type, task = random.choice(non_simple_task_options)
            count = random.randint(1, 2)
            config[device_type][task] = config[device_type].get(task, 0) + count

        return config

    def _calculate_tag_counts(self, turns: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate the count of each tag in the turns."""
        tag_counts = {}

        for turn in turns:
            tag = turn.get("tag")
            if isinstance(tag, list):
                tag = tag[0] if tag else "unknown"
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return tag_counts

    def _get_task_breakdown(self, task_config: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        """Get breakdown of task counts per device type."""
        breakdown = {}

        for device_type, tasks in task_config.items():
            breakdown[device_type] = tasks.copy()

        return breakdown

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def _command_to_natural_language(self, input_command: Dict[str, Any]) -> str:
        """
        Convert structured input_command to natural language using LLM with retry logic.

        Args:
            input_command: Dict containing command parameters

        Returns:
            str: Natural language description of the command
        """
        # Build history context from last 5 conversions
        history_context = ""
        if self.conversion_history:
            history_context = "\nPrevious commands in this session (for context):\n"
            for i, (prev_cmd, prev_nl) in enumerate(self.conversion_history[-5:], 1):
                history_context += f"{i}. Command: {json.dumps(prev_cmd, ensure_ascii=False)}\n   Natural: \"{prev_nl}\"\n"

        # Get unified prompt (handles all device types)
        prompt = get_unified_prompt(input_command, history_context)

        # Call LLM asynchronously
        response = await self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.model_gen_args
        )

        natural_language = response.choices[0].message.content.strip()

        # Add to conversion history for context
        self.conversion_history.append((input_command, natural_language))

        return natural_language

    async def _add_command_text_to_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add command_text field to all turns in a test case.

        Processes turns sequentially to maintain conversion_history dependencies.

        Args:
            test_case: Test case dictionary with turns

        Returns:
            Dict: Test case with command_text added to all non-system turns
        """
        # Reset conversion history for this test case
        self.conversion_history = []

        turns = test_case.get("turns", [])
        processed_turns = []

        # Process turns sequentially (each depends on previous via conversion_history)
        for turn in turns:
            input_command = turn.get("input_command", {})
            action = input_command.get("action", "")

            # Skip time_advance - it's a system action, no command_text needed
            if action == "time_advance":
                processed_turns.append(turn)
                continue

            # Convert to natural language
            try:
                command_text = await self._command_to_natural_language(input_command)
                turn["command_text"] = command_text
                processed_turns.append(turn)
            except Exception as e:
                print(f"Warning: Failed to convert command to natural language, skipping: {e}")
                print(f"Command: {input_command}")
                # Skip this turn

        # Update test case
        test_case["turns"] = processed_turns
        # Update turn count in metadata
        test_case["metadata"]["turn_count"] = len(processed_turns)

        return test_case


async def main():
    """Main function to generate dataset."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Generate SmartMem test dataset")
    parser.add_argument(
        "--output-name",
        type=str,
        default="smartmem_dataset.json",
        help="Output filename (default: smartmem_dataset.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name for NLP conversion (default: gpt-4o)"
    )
    args = parser.parse_args()

    generator = DatasetGenerator(model_name=args.model)

    # Output path (relative from current file: data_synthesizer -> ../../data)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    output_path = os.path.join(output_dir, args.output_name)

    # Generate dataset
    dataset = await generator.generate_dataset(output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET GENERATION SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {dataset['metadata']['total_test_cases']}")
    print(f"Total turns: {dataset['metadata']['total_turns']}")
    print("\nDifficulty levels:")
    for level, info in dataset['metadata']['difficulty_levels'].items():
        print(f"  {level}: {info['description']}")
        print(f"    Test cases: {info['test_case_range']}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
