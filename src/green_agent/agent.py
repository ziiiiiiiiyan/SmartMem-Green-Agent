import os
import json
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .simulator import SmartHomeEnv
from .evaluator import Evaluator

load_dotenv()

logger = logging.getLogger("smartmem_green_agent")

# Get the directory where this module is located
_MODULE_DIR = Path(__file__).parent

# Mapping from Purple Agent's device_id (snake_case) to simulator's device names
DEVICE_ID_TO_NAME = {
    "living_room_light": "Living Room Light",
    "living_room_color": "Living Room Light",  # Color is a property of the light
    "bedroom_light": "Bedroom Light",
    "bedroom_color": "Bedroom Light",  # Color is a property of the light
    "kitchen_light": "Kitchen Light",
    "ac": "AC",
    "ac_temperature": "AC",  # Temperature is a property of AC
    "fan_speed": "AC",  # Fan speed is on AC device
    "music_volume": "Speaker",
    "front_door_lock": "Security",
    "all": None  # Special case for reading all devices
}

# Mapping for parameter names based on device_id
DEVICE_ID_TO_PARAM = {
    "living_room_light": "power",
    "living_room_color": "color",
    "bedroom_light": "power",
    "bedroom_color": "color",
    "kitchen_light": "power",
    "ac": "power",
    "ac_temperature": "temperature",
    "fan_speed": "fan_speed",
    "music_volume": "volume",
    "front_door_lock": "door_lock",
}

# Value translation for specific device_ids
# Purple Agent value -> Simulator value
VALUE_TRANSLATION = {
    "front_door_lock": {
        "locked": "closed",
        "unlocked": "open"
    },
    "fan_speed": {
        "off": "auto",  # Map "off" to "auto" for AC fan
        "low": "1",
        "medium": "2",
        "high": "3"
    }
}


def translate_agent_action_for_eval(agent_action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate Purple Agent's action format to the expected format for evaluation.

    Purple Agent format: {"device_id": "bedroom_light", "action": "read/update", "value": "..."}
    Expected format: {"action": "read/update", "device": "Bedroom Light", "state": {...}}

    Args:
        agent_action: Action in Purple Agent's format

    Returns:
        Action in expected format for evaluation
    """
    device_id = agent_action.get("device_id", "")
    action = agent_action.get("action", "read")
    value = agent_action.get("value")

    # Map device_id to device name
    device_name = DEVICE_ID_TO_NAME.get(device_id, device_id)

    # Build translated action
    translated = {
        "action": action,
        "device": device_name
    }

    # For update actions, build the state dict
    if action == "update" and value is not None:
        param_name = DEVICE_ID_TO_PARAM.get(device_id, "power")

        # Translate value if needed
        translated_value = value
        if device_id in VALUE_TRANSLATION and value in VALUE_TRANSLATION[device_id]:
            translated_value = VALUE_TRANSLATION[device_id][value]

        translated["state"] = {param_name: translated_value}

    return translated


class GreenAgent:
    """Green Agent for testing smart home control capabilities"""

    def __init__(self, test_data_path: str = None):
        """
        Initialize GreenAgent

        - Load environment variables and configure LLM for multi-turn interaction with agent
        - Load test data from test_data_path (list of turns)
        - Initialize simulator and evaluator
        - Maintain test case progress (reset simulator when test case completes)
        - Maintain current turn state (may be multi-turn conversation)
        - Maintain historical performance (per test case + global cumulative)
        - Maintain conversation history for each test case
        """
        # Initialize LLM client for agent interaction
        self.llm_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o")
        self.model_gen_args = json.loads(os.getenv("MODEL_GEN_ARGS", "{}"))

        # Load test data - use default path relative to module if not specified
        if test_data_path is None:
            test_data_path = str(_MODULE_DIR / "data" / "test.json")
        self.test_data_path = test_data_path
        self.test_data: List[Dict[str, Any]] = []
        self._load_test_data()

        # Initialize simulator and evaluator
        self.simulator = SmartHomeEnv()
        self.evaluator = Evaluator()

        # Test case progress tracking
        self.current_turn_index: int = 0
        self.current_turn: Optional[Dict[str, Any]] = None
        self.turn_round: int = 0  # For multi-turn conversations
        self.is_multi_round: bool = False
        self.conversation_complete: bool = False

        # Track agent actions for current turn evaluation
        self.current_turn_actions: List[Dict[str, Any]] = []

        # Performance tracking
        self.global_results: Dict[str, Any] = {
            "total_test_cases": 0,
            "total_turns": 0,
            "overall_score": 0.0,
            "per_tag_scores": {},
            "all_test_case_scores": []
        }

        # Conversation history for current test case
        self.conversation_history: List[Dict[str, Any]] = []

        # Track if we're in the middle of a test sequence
        self.test_active: bool = False

    def _load_test_data(self):
        """Load test data from JSON file

        The test.json structure is:
        {
            "metadata": {...},
            "test_cases": [
                {
                    "metadata": {...},
                    "turns": [...]
                },
                ...
            ]
        }

        We flatten all turns from all test cases into a single list.
        """
        try:
            with open(self.test_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both formats: direct list of turns or nested structure
            if isinstance(data, list):
                self.test_data = data
            elif isinstance(data, dict):
                if "test_cases" in data:
                    # Flatten all turns from all test cases
                    all_turns = []
                    for test_case in data["test_cases"]:
                        turns = test_case.get("turns", [])
                        all_turns.extend(turns)
                    self.test_data = all_turns
                elif "turns" in data:
                    self.test_data = data["turns"]
                else:
                    self.test_data = []
            else:
                self.test_data = []

            logger.info(f"Loaded {len(self.test_data)} turns from {self.test_data_path}")
        except FileNotFoundError:
            logger.warning(f"Test data file not found: {self.test_data_path}")
            self.test_data = []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse test data: {e}")
            self.test_data = []

    def reset_test_case(self):
        """Reset simulator and evaluator for new test case"""
        self.simulator = SmartHomeEnv()
        self.evaluator.reset()
        self.turn_round = 0
        self.is_multi_round = False
        self.conversation_complete = False
        logger.info("Reset simulator and evaluator for new test case")

    def get_current_turn(self) -> Optional[Dict[str, Any]]:
        """Get current turn data"""
        if self.current_turn_index < len(self.test_data):
            return self.test_data[self.current_turn_index]
        return None

    def advance_to_next_turn(self) -> bool:
        """
        Advance to next turn

        Returns:
            bool: True if successfully advanced, False if no more turns
        """
        self.current_turn_index += 1
        self.turn_round = 0
        self.current_turn = self.get_current_turn()

        # Clear previous turn's actions
        self.current_turn_actions = []

        if self.current_turn is None:
            logger.info("No more turns available")
            return False

        # Check if this is a multi-round conversation
        input_command = self.current_turn.get("input_command", {})
        self.is_multi_round = input_command.get("multi_round", False)
        self.conversation_complete = False

        return True

    async def step(self, input_text: str) -> tuple[Optional[str], Optional[bool]]:
        """
        Process a step in the conversation

        Args:
            input_text: JSON string of a single message - {"message_type": "tool" or "text", "message_content": content}

        Returns:
            tuple: (response_content, is_new_test_case)
                - response_content: Response content, or None if no response
                - is_new_test_case: True if test data exhausted, otherwise None

        Raises:
            ValueError: If message_type is invalid
        """
        # Parse single message
        message = self._parse_single_message(input_text)

        message_type = message.get("message_type")
        message_content = message.get("message_content", "")
        logger.info(f"Processing step - message_type: {message_type}")

        if message_type == "tool":
            # Forward to simulator - message_content is a JSON string of tool calls
            tool_calls = json.loads(message_content) if isinstance(message_content, str) else message_content
            # Handle single tool call or list of tool calls
            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                tool_call = tool_calls[0]  # Take the first tool call
            else:
                tool_call = tool_calls
            return await self._handle_tool_call(tool_call)
        elif message_type == "text":
            # Use LLM to determine intent
            logger.info(f"Handling text message: {message_content[:100]}...")
            return await self._handle_text_message(message_content)
        else:
            # Invalid message_type - raise error
            error_msg = f"Invalid message_type: {message_type}. Must be 'tool' or 'text'."
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_single_message(self, input_text: str) -> Dict[str, Any]:
        """Parse single message from JSON string"""
        try:
            message = json.loads(input_text)
            if isinstance(message, dict):
                return message
            else:
                raise ValueError(f"Expected a single message object, got {type(message)}")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse input as JSON: {input_text}")

    async def _handle_tool_call(self, tool_call: Dict[str, Any]) -> tuple[str, Optional[bool]]:
        """
        Handle tool/function call from agent

        Args:
            tool_call: Tool call data from Purple Agent
                Format: {"device_id": "bedroom_light", "action": "read/update", "value": "..."}

        Returns:
            tuple: (response_content, is_new_test_case)
        """
        try:
            # Translate Purple Agent's format to simulator's format
            device_id = tool_call.get("device_id", "")
            action = tool_call.get("action", "read")
            value = tool_call.get("value")

            # Map device_id to simulator device name
            device_name = DEVICE_ID_TO_NAME.get(device_id)

            if device_id == "all":
                # Special case: read all devices
                simulator_command = {
                    "device": None,
                    "action": "read_all"
                }
                # Get status of all devices
                all_status = {}
                for dev_name, device in self.simulator.devices.items():
                    all_status[dev_name] = device.read()
                result = {"status": True, "message": "Read all devices", "data": all_status}
            elif device_name is None:
                result = {"device": device_id, "status": False, "message": f"Unknown device_id: {device_id}"}
            else:
                # Build simulator command
                if action == "read":
                    simulator_command = {
                        "device": device_name,
                        "action": "read"
                    }
                else:  # update
                    # Map the value to the correct parameter name
                    param_name = DEVICE_ID_TO_PARAM.get(device_id, "power")

                    # Translate value if needed (e.g., locked -> closed for door lock)
                    translated_value = value
                    if device_id in VALUE_TRANSLATION and value in VALUE_TRANSLATION[device_id]:
                        translated_value = VALUE_TRANSLATION[device_id][value]

                    simulator_command = {
                        "device": device_name,
                        "action": "update",
                        "params": {param_name: translated_value}
                    }

                # Execute command on simulator
                result = self.simulator.execute(simulator_command)

            # Add to conversation history
            self.conversation_history.append({
                "role": "purple",
                "content": json.dumps(tool_call),
                "tool_result": result
            })

            # Record action for current turn evaluation
            self.current_turn_actions.append(tool_call)

            # Don't evaluate here - evaluate when turn is complete
            # Include device_id in response for Purple agent to match results
            response = {
                "message_type": "tool_result",
                "message_content": result,
                "device_id": device_id
            }
            return (json.dumps(response, ensure_ascii=False), None)

        except Exception as e:
            logger.error(f"Error executing tool call: {e}")
            raise

    async def _handle_text_message(self, message_content: str) -> tuple[Optional[str], Optional[bool]]:
        """
        Handle text message from agent

        Use LLM to determine:
        - If current turn is multi-round conversation, generate further response
        - If conversation is complete, fetch next turn data and send command
        - Handle special system actions (e.g., time_advance)

        Args:
            message_content: Text message from agent

        Returns:
            tuple: (response_content, is_new_test_case)
        """
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message_content
            })

            # Check if we should continue conversation or move to next turn
            if self.is_multi_round and not self.conversation_complete:
                if self.turn_round < 1:  # Two rounds total
                    self.turn_round += 1
                    # Generate further response for multi-round conversation
                    response = await self._generate_multi_round_response()
                    response_dict = {
                        "message_type": "text",
                        "message_content": response
                    }
                    return (json.dumps(response_dict, ensure_ascii=False), None)

            # Check if conversation is complete
            intent = await self._determine_conversation_intent(message_content)
            logger.info(f"Conversation intent for '{message_content[:50]}...': {intent}")

            if intent == "complete":
                # Mark conversation as complete
                self.conversation_complete = True

                # Evaluate this turn before advancing
                await self._evaluate_completed_turn()

                # Move to next turn
                if not self.advance_to_next_turn():
                    # All tests complete
                    return (None, True)

                # Check if this turn has command_text
                command_text = self.current_turn.get("command_text")

                if command_text is not None:
                    # This turn requires agent action - send command_text
                    return (command_text, None)
                else:
                    # This is a system instruction (no command_text)
                    # Execute system action and skip to next turn
                    input_command = self.current_turn.get("input_command", {})
                    action = input_command.get("action", "")

                    if action == "time_advance":
                        # Execute system time advance
                        duration = input_command.get("duration", 0)
                        self.simulator.tick(duration)

                        # Move to next turn (don't evaluate system actions)
                        if not self.advance_to_next_turn():
                            return (None, True)

                        # Check next turn
                        command_text = self.current_turn.get("command_text")
                        if command_text is not None:
                            return (command_text, None)
                        else:
                            # Another system action - loop
                            return (None, None)

            # Continue conversation - no response needed
            return (None, None)

        except Exception as e:
            logger.error(f"Error handling text message: {e}")
            raise  # Re-raise exception instead of returning error message

    async def _determine_conversation_intent(self, message: str) -> str:
        """
        Determine if agent has completed the current task.

        Uses heuristic detection first, then falls back to LLM if needed.

        Args:
            message: Agent's message

        Returns:
            str: "complete" if task is done, "continue" otherwise
        """
        # Heuristic detection for common completion patterns
        message_lower = message.lower().strip()

        # Patterns that indicate task completion
        completion_patterns = [
            # Device state reports
            "is currently", "is now", "has been", "was set to",
            "turned on", "turned off", "is on", "is off",
            # Status confirmations
            "the light", "the ac", "the fan", "the door", "the speaker",
            "temperature is", "volume is", "speed is",
            # Memory recall patterns (offline mode)
            "i recall", "i remember", "last time we checked",
            "the last known", "according to my memory",
            # Confirmation patterns
            "done", "completed", "finished", "success",
            # Direct answers
            "yes", "no", "correct", "that's right",
        ]

        # Check if message contains completion patterns
        for pattern in completion_patterns:
            if pattern in message_lower:
                logger.info(f"Heuristic detected completion pattern: '{pattern}'")
                return "complete"

        # If message is a short definitive statement (likely an answer)
        if len(message) < 200 and not message_lower.endswith("?"):
            # Check if it's providing information (not asking)
            info_indicators = ["is", "are", "was", "were", "has", "have", "can", "will"]
            for indicator in info_indicators:
                if f" {indicator} " in f" {message_lower} ":
                    logger.info(f"Heuristic detected informative statement")
                    return "complete"

        # Fall back to LLM for ambiguous cases
        try:
            prompt = f"""Analyze the following message from a smart home AI assistant and determine if they have completed the user's request.

Assistant's message: "{message}"

The assistant is responding to a user command about smart home devices (lights, AC, fan, door lock, speaker, etc.).

Respond with ONLY "complete" or "continue" (lowercase).
- "complete": The assistant has provided a definitive answer, status report, or confirmation that the task is done
- "continue": The assistant is asking a question, requesting clarification, or the task is clearly incomplete

Examples of COMPLETE responses:
- "The bedroom light is currently off."
- "I've turned on the living room light."
- "The AC temperature is set to 24 degrees."
- "Since the network is down, I recall the light was off."

Examples of CONTINUE responses:
- "Which light would you like me to turn on?"
- "I'm checking the device status..."
- "Could you please clarify which room?"
"""

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.model_gen_args
            )

            result = response.choices[0].message.content.strip().lower()
            return result if result in ["complete", "continue"] else "complete"  # Default to complete

        except Exception as e:
            logger.error(f"Error determining intent: {e}")
            return "complete"  # Default to complete on error to avoid infinite loops

    async def _generate_multi_round_response(self) -> str:
        """
        Generate further response for multi-round conversation

        Returns:
            str: Generated response text
        """
        try:
            # Build conversation context
            context = self._build_conversation_context()

            prompt = f"""Continue the conversation naturally based on the context:

{context}

Generate a brief, natural response to continue the conversation."""

            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.model_gen_args
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating multi-round response: {e}")
            return "I understand. Please continue."

    def _build_conversation_context(self) -> str:
        """Build conversation context string from history"""
        context_parts = []
        for item in self.conversation_history:
            role = item.get("role", "user")
            content = item.get("content", "")
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    async def _evaluate_completed_turn(self):
        """
        Evaluate the completed turn performance.

        When a turn is complete (conversation_complete=True):
        1. Use all agent actions from current_turn_actions
        2. Compare with expected_choices
        3. Get actual simulator state for state evaluation
        """
        if not self.current_turn:
            return

        try:
            input_command = self.current_turn.get("input_command", {})
            expected_choices = self.current_turn.get("expected_choices", [])
            tag = self.current_turn.get("tag")

            # Get actual simulator state for this turn
            simulator_state = self._get_simulator_state_for_turn(input_command)

            # Use the last agent action for evaluation
            if self.current_turn_actions:
                last_agent_action = self.current_turn_actions[-1]
                # Translate Purple Agent's action format to expected format for evaluation
                translated_action = translate_agent_action_for_eval(last_agent_action)
            else:
                # No actions taken - empty action
                translated_action = {"action": "none"}

            # Evaluate turn
            action_score, state_score = self.evaluator.eval_turn(
                input_command=input_command,
                agent_action=translated_action,
                expected_choices=expected_choices,
                simulator_state=simulator_state,
                tag=tag
            )

            logger.info(f"Turn evaluation - Action: {action_score}, State: {state_score}")

        except Exception as e:
            logger.error(f"Error evaluating completed turn: {e}", exc_info=True)

    def _get_simulator_state_for_turn(self, input_command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get actual simulator state for the given input command.

        Args:
            input_command: The input command for this turn

        Returns:
            Dict: The actual state from simulator
        """
        try:
            # Determine which device to query based on input_command
            device_name = input_command.get("device")

            if device_name and device_name in self.simulator.devices:
                # Get state for specific device
                device = self.simulator.devices[device_name]
                return device.read()
            else:
                # For commands without specific device
                action = input_command.get("action", "")
                if action == "chat":
                    return {}  # No state for chat commands
                elif action == "time_advance":
                    return {"time_elapsed": self.simulator.time_simulation_elapsed}
                else:
                    return {}

        except Exception as e:
            logger.error(f"Error getting simulator state: {e}")
            return {}

    async def generate_report(self) -> str:
        """
        Generate overall performance report

        Returns:
            str: JSON string with overall results
        """
        try:
            # Get results from evaluator
            current_test_results = self.evaluator.view_results()

            # Update global results
            self.global_results["total_test_cases"] += 1
            self.global_results["total_turns"] += current_test_results.get("total_turns", 0)

            # Accumulate scores
            self.global_results["all_test_case_scores"].append(current_test_results)

            # Calculate overall average
            all_scores = [r.get("overall_score", 0) for r in self.global_results["all_test_case_scores"]]
            if all_scores:
                self.global_results["overall_score"] = sum(all_scores) / len(all_scores)

            # Aggregate per-tag scores
            for tag, score in current_test_results.get("per_tag_scores", {}).items():
                if tag not in self.global_results["per_tag_scores"]:
                    self.global_results["per_tag_scores"][tag] = []
                self.global_results["per_tag_scores"][tag].append(score)

            # Calculate average per-tag scores
            averaged_tag_scores = {}
            for tag, scores in self.global_results["per_tag_scores"].items():
                averaged_tag_scores[tag] = sum(scores) / len(scores) if scores else 0.0

            report = {
                "summary": {
                    "total_test_cases": self.global_results["total_test_cases"],
                    "total_turns": self.global_results["total_turns"],
                    "overall_score": round(self.global_results["overall_score"], 2)
                },
                "current_test_case": current_test_results,
                "per_tag_scores": {k: round(v, 2) for k, v in averaged_tag_scores.items()},
                "max_streak": current_test_results.get("max_streak", 0)
            }

            return json.dumps(report, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return json.dumps({"error": str(e)})
''