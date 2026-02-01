import json
import json_repair
import logging
import base64
from typing import Any, List, Dict, Optional
from pathlib import Path
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart, FilePart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from green_agent import GreenAgent, generate_report_charts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartmem_green_agent")


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class Agent:
    """
    SmartMem Green Agent - Orchestrates evaluation of Purple agents.

    The Green Agent:
    1. Receives assessment requests from AgentBeats platform
    2. Sends test instructions to Purple agent via A2A
    3. Processes Purple agent responses (tool calls or text)
    4. Evaluates performance and generates reports
    """
    required_roles: list[str] = ['purple']
    required_config_keys: list[str] = []

    def __init__(self, use_static: bool = False):
        self.messenger = Messenger()
        self.green_agent = GreenAgent()
        self.use_static = use_static
        self.output_dir = Path("artifacts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Main evaluation loop that orchestrates the assessment.

        Args:
            message: The incoming message from AgentBeats platform
            updater: Report progress (update_status) and results (add_artifact)

        Flow:
        1. Parse EvalRequest to get Purple agent URL and config
        2. Initialize test sequence
        3. Loop: send instruction -> receive response -> evaluate
        4. Generate final report and artifacts
        """
        input_text = get_message_text(message)

        # Try to parse as EvalRequest (from AgentBeats platform)
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.complete(new_agent_text_message(f"Request validation failed: {msg}"))
                return
        except ValidationError as e:
            logger.error(f"Invalid request format: {e}")
            await updater.failed(new_agent_text_message(f"Invalid request format: {e}"))
            return

        purple_addr = str(request.participants['purple'])
        config = request.config

        # Extract config parameters with defaults
        max_test_rounds = config.get("max_test_rounds", 5)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting SmartMem evaluation against {purple_addr}")
        )

        try:
            # Reset messenger for new conversation
            self.messenger.reset()

            # Initialize test sequence - advance to first turn
            if not self.green_agent.advance_to_next_turn():
                await updater.failed(new_agent_text_message("No test data available"))
                return

            self.green_agent.test_active = True
            turns_completed = 0
            # Limit turns based on config (for testing) or use all test data
            max_turns_from_config = config.get("max_turns", None)
            max_turns = min(len(self.green_agent.test_data), max_turns_from_config) if max_turns_from_config else len(self.green_agent.test_data)

            # Get first command to send
            current_turn = self.green_agent.current_turn
            if current_turn is None:
                await updater.failed(new_agent_text_message("Failed to get first test turn"))
                return

            command_text = current_turn.get("command_text")
            if command_text is None:
                # System action turn - handle internally
                input_command = current_turn.get("input_command", {})
                action = input_command.get("action", "")
                if action == "time_advance":
                    duration = input_command.get("duration", 0)
                    self.green_agent.simulator.tick(duration)
                    self.green_agent.advance_to_next_turn()
                    current_turn = self.green_agent.current_turn
                    command_text = current_turn.get("command_text") if current_turn else None

            # Main evaluation loop
            while self.green_agent.test_active and turns_completed < max_turns:
                if command_text is None:
                    # No more commands to send
                    break

                # Update status with progress
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Turn {turns_completed + 1}/{max_turns}: Sending instruction to Purple agent")
                )

                # Send instruction to Purple agent
                try:
                    purple_response = await self.messenger.talk_to_agent(
                        message=command_text,
                        url=purple_addr,
                        new_conversation=(turns_completed == 0),
                        timeout=120
                    )
                except Exception as e:
                    logger.error(f"Failed to communicate with Purple agent: {e}")
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(f"Communication error with Purple agent: {e}")
                    )
                    break

                # Parse Purple agent response
                try:
                    response_data = json_repair.loads(purple_response)
                except Exception as parse_err:
                    logger.warning(f"json_repair.loads failed: {parse_err}")
                    response_data = {"message_type": "text", "message_content": purple_response}

                # Process response through Green agent
                response_json = json.dumps(response_data, ensure_ascii=False)
                result = await self.green_agent.step(response_json)

                if result is None:
                    # No response needed, continue
                    turns_completed += 1
                    continue

                response_content, is_test_complete = result

                if is_test_complete:
                    # All tests completed
                    self.green_agent.test_active = False
                    break

                if response_content is not None:
                    # Green agent has a response to send back
                    try:
                        response_parsed = json.loads(response_content)
                        msg_type = response_parsed.get("message_type", "")
                        msg_content = response_parsed.get("message_content", "")

                        if msg_type == "tool_result":
                            # Send tool result back to Purple agent
                            device_id = response_parsed.get("device_id", "environment")
                            tool_result_msg = json.dumps([{
                                "message": msg_content,
                                "metadata": {"operation_object": device_id}
                            }], ensure_ascii=False)

                            purple_response = await self.messenger.talk_to_agent(
                                message=tool_result_msg,
                                url=purple_addr,
                                new_conversation=False,
                                timeout=120
                            )
                            # Process the follow-up response
                            response_data = json_repair.loads(purple_response)
                            response_json = json.dumps(response_data, ensure_ascii=False)
                            result = await self.green_agent.step(response_json)

                            if result:
                                response_content, is_test_complete = result
                                if is_test_complete:
                                    self.green_agent.test_active = False
                                    break
                                # If we got a new command, use it for the next iteration
                                if response_content:
                                    command_text = response_content
                                    turns_completed += 1
                                    continue
                        else:
                            # Text response - this is the next command
                            command_text = msg_content
                            turns_completed += 1
                            continue
                    except json.JSONDecodeError:
                        command_text = response_content
                        turns_completed += 1
                        continue
                else:
                    # Move to next turn
                    current_turn = self.green_agent.current_turn
                    command_text = current_turn.get("command_text") if current_turn else None

                turns_completed += 1

            # Generate final report
            report = await self.green_agent.generate_report()
            report_data = json.loads(report)

            # Create result summary
            summary = report_data.get("summary", {})
            overall_score = summary.get("overall_score", 0)
            total_turns = summary.get("total_turns", 0)

            summary_text = (
                f"SmartMem Evaluation Complete\n"
                f"Total Turns: {total_turns}\n"
                f"Overall Score: {overall_score}%\n"
            )

            # Add per-tag breakdown
            per_tag_scores = report_data.get("per_tag_scores", {})
            if per_tag_scores:
                summary_text += "\nPer-Category Scores:\n"
                for tag, score in per_tag_scores.items():
                    summary_text += f"  - {tag}: {score}%\n"

            # Generate visualization charts
            chart_paths = generate_report_charts(
                report_data,
                output_dir=str(self.output_dir),
                agent_name="Purple Agent"
            )

            # Create artifacts
            artifacts = [
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=report_data))
            ]

            # Add chart images as artifacts
            if chart_paths:
                chart_payload = []
                for path in chart_paths:
                    try:
                        file_bytes = Path(path).read_bytes()
                        chart_payload.append({
                            "name": Path(path).name,
                            "content_type": "image/png",
                            "data_base64": base64.b64encode(file_bytes).decode("ascii"),
                        })
                    except OSError as e:
                        logger.warning(f"Could not read chart file {path}: {e}")
                        chart_payload.append({
                            "name": Path(path).name,
                            "content_type": "image/png",
                            "path": str(path),
                            "error": "could not read file",
                        })
                if chart_payload:
                    artifacts.append(Part(root=DataPart(data={"charts": chart_payload})))

            await updater.add_artifact(
                parts=artifacts,
                name="SmartMem Evaluation Results"
            )

            await updater.complete(new_agent_text_message(summary_text))

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            await updater.failed(new_agent_text_message(f"Evaluation failed: {e}"))
