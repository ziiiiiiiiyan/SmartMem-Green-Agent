import json
import json_repair
import logging
from typing import Any, List, Dict
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart, FilePart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from green_agent import GreenAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartmem_green_agent")


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class Agent:
    """
    """
    required_roles: list[str] = ['purple']
    required_config_keys: list[str] = []

    def __init__(self, use_static: bool = False):
        self.messenger = Messenger()
        self.agent = GreenAgent()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"
    
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        input_text直接就是GreenAgent.step()可以处理的{"message_type": "tool" or "text", "message_content": content}格式
        """
        input_text = get_message_text(message)

        # Try to parse as EvalRequest (from AgentBeats platform)
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.complete(new_agent_text_message(f"Request validation failed: {msg}"))
                return
        except ValidationError:

            return

        purple_addr = str(request.participants['purple'])


        
