from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from app.environment import SmartHomeEnv
from green_agent.adaptive_loop import AdaptiveTestLoop, BaselineAgent
from green_agent.blackbox_eval import BlackBoxEvaluator
from green_agent.green_agent import GreenAgent
from green_agent.api_config import get_api_config
from green_agent.visualize import generate_radar_chart, generate_full_report_charts


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""

    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class Agent:
    # Accept common aliases for the Purple agent role
    required_roles: list[str] = ["purple_agent", "purple"]
    # Config keys are optional; we validate presence when required in logic
    required_config_keys: list[str] = []

    def __init__(self):
        self.messenger = Messenger()
        self.output_dir = Path("artifacts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        has_role = any(alias in request.participants for alias in self.required_roles)
        if not has_role:
            return False, f"Missing required participant role(s): {self.required_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    def _get_participant_url(self, request: EvalRequest) -> str | None:
        for alias in self.required_roles:
            if alias in request.participants:
                return str(request.participants[alias])
        return None

    def _build_green_agent(self, config: dict[str, Any]) -> GreenAgent | None:
        """Build GreenAgent from config, returning None if unavailable."""

        provider = config.get("generator_provider") or config.get("green_provider")
        model = config.get("green_model")
        api_key = config.get("green_api_key")
        base_url = config.get("green_base_url")

        if not provider:
            return None

        try:
            api_cfg = get_api_config(
                provider,
                model=model if model else None,
                api_key=api_key,
                base_url=base_url if base_url else None,
            )
            return GreenAgent.from_config(api_cfg, max_retries=int(config.get("max_retries", 2)))
        except Exception:
            return None

    def _adaptive_run(self, purple_url: str, config: dict[str, Any]) -> dict[str, Any]:
        env = SmartHomeEnv()
        baseline_agent = BaselineAgent(env, agent_type="a2a", agent_url=purple_url)
        green_agent = self._build_green_agent(config)

        if green_agent is None:
            raise RuntimeError("GreenAgent unavailable; provide generator_provider and credentials")

        loop = AdaptiveTestLoop(
            green_agent=green_agent,
            baseline_agent=baseline_agent,
            output_dir=self.output_dir,
        )

        rounds = int(config.get("rounds", 2))
        initial_per_dim = int(config.get("initial_per_dim", 2))
        targeted_per_weakness = int(config.get("targeted_per_weakness", 2))
        convergence = float(config.get("convergence", 0.05))
        agent_name = config.get("agent_name", "Purple Agent")

        report_path = loop.run(
            max_rounds=rounds,
            initial_per_dim=initial_per_dim,
            targeted_per_weakness=targeted_per_weakness,
            convergence_threshold=convergence,
            agent_name=agent_name,
        )

        data_path = loop.reporter.last_data_path
        chart_paths = []
        if data_path:
            chart_paths = generate_full_report_charts(str(data_path), output_dir=str(self.output_dir))

        latest_round = loop.round_history[-1] if loop.round_history else {}
        summary_text = (
            f"Adaptive evaluation finished for {agent_name}. "
            f"Rounds: {len(loop.round_history)}, Last pass rate: {latest_round.get('pass_rate', 0):.2f}."
        )

        return {
            "mode": "adaptive",
            "summary": summary_text,
            "round_history": loop.round_history,
            "report_path": report_path,
            "data_path": str(data_path) if data_path else None,
            "chart_paths": chart_paths,
            "weaknesses": loop.analyzer.get_top_weaknesses(5),
        }

    def _smoke_run(self, purple_url: str | None, agent_type: str = "a2a") -> dict[str, Any]:
        env = SmartHomeEnv()
        agent_kwargs = {"agent_type": agent_type}
        if agent_type == "a2a" and purple_url:
            agent_kwargs["agent_url"] = purple_url
        baseline_agent = BaselineAgent(env, **agent_kwargs)
        evaluator = BlackBoxEvaluator(env)

        smoke_cases = [
            {
                "scenario_id": "smoke_light_on",
                "difficulty": "easy",
                "dimension": "precision",
                "description": "Turn on living room light",
                "initial_state": {},
                "turns": [
                    {
                        "turn_id": 1,
                        "gm_instruction": "Please turn on the living room light.",
                        "expected_agent_action": [
                            {"action": "update", "key": "living_room_light", "value": "on"}
                        ],
                        "expected_final_state": {"living_room_light": "on"},
                    }
                ],
            },
            {
                "scenario_id": "smoke_temperature",
                "difficulty": "easy",
                "dimension": "memory",
                "description": "Set AC temperature to 24",
                "initial_state": {},
                "turns": [
                    {
                        "turn_id": 1,
                        "gm_instruction": "Set the AC to 24 degrees.",
                        "expected_agent_action": [
                            {"action": "update", "key": "ac_temperature", "value": 24},
                            {"action": "update", "key": "ac", "value": "on"},
                        ],
                        "expected_final_state": {"ac_temperature": 24, "ac": "on"},
                    }
                ],
            },
        ]

        agent_for_eval = getattr(baseline_agent, 'agent', baseline_agent)
        results = [evaluator.evaluate_test_case(agent_for_eval, case) for case in smoke_cases]
        pass_count = sum(1 for r in results if r.passed)
        pass_rate = pass_count / max(1, len(results))

        dimension_scores: dict[str, float] = {}
        for result in results:
            score = (result.total_score / max(1, result.max_score)) * 100
            dimension_scores[result.dimension] = score

        chart_paths: list[str] = []
        radar_path = self.output_dir / "smoke_radar.png"
        radar_file = generate_radar_chart(dimension_scores, title="Smoke Check", output_path=str(radar_path))
        if radar_file:
            chart_paths.append(radar_file)

        summary_text = f"Smoke evaluation: {pass_count}/{len(results)} passed (pass rate {pass_rate:.2f})."

        return {
            "mode": "smoke",
            "summary": summary_text,
            "pass_rate": pass_rate,
            "results": [r.model_dump() if hasattr(r, "model_dump") else r.__dict__ for r in results],
            "chart_paths": chart_paths,
        }

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        istoolcalling = message.metadata['message_type'] == 'tool_calling'
        
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        purple_url = self._get_participant_url(request)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Starting evaluation loop...")
        )

        mode = request.config.get("mode", "adaptive")
        agent_type = request.config.get("agent_type", "a2a")

        try:
            if mode == "adaptive":
                if not purple_url:
                    await updater.reject(new_agent_text_message("Participant Purple agent URL missing for adaptive mode."))
                    return
                result = await asyncio.to_thread(self._adaptive_run, purple_url, request.config)
            else:
                result = await asyncio.to_thread(self._smoke_run, purple_url, agent_type)
        except Exception as exc:
            # Fallback to smoke if adaptive fails
            try:
                fallback = await asyncio.to_thread(self._smoke_run, purple_url, agent_type)
                result = fallback
                result["warning"] = f"Adaptive run failed: {exc}" if mode == "adaptive" else str(exc)
            except Exception as smoke_exc:
                await updater.failed(new_agent_text_message(f"Evaluation failed: {smoke_exc}"))
                return

        text_part = Part(root=TextPart(text=result.get("summary", "")))
        data_part = Part(root=DataPart(data={k: v for k, v in result.items() if k != "summary"}))

        artifacts = [text_part, data_part]
        chart_paths = result.get("chart_paths") or []
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
                except OSError:
                    chart_payload.append({
                        "name": Path(path).name,
                        "content_type": "image/png",
                        "path": str(path),
                        "error": "could not read file",
                    })
            artifacts.append(Part(root=DataPart(data={"charts": chart_payload})))

        await updater.add_artifact(parts=artifacts, name="Evaluation Result")
        await updater.complete()
