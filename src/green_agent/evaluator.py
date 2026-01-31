"""
Evaluator Module for Agent Performance Assessment

This module provides an evaluator to assess agent performance on test cases.
Each test case contains multiple turns, and each turn includes an instruction
with expected choices.
"""

import os
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_llm_client():
    """Get OpenAI client configured from .env"""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )


class Evaluator:
    """
    Test case evaluator for assessing agent performance in each turn.

    A test case contains multiple turns, each with an instruction and expected_choices.

    Evaluation Logic:
    1. Evaluate each instruction's action correctness and state correctness
       1.1 If action matches any option in "expected_choices", action scores (0.5 points)
           1.1.1 For "read" action: check if returned keys match state keys
           1.1.2 For "update" action: check if returned result matches complete state (keys and values)
       1.2 If current simulator state matches "expected_state", state scores (0.5 points)
       1.3 For "send_user_msg" or "send_visitor_msg" with non-empty expected_choices:
            use LLM to evaluate if text meaning matches expected content
       1.4 If command action is not "read", "update", or the special cases in 1.3,
            no evaluation is triggered
       1.5 Each evaluable item is worth 1 point total (0.5 for action, 0.5 for state)

    2. Maintain evaluation results for complete test case, including:
       - Percentage score for each tag
       - Overall percentage score
       - Maximum winning streak (consecutive turns with both correct action and state)
       2.1 eval_turn method returns current turn's score
       2.2 view_results method returns overall score, per-tag scores, and max streak

    3. reset method initializes evaluator for new test case evaluation
    """

    def __init__(self):
        """Initialize the evaluator"""
        self.reset()
        self.llm_client = None
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client for text similarity evaluation"""
        try:
            self.llm_client = get_llm_client()
        except Exception as e:
            print(f"Warning: Failed to initialize LLM client: {e}")
            self.llm_client = None

    def reset(self) -> None:
        """
        Reset evaluator state to prepare for a new test case evaluation.

        Clears all previous scores, results, and tracking data.
        """
        self.turn_scores: List[float] = []  # Score for each turn
        self.turn_action_scores: List[float] = []  # Action score for each turn
        self.turn_state_scores: List[float] = []  # State score for each turn
        self.turn_tags: List[str] = []  # Tag for each turn (for per-tag analysis)

        # Per-tag score tracking
        self.tag_scores: Dict[str, List[float]] = defaultdict(list)

        # Winning streak tracking
        self.current_streak: int = 0
        self.max_streak: int = 0

    def eval_turn(
        self,
        input_command: Dict[str, Any],
        agent_action: Dict[str, Any],
        expected_choices: List[List[Dict[str, Any]]],
        simulator_state: Dict[str, Any],
        tag: Optional[str] = None
    ) -> float:
        """
        Evaluate agent's action and state for the current turn.

        Args:
            input_command: The user's input command
                Example: {"action": "read", "targets": ["power", "temperature"]}
            agent_action: The action taken by the agent
                Example: {"action": "read", "state": {"power": "on", "temperature": 26}}
            expected_choices: List of possible valid outcomes
                Each outcome is a list of action dicts
                Example: [[{"action": "read", "state": {...}}], [...] ]
            simulator_state: Current state of the simulator (ground truth)
                Example: {"power": "on", "temperature": 26}
            tag: Optional tag for this turn (for per-tag scoring)

        Returns:
            Tuple[float, float]: (action_score, state_score)
                - action_score: 0.0 or 0.5 for action correctness
                - state_score: 0.0 or 0.5 for state correctness
                - Total: 1.0 for perfect turn
        """
        action_score = 0.0
        state_score = 0.0

        # Get input command action type
        input_action_type = input_command.get("action", "")

        # Skip evaluation for non-evaluable action types
        if input_action_type not in ["read", "update", "send_user_msg", "send_visitor_msg"]:
            self._record_turn(0.0, 0.0, 0.0, tag)
            return (0.0, 0.0)

        # Evaluate action correctness (0.5 points)
        action_score = self._eval_action(input_command, agent_action, expected_choices)

        # Evaluate state correctness (0.5 points) - only for read and update
        if input_action_type in ["read", "update"]:
            state_score = self._eval_state(agent_action, simulator_state)
        else:
            # For message actions, state is not applicable, give full points if action is correct
            state_score = 0.5 if action_score > 0 else 0.0

        # Calculate total score for this turn
        total_score = action_score + state_score

        # Record this turn's results
        self._record_turn(total_score, action_score, state_score, tag)

        return (action_score, state_score)

    def _eval_action(
        self,
        input_command: Dict[str, Any],
        agent_action: Dict[str, Any],
        expected_choices: List[List[Dict[str, Any]]]
    ) -> float:
        """
        Evaluate action correctness based on input command type.

        Args:
            input_command: User's input command
            agent_action: Action taken by agent
            expected_choices: List of valid expected outcomes

        Returns:
            float: Action score (0.0 or 0.5)
        """
        if not agent_action or not expected_choices:
            return 0.0

        input_action_type = input_command.get("action", "")

        # Check if agent action matches any expected choice
        for expected_outcome in expected_choices:
            if not expected_outcome:
                continue

            for expected_action in expected_outcome:
                if self._actions_match(agent_action, expected_action, input_action_type):
                    return 0.5

        return 0.0

    def _actions_match(
        self,
        agent_action: Dict[str, Any],
        expected_action: Dict[str, Any],
        input_action_type: str
    ) -> bool:
        """
        Check if agent action matches expected action based on input command type.

        Args:
            agent_action: Action from agent
            expected_action: Expected action from choices
            input_action_type: Type of input command (read, update, etc.)

        Returns:
            bool: True if actions match
        """
        # For message-based input commands, use LLM evaluation
        if input_action_type in ["send_user_msg", "send_visitor_msg"]:
            return self._evaluate_text_similarity(agent_action, expected_action)

        # For read and update actions, check structure
        if "action" not in agent_action or "action" not in expected_action:
            return False

        if agent_action["action"] != expected_action["action"]:
            return False

        # For read: check if keys match
        if input_action_type == "read":
            agent_state = agent_action.get("state", {})
            expected_state = expected_action.get("state", {})
            return set(agent_state.keys()) == set(expected_state.keys())

        # For update: check if complete state matches (keys and values)
        if input_action_type == "update":
            agent_state = agent_action.get("state", {})
            expected_state = expected_action.get("state", {})
            return agent_state == expected_state

        return False

    def _eval_state(
        self,
        agent_action: Dict[str, Any],
        simulator_state: Dict[str, Any]
    ) -> float:
        """
        Evaluate state correctness.

        Args:
            agent_action: Action taken by agent (contains state in agent's response)
            simulator_state: Ground truth state from simulator

        Returns:
            float: State score (0.0 or 0.5)
        """
        if not simulator_state:
            return 0.0

        # Extract state from agent action
        agent_state = agent_action.get("state", {})

        # Check if agent's state matches simulator's state
        if agent_state == simulator_state:
            return 0.5

        return 0.0

    def _evaluate_text_similarity(
        self,
        agent_action: Dict[str, Any],
        expected_action: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if two text-based actions have similar meaning using LLM.

        Args:
            agent_action: Action from agent
            expected_action: Expected action from choices

        Returns:
            bool: True if the meanings are similar enough
        """
        if not self.llm_client:
            # Fallback to exact comparison if LLM client not available
            return agent_action == expected_action

        # Extract text content from actions
        agent_text = self._extract_text_content(agent_action)
        expected_text = self._extract_text_content(expected_action)

        if not agent_text or not expected_text:
            return False

        try:
            # Use LLM to evaluate similarity
            prompt = f"""Compare the following two texts and determine if they convey similar meaning.

Text 1 (Agent's response): "{agent_text}"
Text 2 (Expected response): "{expected_text}"

Consider the semantic meaning, not exact wording. The texts should be considered similar if:
- They convey the same core information
- They serve the same purpose
- Minor wording differences are acceptable

Respond with ONLY "true" or "false" (lowercase)."""

            response = self.llm_client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                **json.loads(os.getenv("MODEL_GEN_ARGS", "{}"))
            )

            result = response.choices[0].message.content.strip().lower()
            return result == "true"

        except Exception as e:
            print(f"Warning: LLM evaluation failed: {e}, falling back to exact comparison")
            return agent_action == expected_action

    def _extract_text_content(self, action: Dict[str, Any]) -> str:
        """
        Extract text content from an action.

        Args:
            action: Action dictionary

        Returns:
            str: Extracted text content
        """
        # Try different fields that might contain text
        if "msg_content" in action:
            return action["msg_content"]
        if "state" in action and isinstance(action["state"], dict):
            if "msg_content" in action["state"]:
                return action["state"]["msg_content"]
            if "intercom_reply" in action["state"]:
                return action["state"]["intercom_reply"]
        if "note" in action:
            return action["note"]

        # Fallback: return JSON representation
        return json.dumps(action, ensure_ascii=False)

    def _record_turn(
        self,
        total_score: float,
        action_score: float,
        state_score: float,
        tag: Optional[str]
    ) -> None:
        """
        Record turn results and update tracking metrics.

        Args:
            total_score: Total score for this turn
            action_score: Action score for this turn
            state_score: State score for this turn
            tag: Tag for this turn
        """
        self.turn_scores.append(total_score)
        self.turn_action_scores.append(action_score)
        self.turn_state_scores.append(state_score)

        # Record tag
        if tag:
            if isinstance(tag, list):
                tag = tag[0] if tag else "unknown"
            self.turn_tags.append(tag)
            self.tag_scores[tag].append(total_score)
        else:
            self.turn_tags.append("unknown")
            self.tag_scores["unknown"].append(total_score)

        # Update winning streak
        if total_score >= 1.0:  # Both action and state correct
            self.current_streak += 1
            self.max_streak = max(self.max_streak, self.current_streak)
        else:
            self.current_streak = 0

    def view_results(self) -> Dict[str, Any]:
        """
        View overall evaluation results for the complete test case.

        Returns:
            Dict with:
                - overall_score: Overall percentage score (0-100)
                - overall_action_score: Overall action percentage score (0-100)
                - overall_state_score: Overall state percentage score (0-100)
                - total_turns: Total number of turns evaluated
                - per_tag_scores: Dict of percentage scores for each tag
                - max_streak: Maximum winning streak (consecutive perfect turns)
        """
        if not self.turn_scores:
            return {
                "overall_score": 0.0,
                "overall_action_score": 0.0,
                "overall_state_score": 0.0,
                "total_turns": 0,
                "per_tag_scores": {},
                "max_streak": 0
            }

        # Calculate overall scores
        total_turns = len(self.turn_scores)
        overall_total = sum(self.turn_scores) / total_turns * 100
        overall_action = sum(self.turn_action_scores) / total_turns * 100
        overall_state = sum(self.turn_state_scores) / total_turns * 100

        # Calculate per-tag scores
        per_tag_scores = {}
        for tag, scores in self.tag_scores.items():
            if scores:
                per_tag_scores[tag] = sum(scores) / len(scores) * 100

        return {
            "overall_score": round(overall_total, 2),
            "overall_action_score": round(overall_action, 2),
            "overall_state_score": round(overall_state, 2),
            "total_turns": total_turns,
            "per_tag_scores": {tag: round(score, 2) for tag, score in per_tag_scores.items()},
            "max_streak": round(self.max_streak / total_turns * 100, 2) if total_turns > 0 else 0.0
        }

    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Get detailed results including turn-by-turn breakdown.

        Returns:
            Dict with:
                - summary: Overall summary (same as view_results)
                - turn_by_turn: List of individual turn results
                - per_tag_breakdown: Detailed breakdown per tag
        """
        summary = self.view_results()

        # Turn-by-turn breakdown
        turn_by_turn = []
        for i, (total, action, state, tag) in enumerate(zip(
            self.turn_scores,
            self.turn_action_scores,
            self.turn_state_scores,
            self.turn_tags
        ), 1):
            turn_by_turn.append({
                "turn": i,
                "tag": tag,
                "total_score": total,
                "action_score": action,
                "state_score": state,
                "is_perfect": total >= 1.0
            })

        # Per-tag detailed breakdown
        per_tag_breakdown = {}
        for tag, scores in self.tag_scores.items():
            perfect_count = sum(1 for s in scores if s >= 1.0)
            per_tag_breakdown[tag] = {
                "total_turns": len(scores),
                "perfect_turns": perfect_count,
                "average_score": round(sum(scores) / len(scores) * 100, 2) if scores else 0.0,
                "perfect_rate": round(perfect_count / len(scores) * 100, 2) if scores else 0.0
            }

        return {
            "summary": summary,
            "turn_by_turn": turn_by_turn,
            "per_tag_breakdown": per_tag_breakdown
        }