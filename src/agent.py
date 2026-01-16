import json
# from json_repair import repair_json
import json_repair
from typing import Any, List, Dict
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart, FilePart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from green_agent_v2 import AdaptiveGenerator, WeaknessAnalyzer, TurnEvaluator, TestResult
from green_agent_v2.visualize import ReportGenerator
from app import SmartHomeEnv

class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class Agent:
    """
    Green Agent Testing Pipeline.

    In this pipeline, the Green Agent acts as the examiner. It fetches questions, 
    administers them to the examinee (Purple Agent), manages the necessary multi-turn 
    interactions, and finally collects and forwards the responses to the evaluator.

    The pipeline operates adaptively:
    1. The initial round uses a default 'pyramid' distribution strategy to sample questions.
    2. Subsequent rounds generate new question sets targeting the specific weaknesses 
       identified in the previous round's evaluation.
    3. The process continues until the maximum number of test rounds is reached or 
       the results converge. Finally, the Green Agent returns the comprehensive evaluation results.

    Required configurations in `scenario.toml`:

    - max_test_rounds (int): 
        Specifies the maximum number of testing iterations. 一个test round包含n个test case，test case的数量 = weakness_num * targeted_per_weakness
        Round 1 uses the default pyramid distribution. For the remaining (max_test_rounds - 1) 
        rounds, questions are dynamically generated based on the top `weakness_num` 
        weaknesses and `targeted_per_weakness`.

    - weakness_num (int): 
        The number of top weaknesses (k) to prioritize when generating adaptive question 
        sets for subsequent rounds.

    - targeted_per_weakness (int): 
        The number of new test cases to generate for each identified weakness category. 
        For example, if `weakness_num` is 2 and this value is 3, a total of 6 new 
        questions will be generated for the next round.

    - convergence_threshold (float): 
        A value between 0.0 and 1.0. This defines the stopping criterion based on stability; 
        testing stops early if the change in pass rate between two consecutive rounds 
        is less than this threshold.
    """
    required_roles: list[str] = ['purple']
    required_config_keys: list[str] = ['max_test_rounds', 'weakness_num', 'targeted_per_weakness', 'convergence_threshold']

    def __init__(self):
        self.messenger = Messenger()
        self.test_case_generator = AdaptiveGenerator()
        self.env = SmartHomeEnv()
        self.analyser = WeaknessAnalyzer()
        self.report_generator = ReportGenerator()
        self.round_history = []
        self.all_results: List[TestResult] = []

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

        Use self.messenger.talk_to_agent(message, url) to call other agents.
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
            # Not a valid EvalRequest - this is a regular message (e.g., smoke test)
            # Return a friendly response for A2A protocol compliance
            await updater.complete(new_agent_text_message(
                "Hello! I'm the SmartMem Green Agent. "
                "I evaluate Purple Agents on smart home memory tasks. "
                "Send me an EvalRequest JSON with 'participants' and 'config' to start an evaluation."
            ))
            return

        purple_addr = request.participants['purple']
        max_rounds = request.config['max_test_rounds']
        weakness_num = request.config['weakness_num']
        target_count = request.config['targeted_per_weakness']
        threshold = request.config['convergence_threshold']
        

        # 题目的格式：
        # {{
        # "scenario_id": "scenario_A1_B1_C0_N0",
        # "difficulty": "easy|medium|difficult",
        # "dimension": "precision|ambiguous|conflict|memory|noise",
        # "description": "Brief English description",
        # "initial_state": {{"device_key": "valid_value"}},
        # "turns": [
        #     {{
        #     "turn_id": 1,
        #     "gm_instruction": "User instruction",
        #     "expected_agent_action": [
        #         {{"action": "update", "key": "device_key", "value": "valid_value"}}
        #     ],
        #     "expected_final_state": {{"device_key": "valid_value"}}
        #     }}
        # ]
        # }}

        # cold start: 
        new_test_cases = self.test_case_generator.generate_initial_pyramid() # Here we generate 6 test cases for 5 evaluation dimensions in a pyramid shape (easy: medium: hard = 3:2:1)
        all_test_cases = [new_test_cases, ]
        
        last_round_pass_rate = 0
        for round_cnt in range(max_rounds):
            round_num = round_cnt + 1
            focus = "General Pyramid" if round_cnt == 0 else "Weakness Targeted"
            
            await updater.update_status(
                TaskState.working, 
                new_agent_text_message(f"Round {round_num}/{max_rounds}: Testing ({focus})...")
            )
            
            current_round_results: List[TestResult] = []
            for test_case in new_test_cases:
                self.env.reset_turn_history()
                self.env.reset(initial_state=test_case['initial_state'])
                is_new_conversation = True
                
                turn_res = {}
                test_case_total_score, test_case_max_score = 0.0, 0.0
                turn_details = []
                test_case_all_errors = []
                # result ds:
                # {
                #     "score": score,
                #     "details": {
                #         "sequence_match": seq_match,
                #         "state_match": state_match,
                #         "errors": errors
                #     },
                #     "message": msg
                # }
                for turn in test_case['turns']:
                    turn_id = turn.get('turn_id', 0)
                    instruction = turn.get('gm_instruction', '')
                    expected_actions = turn.get('expected_agent_action', [])
                    expected_state = turn.get('expected_final_state', {})
                    
                    evaluator = TurnEvaluator(expected_actions=expected_actions, expected_final_state=expected_state)
                    # Interaction Loop
                    current_input = instruction
                    while True:
                        agent_reply = self.messenger.talk_to_agent(
                            message=current_input, 
                            url=purple_addr, 
                            new_conversation=is_new_conversation
                        )
                        is_new_conversation = False 

                        try:
                            parsed_actions = json_repair(agent_reply, return_objects=True)
                        except Exception:
                            parsed_actions = None

                        # 1. Tool Use
                        if isinstance(parsed_actions, list) and len(parsed_actions) > 0:
                            env_res: List[Dict] = []
                            for action in parsed_actions:
                                _res = self.env.update_state(action)
                                env_res.append(_res)
                            current_input = json.dumps(env_res)
                        
                        # 2. Turn Complete
                        else:
                            # evaluate turn
                            turn_res = evaluator.evaluate(actual_actions=self.env.get_action_history(), actual_final_state=self.env.get_state())
                            break
                    
                    turn_score = turn_res['score']
                    test_case_total_score += turn_score
                    test_case_max_score += 1 # 实际上应该就是turn的数量？
                    turn_details.append(
                        {
                            'turn_id': turn_id,
                            'instruction': instruction,
                            'score': turn_score,
                            'max_score': 1.0,
                            'passed': turn_score == 1.0,
                            'errors': turn_res.get('details', {}).get('errors', [])
                        }
                    )
                    if turn_res.get('details', {}).get('errors'):
                        test_case_all_errors.extend(turn_res['details']['errors'])
                    
                test_case_final_score = test_case_total_score / max(1, test_case_max_score)
                ifthistc_passed = test_case_final_score >= 1.0
                current_round_results.append(
                    TestResult(
                        test_case=test_case,
                        score=test_case_total_score,
                        max_score=test_case_max_score,
                        passed=ifthistc_passed,
                        errors=test_case_all_errors,
                        turn_details=turn_details
                    )
                )

            self.all_results += current_round_results
            self.analyser.analyze(current_round_results)  # Update weakness profile
            current_round_pass_rate = sum(1 for r in current_round_results if r.passed) / max(1, len(current_round_results))
            
            # Record round history
            self.round_history.append({
                "round": round_num,
                "focus": focus,
                "total_cases": len(current_round_results),
                "passed": sum(1 for r in current_round_results if r.passed),
                "pass_rate": current_round_pass_rate
            })
            
            if abs(current_round_pass_rate-last_round_pass_rate) < threshold:
                break  # Convergence reached
            
            last_round_pass_rate = current_round_pass_rate
            
            # Generate targeted test cases for next round (if not the last round)
            if round_cnt < max_rounds - 1:
                top_weaknesses = self.analyser.get_top_weaknesses(weakness_num)
                new_test_cases = self.test_case_generator.generate_targeted(top_weaknesses, target_count)
                all_test_cases.append(new_test_cases)
        
        # Generate final report
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message("Generating assessment report...")
        )
        
        report = self.report_generator.generate_report(
            profile=self.analyser.profile,
            round_history=self.round_history,
            all_results=self.all_results,
            agent_name="Purple Agent"
        )
        
        # Build artifact parts
        artifact_parts = [
            Part(root=TextPart(text=report['text']))
        ]
        
        # Add structured data
        summary_data = report['data'].get('summary', {})
        dimension_stats = report['data'].get('dimension_stats', {})
        artifact_parts.append(Part(root=DataPart(data={
            "summary": summary_data,
            "dimension_stats": dimension_stats,
            "radar_data": report['data'].get('radar_data', {}),
            "boundaries": report['data'].get('boundaries', {}),
            "round_history": self.round_history
        })))
        
        # Add chart files if available
        for chart_name, chart_bytes in report.get('charts', {}).items():
            import base64
            # FilePart expects a file URI, so we'll include charts in DataPart as base64
            pass  # Charts are included in the DataPart above if needed
        
        await updater.add_artifact(
            parts=artifact_parts,
            name="Assessment Result",
        )
        
        await updater.complete(new_agent_text_message(
            f"Assessment completed. Pass rate: {summary_data.get('pass_rate', 0)*100:.1f}%"
        ))