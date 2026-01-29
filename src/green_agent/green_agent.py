from collections import deque
from typing import List, Dict
import logging
import json

from dotenv import load_dotenv

from .base import TestCase, TestResult, TurnCache
from .simulator import SmartHomeEnv
from .evaluator import WeaknessAnalyzer, TurnEvaluator
from .instruction_generator import AdaptiveGenerator

load_dotenv()

logger = logging.getLogger("smartmem_purple_agent")



class GreenAgent:
    """
    现在需要实现的功能：
    1. 调动题库继续生成 question_bank
    2. 根据purple agent回答的内容做调度 (转发给工具、发送下一条题目+触发评估) router
    3. 维护评估结果和历史信息 history, results
    4. 把响应速度也加入评估结果
    5. 维护/管理虚拟环境 simulator
    
    
    进阶TODO:
    1. 让交互更真实：如果反复错误...如果意图理解错误...
    2. 增加个性化提醒
    3. 引入随机错误 (考虑语音识别错误) - 首先给purple发一个指令, 然后告诉它听错了...
    4. 重构generator和evaluator
    """
    def  __init__(self, max_rounds: int, top_k_weakness: int, targeted_per_weakness: int, convergence_threshold: float):
        """
        Args:
        
            max_rounds: Specifies the maximum number of testing iterations if not converged. One test round includes n test case.
            
            top_k_weakness: The number of top weaknesses (k) to prioritize when generating adaptive question sets for subsequent rounds.
            
            targeted_per_weakness: The number of new test cases to generate for each identified weakness category.
            
            convergence_thershold: Testing stops early if the change in pass rate between two consecutive rounds is less than this threshold.
        """
        self.simulator = SmartHomeEnv()
        self.analyser = WeaknessAnalyzer()
        self.test_case_generator = AdaptiveGenerator(use_static=False)
        self.generator_config = {"top_k_weakness": top_k_weakness, 'targeted_per_weakness': targeted_per_weakness}
        self.remaining_test_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        
        self.pending_cases: List[TestCase] = deque() # 包含n个test_case, test_case_i有k_i个turn, 每个turn包含一条instruction
        self.pending_cases.extend(self.test_case_generator.generate_initial_pyramid())
        self.archived_cases: List[TestCase] = []
        
        self.results: List[TestResult]= []
        self.case_history: List[Dict] = [] # 对话历史, 每个case中green和purple都说了什么
        self.last_round_pass_rate = 0
        
        self.current_case: TestCase = self.pending_cases.popleft()
        first_turn = self.current_case.turns[0]
        self.turn_cache = TurnCache(
            instruction=first_turn.gm_instruction,
            evaluator=TurnEvaluator(expected_actions=first_turn.expected_agent_action),
            agent_actions=[],
            chat_history=[{"role": "user", "content": first_turn.gm_instruction}]
        )
        self.simulator.reset(self.current_case.initial_state)
        
    async def step(self, input_text: str):
        """
        进行一次green-purple交互。如果pending_cases空了, 那么一个test round结束。
        """
        if self.remaining_test_rounds == 0:
            eval_report = self.generate_report()
            return ...#TODO: 评测结束, 返回结果
        
        # 根据内容决定下一步做什么
        agent_reply = json.loads(input_text)
        msg_type, msg_content = agent_reply['message_type'], agent_reply['message_content']
        msg2send = {"message_type": "", "message_content": ""}
        
        if msg_type == "tool":
            # 打包返回的工具结果
            tool_res = []
            agent_actions = []
            for tool_info in msg_content:
                if tool_info['action'] == 'update':
                    _res = self.simulator.update_state(key=tool_info['device_id'], value=tool_info['value'])
                else:
                    key = tool_info['device_id']
                    if key == 'all':
                        _res = self.simulator.get_state()
                    else:
                        _res = self.simulator.get_state(key=key)
                tool_res.append(_res)
                agent_actions.append({"action": tool_info['action'], 'key': None if key=='all' else key, 'value': tool_info.get('value')})
                
            msg2send['message_type'] = "tool"
            msg2send['message_content'] = json.dumps(tool_res)
            self.turn_cache.agent_actions.extend(agent_actions)

        else:
            msg2send['message_type'] = 'text'
            # 暂时约定如果开始返回文本说明这个turn的任务执行完了
            # 评估一下当前的turn
            
            # 新的题目
            # 当前case中还有turn没完成 - 更新turn cache
            
            # 当前case已经执行完成
            # 1. 拿新的case  2. 更新turn_cache
            if self.pending_cases:
                msg2send['message_content'] = self.pending_cases.popleft()
            else:
                self.remaining_test_rounds -= 1
                # 校验是否收敛
                # 没收敛就继续生成下一个题组加到pending cases
                # 发布题目
        
    
    def generate_report(self):
        """生成报告, 套在ReportGenerator上"""