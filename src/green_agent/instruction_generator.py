import os
import json
import logging
from typing import List, Tuple, Dict, Optional

import asyncio
import json_repair
from pydantic import ValidationError
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .prompts import SYSTEM_PROMPT, USER_PROMPT, difficulty_specs, dimension_specs
from .base import TestCase

load_dotenv() # 如果我们在本地测试

logger = logging.getLogger('smartmem_green_agent')

class LLMCaseGenerator:
    """
    Generator that interfaces with the LLM to create specific test cases.
    Replaces the original 'GreenAgent'.
    """
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY") # 直接从环境读取api key和base url, 不再做fallback和分类处理
        base_url = os.getenv('OPENAI_BASE_URL')
        assert api_key and base_url, "Missing API KEY and BASE URL. Please set them in environment."

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.model_name = os.getenv("MODEL_NAME")
        assert self.model_name, "Please specify the backbone model you want to use."

        gen_args_str = os.getenv("MODEL_GEN_ARGS", '{}')
        try:
            self.model_gen_args = json_repair.loads(gen_args_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse MODEL_GEN_ARGS from .env: {e}. Using empty dict.")
            self.model_gen_args = {}

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APIError, RateLimitError)),
        stop=stop_after_attempt(3),      # Retry up to 3 times
        wait=wait_exponential(multiplier=1, min=4, max=10) # Wait 4s, 8s, 10s...
    )
    async def _call_llm_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Internal method to call LLM with retry logic and dynamic parameters.
        """
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_gen_args
        )
        return response.choices[0].message.content

    async def generate_single_case(
        self, 
        difficulty: str, 
        dimension: str, 
        scenario_number: int
    ) -> Optional[TestCase]:
        """
        Generates a single test case based on difficulty and dimension.
        """
        try:
            # Construct the prompt
            #FIXME: 目前LLM同时给出题目和答案的范式下, 对于提问式的指令通常没有给出的expected_actions, 参考/home/crema/SmartMem/test_cases/batch_15_difficult_noise.json
            formatted_user_prompt = USER_PROMPT.format(
                difficulty=difficulty,
                difficulty_specs=difficulty_specs.get(difficulty, ""),
                dimension=dimension,
                dimension_specs=dimension_specs.get(dimension, ""),
                scenario_number=scenario_number
            )#TODO: 这里的scenario_number放在prompt里喂给LLM好像没有意义啊

            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_user_prompt}
            ]

            content = await self._call_llm_api(messages)
            
            if not content:
                logger.warning("Received empty content from LLM.")
                return None
                
            # 先转json
            try:
                case_data = json_repair.loads(content)
            except Exception as e:
                logger.error(f"JSON parsing failed. Content snippet: {content[:100]}... Error: {e}")
                return None
            
            # 然后校验格式
            try:
                case = TestCase.model_validate(case_data)
            except ValidationError as e:
                logger.warning(f"Test Case Generation Error: {e.json}")
                print(f"Generated Content: {case_data}")

            # Add metadata
            case_data['difficulty'] = difficulty
            case_data['dimension'] = dimension
            
            return case_data

        except Exception as e:
            logger.error(f"Error in generate_single_case: {e}")
            return None


# Static test cases for testing without LLM API calls
STATIC_TEST_CASES = [
    {
        "scenario_id": "static_precision_easy_001",
        "difficulty": "easy", 
        "dimension": "precision",
        "description": "Turn on the living room light",
        "initial_state": {"living_room_light": "off", "living_room_brightness": 50},
        "turns": [{
            "turn_id": 1,
            "gm_instruction": "Please turn on the living room light.",
            "expected_agent_action": [{"action": "update", "key": "living_room_light", "value": "on"}],
            "expected_final_state": {"living_room_light": "on", "living_room_brightness": 50}
        }]
    },
    {
        "scenario_id": "static_memory_medium_001",
        "difficulty": "medium",
        "dimension": "memory",
        "description": "Remember and recall a preference",
        "initial_state": {"bedroom_temperature": 22, "bedroom_ac": "off"},
        "turns": [
            {
                "turn_id": 1,
                "gm_instruction": "I prefer the bedroom temperature at 25 degrees.",
                "expected_agent_action": [{"action": "update", "key": "bedroom_temperature", "value": 25}],
                "expected_final_state": {"bedroom_temperature": 25, "bedroom_ac": "off"}
            },
            {
                "turn_id": 2, 
                "gm_instruction": "Set the bedroom to my preferred temperature.",
                "expected_agent_action": [{"action": "update", "key": "bedroom_temperature", "value": 25}],
                "expected_final_state": {"bedroom_temperature": 25, "bedroom_ac": "off"}
            }
        ]
    },
    {
        "scenario_id": "static_conflict_easy_001",
        "difficulty": "easy",
        "dimension": "conflict",
        "description": "Handle conflicting instructions",
        "initial_state": {"kitchen_light": "off"},
        "turns": [{
            "turn_id": 1,
            "gm_instruction": "Turn on the kitchen light, but actually keep it off.",
            "expected_agent_action": [],
            "expected_final_state": {"kitchen_light": "off"}
        }]
    },
    {
        "scenario_id": "static_ambiguous_easy_001",
        "difficulty": "easy",
        "dimension": "ambiguous",
        "description": "Handle ambiguous instruction",
        "initial_state": {"living_room_light": "off", "bedroom_light": "off"},
        "turns": [{
            "turn_id": 1,
            "gm_instruction": "Turn on the light.",
            "expected_agent_action": [{"action": "update", "key": "living_room_light", "value": "on"}],
            "expected_final_state": {"living_room_light": "on", "bedroom_light": "off"}
        }]
    },
    {
        "scenario_id": "static_noise_easy_001",
        "difficulty": "easy",
        "dimension": "noise",
        "description": "Filter noise from instructions",
        "initial_state": {"tv": "off"},
        "turns": [{
            "turn_id": 1,
            "gm_instruction": "Um, could you please, like, turn on the TV? Thanks!",
            "expected_agent_action": [{"action": "update", "key": "tv", "value": "on"}],
            "expected_final_state": {"tv": "on"}
        }]
    }
]


class AdaptiveGenerator:
    """
    Adaptive strategy for generating test cases based on weaknesses or initial setup.
    """
    
    def __init__(self, use_static: bool = False): #TODO: 这里的使用静态数据跟像是做测试用的, 改成允许从某个地方读取题库吧
        # Lazy initialization - don't create LLMCaseGenerator until needed
        # This allows smoke tests to pass without API keys
        self._case_generator = None
        self._use_static = use_static
    
    @property
    def case_generator(self) -> LLMCaseGenerator:
        """Lazily initialize the LLM case generator when first accessed."""
        if self._case_generator is None:
            self._case_generator = LLMCaseGenerator()
        return self._case_generator
    
    async def generate_targeted(
        self, 
        weaknesses: List[Tuple[str, str, float]], 
        count_per_weakness: int = 5,
        difficulty_boost: bool = True
    ) -> List[TestCase]:
        """
        Generates test cases targeted at identified weaknesses.
        
        If use_static=True, returns filtered static test cases.
        
        Args:
            weaknesses: List of (type, name, score) tuples.
            count_per_weakness: Number of cases to generate per weakness.
            difficulty_boost: Whether to progressively increase difficulty.
        """
        # Use static test cases if configured
        if self._use_static:
            # Filter static cases by weakness dimensions
            weakness_dims = {w[1] for w in weaknesses if w[0] == 'dimension'}
            filtered = [c for c in STATIC_TEST_CASES if c.get('dimension') in weakness_dims]
            print(f"  🎯 Using STATIC targeted cases ({len(filtered)} cases for {weakness_dims})")
            return filtered[:count_per_weakness * len(weaknesses)] if filtered else STATIC_TEST_CASES[:2]
        
        generated = []
        tasks = []
        scenario_counter = 1
        difficulties_order = ['easy', 'medium', 'difficult'] 
        
        for weakness_type, weakness_name, weakness_score in weaknesses:
            # Determine generation parameters based on weakness type
            if weakness_type == 'dimension':
                dimension = weakness_name
                # Determine difficulty based on weakness score
                if weakness_score > 0.7:
                    difficulty = 'easy'  # Obvious weakness
                elif weakness_score > 0.4:
                    difficulty = 'medium'
                else:
                    difficulty = 'difficult'  # Hidden weakness
            else:
                # Device weakness, default to precision dimension
                dimension = 'precision'
                difficulty = 'medium'
            
            base_difficulty = difficulty
            
            print(f"  🎯 Targeting weakness [{weakness_type}: {weakness_name}] | Base difficulty: {base_difficulty}")
            
            difficulties_order = ['easy', 'medium', 'difficult']
            
            for i in range(count_per_weakness):
                current_difficulty = base_difficulty
                
                # Logic to boost difficulty progressively
                if difficulty_boost and i > 0:
                    try:
                        idx = difficulties_order.index(base_difficulty)
                        # Every 2 cases, try to increase difficulty
                        if i >= 2 and idx < len(difficulties_order) - 1:
                            current_difficulty = difficulties_order[idx + 1]
                    except ValueError:
                        pass
                
                task = self.case_generator.generate_single_case(
                    difficulty=current_difficulty,
                    dimension=dimension,
                    scenario_number=scenario_counter
                )
                tasks.append(task)
                scenario_counter += 1
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"  ❌ Task failed with error: {res}")
            elif res is None:
                print("  ⚠️ Task returned None (generation failed)")
            else:
                generated.append(res)
                
        return generated

    def generate_initial_pyramid(self) -> List[TestCase]:
        """从data/init_cases.json中加载冷启动数据, 这个初始数据一共30题, 每个维度按照简单：中等：困难=3:2:1进行采样"""
        from pathlib import Path
        dir_path = Path(__file__).parent
        with open(dir_path/"data"/"init_cases.json", 'r') as f:
            init_cases = json.load(f)
        
        return init_cases


async def test_generator():
    """
    Test function to verify the LLM-based generator works correctly.

    This function tests:
    1. LLMCaseGenerator initialization (API key and model config)
    2. Single test case generation via LLM API
    3. AdaptiveGenerator pyramid generation
    4. AdaptiveGenerator targeted generation

    Requires: GOOGLE_API_KEY or OPENAI_API_KEY in environment/.env
    """
    # Configure logging to see INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Testing LLM Instruction Generator")
    print("=" * 60)

    # Test 1: Initialize generator and verify API configuration
    print("\n[TEST 1] Initializing LLMCaseGenerator...")
    try:
        generator = LLMCaseGenerator()
        print(f"    ✓ Generator initialized successfully")
        print(f"    Base URL: {generator.client.base_url}")
        api_key = generator.client.api_key
        masked_key = api_key[:8] + "..." + api_key[-4:] if api_key and len(api_key) > 12 else "***"
        print(f"    API Key: {masked_key}")
        print(f"    Model: {generator.model_name}")
        print(f"    Generation args: {generator.model_gen_args}")
    except ValueError as e:
        print(f"    ✗ Initialization failed: {e}")
        print("    Please set GOOGLE_API_KEY or OPENAI_API_KEY in .env file")
        return
    except Exception as e:
        print(f"    ✗ Unexpected error: {e}")
        return

    # Test 2: Generate a single test case
    print("\n[TEST 2] Generating single test case...")
    print("    Calling LLM API (difficulty=easy, dimension=precision)...")
    try:
        case = await generator.generate_single_case(
            difficulty='easy',
            dimension='precision',
            scenario_number=1
        )
        if case:
            print(f"    ✓ Successfully generated test case!")
            print(f"      Scenario ID: {case.get('scenario_id', 'N/A')}")
            print(f"      Difficulty: {case.get('difficulty', 'N/A')}")
            print(f"      Dimension: {case.get('dimension', 'N/A')}")
            print(f"      Description: {case.get('description', 'N/A')[:80]}...")
            print(f"      Turns: {len(case.get('turns', []))}")
            print("\n    --- Generated Content (JSON) ---")
            print(json.dumps(case, indent=2, ensure_ascii=False))
            print("    ------------------------------")
        else:
            print("    ✗ Failed to generate test case (returned None)")
    except Exception as e:
        print(f"    ✗ Error during generation: {e}")
        return

    # Test 3: Generate multiple cases with AdaptiveGenerator
    print("\n[TEST 3] Testing AdaptiveGenerator (load initial test cases)...")
    try:
        adaptive_gen = AdaptiveGenerator(use_static=False)
        print("    Generating initial pyramid test cases...")
        init_cases = await adaptive_gen.generate_initial_pyramid()
        if init_cases:
            print("    ✓ Generated initial test cases")
            print("\n    --- Generated Content (JSON) ---")
            print(json.dumps(init_cases, indent=2, ensure_ascii=False))
            print("    ------------------------------")
        else:
            print("    ✗ Failed to generate initial test case (returned None)")

        # Test targeted generation instead (fewer API calls)
        print("\n[TEST 4] Testing targeted generation...")
        weaknesses = [('dimension', 'precision', 0.8)]
        targeted_cases = await adaptive_gen.generate_targeted(
            weaknesses=weaknesses,
            count_per_weakness=1
        )
        print(f"    ✓ Generated {len(targeted_cases)} targeted cases")
        if targeted_cases:
            print("\n    --- Generated Content (JSON) ---")
            print(json.dumps(targeted_cases, indent=2, ensure_ascii=False))
            print("    ------------------------------")
        else:
            print("    ✗ Failed to generate targeted test case (returned None)")
            

    except Exception as e:
        print(f"    ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_generator())