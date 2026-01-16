import os
import json
import logging
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIError, RateLimitError
from json_repair import repair_json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .prompts import SYSTEM_PROMPT, USER_PROMPT, difficulty_specs, dimension_specs
from .base import TestCase

load_dotenv()

logger = logging.getLogger(__name__)

class LLMCaseGenerator:
    """
    Generator that interfaces with the LLM to create specific test cases.
    Replaces the original 'GreenAgent'.
    """
    def __init__(self):
        # 1. Load API Credentials - Support multiple providers
        # Priority: GOOGLE_API_KEY (Gemini) > OPENAI_API_KEY > error
        google_api_key = os.getenv("GOOGLE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if google_api_key:
            # Use Gemini via OpenAI-compatible endpoint
            api_key = google_api_key
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            default_model = "gemini-2.0-flash"
            logger.info("Using Google Gemini API")
        elif openai_api_key:
            api_key = openai_api_key
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            default_model = "gpt-4o"
            logger.info("Using OpenAI API")
        else:
            raise ValueError("No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in environment.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # 2. Load Model Configurations from .env
        self.model_name = os.getenv("MODEL_NAME", default_model)

        gen_args_str = os.getenv("MODEL_GEN_ARGS", '{}')
        try:
            self.model_gen_args = repair_json(gen_args_str, return_objects=True)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse MODEL_GEN_ARGS from .env: {e}. Using empty dict.")
            self.model_gen_args = {}

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APIError, RateLimitError)),
        stop=stop_after_attempt(3),      # Retry up to 3 times
        wait=wait_exponential(multiplier=1, min=4, max=10) # Wait 4s, 8s, 10s...
    )
    def _call_llm_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Internal method to call LLM with retry logic and dynamic parameters.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_gen_args # Unpack arguments loaded from .env
        )
        return response.choices[0].message.content

    def generate_single_case(
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
            formatted_user_prompt = USER_PROMPT.format(
                difficulty=difficulty,
                difficulty_specs=difficulty_specs.get(difficulty, ""),
                dimension=dimension,
                dimension_specs=dimension_specs.get(dimension, ""),
                scenario_number=scenario_number
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_user_prompt}
            ]

            # Call LLM with retry logic
            content = self._call_llm_api(messages)
            
            if not content:
                logger.warning("Received empty content from LLM.")
                return None
                
            # Parse the JSON response using repair_json
            try:
                case_data = repair_json(content, return_objects=True)
            except Exception as e:
                logger.error(f"JSON parsing failed. Content snippet: {content[:100]}... Error: {e}")
                return None
            
            # Validate format (Assuming TestCase behaves like a dict or can be cast to one)
            if not isinstance(case_data, dict):
                logger.warning(f"Parsed data is not a dict, got {type(case_data)}.")
                return None

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
    
    def __init__(self, use_static: bool = False):
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
    
    def generate_targeted(
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

                case = self.case_generator.generate_single_case(
                    difficulty=current_difficulty,
                    dimension=dimension,
                    scenario_number=len(generated) + 1
                )
                if case:
                    generated.append(case)
        
        return generated

    def generate_initial_pyramid(self) -> List[TestCase]:
        """
        Generates a fixed set of test cases using a pyramid distribution.
        
        If use_static=True, returns predefined static test cases (for testing/CI).
        Otherwise, uses LLM to generate test cases.
        
        Hardcoded Strategy (6 cases per dimension):
        - Dimensions: ["precision", "ambiguous", "conflict", "memory", "noise"]
        - Distribution: 3 Easy, 2 Medium, 1 Difficult
        """
        # Use static test cases if configured
        if self._use_static:
            print(f"  ▲ Using STATIC test cases ({len(STATIC_TEST_CASES)} cases)")
            return STATIC_TEST_CASES.copy()
        
        generated = []
        
        dimensions = ["precision", "ambiguous", "conflict", "memory", "noise"]
        
        difficulty_counts = {
            'easy': 3,
            'medium': 2,
            'difficult': 1
        }
        
        print(f"  ▲ Starting Fixed Pyramid Generation (Target: {len(dimensions) * 6} cases)...")

        for dimension in dimensions:
            print(f"    > Processing dimension: {dimension}")
            
            for difficulty, count in difficulty_counts.items():
                # print(f"      - Generating {count} {difficulty} cases") 
                
                for _ in range(count):
                    case = self.case_generator.generate_single_case(
                        difficulty=difficulty,
                        dimension=dimension,
                        scenario_number=len(generated) + 1
                    )
                    if case:
                        generated.append(case)
                        
        return generated