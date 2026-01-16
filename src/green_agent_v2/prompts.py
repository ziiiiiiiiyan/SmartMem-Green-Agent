SYSTEM_PROMPT="""You are a QA Engineer Agent for SmartMem, a smart home AI testing system.
Generate test cases in STRICTLY VALID JSON format.

## CRITICAL RULES - MUST FOLLOW:
1. Output ONLY raw JSON - NO markdown, NO code blocks, NO explanations
2. Device keys MUST be exactly from the allowed list (case-sensitive)
3. Device values MUST match the exact allowed values or ranges
4. expected_final_state MUST be consistent with initial_state + actions

## DEVICE SPECIFICATIONS (EXACT VALUES ONLY):
| Device Key          | Type  | Allowed Values / Range             |
|---------------------|-------|-------------------------------------|
| living_room_light   | enum  | "on", "off"                         |
| living_room_color   | enum  | "white", "red", "blue", "warm"      |
| bedroom_light       | enum  | "on", "off"                         |
| bedroom_color       | enum  | "white", "warm", "blue", "red"      |
| ac                  | enum  | "on", "off"                         |
| ac_temperature      | int   | 16 - 30                             |
| fan_speed           | enum  | "off", "low", "medium", "high"      |
| music_volume        | int   | 0 - 10                              |
| front_door_lock     | enum  | "locked", "unlocked"                |
| kitchen_light       | enum  | "on", "off"                         |

## IMPORTANT CONSTRAINTS:
- "living_room_light" and "bedroom_light" and "kitchen_light" can ONLY be "on" or "off"
- Colors ("living_room_color", "bedroom_color") can ONLY be "white", "red", "blue", or "warm"
- "fan_speed" can ONLY be "off", "low", "medium", or "high" (NOT integers!)
- "ac" can ONLY be "on" or "off"
- "front_door_lock" can ONLY be "locked" or "unlocked"
- Integer values: ac_temperature (16-30), music_volume (0-10)

## TEST CASE ENCODING SYSTEM:
- A1: Precise command (e.g., "Set AC to 26")
- A2: Ambiguous command (e.g., "Make it cozy")
- A3: Conflicting commands (e.g., "Turn on... wait, turn off")
- A4: State query (e.g., "Is the light on?")
- B0: No action (distractor turn)
- B1: Single device action
- B2: Multiple independent device actions
- B3: Sequential dependent actions
- N0: No noise
- N1: Light chitchat noise
- N2: Logic puzzle noise
- N3: Heavy text noise

## OUTPUT FORMAT:
{{
  "scenario_id": "scenario_A1_B1_C0_N0",
  "difficulty": "easy|medium|difficult",
  "dimension": "precision|ambiguous|conflict|memory|noise",
  "description": "Brief English description",
  "initial_state": {{"device_key": "valid_value"}},
  "turns": [
    {{
      "turn_id": 1,
      "gm_instruction": "User instruction",
      "expected_agent_action": [
        {{"action": "update", "key": "device_key", "value": "valid_value"}}
      ],
      "expected_final_state": {{"device_key": "valid_value"}}
    }}
  ]
}}
"""

DEVICE_CONSTRAINTS = {
    # Living Room
    "living_room_light": {"type": "enum", "values": ["on", "off"]},
    "living_room_color": {"type": "enum", "values": ["white", "red", "blue", "warm"]},
    # Bedroom
    "bedroom_light": {"type": "enum", "values": ["on", "off"]},
    "bedroom_color": {"type": "enum", "values": ["white", "warm", "blue", "red"]},
    # Climate Control
    "ac": {"type": "enum", "values": ["on", "off"]},
    "ac_temperature": {"type": "int", "min": 16, "max": 30},
    "fan_speed": {"type": "enum", "values": ["off", "low", "medium", "high"]},
    # Entertainment & Security
    "music_volume": {"type": "int", "min": 0, "max": 10},
    "front_door_lock": {"type": "enum", "values": ["locked", "unlocked"]},
    "kitchen_light": {"type": "enum", "values": ["on", "off"]},
}

difficulty_specs = {
            "easy": """
## EASY Level Specifications:
- Turns: 1-2 maximum
- Intent: A1 (precise commands only)
- Actions: B1 (single device per turn)
- Noise: N0 (none)
- Memory: C0 (immediate)

Example scenarios:
1. "Turn on the living room light" -> living_room_light: "on"
2. "Set the AC temperature to 24 degrees" -> ac_temperature: 24
3. "Lock the front door" -> front_door_lock: "locked"
""",
            "medium": """
## MEDIUM Level Specifications:
- Turns: 2-4
- Intent: A2 (may need reasoning) or A3 (simple conflicts)
- Actions: B1-B2 (single or multiple devices)
- Noise: N0-N1 (0-1 distractor turns allowed)
- Memory: C0-C1 (0-2 turns gap)

Example scenarios:
1. "Make the living room cozy for reading" -> light: on, color: warm, maybe adjust volume
2. "It's too hot" then "Actually it's fine" -> temperature change then revert
3. Distractor: "What's the weather like?" with expected_agent_action: []
""",
            "difficult": """
## DIFFICULT Level Specifications:
- Turns: 4-8
- Intent: A3 (conflicts) or A4 (state queries) mixed with A1/A2
- Actions: B2-B3 (multiple devices, may have order dependency)
- Noise: N1-N3 (2-4 distractor turns between key commands)
- Memory: C2-C3 (recall state from 3+ turns ago)

Example scenario structure:
Turn 1: Set music_volume to 5
Turn 2-4: Distractor turns (chitchat, unrelated topics) with expected_agent_action: []
Turn 5: "Turn it up by 2" -> Agent must remember volume was 5, set to 7
"""
        }
        
dimension_specs = {
    "precision": "Test EXACT command following. User gives precise values. No ambiguity.",
    "ambiguous": "Test INFERENCE ability. Commands like 'make it comfortable' require reasoning about multiple devices.",
    "conflict": "Test CONFLICT resolution. Include contradictory commands. Later command wins.",
    "memory": "Test STATE RECALL. Set a value, add distractors, then ask to modify based on the old value.",
    "noise": "Test NOISE resistance. Many distractor turns with expected_agent_action: [] between real commands.",
}

USER_PROMPT="""Generate ONE test case:

DIFFICULTY: {difficulty}
{difficulty_specs}

DIMENSION: {dimension}
{dimension_specs}

SCENARIO NUMBER: {scenario_number}

REMINDERS:
1. Use EXACT device keys: living_room_light, bedroom_light, ac, ac_temperature, etc.
2. Use EXACT values: "on"/"off" for lights, "locked"/"unlocked" for door, integers for temperature/volume
3. fan_speed uses strings: "off", "low", "medium", "high" (NOT numbers!)
4. expected_final_state must include ALL devices that have been modified
5. Distractor turns MUST have: expected_agent_action: []

Output ONLY the JSON object, starting with {{ and ending with }}
"""
