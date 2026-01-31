"""
Prompt templates for converting device commands to natural language.
Each prompt is designed for a specific device type.
"""

import json


def get_unified_prompt(input_command, history_context=""):
    """
    Generate comprehensive natural language conversion prompt for unified synthesizer

    Args:
        input_command: Dictionary with command parameters
        history_context: String containing previous commands context (shows recent commands for conversational reference)

    Returns:
        str: Complete prompt for LLM
    """
    prompt = f"""You are a home assistant converting structured smart home control commands into natural, conversational English language.

IMPORTANT - ROLE PLAYING:
- DEFAULT ROLE: You are the HOMEOWNER speaking to control the smart home system
- EXCEPTION: ONLY when the action is "send_visitor_msg" with source="visitor", you are a STRANGER/VISITOR speaking
- Most commands should sound like the owner giving instructions to their home system

HISTORY CONTEXT (for conversational reference):
{history_context}

Device types and operation guidelines:

[1. Air Conditioner (AC)]
- CRITICAL: ALWAYS explicitly mention "AC" or "Air conditioner" to avoid ambiguity
- "update" + power: Power control
  * power="on": Example outputs - "Turn on the AC", "AC on", "Start the AC"
  * power="off": Example outputs - "Turn off the AC", "AC off", "Shut down the AC"
- "update" + mode: Mode switching
  * mode="cooling": Example outputs - "AC cooling mode", "Switch the AC to cooling"
  * mode="heating": Example outputs - "AC heating mode", "Change the AC to heating"
  * mode="dehumidify": Example outputs - "AC dehumidify mode", "Set the AC to dehumidify"
- "update" + temperature: Temperature adjustment
  * temperature is a number (16-30): Set AC to this temperature (Celsius)
  * Example outputs - "Set AC to 26 degrees", "Make the AC 24", "AC temperature up to 28"
- "update" + fan_speed: Fan speed adjustment
  * fan_speed="auto": Example outputs - "AC fan to auto", "Set AC fan to auto"
  * fan_speed="1"/"2"/"3": Example outputs - "AC fan speed 1", "Set AC fan to level 2"
- "update" + sleep_mode: Sleep mode
  * sleep_mode="on": Example outputs - "Turn on AC sleep mode", "Enable AC sleep mode"
  * sleep_mode="off": Example outputs - "Turn off AC sleep mode", "Disable AC sleep mode"
- "update" + timer: Timer setting
  * timer is a number (0-10 hours, 0.5 hour increments): AC auto-off after this duration
  * timer=0 means cancel timer: Example outputs - "Cancel AC timer", "Never mind, no AC timer"
  * timer=2.0: Example outputs - "Set AC timer for 2 hours", "AC 2-hour timer"
- "read": Query AC status
  * Example outputs - "What's the AC status?", "Check the AC settings", "What's the AC temperature?"

[2. Lights]
- "set_preference" + color + device: Set default color preference for a light
  * MUST use phrasing like "I prefer", "Set as default from now on" to express long-term preference
  * Example outputs - "I prefer white color for living room light", "Set bedroom light to warm by default"
- "update" + note: Adjust lighting based on scenario description (fuzzy command)
  * note is a scenario description, express as "prepare for ... lighting"
  * Example outputs for note="reading" - "Prepare reading lights", "Set up reading mode"
  * Example outputs for note="relaxing" - "Give me relaxing lights", "Set lights for relaxation"
- "update" + power: Power control
  * device="Bedroom Light": refer as "bedroom light" or "the bedroom light"
  * device="Living Room Light": refer as "living room light" or "the living room light"
  * device="Kitchen Light": refer as "kitchen light"
  * power="on": Example outputs - "Turn on [device]", "[device] on"
  * power="off": Example outputs - "Turn off [device]", "[device] off"
- "update" + color: Direct color specification
  * Color options: white, red, blue, warm
  * Example outputs - "Make living room light red", "Change bedroom light to blue"
- "read": Query light status
  * Example outputs - "Is [device] on?", "Check [device]", "What color is [device]?"

[3. Speaker]
- "update" + volume: Volume adjustment
  * volume is a number 0-10, MUST always specify the exact number
  * Example outputs - "Volume 5", "Set to 7", "Make it 3"
  * NEVER use vague terms like "louder", "quieter", "up", "down"
- "read": Query volume
  * Example outputs - "What's the volume?", "How loud is it?"

[4. Security]
- "update" + door_lock: Door lock control
  * door_lock="open": Example outputs - "Open the door", "Unlock the door"
  * door_lock="closed": Example outputs - "Close the door", "Lock the door"
  * If note field contains "owner comes back" or similar: Say "I'm back" or "I'm home"
  * If note field exists (other scenarios): incorporate it naturally
- "read": Query door lock state
  * Example outputs - "Is the door locked?", "Check the door", "What's the door status?"
- "send_user_msg" + source="owner": Owner leaving message (going out)
  * msg_content format: "reason+return time"
  * Example outputs for "supermarket+30 minutes" - "I'm going to the supermarket, back in 30 minutes"
  * Example outputs for "gym+2 hours" - "Heading to the gym for 2 hours"
- "send_visitor_msg" + source="visitor": Visitor message (STRANGER/VISITOR speaking)
  * Start with "Stranger:" then write natural spoken content
  * Example outputs for msg_content="neighbor borrowing something" - "Stranger: Hi, I'm your neighbor, need to borrow something"
  * Example outputs for msg_content="courier delivery" - "Stranger: Hello, delivery needs signature"

[5. Daily Chat]
- "chat": Casual conversation command
  * topic is a direction, generate a specific conversation question/topic to start a dialogue with the agent
  * topic="books": Example outputs - "What's 'Strait Gate' about?", "Is 'The Three-Body Problem' good?", "What are Haruki Murakami's novels like?"
  * topic="life tips": Example outputs - "How to cook a perfect soft-boiled egg?", "How to remove oil stains from clothes quickly?", "How to take care of indoor plants?"
  * topic="AI knowledge concepts": Example outputs - "What is a large language model?", "What is quantum computing?", "How does blockchain work?"
  * topic="sports and health": Example outputs - "How much exercise should I do daily?", "How to start running training?", "What should yoga beginners know?"

Conversational guidelines:
1. Keep it casual and natural, like a real person talking
2. USE CONTEXT from previous commands: If similar to a previous command, use words like "again", "change to", "actually", "make it"
   * Example evolution - First: "Turn on the AC" → Later: "Turn off the AC" or "Turn it on again"
   * Example evolution - First: "Set to 26 degrees" → Later: "Change to 24" or "Actually, make it 28"
3. Don't stuff too many things into one phrase, keep it simple and natural
4. For "read" action with multiple targets, use natural conversational flow
   * Example - "What's the fan speed? Is sleep mode on?" instead of "Query fan speed and sleep mode"

Input command (JSON):
{json.dumps(input_command, ensure_ascii=False)}

Output ONLY the natural language phrase, nothing else. Be concise and conversational.
"""
    return prompt


def get_ac_prompt(input_command, history_context=""):
    """
    Generate prompt for AC (Air Conditioner) control command conversion.

    Args:
        input_command: Dict with AC command parameters
        history_context: String containing previous commands context

    Returns:
        str: Complete prompt for LLM
    """
    prompt = f"""You are a home assistant converting AC control commands into natural, conversational English language.

Convert the following command into a casual, spoken English phrase that a person would say to control their air conditioner.
{history_context}
Action types:
- "update": Change/set AC parameters (this is the action for making changes)
- "read": Query/read current AC status

Parameter meanings:
- power: "on" = turn on, "off" = turn off
- mode: "cooling" = cooling mode, "heating" = heating mode, "dehumidify" = dehumidify mode
- fan_speed: "auto" = auto fan, "1"/"2"/"3" = fan speed level 1/2/3
- sleep_mode: "on" = enable sleep mode, "off" = disable sleep mode
- temperature: Number (16-30) = temperature in Celsius
- timer: Number (0-5, in 0.5 hour increments) = timer in hours

Conversational guidelines:
- When timer is set to 0, it means canceling/disabling the timer. Use casual tone like "Never mind, no timer" or "Cancel the timer"
- For "read" action with multiple targets, use natural conversational flow like "What's the fan speed? Is sleep mode on?" instead of listing them robotically
- Keep it simple - don't stuff too many things into one phrase. Focus on what feels natural to say
- USE CONTEXT from previous commands: If similar to a previous command, use words like "again", "change to", "actually", "make it", "let's try" to sound more natural
- Examples of contextual phrasing:
  * First: "Set to 26 degrees" -> Later: "Change it to 24" or "Actually, make it 24"
  * First: "Turn on the AC" -> Later: "Turn it off" or "AC off"
  * First: "Cooling mode" -> Later: "Switch to heating instead" or "Let's do dehumidify"

Input command (JSON):
{json.dumps(input_command, ensure_ascii=False)}

Output ONLY the natural language phrase, nothing else. Be concise and conversational.
"""
    return prompt


def get_lights_prompt(input_command, history_context=""):
    """
    Generate prompt for Light control command conversion.

    Args:
        input_command: Dict with light command parameters
        history_context: String containing previous commands context

    Returns:
        str: Complete prompt for LLM
    """
    prompt = f"""You are converting structured light control commands into natural, conversational English that a person would say to their home assistant.

{history_context}
Task interpretation (from natural language perspective):
- "set_preference" + color: User is setting a LONG-TERM default color preference for a device
  * MUST use phrasing: "I prefer [color] color for [device]" or "Set [device] to [color] by default from now on"
  * This is a persistent preference, not a temporary scenario or immediate action
  * Example: color="warm", device="Living Room Light" -> "I prefer warm color for Living Room Light"
  * Example: color="white", device="Bedroom Light" -> "Set Bedroom Light to white by default from now on"
  * Example: color="blue", device="Kitchen Light" -> "I prefer blue color for Kitchen Light"
- "update" + note only (no explicit color): User describes a scenario, implicitly expecting light color adjustment
  * Express the need for a specific lighting environment (not just turning on)
  * Example: note="reading" -> "Prepare the lights for reading" or "Set up reading lighting"
  * Example: note="relaxing" -> "Give me relaxing lighting" or "Set the lights for relaxation"
- "update" + power: Direct power control
  * power="on": "Turn on [device]" or "[device] on"
  * power="off": "Turn off [device]" or "[device] off"
- "update" + color: Direct color specification
  * "Make the [device] [color]" or "Change [device] to [color]"
- "read": Query current state
  * "What's the [device] state?" or "Check the [device]"

Device names:
- Bedroom Light
- Living Room Light
- Kitchen Light

Color options: white, red, blue, warm

Conversational guidelines:
- Keep it casual and natural, like talking to a real assistant
- For implicit commands (note only): Clearly express intent to set up lighting environment, not just power on
- USE CONTEXT from previous commands: If similar to a previous command, use words like "again", "change to", "actually", "make it"
- Examples of contextual phrasing:
  * First: "Turn on the bedroom light" -> Later: "Turn it off" or "Bedroom light off"
  * First: "Set to red" -> Later: "Change it to blue" or "Actually, make it warm"
  * First: "I prefer white for bedroom" -> Later: "Change the preference to blue"

Input command (JSON):
{json.dumps(input_command, ensure_ascii=False)}

Output ONLY the natural language phrase, nothing else. Be concise and conversational.
"""
    return prompt


def get_speaker_prompt(input_command, history_context=""):
    """
    Generate prompt for Speaker control command conversion.

    Args:
        input_command: Dict with speaker command parameters
        history_context: String containing previous commands context

    Returns:
        str: Complete prompt for LLM
    """
    prompt = f"""You are converting structured speaker control commands into natural, conversational English that a person would say to their home assistant.

{history_context}
Task interpretation (from natural language perspective):
- "update" + volume: Set speaker to a specific volume level
  * volume is a number from 0-10 (0=muted/silent, 10=maximum volume)
  * MUST always specify the exact number, never use vague terms like "louder", "quieter", "a bit", "up", "down"
  * Correct: "Set volume to 7", "Volume 5", "Make it 3", "Speaker volume 8"
  * Wrong: "Turn it up", "Make it louder", "Volume down", "A bit quieter"
- "read": Query current volume
  * "What's the volume?" or "Check the speaker volume" or "How loud is it?"

Device name: Speaker

Volume range: 0-10 (must use exact numbers)

Conversational guidelines:
- Keep it casual and natural, like talking to a real assistant
- ALWAYS use specific volume numbers, never relative descriptions
- USE CONTEXT from previous commands: If similar to a previous command, use words like "again", "change it to", "make it [number]"
- Examples of contextual phrasing:
  * First: "Set volume to 5" -> Later: "Change it to 7" or "Make it 3" or "Volume 8 again"
  * First: "What's the volume?" -> Later: "Check the volume again"

Input command (JSON):
{json.dumps(input_command, ensure_ascii=False)}

Output ONLY the natural language phrase, nothing else. Be concise and conversational. ALWAYS specify exact volume numbers.
"""
    return prompt


def get_security_prompt(input_command, history_context=""):
    """
    Generate prompt for Security System control command conversion.

    Args:
        input_command: Dict with security command parameters
        history_context: String containing previous commands context

    Returns:
        str: Complete prompt for LLM
    """
    prompt = f"""You are converting structured security system commands into natural, conversational English that a person would say to their home assistant.

{history_context}
Task interpretation:
- "update" + door_lock: "open"/"closed": Direct door control
  * door_lock="open": "Open the door", "Unlock the door"
  * door_lock="closed": "Close the door", "Lock the door"
  * If note field is present (e.g., "The owner comes back"), incorporate it: "Open the door, I'm back", "Unlock, I'm home"
  * Otherwise, add brief context when appropriate: "Open the door for delivery", "Lock up, going to bed"
- "read": Query door lock state
  * "Is the door locked?", "Check the door", "What's the door status?"
- "send_user_msg" + source="owner": Owner speaking to agent (leaving home message)
  * msg_content format: "reason+return time" (e.g., "supermarket+30 minutes", "gym+2 hours")
  * Convert to natural message: "I'm going to the supermarket, back in 30 minutes", "Heading to the gym for 2 hours"
- "send_visitor_msg" + source="visitor": Visitor speaking to system
  * msg_content is a hint (e.g., "neighbor borrowing something", "courier delivery signature required")
  * MUST start with "Stranger:" then write natural spoken content
  * Keep core meaning but be flexible: "Stranger: Hi, I'm your neighbor, need to borrow something", "Stranger: Hello, delivery needs signature"

Conversational guidelines:
- Keep it casual and natural, like talking to a real assistant
- USE CONTEXT from previous commands: "Open it again", "Lock it this time"
- For visitor messages, vary the phrasing while keeping core meaning
- Examples of contextual phrasing:
  * First: "Lock the door" -> Later: "Unlock it"
  * First: "I'm going to the store" -> Later with note: "Open the door, I'm back"
  * First: "Open for delivery" -> Later: "Lock up now"

Input command (JSON):
{json.dumps(input_command, ensure_ascii=False)}

Output ONLY the natural language phrase, nothing else. Be concise and conversational.
"""
    return prompt
