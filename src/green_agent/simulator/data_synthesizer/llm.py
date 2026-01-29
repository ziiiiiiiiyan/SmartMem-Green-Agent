import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from .prompts import get_ac_prompt, get_lights_prompt, get_speaker_prompt, get_security_prompt


# Device type to prompt function mapping
DEVICE_PROMPTS = {
    "ac": get_ac_prompt,
    "lights": get_lights_prompt,
    "speaker": get_speaker_prompt,
    "security": get_security_prompt
}


def load_env():
    """Load environment variables from .env file"""
    # Get the directory of this file, then go up to find .env
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "../../.env")
    load_dotenv(env_path)


def get_client():
    """Get OpenAI client configured from .env"""
    load_env()
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )


def command_to_natural_language(input_command, history=None, device_type="ac"):
    """
    Convert device control command to natural language using LLM.

    Args:
        input_command: Dict with device command parameters
        history: List of previous (input_command, natural_language) pairs for context
        device_type: Type of device - "ac", "lights", "speaker", or "security" (default: "ac")

    Returns:
        str: Natural language description of the command
    """
    client = get_client()

    # Build history context if available
    history_context = ""
    if history:
        history_context = "\nPrevious commands in this session (for context):\n"
        for i, (prev_cmd, prev_nl) in enumerate(history[-5:], 1):  # Last 5 for context
            history_context += f"{i}. Command: {json.dumps(prev_cmd, ensure_ascii=False)}\n   Natural: \"{prev_nl}\"\n"

    # Get prompt function based on device type
    prompt_func = DEVICE_PROMPTS.get(device_type, get_ac_prompt)
    prompt = prompt_func(input_command, history_context)

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        **json.loads(os.getenv("MODEL_GEN_ARGS", "{}"))
    )

    return response.choices[0].message.content.strip()


def convert_test_json_to_natural_language(input_json_path, output_json_path, device_type="ac"):
    """
    Convert all input_command in test.json to natural language.

    Args:
        input_json_path: Path to input test.json
        output_json_path: Path to save output with natural language descriptions
        device_type: Type of device - "ac", "lights", or "speaker" (default: "ac")

    Returns:
        list: List of entries with natural language descriptions
    """
    # Load test data
    with open(input_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    total = len(test_data)
    history = []  # Store (input_command, natural_language) pairs

    for i, entry in enumerate(test_data, 1):
        input_command = entry["input_command"]
        print(f"[{i}/{total}] Converting: {input_command}")

        # Convert to natural language with history context
        natural_language = command_to_natural_language(input_command, history, device_type)

        # Add to history
        history.append((input_command, natural_language))

        results.append({
            "input_command": input_command,
            "natural_language": natural_language,
            "expected_choices": entry["expected_choices"]
        })

    # Save to output file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted! Saved {len(results)} entries to: {output_json_path}")
    return results


if __name__ == "__main__":
    import sys

    # Example usage: convert test.json to natural language
    # Usage: python llm.py [device_type]
    # device_type can be: ac, lights, speaker (default: ac)

    device_type = sys.argv[1] if len(sys.argv) > 1 else "ac"

    if device_type not in DEVICE_PROMPTS:
        print(f"Error: Unknown device type '{device_type}'")
        print(f"Supported device types: {', '.join(DEVICE_PROMPTS.keys())}")
        sys.exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Map device type to input/output files
    file_mapping = {
        "ac": ("test_data/ac_dict.json", "test_data/ac_nl.json"),
        "lights": ("test_data/lights_dict.json", "test_data/lights_nl.json"),
        "speaker": ("test_data/speaker_dict.json", "test_data/speaker_nl.json"),
        "security": ("test_data/security_dict.json", "test_data/security_nl.json")
    }

    input_file, output_file = file_mapping.get(device_type, ("test.json", "test_nl.json"))
    input_path = os.path.join(current_dir, input_file)
    output_path = os.path.join(current_dir, output_file)

    print(f"Converting {device_type} commands...")
    convert_test_json_to_natural_language(input_path, output_path, device_type)
