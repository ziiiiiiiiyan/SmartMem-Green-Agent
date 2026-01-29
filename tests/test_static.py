"""
Local test with STATIC test cases - no LLM API calls needed.
This tests the full pipeline: Purple Agent interaction, environment, evaluation.
Uses the proper A2A client from messenger.py
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import asyncio
import httpx
from messenger import send_message

# Static test cases (no LLM generation needed)
STATIC_TEST_CASES = [
    {
        "scenario_id": "static_001",
        "difficulty": "easy", 
        "dimension": "precision",
        "description": "Turn on the living room light",
        "initial_state": {
            "living_room_light": "off",
            "living_room_brightness": 50
        },
        "turns": [
            {
                "turn_id": 1,
                "gm_instruction": "Please turn on the living room light.",
                "expected_agent_action": [
                    {"action": "update", "key": "living_room_light", "value": "on"}
                ],
                "expected_final_state": {"living_room_light": "on", "living_room_brightness": 50}
            }
        ]
    },
    {
        "scenario_id": "static_002",
        "difficulty": "medium",
        "dimension": "memory",
        "description": "Remember and recall a preference",
        "initial_state": {
            "bedroom_temperature": 22,
            "bedroom_ac": "off"
        },
        "turns": [
            {
                "turn_id": 1,
                "gm_instruction": "I prefer the bedroom temperature at 25 degrees.",
                "expected_agent_action": [
                    {"action": "update", "key": "bedroom_temperature", "value": 25}
                ],
                "expected_final_state": {"bedroom_temperature": 25, "bedroom_ac": "off"}
            },
            {
                "turn_id": 2, 
                "gm_instruction": "Set the bedroom to my preferred temperature.",
                "expected_agent_action": [
                    {"action": "update", "key": "bedroom_temperature", "value": 25}
                ],
                "expected_final_state": {"bedroom_temperature": 25, "bedroom_ac": "off"}
            }
        ]
    }
]


async def test_purple_agent():
    """Test Purple Agent with static test cases."""
    
    purple_url = "http://localhost:9011"
    
    print("=" * 60)
    print("STATIC TEST - Using A2A Client")
    print("=" * 60)
    
    # Check if Purple Agent is running
    print("\n1. Checking Purple Agent availability...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try the agent card endpoint
            resp = await client.get(f"{purple_url}/.well-known/agent.json")
            if resp.status_code == 200:
                print(f"   ✅ Purple Agent is running at {purple_url}")
                agent_card = resp.json()
                print(f"   Agent Name: {agent_card.get('name', 'N/A')}")
            else:
                print(f"   ⚠️  Purple Agent returned status {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Cannot connect to Purple Agent: {e}")
        print("   Please start Purple Agent first:")
        print("   cd SmartMem-Purple-Agent && python src/server.py --port 9011")
        return
    
    print(f"\n2. Running {len(STATIC_TEST_CASES)} static test cases...")
    
    results = []
    context_id = None  # For conversation continuity
    
    for i, test_case in enumerate(STATIC_TEST_CASES, 1):
        print(f"\n   --- Test Case {i}/{len(STATIC_TEST_CASES)} ---")
        print(f"   Scenario: {test_case['scenario_id']}")
        print(f"   Difficulty: {test_case['difficulty']}")
        print(f"   Dimension: {test_case['dimension']}")
        print(f"   Description: {test_case['description']}")
        
        case_passed = True
        context_id = None  # New conversation for each test case
        
        for turn in test_case['turns']:
            instruction = turn['gm_instruction']
            print(f"\n   Turn {turn['turn_id']}: {instruction[:50]}...")
            
            # Send to Purple Agent using proper A2A client
            try:
                outputs = await send_message(
                    message=instruction,
                    base_url=purple_url,
                    context_id=context_id,
                    timeout=60
                )
                
                context_id = outputs.get("context_id")
                response = outputs.get("response", "")
                status = outputs.get("status", "completed")
                
                print(f"   Status: {status}")
                print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
                
                if status != "completed":
                    print(f"   ⚠️  Unexpected status: {status}")
                    case_passed = False
                        
            except Exception as e:
                print(f"   ❌ Error: {e}")
                case_passed = False
        
        results.append({
            "scenario_id": test_case['scenario_id'],
            "dimension": test_case['dimension'],
            "difficulty": test_case['difficulty'],
            "passed": case_passed
        })
        
        status = "✅ PASS" if case_passed else "❌ FAIL"
        print(f"\n   Result: {status}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    pass_rate = passed / total if total > 0 else 0
    
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1%}")
    
    # Generate results JSON
    output = {
        "participants": {"purple_agent": "baseline-agent"},
        "results": [{
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": pass_rate,
            "dimension_stats": []
        }]
    }
    
    # Group by dimension
    dimension_results = {}
    for r in results:
        dim = r['dimension']
        if dim not in dimension_results:
            dimension_results[dim] = {"total": 0, "passed": 0}
        dimension_results[dim]["total"] += 1
        if r['passed']:
            dimension_results[dim]["passed"] += 1
    
    for dim, stats in dimension_results.items():
        output["results"][0]["dimension_stats"].append({
            "key": dim,
            "pass_rate": stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        })
    
    print(f"\nResults JSON:\n{json.dumps(output, indent=2)}")
    
    return output


if __name__ == "__main__":
    asyncio.run(test_purple_agent())
