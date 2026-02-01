"""
Full integration test - Test Green Agent with EvalRequest format.
This mimics how AgentBeats platform would call the Green Agent.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import asyncio
import httpx
import pytest

GREEN_AGENT_URL = "http://localhost:9010"
PURPLE_AGENT_URL = "http://localhost:9011"


@pytest.mark.asyncio
async def test_full_eval():
    """Send an EvalRequest to Green Agent and monitor progress."""
    
    print("=" * 60)
    print("FULL INTEGRATION TEST")
    print("=" * 60)
    
    # Check agents
    print("\n1. Checking agent availability...")
    for name, url in [("Green", GREEN_AGENT_URL), ("Purple", PURPLE_AGENT_URL)]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/.well-known/agent.json")
                if resp.status_code == 200:
                    print(f"   ✅ {name} Agent running at {url}")
                else:
                    print(f"   ❌ {name} Agent error: {resp.status_code}")
                    return
        except Exception as e:
            print(f"   ❌ {name} Agent not reachable: {e}")
            return
    
    # Create EvalRequest
    eval_request = {
        "participants": {
            "purple": PURPLE_AGENT_URL
        },
        "config": {
            "max_test_rounds": 1,
            "weakness_num": 1,
            "targeted_per_weakness": 1,
            "convergence_threshold": 0.1
        }
    }
    
    print(f"\n2. Sending EvalRequest to Green Agent...")
    print(f"   Config: {json.dumps(eval_request['config'])}")
    
    # Send using A2A client
    from messenger import send_message
    
    try:
        print("\n   ⏳ Waiting for response (this may take a while for LLM generation)...")
        
        outputs = await send_message(
            message=json.dumps(eval_request),
            base_url=GREEN_AGENT_URL,
            timeout=600  # 10 minutes for LLM generation
        )
        
        print(f"\n3. Response received!")
        print(f"   Status: {outputs.get('status', 'N/A')}")
        response = outputs.get('response', '')
        
        # Try to parse as JSON
        try:
            result_json = json.loads(response)
            print(f"\n   Results:\n{json.dumps(result_json, indent=2)}")
        except:
            print(f"\n   Response (raw): {response[:1000]}...")
            
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_full_eval())
