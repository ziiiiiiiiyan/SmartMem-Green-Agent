# SmartMem Green Agent

## Quick Start

### Prerequisites

- Python 3.10+
- uv package manager
- API key for OpenAI-compatible LLM service

### Running the Agents

1. **Start Purple Agent** (port 9011):
   ```bash
   cd SmartMem-Purple-Agent
   uv run src/server.py --port 9011
   ```

2. **Start Green Agent** (port 9010):
   ```bash
   cd SmartMem-Green-Agent
   uv run src/server.py --port 9010
   ```

3. **Run evaluation test**:
   ```bash
   cd SmartMem-Green-Agent-Leaderboard
   python test_local.py
   ```

### Configuration

Set API keys in `.env` files or environment variables:

**Green Agent** (`src/green_agent/.env`):
```
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api2.aigcbest.top/v1
MODEL_NAME=gpt-4o
```

**Purple Agent** (`src/purple_agent/.env`):
```
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api2.aigcbest.top/v1
MODEL_NAME=qwen2.5-1.5b-instruct
```

### Test Configuration Options

In `test_local.py` or evaluation requests, you can configure:

```python
{
    "config": {
        "max_test_rounds": 1,        # Number of adaptive test rounds
        "targeted_per_weakness": 1,  # Test cases per identified weakness
        "convergence_threshold": 0.05,
        "weakness_num": 1,
        "max_turns": 5               # Limit turns for quick testing (optional)
    }
}
```

## Visualization

When evaluation completes, charts are automatically generated in `artifacts/`:

- `capability_radar.png` - Radar chart showing per-category capability scores
- `category_scores.png` - Bar chart of category performance with pass/fail thresholds

Charts require matplotlib: `pip install matplotlib`

---

## A2A Protocol Message Formats

This section describes the message formats used for communication between the Green Agent (evaluator) and Purple Agent (smart home assistant) via the A2A protocol.

### Overview

The Green Agent sends test instructions to the Purple Agent and evaluates its responses. Communication happens through JSON messages with specific formats.

### Message Types

#### 1. Green → Purple: User Instruction

Plain text instruction sent to the Purple Agent:

```
Is the bedroom light on?
```

or

```
Turn on the living room light.
```

#### 2. Purple → Green: Response Message

Purple Agent responds with a JSON object:

```json
{
  "message_type": "text" | "tool",
  "message_content": "<content>"
}
```

**Text Response** (when Purple responds with natural language):
```json
{
  "message_type": "text",
  "message_content": "The bedroom light is currently off."
}
```

**Tool Response** (when Purple needs to interact with devices):
```json
{
  "message_type": "tool",
  "message_content": "[{\"device_id\": \"bedroom_light\", \"action\": \"read\"}]"
}
```

Note: `message_content` for tool responses is a JSON-encoded string containing an array of tool calls.

#### 3. Green → Purple: Tool Execution Result

After executing tool calls on the simulator, Green sends results back:

```json
[
  {
    "message": {"status": "success", "value": "off", ...},
    "metadata": {"operation_object": "bedroom_light"}
  }
]
```

**Important**: The `metadata.operation_object` must match the `device_id` from the original tool call so Purple can correlate results.

### Tool Call Format

Each tool call in the `message_content` array has this structure:

```json
{
  "device_id": "bedroom_light",
  "action": "read" | "update",
  "value": "<optional, for update actions>"
}
```

**Available device_ids:**
- `living_room_light`, `living_room_color`
- `bedroom_light`, `bedroom_color`
- `ac`, `ac_temperature`
- `fan_speed`
- `music_volume`
- `front_door_lock`
- `kitchen_light`
- `all` (read-only, returns all device states)

**Actions:**
- `read`: Get current device state
- `update`: Change device state (requires `value`)

**Values by device type:**
- Lights/AC Power: `on`, `off`
- Colors: `white`, `red`, `blue`, `warm`
- Temperature: `16` to `30` (string)
- Fan Speed: `off`, `low`, `medium`, `high`
- Volume: `0` to `10` (string)
- Lock: `locked`, `unlocked`

---

# 原有文档 (Original Documentation)

## agentbeats的主要交互逻辑
agentbeats平台把scenario.toml里的参赛者等配置发给green， green开始评估流程，中间状态变化通过update_status更新，最终的评估结果通过add_artifacts提交，这会被更新在leaderboard上（默认是results文件夹里）

green评估过程中通过talk_to_agent和purple交互

## green_agent_v2
它继承自原来的green agent文件夹，去除了各种通用设施，保留了核心的试题生成和评估逻辑。

- 各种基础的数据结构定义被统一存放到了base.py中

- 原先的AdaptiveGenerator(https://vscode.dev/github/ziiiiiiiiyan/SmartMem-Green-Agent/blob/main/archieved/green_agent/adaptive_loop.py#L508)改造成了instruction_generator.py中的AdaptiveGenerator，添加了生成初始题组的逻辑，强制生成了金字塔形的配比。抽离出了生成题目的prompt放在prompt.py中，负责生成题目的LLM被包装在LLMCaseGenerator中。

- 原先的adaptive_loop.py中的WeaknessAnalyzer、AdaptiveEvaluator(x 逻辑搬错了已删除，直接放agent.py/run里去)被放到现在evaluator.py中。考虑green agent发出instruction后未必purple agent能够一次性执行完成、同时单纯的对话指令实际上没有评估对象，我把evaluator调整为和analyser联动的版本，即evaluator接收每个turn的执行历史进行评估，调用analyser同步弱点信息。green可以调用evaluator.analyser.get_top_weaknesses(k)来获取前k个弱点丢给AdaptiveGenerator生成新的题组。

- adaptive_loop.py中的主循环AdaptiveTestLoop逻辑被放在了src/agent.py中

- **还没有完成的**

   1. 可视化和报告生成这部分还没搬运

   2. prompt里直接把设备信息写死，但是现在还保留了占位符

## MEMO
一个test round包含n个test case，test case的数量=当前关注的前k个弱点的数量\times希望针对每个弱点生成的test case的数量