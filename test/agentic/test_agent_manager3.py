  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
import asyncio

from aegis_multimodal_ai_system.agentic.agent_manager import (
    AgentManager,
    example_echo_tool,
    async_wrapper_example,
)


def test_sync_tool_invocation():
    am = AgentManager()
    am.register_tool("echo", example_echo_tool)
    action = {
        "tool": "echo",
        "payload": {"text": "hello"},
    }
    res = asyncio.run(am.dispatch_action(action))
    assert res["status"] == "ok"
    assert res["result"]["echo_text"] == "hello"


def test_async_tool_invocation():
    am = AgentManager()
    am.register_tool("async_echo", async_wrapper_example)
    action = {
        "tool": "async_echo",
        "payload": {"text": "async"},
    }
    res = asyncio.run(am.dispatch_action(action))
    assert res["status"] == "ok"
    assert res["result"]["echo_text"] == "async"


def test_missing_tool():
    am = AgentManager()
    res = asyncio.run(am.dispatch_action({"tool": "nope"}))
    assert res["status"] == "error"


def test_timeout_behavior():
    async def slow_tool(payload):
        await asyncio.sleep(1)
        return {"ok": True}

    am = AgentManager()
    am.register_tool("slow", slow_tool)
    res = asyncio.run(
        am.dispatch_action({"tool": "slow", "timeout_seconds": 0.01})
    )
    assert res["status"] == "error"
    assert "timeout" in res.get("error", "")
