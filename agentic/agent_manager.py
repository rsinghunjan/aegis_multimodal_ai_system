"""
AgentManager - a minimal async multimodal agent orchestration scaffold.

Features:
- Register tools/plugins (callables or async callables)
- Dispatch simple multimodal actions (text, image, audio payloads)
- Extensible hooks for auth, rate-limiting, and result aggregation

This file is intentionally minimal and dependency-light to be unit-testable.
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, Optional

class ToolRegistrationError(Exception):
    pass

class AgentManager:
    """
    Async AgentManager that registers tools and dispatches multimodal actions.

    Example tool signature:
        async def my_tool(action: Dict[str, Any]) -> Dict[str, Any]:
            ...

    or sync:
        def my_tool(action): ...
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        # a simple in-memory store for agent state (replace with DB in prod)
        self.state: Dict[str, Any] = {}

    def register_tool(self, name: str, tool: Callable):
        """Register a tool by name. Tool may be sync or async callable."""
        if not callable(tool):
            raise ToolRegistrationError("Tool must be callable")
        self._tools[name] = tool

    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    async def _call_tool(self, tool: Callable, action: Dict[str, Any]) -> Any:
        """Call a tool and return result, handling sync/async functions."""
        if inspect.iscoroutinefunction(tool):
            return await tool(action)
        # run sync functions in threadpool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: tool(action))

    async def dispatch_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch a multimodal action.

        action (example):
        {
            "tool": "vision_tool",
            "payload": {
                "text": "...",
                "image_bytes": b"...",
                "metadata": {...}
            },
            "timeout_seconds": 10
        }

        Returns:
            {"status": "ok", "result": ...} or {"status":"error", "error": "..."}
        """
        tool_name = action.get("tool")
        if not tool_name:
            return {"status": "error", "error": "missing tool name"}

        tool = self.get_tool(tool_name)
        if not tool:
            return {"status": "error", "error": f"tool not found: {tool_name}"}

        timeout = action.get("timeout_seconds", 30)
        try:
            coro = self._call_tool(tool, action.get("payload", {}))
            result = await asyncio.wait_for(coro, timeout=timeout)
            # simple state mutation example (replace with better state handling)
            self.state[f"last_{tool_name}"] = {"result": result}
            return {"status": "ok", "result": result}
        except asyncio.TimeoutError:
            return {"status": "error", "error": "tool timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Example tool (keeps dependencies out of core)
def example_echo_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    A sync example tool that echoes multimodal payload.
    Good for unit tests and as a template for real tools.
    """
    return {
        "echo_text": payload.get("text"),
        "metadata": payload.get("metadata")
    }

# Lightweight async wrapper example
async def async_wrapper_example(payload: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0)  # placeholder for async work
    return example_echo_tool(payload)
