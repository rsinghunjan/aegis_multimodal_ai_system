"""
Async AgentManager scaffold for multimodal tool registration & dispatch.

Improvements in this patch:
- Use asyncio.Lock to protect state mutations
- _call_tool handles coroutine functions, coroutine objects, and awaitables returned by sync functions
- Add structured logging (logger) and lightweight in-memory metrics counters on the AgentManager instance
- Avoid storing large raw payloads in state; only store minimal metadata reference
"""

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

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
        # lock to protect state mutations in concurrent dispatches
        self._state_lock = asyncio.Lock()
        # lightweight metrics counters (optional integration point for Prometheus)
        self.metrics: Dict[str, int] = {
            "invocations": 0,
            "success": 0,
            "errors": 0,
            "timeouts": 0,
        }

    def register_tool(self, name: str, tool: Callable):
        """Register a tool by name. Tool may be sync or async callable."""
        if not callable(tool):
            raise ToolRegistrationError("Tool must be callable")
        self._tools[name] = tool
        logger.debug("Registered tool: %s", name)

    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    async def _call_tool(self, tool: Callable, action: Dict[str, Any]) -> Any:
        """Call a tool and return result, handling multiple callable types.

        Handles:
        - async def tool(...)
        - sync def tool(...) that returns an awaitable
        - coroutine objects passed in directly
        - sync def tool(...) that returns a direct result
        """
        # If the tool is a coroutine function, call and await it
        if inspect.iscoroutinefunction(tool):
            return await tool(action)

        # If the tool is a coroutine object already, await it
        if inspect.isawaitable(tool):
            # tool is an awaitable/coroutine object
            return await tool

        # Call sync function. It might return an awaitable (e.g., partial of coroutine)
        result = tool(action)
        if inspect.isawaitable(result):
            return await result
        return result

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
            "timeout_seconds": 10,
            "request_id": "..."
        }

        Returns:
            {"status": "ok", "result": ...} or {"status":"error", "error": "..."}
        """
        tool_name = action.get("tool")
        request_id = action.get("request_id")
        self.metrics["invocations"] += 1

        if not tool_name:
            logger.warning("dispatch_action missing tool name, request_id=%s", request_id)
            self.metrics["errors"] += 1
            return {"status": "error", "error": "missing tool name"}

        tool = self.get_tool(tool_name)
        if not tool:
            logger.warning("tool not found: %s, request_id=%s", tool_name, request_id)
            self.metrics["errors"] += 1
            return {"status": "error", "error": f"tool not found: {tool_name}"}

        timeout = action.get("timeout_seconds", 30)
        try:
            coro = self._call_tool(tool, action.get("payload", {}))
            result = await asyncio.wait_for(coro, timeout=timeout)

            # Only store lightweight metadata about the result to avoid storing raw payloads.
            async with self._state_lock:
                self.state[f"last_{tool_name}"] = {"status": "ok", "timestamp": asyncio.get_event_loop().time()}

            self.metrics["success"] += 1
            logger.info("tool invocation success: %s, request_id=%s", tool_name, request_id)
            return {"status": "ok", "result": result}
        except asyncio.TimeoutError:
            self.metrics["timeouts"] += 1
            logger.error("tool timeout: %s, timeout=%s, request_id=%s", tool_name, timeout, request_id)
            return {"status": "error", "error": "tool timeout"}
        except Exception as e:
            self.metrics["errors"] += 1
            logger.exception("tool invocation error: %s, request_id=%s", tool_name, request_id)
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
