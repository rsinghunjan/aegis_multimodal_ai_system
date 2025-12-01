"""
Agent Manager for multimodal agentic orchestration.

This module provides an async AgentManager class that supports:
- Plugin/tool registration
- Multimodal action dispatching
- Example tool implementations

To extend for production:
- Add authentication/authorization for tool execution
- Implement persistent tool registry (e.g., Redis, database)
- Add monitoring and observability hooks
- Integrate with message queues for async task processing
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Async agent manager for multimodal AI orchestration.

    This class manages tool/plugin registration and provides a simple
    multimodal action dispatcher for coordinating AI agent tasks.

    Example usage:
        manager = AgentManager()
        manager.register_tool("echo", echo_tool)
        result = await manager.dispatch("echo", {"message": "Hello"})
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AgentManager.

        Args:
            config: Optional configuration dictionary for customization.
                    Supported keys:
                    - max_concurrent_tasks: Maximum concurrent task limit
                    - default_timeout: Default timeout for tool execution
        """
        self._tools: Dict[str, Callable] = {}
        self._config = config or {}
        self._max_concurrent_tasks = self._config.get("max_concurrent_tasks", 10)
        self._default_timeout = self._config.get("default_timeout", 30.0)
        self._semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
        logger.info("AgentManager initialized with config: %s", self._config)

    def register_tool(
        self,
        name: str,
        handler: Callable,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a tool/plugin with the agent manager.

        Args:
            name: Unique identifier for the tool.
            handler: Async or sync callable that implements the tool logic.
            metadata: Optional metadata about the tool (description, schema).

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self._tools[name] = {
            "handler": handler,
            "metadata": metadata or {}
        }
        logger.info("Registered tool: %s", name)

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: The name of the tool to unregister.

        Returns:
            True if the tool was unregistered, False if not found.
        """
        if name in self._tools:
            del self._tools[name]
            logger.info("Unregistered tool: %s", name)
            return True
        return False

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of registered tool names.
        """
        return list(self._tools.keys())

    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a registered tool.

        Args:
            name: The name of the tool.

        Returns:
            Tool metadata dictionary or None if not found.
        """
        if name in self._tools:
            return self._tools[name].get("metadata")
        return None

    async def dispatch(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Dispatch an action to a registered tool.

        Args:
            tool_name: Name of the tool to invoke.
            payload: Input payload for the tool.
            timeout: Optional timeout override in seconds.

        Returns:
            Result dictionary with 'success', 'result' or 'error' keys.

        Example:
            result = await manager.dispatch("echo", {"message": "test"})
            if result["success"]:
                print(result["result"])
        """
        if tool_name not in self._tools:
            logger.warning("Tool not found: %s", tool_name)
            return {"success": False, "error": f"Tool '{tool_name}' not found"}

        tool = self._tools[tool_name]
        handler = tool["handler"]
        effective_timeout = timeout or self._default_timeout

        async with self._semaphore:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await asyncio.wait_for(
                        handler(payload),
                        timeout=effective_timeout
                    )
                else:
                    # Run sync handlers in executor
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, handler, payload),
                        timeout=effective_timeout
                    )

                logger.info("Tool '%s' executed successfully", tool_name)
                return {"success": True, "result": result}

            except asyncio.TimeoutError:
                logger.error("Tool '%s' timed out", tool_name)
                return {"success": False, "error": "Tool execution timed out"}
            except Exception as e:
                logger.error("Tool '%s' failed: %s", tool_name, str(e))
                return {"success": False, "error": str(e)}

    async def dispatch_multimodal(
        self,
        actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Dispatch multiple actions concurrently for multimodal processing.

        Args:
            actions: List of action dictionaries, each containing:
                     - tool: Name of the tool to invoke
                     - payload: Input payload for the tool
                     - timeout: Optional timeout override

        Returns:
            List of result dictionaries in the same order as input actions.

        Example:
            actions = [
                {"tool": "text_processor", "payload": {"text": "hello"}},
                {"tool": "image_analyzer", "payload": {"url": "..."}},
            ]
            results = await manager.dispatch_multimodal(actions)
        """
        tasks = []
        for action in actions:
            tool_name = action.get("tool", "")
            payload = action.get("payload", {})
            timeout = action.get("timeout")
            tasks.append(self.dispatch(tool_name, payload, timeout))

        return await asyncio.gather(*tasks)


# Example tool implementations for testing and demonstration
def echo_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple echo tool for testing.

    Args:
        payload: Dictionary containing 'message' key.

    Returns:
        Dictionary with echoed message.
    """
    message = payload.get("message", "")
    return {"echoed": message}


async def async_delay_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async tool that simulates processing delay.

    Args:
        payload: Dictionary containing 'delay' (seconds) and 'data' keys.

    Returns:
        Dictionary with processed data and timing info.
    """
    delay = payload.get("delay", 0.1)
    data = payload.get("data", {})
    await asyncio.sleep(delay)
    return {"processed": True, "data": data, "delay_applied": delay}


def create_default_manager() -> AgentManager:
    """
    Create an AgentManager with default example tools registered.

    Returns:
        Configured AgentManager instance.
    """
    manager = AgentManager()
    manager.register_tool(
        "echo",
        echo_tool,
        metadata={"description": "Echoes the input message"}
    )
    manager.register_tool(
        "async_delay",
        async_delay_tool,
        metadata={"description": "Simulates async processing with delay"}
    )
    return manager
