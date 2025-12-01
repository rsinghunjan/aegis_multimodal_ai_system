"""
Tests for the Agent Manager module.
"""

import pytest
from aegis_multimodal_ai_system.agentic.agent_manager import (
    AgentManager,
    echo_tool,
    async_delay_tool,
    create_default_manager,
)


class TestAgentManager:
    """Test cases for AgentManager class."""

    def test_init_default_config(self):
        """Test AgentManager initialization with default config."""
        manager = AgentManager()
        assert manager.list_tools() == []

    def test_init_custom_config(self):
        """Test AgentManager initialization with custom config."""
        config = {"max_concurrent_tasks": 5, "default_timeout": 60.0}
        manager = AgentManager(config=config)
        assert manager._max_concurrent_tasks == 5
        assert manager._default_timeout == 60.0

    def test_register_tool(self):
        """Test tool registration."""
        manager = AgentManager()
        manager.register_tool("test_tool", echo_tool)
        assert "test_tool" in manager.list_tools()

    def test_register_tool_with_metadata(self):
        """Test tool registration with metadata."""
        manager = AgentManager()
        metadata = {"description": "Test tool", "version": "1.0"}
        manager.register_tool("test_tool", echo_tool, metadata=metadata)
        assert manager.get_tool_metadata("test_tool") == metadata

    def test_register_duplicate_tool_raises(self):
        """Test that registering duplicate tool raises ValueError."""
        manager = AgentManager()
        manager.register_tool("test_tool", echo_tool)
        with pytest.raises(ValueError, match="already registered"):
            manager.register_tool("test_tool", echo_tool)

    def test_unregister_tool(self):
        """Test tool unregistration."""
        manager = AgentManager()
        manager.register_tool("test_tool", echo_tool)
        assert manager.unregister_tool("test_tool") is True
        assert "test_tool" not in manager.list_tools()

    def test_unregister_nonexistent_tool(self):
        """Test unregistering a tool that doesn't exist."""
        manager = AgentManager()
        assert manager.unregister_tool("nonexistent") is False

    def test_list_tools(self):
        """Test listing registered tools."""
        manager = AgentManager()
        manager.register_tool("tool1", echo_tool)
        manager.register_tool("tool2", echo_tool)
        tools = manager.list_tools()
        assert "tool1" in tools
        assert "tool2" in tools
        assert len(tools) == 2

    def test_get_tool_metadata_nonexistent(self):
        """Test getting metadata for nonexistent tool."""
        manager = AgentManager()
        assert manager.get_tool_metadata("nonexistent") is None


class TestAgentManagerDispatch:
    """Test cases for AgentManager dispatch methods."""

    @pytest.mark.asyncio
    async def test_dispatch_sync_tool(self):
        """Test dispatching to a synchronous tool."""
        manager = AgentManager()
        manager.register_tool("echo", echo_tool)
        result = await manager.dispatch("echo", {"message": "hello"})
        assert result["success"] is True
        assert result["result"]["echoed"] == "hello"

    @pytest.mark.asyncio
    async def test_dispatch_async_tool(self):
        """Test dispatching to an async tool."""
        manager = AgentManager()
        manager.register_tool("delay", async_delay_tool)
        result = await manager.dispatch("delay", {"delay": 0.01, "data": {"key": "value"}})
        assert result["success"] is True
        assert result["result"]["processed"] is True
        assert result["result"]["data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_dispatch_nonexistent_tool(self):
        """Test dispatching to a nonexistent tool."""
        manager = AgentManager()
        result = await manager.dispatch("nonexistent", {})
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_dispatch_timeout(self):
        """Test dispatch timeout handling."""
        manager = AgentManager()
        manager.register_tool("delay", async_delay_tool)
        result = await manager.dispatch("delay", {"delay": 10}, timeout=0.01)
        assert result["success"] is False
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_dispatch_multimodal(self):
        """Test multimodal dispatching."""
        manager = AgentManager()
        manager.register_tool("echo", echo_tool)
        manager.register_tool("delay", async_delay_tool)

        actions = [
            {"tool": "echo", "payload": {"message": "test1"}},
            {"tool": "echo", "payload": {"message": "test2"}},
        ]
        results = await manager.dispatch_multimodal(actions)

        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert results[0]["result"]["echoed"] == "test1"
        assert results[1]["result"]["echoed"] == "test2"


class TestExampleTools:
    """Test cases for example tool implementations."""

    def test_echo_tool(self):
        """Test echo tool."""
        result = echo_tool({"message": "hello world"})
        assert result["echoed"] == "hello world"

    def test_echo_tool_empty_message(self):
        """Test echo tool with empty message."""
        result = echo_tool({})
        assert result["echoed"] == ""

    @pytest.mark.asyncio
    async def test_async_delay_tool(self):
        """Test async delay tool."""
        result = await async_delay_tool({"delay": 0.01, "data": {"test": True}})
        assert result["processed"] is True
        assert result["data"] == {"test": True}
        assert result["delay_applied"] == 0.01


class TestCreateDefaultManager:
    """Test cases for create_default_manager function."""

    def test_create_default_manager(self):
        """Test creating default manager with example tools."""
        manager = create_default_manager()
        tools = manager.list_tools()
        assert "echo" in tools
        assert "async_delay" in tools

    @pytest.mark.asyncio
    async def test_default_manager_tools_work(self):
        """Test that default manager tools are functional."""
        manager = create_default_manager()
        result = await manager.dispatch("echo", {"message": "test"})
        assert result["success"] is True
