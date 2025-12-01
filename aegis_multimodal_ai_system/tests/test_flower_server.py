"""
Tests for the Federated Learning module.
"""

import pytest
from aegis_multimodal_ai_system.federated.flower_server import (
    FlowerServerWrapper,
    check_flwr_available,
    create_simple_server,
)


class TestFlowerServerWrapper:
    """Test cases for FlowerServerWrapper class."""

    def test_init_default(self):
        """Test FlowerServerWrapper initialization with defaults."""
        server = FlowerServerWrapper()
        config = server.get_config()
        assert config["num_rounds"] == 3
        assert config["min_clients"] == 2

    def test_init_custom(self):
        """Test FlowerServerWrapper initialization with custom values."""
        server = FlowerServerWrapper(
            num_rounds=10,
            min_clients=5,
            fraction_fit=0.5,
        )
        config = server.get_config()
        assert config["num_rounds"] == 10
        assert config["min_clients"] == 5
        assert config["fraction_fit"] == 0.5

    def test_is_available(self):
        """Test is_available method."""
        server = FlowerServerWrapper()
        # Result depends on whether flwr is installed
        assert isinstance(server.is_available(), bool)

    def test_check_flwr_available(self):
        """Test check_flwr_available function."""
        result = check_flwr_available()
        assert isinstance(result, bool)

    def test_get_installation_instructions(self):
        """Test get_installation_instructions returns valid string."""
        server = FlowerServerWrapper()
        instructions = server.get_installation_instructions()
        assert "pip install flwr" in instructions
        assert "flower.dev" in instructions

    def test_get_config(self):
        """Test get_config method returns complete config."""
        server = FlowerServerWrapper(num_rounds=5, min_clients=3)
        config = server.get_config()

        assert "num_rounds" in config
        assert "min_clients" in config
        assert "min_fit_clients" in config
        assert "min_evaluate_clients" in config
        assert "fraction_fit" in config
        assert "fraction_evaluate" in config
        assert "flwr_available" in config

    def test_start_without_flwr_raises(self):
        """Test that start() raises if flwr not installed."""
        server = FlowerServerWrapper()
        if not server.is_available():
            with pytest.raises(RuntimeError, match="not installed"):
                server.start()


class TestCreateSimpleServer:
    """Test cases for create_simple_server function."""

    def test_create_simple_server_default(self):
        """Test creating simple server with defaults."""
        server = create_simple_server()
        config = server.get_config()
        assert config["num_rounds"] == 3
        assert config["min_clients"] == 2

    def test_create_simple_server_custom(self):
        """Test creating simple server with custom values."""
        server = create_simple_server(num_rounds=10, min_clients=4)
        config = server.get_config()
        assert config["num_rounds"] == 10
        assert config["min_clients"] == 4
