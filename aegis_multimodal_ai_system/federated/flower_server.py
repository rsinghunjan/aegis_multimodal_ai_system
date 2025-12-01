"""
Flower (flwr) Federated Learning Server Wrapper.

This module provides a wrapper for running federated learning training
using the Flower framework. It includes fallback instructions if flwr
is not installed.

To install flwr:
    pip install flwr

For production deployments:
- Configure TLS/SSL for secure communication
- Set up proper authentication for clients
- Implement custom aggregation strategies
- Add monitoring and metrics collection
- Configure persistent storage for model checkpoints

Example usage:
    server = FlowerServerWrapper(num_rounds=10)
    server.start(server_address="0.0.0.0:8080")
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Check if flwr is available
FLWR_AVAILABLE = False
try:
    import flwr as fl
    from flwr.server.strategy import FedAvg
    FLWR_AVAILABLE = True
except ImportError:
    fl = None
    FedAvg = None
    logger.warning(
        "Flower (flwr) is not installed. "
        "Install with: pip install flwr"
    )


def check_flwr_available() -> bool:
    """
    Check if the Flower library is available.

    Returns:
        True if flwr is installed and importable.
    """
    return FLWR_AVAILABLE


class FlowerServerWrapper:
    """
    Wrapper for Flower federated learning server.

    This class simplifies the setup and configuration of a Flower server
    for federated training scenarios.

    Attributes:
        num_rounds: Number of federated training rounds.
        min_clients: Minimum number of clients required for training.
        strategy: Aggregation strategy (defaults to FedAvg).

    Example:
        wrapper = FlowerServerWrapper(num_rounds=5, min_clients=2)
        if wrapper.is_available():
            wrapper.start("0.0.0.0:8080")
    """

    def __init__(
        self,
        num_rounds: int = 3,
        min_clients: int = 2,
        min_fit_clients: Optional[int] = None,
        min_evaluate_clients: Optional[int] = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        strategy_class: Optional[Any] = None,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Flower server wrapper.

        Args:
            num_rounds: Number of federated training rounds.
            min_clients: Minimum clients for aggregation.
            min_fit_clients: Minimum clients for fit (defaults to min_clients).
            min_evaluate_clients: Minimum clients for evaluate.
            fraction_fit: Fraction of clients for training.
            fraction_evaluate: Fraction of clients for evaluation.
            strategy_class: Custom strategy class (defaults to FedAvg).
            strategy_kwargs: Additional kwargs for strategy initialization.
        """
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.min_fit_clients = min_fit_clients or min_clients
        self.min_evaluate_clients = min_evaluate_clients or min_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self._strategy_class = strategy_class
        self._strategy_kwargs = strategy_kwargs or {}
        self._strategy = None

        if FLWR_AVAILABLE:
            self._initialize_strategy()
        else:
            logger.warning(
                "FlowerServerWrapper created but flwr is not installed. "
                "Call is_available() before using server methods."
            )

    def _initialize_strategy(self) -> None:
        """Initialize the aggregation strategy."""
        if not FLWR_AVAILABLE:
            return

        strategy_cls = self._strategy_class or FedAvg
        self._strategy = strategy_cls(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_evaluate_clients,
            min_available_clients=self.min_clients,
            **self._strategy_kwargs
        )
        logger.info("Initialized strategy: %s", strategy_cls.__name__)

    def is_available(self) -> bool:
        """
        Check if the Flower framework is available.

        Returns:
            True if flwr is installed and ready to use.
        """
        return FLWR_AVAILABLE

    def get_installation_instructions(self) -> str:
        """
        Get installation instructions for the Flower framework.

        Returns:
            String with installation instructions.
        """
        return """
Flower (flwr) Installation Instructions:
=========================================

1. Basic installation:
   pip install flwr

2. With simulation support:
   pip install 'flwr[simulation]'

3. For development:
   pip install 'flwr[rest]'

Documentation: https://flower.dev/docs/

Example client code:
-------------------
import flwr as fl

class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=MyClient()
)
"""

    def start(
        self,
        server_address: str = "0.0.0.0:8080",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start the Flower server.

        Args:
            server_address: Address to bind the server to.
            config: Optional server configuration.

        Raises:
            RuntimeError: If flwr is not installed.
        """
        if not FLWR_AVAILABLE:
            raise RuntimeError(
                "Flower (flwr) is not installed. "
                f"{self.get_installation_instructions()}"
            )

        server_config = fl.server.ServerConfig(num_rounds=self.num_rounds)

        logger.info(
            "Starting Flower server on %s for %d rounds",
            server_address,
            self.num_rounds
        )

        fl.server.start_server(
            server_address=server_address,
            config=server_config,
            strategy=self._strategy,
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current server configuration.

        Returns:
            Dictionary with server configuration.
        """
        return {
            "num_rounds": self.num_rounds,
            "min_clients": self.min_clients,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "flwr_available": FLWR_AVAILABLE,
        }


def create_simple_server(
    num_rounds: int = 3,
    min_clients: int = 2,
    server_address: str = "0.0.0.0:8080"
) -> FlowerServerWrapper:
    """
    Create and configure a simple Flower server.

    This is a convenience function for quick server setup.

    Args:
        num_rounds: Number of training rounds.
        min_clients: Minimum number of clients required.
        server_address: Address for the server (used in logging).

    Returns:
        Configured FlowerServerWrapper instance.

    Example:
        server = create_simple_server(num_rounds=10, min_clients=3)
        if server.is_available():
            server.start("0.0.0.0:8080")
    """
    logger.info(
        "Creating simple Flower server: rounds=%d, min_clients=%d",
        num_rounds,
        min_clients
    )
    return FlowerServerWrapper(num_rounds=num_rounds, min_clients=min_clients)
