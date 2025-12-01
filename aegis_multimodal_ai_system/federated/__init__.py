"""
Federated Learning module for AEGIS Multimodal AI System.

This package provides federated learning capabilities using the Flower (flwr)
framework for privacy-preserving distributed training.
"""

from .flower_server import FlowerServerWrapper, check_flwr_available

__all__ = ["FlowerServerWrapper", "check_flwr_available"]
