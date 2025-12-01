"""
Carbon-aware scheduling module for AEGIS Multimodal AI System.

This package provides carbon-aware scheduling capabilities to help
reduce the environmental impact of compute-intensive AI workloads.
"""

from .carbon_scheduler import (
    CarbonScheduler,
    CarbonIntensityData,
    MockCarbonScheduler,
    create_scheduler,
)

__all__ = [
    "CarbonScheduler",
    "CarbonIntensityData",
    "MockCarbonScheduler",
    "create_scheduler",
]
