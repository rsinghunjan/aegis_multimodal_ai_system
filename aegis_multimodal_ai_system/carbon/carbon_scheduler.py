"""
Carbon-Aware Scheduler Helper for AEGIS Multimodal AI System.

This module provides carbon-aware scheduling capabilities to help
reduce the carbon footprint of compute-intensive AI workloads.

Features:
- Query carbon intensity from configurable API endpoints
- Caching to reduce API calls
- Retry logic for resilience
- Threshold-based scheduling decisions

Environment Variables:
- CARBON_API_URL: URL of the carbon intensity API
- CARBON_API_KEY: API key for authentication (if required)
- CARBON_CACHE_TTL: Cache TTL in seconds (default: 300)
- CARBON_INTENSITY_THRESHOLD: Default threshold for scheduling

For production:
- Integrate with real carbon APIs (WattTime, ElectricityMap, etc.)
- Add metrics collection for carbon savings tracking
- Implement predictive scheduling based on forecasts
- Consider regional carbon intensity variations
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Check if requests is available
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    logger.warning(
        "requests library not installed. "
        "Install with: pip install requests"
    )


@dataclass
class CarbonIntensityData:
    """
    Carbon intensity data from API response.

    Attributes:
        intensity: Carbon intensity value (gCO2/kWh).
        timestamp: Unix timestamp when data was retrieved.
        region: Geographic region for the intensity data.
        source: Data source/API name.
    """
    intensity: float
    timestamp: float
    region: str = "unknown"
    source: str = "unknown"


class CarbonCache:
    """
    Simple in-memory cache for carbon intensity data.

    Attributes:
        ttl: Time-to-live for cached data in seconds.
    """

    def __init__(self, ttl: int = 300):
        """
        Initialize the cache.

        Args:
            ttl: Cache TTL in seconds (default: 5 minutes).
        """
        self.ttl = ttl
        self._cache: Dict[str, tuple] = {}

    def get(self, key: str) -> Optional[CarbonIntensityData]:
        """
        Get cached data if not expired.

        Args:
            key: Cache key.

        Returns:
            Cached CarbonIntensityData or None if expired/missing.
        """
        if key in self._cache:
            data, expiry = self._cache[key]
            if time.time() < expiry:
                return data
            del self._cache[key]
        return None

    def set(self, key: str, data: CarbonIntensityData) -> None:
        """
        Store data in cache.

        Args:
            key: Cache key.
            data: CarbonIntensityData to cache.
        """
        expiry = time.time() + self.ttl
        self._cache[key] = (data, expiry)

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


class CarbonScheduler:
    """
    Carbon-aware scheduler helper.

    This class provides methods to query carbon intensity and make
    scheduling decisions based on current grid conditions.

    Example usage:
        scheduler = CarbonScheduler()
        if scheduler.should_schedule_now(threshold=200):
            run_training_job()
        else:
            defer_training_job()
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        default_threshold: Optional[float] = None,
        region: str = "default",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the carbon scheduler.

        Args:
            api_url: Carbon API endpoint URL (or env var CARBON_API_URL).
            api_key: API key for authentication (or env var CARBON_API_KEY).
            cache_ttl: Cache TTL in seconds (or env var CARBON_CACHE_TTL).
            default_threshold: Default intensity threshold for scheduling.
            region: Geographic region for intensity queries.
            max_retries: Maximum retry attempts for API calls.
            retry_delay: Delay between retries in seconds.
        """
        self.api_url = api_url or os.environ.get(
            "CARBON_API_URL",
            "https://api.carbonintensity.org.uk/intensity"
        )
        self.api_key = api_key or os.environ.get("CARBON_API_KEY")
        cache_ttl_value = cache_ttl or int(
            os.environ.get("CARBON_CACHE_TTL", "300")
        )
        self.default_threshold = default_threshold or float(
            os.environ.get("CARBON_INTENSITY_THRESHOLD", "200")
        )
        self.region = region
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cache = CarbonCache(ttl=cache_ttl_value)

        logger.info(
            "CarbonScheduler initialized: api_url=%s, region=%s, threshold=%s",
            self.api_url,
            self.region,
            self.default_threshold
        )

    def _make_request(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with retry logic.

        Args:
            url: URL to request.
            headers: Optional request headers.

        Returns:
            JSON response as dictionary or None on failure.
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available")
            return None

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "Request attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    str(e)
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        logger.error("All request attempts failed for %s", url)
        return None

    def get_current_intensity(
        self,
        use_cache: bool = True
    ) -> Optional[CarbonIntensityData]:
        """
        Get current carbon intensity from the API.

        Args:
            use_cache: Whether to use cached data if available.

        Returns:
            CarbonIntensityData or None if unavailable.

        Example:
            intensity = scheduler.get_current_intensity()
            if intensity:
                print(f"Current intensity: {intensity.intensity} gCO2/kWh")
        """
        cache_key = f"intensity:{self.region}"

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug("Using cached carbon intensity data")
                return cached

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["Accept"] = "application/json"

        data = self._make_request(self.api_url, headers)
        if not data:
            return None

        # Parse response - adapt this to your specific API
        try:
            intensity_value = self._parse_intensity_response(data)
            result = CarbonIntensityData(
                intensity=intensity_value,
                timestamp=time.time(),
                region=self.region,
                source=self.api_url
            )
            self._cache.set(cache_key, result)
            return result
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to parse carbon intensity response: %s", e)
            return None

    def _parse_intensity_response(self, data: Dict[str, Any]) -> float:
        """
        Parse carbon intensity from API response.

        This method should be adapted to match your specific API's response format.

        Args:
            data: JSON response from API.

        Returns:
            Carbon intensity value in gCO2/kWh.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If intensity cannot be parsed.
        """
        # UK Carbon Intensity API format
        if "data" in data and isinstance(data["data"], list):
            return float(data["data"][0]["intensity"]["actual"])

        # Generic format
        if "intensity" in data:
            return float(data["intensity"])

        # WattTime-like format
        if "moer" in data:
            return float(data["moer"])

        raise KeyError("Unable to find intensity value in response")

    def should_schedule_now(
        self,
        threshold: Optional[float] = None,
        fallback: bool = True
    ) -> bool:
        """
        Determine if a job should be scheduled based on current carbon intensity.

        Args:
            threshold: Carbon intensity threshold (gCO2/kWh). Jobs should run
                      when intensity is below this value.
            fallback: Value to return if intensity data is unavailable.

        Returns:
            True if current intensity is below threshold (good time to schedule),
            False otherwise, or fallback value if data unavailable.

        Example:
            if scheduler.should_schedule_now(threshold=150):
                submit_training_job()
            else:
                queue_for_later()
        """
        effective_threshold = threshold or self.default_threshold
        intensity_data = self.get_current_intensity()

        if intensity_data is None:
            logger.warning(
                "Carbon intensity unavailable, using fallback: %s",
                fallback
            )
            return fallback

        should_schedule = intensity_data.intensity < effective_threshold
        logger.info(
            "Carbon scheduling decision: intensity=%.1f, threshold=%.1f, "
            "schedule=%s",
            intensity_data.intensity,
            effective_threshold,
            should_schedule
        )
        return should_schedule

    def get_scheduling_recommendation(
        self,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get detailed scheduling recommendation with reasoning.

        Args:
            threshold: Carbon intensity threshold.

        Returns:
            Dictionary with recommendation details.
        """
        effective_threshold = threshold or self.default_threshold
        intensity_data = self.get_current_intensity()

        if intensity_data is None:
            return {
                "recommendation": "unknown",
                "reason": "Carbon intensity data unavailable",
                "intensity": None,
                "threshold": effective_threshold,
            }

        should_schedule = intensity_data.intensity < effective_threshold

        return {
            "recommendation": "schedule" if should_schedule else "defer",
            "reason": (
                f"Intensity {intensity_data.intensity:.1f} is "
                f"{'below' if should_schedule else 'above'} "
                f"threshold {effective_threshold:.1f}"
            ),
            "intensity": intensity_data.intensity,
            "threshold": effective_threshold,
            "region": intensity_data.region,
            "timestamp": intensity_data.timestamp,
        }

    def clear_cache(self) -> None:
        """Clear the intensity cache."""
        self._cache.clear()
        logger.info("Carbon intensity cache cleared")


def create_scheduler(
    api_url: Optional[str] = None,
    threshold: Optional[float] = None
) -> CarbonScheduler:
    """
    Create a CarbonScheduler with common defaults.

    Args:
        api_url: Optional API URL override.
        threshold: Optional threshold override.

    Returns:
        Configured CarbonScheduler instance.
    """
    return CarbonScheduler(
        api_url=api_url,
        default_threshold=threshold
    )


# Mock implementation for testing without external API
class MockCarbonScheduler(CarbonScheduler):
    """
    Mock carbon scheduler for testing.

    This class provides predictable responses without making API calls.
    """

    def __init__(
        self,
        mock_intensity: float = 100.0,
        **kwargs
    ):
        """
        Initialize mock scheduler.

        Args:
            mock_intensity: Intensity value to return.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self.mock_intensity = mock_intensity

    def get_current_intensity(
        self,
        use_cache: bool = True
    ) -> CarbonIntensityData:
        """Return mock intensity data."""
        return CarbonIntensityData(
            intensity=self.mock_intensity,
            timestamp=time.time(),
            region="mock",
            source="mock"
        )
