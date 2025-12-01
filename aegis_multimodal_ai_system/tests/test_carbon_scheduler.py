"""
Tests for the Carbon Scheduler module.
"""

from aegis_multimodal_ai_system.carbon.carbon_scheduler import (
    CarbonScheduler,
    CarbonIntensityData,
    CarbonCache,
    MockCarbonScheduler,
    create_scheduler,
)


class TestCarbonIntensityData:
    """Test cases for CarbonIntensityData dataclass."""

    def test_create_intensity_data(self):
        """Test creating CarbonIntensityData."""
        data = CarbonIntensityData(
            intensity=150.0,
            timestamp=1234567890.0,
            region="UK",
            source="test"
        )
        assert data.intensity == 150.0
        assert data.timestamp == 1234567890.0
        assert data.region == "UK"
        assert data.source == "test"

    def test_default_values(self):
        """Test CarbonIntensityData default values."""
        data = CarbonIntensityData(intensity=100.0, timestamp=0.0)
        assert data.region == "unknown"
        assert data.source == "unknown"


class TestCarbonCache:
    """Test cases for CarbonCache class."""

    def test_cache_set_get(self):
        """Test basic cache set and get."""
        cache = CarbonCache(ttl=60)
        data = CarbonIntensityData(intensity=100.0, timestamp=0.0)
        cache.set("key1", data)
        result = cache.get("key1")
        assert result is not None
        assert result.intensity == 100.0

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = CarbonCache(ttl=60)
        assert cache.get("nonexistent") is None

    def test_cache_clear(self):
        """Test cache clear."""
        cache = CarbonCache(ttl=60)
        data = CarbonIntensityData(intensity=100.0, timestamp=0.0)
        cache.set("key1", data)
        cache.clear()
        assert cache.get("key1") is None


class TestCarbonScheduler:
    """Test cases for CarbonScheduler class."""

    def test_init_default(self):
        """Test CarbonScheduler initialization with defaults."""
        scheduler = CarbonScheduler()
        assert scheduler.default_threshold == 200.0
        assert scheduler.max_retries == 3

    def test_init_custom(self):
        """Test CarbonScheduler initialization with custom values."""
        scheduler = CarbonScheduler(
            api_url="https://custom.api.com",
            default_threshold=100.0,
            max_retries=5,
        )
        assert scheduler.api_url == "https://custom.api.com"
        assert scheduler.default_threshold == 100.0
        assert scheduler.max_retries == 5

    def test_clear_cache(self):
        """Test cache clearing."""
        scheduler = CarbonScheduler()
        # Should not raise
        scheduler.clear_cache()


class TestMockCarbonScheduler:
    """Test cases for MockCarbonScheduler class."""

    def test_mock_get_current_intensity(self):
        """Test MockCarbonScheduler returns expected intensity."""
        scheduler = MockCarbonScheduler(mock_intensity=150.0)
        data = scheduler.get_current_intensity()
        assert data is not None
        assert data.intensity == 150.0
        assert data.region == "mock"
        assert data.source == "mock"

    def test_mock_should_schedule_below_threshold(self):
        """Test should_schedule_now returns True when below threshold."""
        scheduler = MockCarbonScheduler(mock_intensity=100.0)
        assert scheduler.should_schedule_now(threshold=200.0) is True

    def test_mock_should_schedule_above_threshold(self):
        """Test should_schedule_now returns False when above threshold."""
        scheduler = MockCarbonScheduler(mock_intensity=300.0)
        assert scheduler.should_schedule_now(threshold=200.0) is False

    def test_mock_get_scheduling_recommendation(self):
        """Test get_scheduling_recommendation returns proper structure."""
        scheduler = MockCarbonScheduler(mock_intensity=100.0)
        rec = scheduler.get_scheduling_recommendation(threshold=200.0)

        assert "recommendation" in rec
        assert "reason" in rec
        assert "intensity" in rec
        assert "threshold" in rec
        assert rec["recommendation"] == "schedule"
        assert rec["intensity"] == 100.0

    def test_mock_scheduling_recommendation_defer(self):
        """Test get_scheduling_recommendation recommends defer when high."""
        scheduler = MockCarbonScheduler(mock_intensity=300.0)
        rec = scheduler.get_scheduling_recommendation(threshold=200.0)

        assert rec["recommendation"] == "defer"
        assert "above" in rec["reason"]


class TestCreateScheduler:
    """Test cases for create_scheduler function."""

    def test_create_scheduler_default(self):
        """Test creating scheduler with defaults."""
        scheduler = create_scheduler()
        assert scheduler.default_threshold == 200.0

    def test_create_scheduler_custom(self):
        """Test creating scheduler with custom values."""
        scheduler = create_scheduler(
            api_url="https://custom.api.com",
            threshold=150.0
        )
        assert scheduler.api_url == "https://custom.api.com"
        assert scheduler.default_threshold == 150.0
