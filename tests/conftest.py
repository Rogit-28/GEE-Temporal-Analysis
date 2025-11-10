# Test infrastructure: conftest fixtures and shared utilities
"""
Pytest configuration and fixtures for SatChange tests.

This module provides fixtures and configuration for mocking external dependencies
like Google Earth Engine (ee) that may not be available in all test environments.
"""

import sys
from unittest.mock import MagicMock
import numpy as np

# Check if real cv2/matplotlib are available before mocking
try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False

try:
    import matplotlib
    import matplotlib.pyplot as plt

    _has_matplotlib = True
except ImportError:
    _has_matplotlib = False

try:
    import rasterio

    _has_rasterio = True
except ImportError:
    _has_rasterio = False

# Mock external dependencies that are always mocked
_always_mocked = ["ee", "diskcache", "jinja2", "PIL", "PIL.Image"]

for mod_name in _always_mocked:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Mock jinja2.Template specifically as it's used for rendering
mock_jinja2 = sys.modules["jinja2"]
mock_jinja2.Template = MagicMock()
mock_jinja2.Template.return_value.render = MagicMock(
    return_value="<html>Mock HTML</html>"
)

# Mock jinja2.Environment for autoescape-enabled rendering
mock_env_instance = MagicMock()
mock_env_instance.from_string.return_value.render = MagicMock(
    return_value="<html>Mock HTML</html>"
)
mock_jinja2.Environment = MagicMock(return_value=mock_env_instance)

# Only mock cv2 if not installed - create proper mocks
if not _has_cv2:
    mock_cv2 = MagicMock()

    # Mock cv2.filter2D to return properly shaped array
    def mock_filter2D(src, ddepth, kernel):
        return np.zeros_like(src)

    mock_cv2.filter2D = mock_filter2D

    # Mock cv2.GaussianBlur to return properly shaped array
    def mock_GaussianBlur(src, ksize, sigmaX):
        return src.copy()

    mock_cv2.GaussianBlur = mock_GaussianBlur

    mock_cv2.CV_64F = -1
    sys.modules["cv2"] = mock_cv2

# Only mock matplotlib if not installed
if not _has_matplotlib:
    mock_mpl = MagicMock()
    mock_plt = MagicMock()
    mock_colors = MagicMock()
    mock_patches = MagicMock()

    # Mock plt.subplots to return figure and axes
    def mock_subplots(*args, **kwargs):
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(3)]
        return mock_fig, mock_axes

    mock_plt.subplots = mock_subplots

    sys.modules["matplotlib"] = mock_mpl
    sys.modules["matplotlib.pyplot"] = mock_plt
    sys.modules["matplotlib.colors"] = mock_colors
    sys.modules["matplotlib.patches"] = mock_patches

# Only mock rasterio if not installed
if not _has_rasterio:
    mock_rasterio = MagicMock()
    mock_transform = MagicMock()

    # Mock rasterio.open as context manager
    class MockRasterioOpen:
        def __init__(self, *args, **kwargs):
            self.meta = {}
            self.profile = {}
            self.count = 1
            self.width = 100
            self.height = 100
            self.crs = "EPSG:4326"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write(self, data, band):
            pass

        def update_tags(self, **kwargs):
            pass

        def set_band_description(self, band, desc):
            pass

    mock_rasterio.open = MockRasterioOpen
    mock_transform.from_bounds = MagicMock(
        return_value=(0.0001, 0, 80.0, 0, -0.0001, 13.0)
    )

    sys.modules["rasterio"] = mock_rasterio
    sys.modules["rasterio.transform"] = mock_transform


import pytest
import tempfile
import os


@pytest.fixture
def sample_bands():
    """Create sample band data for testing."""
    return {
        "B4": np.random.rand(100, 100).astype(np.float32) * 1000,  # Red
        "B3": np.random.rand(100, 100).astype(np.float32) * 1000,  # Green
        "B8": np.random.rand(100, 100).astype(np.float32) * 1000,  # NIR
        "B2": np.random.rand(100, 100).astype(np.float32) * 1000,  # Blue
    }


@pytest.fixture
def sample_bands_with_b11():
    """Create sample band data including B11 (SWIR) for urban detection."""
    return {
        "B4": np.random.rand(100, 100).astype(np.float32) * 1000,  # Red
        "B3": np.random.rand(100, 100).astype(np.float32) * 1000,  # Green
        "B8": np.random.rand(100, 100).astype(np.float32) * 1000,  # NIR
        "B2": np.random.rand(100, 100).astype(np.float32) * 1000,  # Blue
        "B11": np.random.rand(50, 50).astype(np.float32) * 1000,  # SWIR (20m res)
        "QA60": np.zeros((100, 100), dtype=np.uint16),  # Cloud mask
    }


@pytest.fixture
def sample_classification():
    """Create sample classification data for testing."""
    return np.random.randint(0, 8, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        "crs": "EPSG:4326",
        "transform": (0.0001, 0.0, 80.2707, 0.0, -0.0001, 13.0827),
        "bounds": {
            "left": 80.2607,
            "right": 80.2807,
            "bottom": 13.0727,
            "top": 13.0927,
        },
        "width": 100,
        "height": 100,
        "date": "2020-01-01",
    }


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = MagicMock()
    config.get = MagicMock(
        side_effect=lambda key, default=None: {
            "cloud_threshold": 20,
            "pixel_size": 100,
            "cache_dir": "/tmp/satchange_cache",
        }.get(key, default)
    )
    return config


# Flags for tests to check if mocking is active
@pytest.fixture
def cv2_is_mocked():
    """Return True if cv2 is mocked."""
    return not _has_cv2


@pytest.fixture
def matplotlib_is_mocked():
    """Return True if matplotlib is mocked."""
    return not _has_matplotlib


@pytest.fixture
def rasterio_is_mocked():
    """Return True if rasterio is mocked."""
    return not _has_rasterio
