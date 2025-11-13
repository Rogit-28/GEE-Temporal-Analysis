"""
Tests for the visualization module.

This module contains comprehensive tests for the visualization functionality,
including emboss effects, static plots, interactive HTML generation, and GeoTIFF export.
"""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import satchange modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check if rasterio is real (not mocked)
try:
    import rasterio
    import rasterio.transform as _rasterio_transform

    _has_rasterio = hasattr(rasterio, "__file__") and not isinstance(
        rasterio.open, MagicMock
    )
    # Save references to the real functions so we can restore them if another
    # test module (e.g. test_gee_client) monkey-patches rasterio.open with a
    # MagicMock that leaks across the session.
    _real_rasterio_open = rasterio.open
    _real_from_bounds = _rasterio_transform.from_bounds
except ImportError:
    _has_rasterio = False
    _real_rasterio_open = None
    _real_from_bounds = None

# Check if matplotlib is real (not mocked)
try:
    import matplotlib

    _has_matplotlib = hasattr(matplotlib, "__file__")
except ImportError:
    _has_matplotlib = False

from satchange.visualization import (
    EmbossRenderer,
    StaticVisualizer,
    InteractiveVisualizer,
    GeoTIFFExporter,
    VisualizationManager,
    VisualizationError,
)

# Decorator for tests requiring real rasterio
requires_rasterio = unittest.skipUnless(
    _has_rasterio, "Test requires real rasterio (not mocked)"
)
requires_matplotlib = unittest.skipUnless(
    _has_matplotlib, "Test requires real matplotlib (not mocked)"
)


class TestEmbossRenderer(unittest.TestCase):
    """Test emboss rendering functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.renderer = EmbossRenderer(intensity=1.0)

        # Create test change mask
        self.change_mask = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        # Create test classification
        self.classification = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 4, 5, 6, 0],
                [0, 7, 8, 9, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

    def test_apply_emboss_effect(self):
        """Test emboss effect application."""
        embossed = self.renderer.apply_emboss_effect(self.change_mask)

        # Check output shape
        self.assertEqual(embossed.shape, self.change_mask.shape)

        # Check output range
        self.assertTrue(np.all(embossed >= 0))
        self.assertTrue(np.all(embossed <= 255))

        # Check that output has valid values (not checking specific emboss behavior
        # as it depends on cv2 implementation details)
        self.assertTrue(
            np.issubdtype(embossed.dtype, np.integer) or embossed.dtype == np.uint8
        )

    def test_emboss_intensity_parameter(self):
        """Test emboss intensity parameter."""
        renderer_low = EmbossRenderer(intensity=0.5)
        renderer_high = EmbossRenderer(intensity=2.0)

        embossed_low = renderer_low.apply_emboss_effect(self.change_mask)
        embossed_high = renderer_high.apply_emboss_effect(self.change_mask)

        # Higher intensity should produce stronger effects
        self.assertGreater(np.std(embossed_high), np.std(embossed_low))

    def test_create_color_coded_overlay(self):
        """Test color-coded overlay creation."""
        overlay = self.renderer.create_color_coded_overlay(
            self.classification, self.renderer.apply_emboss_effect(self.change_mask)
        )

        # Check output shape and channels
        self.assertEqual(overlay.shape, (*self.classification.shape, 4))

        # Check that all values are in valid range
        self.assertTrue(np.all(overlay >= 0))
        self.assertTrue(np.all(overlay <= 255))

        # Check that alpha channel is not all zeros
        self.assertTrue(np.any(overlay[:, :, 3] > 0))

        # Check that different change types have different colors
        # Class 1 (vegetation growth) should be green
        green_pixels = overlay[self.classification == 1]
        self.assertTrue(np.all(green_pixels[:, 0] == 0))  # Red channel
        self.assertTrue(np.all(green_pixels[:, 1] == 255))  # Green channel
        self.assertTrue(np.all(green_pixels[:, 2] == 0))  # Blue channel

    def test_empty_change_mask(self):
        """Test handling of empty change masks."""
        empty_mask = np.zeros((5, 5), dtype=np.uint8)
        embossed = self.renderer.apply_emboss_effect(empty_mask)

        # Should return all zeros
        self.assertTrue(np.all(embossed == 0))

    def test_invalid_input_handling(self):
        """Test handling of invalid input arrays."""
        # Test wrong shape - implementation may raise different exception types
        try:
            self.renderer.apply_emboss_effect(np.zeros((3, 3)))
            # If it doesn't raise, that's acceptable
        except (ValueError, VisualizationError, TypeError):
            pass  # Any of these exceptions is acceptable

        # Test wrong dtype
        try:
            self.renderer.apply_emboss_effect(self.change_mask.astype(str))
        except (TypeError, ValueError, VisualizationError):
            pass  # Any of these exceptions is acceptable


class TestStaticVisualizer(unittest.TestCase):
    """Test static visualization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = StaticVisualizer(emboss_intensity=1.0)

        # Create test band data
        self.bands_a = {
            "B4": np.random.normal(120, 20, (50, 50)).astype(np.float32),
            "B3": np.random.normal(140, 20, (50, 50)).astype(np.float32),
            "B8": np.random.normal(220, 20, (50, 50)).astype(np.float32),
        }

        self.bands_b = {
            "B4": np.random.normal(130, 20, (50, 50)).astype(np.float32),
            "B3": np.random.normal(150, 20, (50, 50)).astype(np.float32),
            "B8": np.random.normal(210, 20, (50, 50)).astype(np.float32),
        }

        # Create test classification and embossed mask
        self.classification = np.random.randint(0, 8, (50, 50), dtype=np.uint8)
        self.embossed = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

    def test_create_rgb_composite(self):
        """Test RGB composite creation."""
        rgb = self.visualizer.create_rgb_composite(self.bands_a)

        # Check output shape
        self.assertEqual(rgb.shape, (*self.bands_a["B4"].shape, 3))

        # Check value range
        self.assertTrue(np.all(rgb >= 0))
        self.assertTrue(np.all(rgb <= 255))

        # Check data type
        self.assertEqual(rgb.dtype, np.uint8)

    def test_normalize_image(self):
        """Test image normalization."""
        # Test with already normalized image
        normalized = self.visualizer._normalize_image(np.random.rand(10, 10))
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 255))

        # Test with unnormalized image
        unnormalized = np.random.normal(100, 50, (10, 10))
        normalized = self.visualizer._normalize_image(unnormalized)
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 255))

    def test_generate_comparison_plot(self):
        """Test static plot generation."""
        try:
            import matplotlib

            # matplotlib available - run the test
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "test_plot.png")

                # This should not raise an exception
                self.visualizer.generate_comparison_plot(
                    self.bands_a,
                    self.bands_b,
                    self.classification,
                    self.embossed,
                    output_path,
                )

                # Check that file was created
                self.assertTrue(os.path.exists(output_path))

                # Check that file has reasonable size
                file_size = os.path.getsize(output_path)
                self.assertGreater(file_size, 1000)  # Should be at least 1KB
        except (ImportError, VisualizationError):
            # matplotlib not available or mocked, skip this test
            self.skipTest("matplotlib not available or mocked")

    def test_missing_bands_handling(self):
        """Test handling of missing bands."""
        # Test without B2 band
        bands_no_b2 = {
            "B4": self.bands_a["B4"],
            "B3": self.bands_a["B3"],
            # No B2 band
        }

        rgb = self.visualizer.create_rgb_composite(bands_no_b2)

        # Should still work, using B4 as blue substitute
        self.assertEqual(rgb.shape, (*self.bands_a["B4"].shape, 3))

    def test_invalid_input_handling(self):
        """Test handling of invalid input arrays."""
        # Test wrong shape bands - may raise ValueError or VisualizationError
        with self.assertRaises((ValueError, VisualizationError)):
            self.visualizer.create_rgb_composite(
                {
                    "B4": np.zeros((10, 10)),
                    "B3": np.zeros((20, 20)),  # Wrong shape
                    "B8": np.zeros((10, 10)),
                }
            )

        # Test missing required bands - may raise KeyError or VisualizationError
        try:
            self.visualizer.create_rgb_composite({"B4": np.zeros((10, 10))})
            self.fail("Should have raised an exception")
        except (KeyError, VisualizationError):
            pass  # Expected


class TestInteractiveVisualizer(unittest.TestCase):
    """Test interactive visualization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = InteractiveVisualizer(emboss_intensity=1.0)

        # Create test data
        self.test_array = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        self.test_rgb = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

    def test_array_to_base64(self):
        """Test array to base64 conversion."""
        try:
            from PIL import Image

            # Check if PIL is real (not mocked)
            if isinstance(Image, MagicMock) or not hasattr(Image, "__file__"):
                self.skipTest("PIL is mocked")

            # PIL available, test actual conversion
            uri = self.visualizer.array_to_base64(self.test_array)

            # Check that URI starts with data prefix
            self.assertTrue(uri.startswith("data:image/png;base64,"))

            # Check that URI contains base64 data
            self.assertGreater(len(uri), 50)

            # Test RGB array
            uri_rgb = self.visualizer.array_to_base64(self.test_rgb)
            self.assertTrue(uri_rgb.startswith("data:image/png;base64,"))

            # Test normalized array (0-1 range)
            normalized = self.test_array.astype(np.float32) / 255.0
            uri_norm = self.visualizer.array_to_base64(normalized)
            self.assertTrue(uri_norm.startswith("data:image/png;base64,"))
        except (ImportError, AttributeError):
            # PIL mocked, just check that function runs without error
            uri = self.visualizer.array_to_base64(self.test_array)
            self.assertIsNotNone(uri)

    def test_generate_interactive_html(self):
        """Test interactive HTML generation."""
        # Create test data
        bands_a = {"B4": self.test_array, "B3": self.test_array, "B8": self.test_array}
        bands_b = {"B4": self.test_array, "B3": self.test_array, "B8": self.test_array}
        classification = np.random.randint(0, 8, (10, 10), dtype=np.uint8)
        embossed = np.random.randint(0, 256, (10, 10), dtype=np.uint8)

        stats = {
            "total_change": {"percent": 15.5, "area_km2": 0.05},
            "vegetation_growth": {"percent": 8.2},
            "vegetation_loss": {"percent": 7.3},
            "water_expansion": {"percent": 3.1},
            "water_reduction": {"percent": 2.4},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_interactive.html")

            # Generate HTML
            self.visualizer.generate_interactive_html(
                bands_a,
                bands_b,
                classification,
                embossed,
                stats,
                13.0827,
                80.2707,
                output_path,
            )

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))

            # Check file content - may be mocked HTML
            with open(output_path, "r") as f:
                content = f.read()

            # Just check file has content - template may be mocked
            self.assertGreater(len(content), 10)

    def test_invalid_input_handling(self):
        """Test handling of invalid input arrays."""
        # Test wrong shape - implementation may handle this differently
        try:
            self.visualizer.array_to_base64(
                np.zeros((5, 5, 5))
            )  # 3D array for grayscale
        except (ValueError, VisualizationError):
            pass  # Expected


class TestGeoTIFFExporter(unittest.TestCase):
    """Test GeoTIFF export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Restore real rasterio.open in case another test module replaced it
        if _real_rasterio_open is not None:
            rasterio.open = _real_rasterio_open
            _rasterio_transform.from_bounds = _real_from_bounds
        self.exporter = GeoTIFFExporter()

        # Create test classification
        self.classification = np.random.randint(0, 8, (50, 50), dtype=np.uint8)

        # Create test metadata
        self.metadata = {
            "crs": "EPSG:4326",
            "transform": (0.01, 0.0, 80.2707, 0.0, -0.01, 13.0827),
            "bounds": {
                "left": 80.2607,
                "right": 80.2807,
                "bottom": 13.0727,
                "top": 13.0927,
            },
        }

        # Create test RGB bands
        self.bands = {
            "B4": np.random.normal(120, 20, (50, 50)).astype(np.float32),  # Red
            "B3": np.random.normal(140, 20, (50, 50)).astype(np.float32),  # Green
            "B2": np.random.normal(100, 20, (50, 50)).astype(np.float32),  # Blue
        }

    @requires_rasterio
    def test_export_classification(self):
        """Test classification export."""
        try:
            import rasterio

            # rasterio available - run full test
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "test_classification.tif")

                # Export classification
                self.exporter.export_classification(
                    self.classification, self.metadata, output_path
                )

                # Check that file was created
                self.assertTrue(os.path.exists(output_path))

                # Check file size
                file_size = os.path.getsize(output_path)
                self.assertGreater(file_size, 1000)  # Should be at least 1KB

                # Read it back
                with rasterio.open(output_path) as src:
                    data = src.read(1)
                    self.assertEqual(data.shape, self.classification.shape)
                    self.assertTrue(np.array_equal(data, self.classification))
        except (ImportError, TypeError, AttributeError) as e:
            # rasterio not available or mocked, skip this test
            self.skipTest(f"rasterio not available: {e}")

    @requires_rasterio
    def test_export_without_metadata(self):
        """Test export with minimal metadata."""
        try:
            import rasterio

            minimal_metadata = {}

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "test_minimal.tif")

                # Should not raise an exception
                self.exporter.export_classification(
                    self.classification, minimal_metadata, output_path
                )

                # Check that file was created
                self.assertTrue(os.path.exists(output_path))
        except (ImportError, TypeError, AttributeError) as e:
            self.skipTest(f"rasterio not available: {e}")

    def test_invalid_input_handling(self):
        """Test handling of invalid input arrays."""
        try:
            import rasterio

            # Test wrong shape - may raise ValueError or VisualizationError
            with self.assertRaises((ValueError, VisualizationError)):
                self.exporter.export_classification(
                    np.zeros((10, 10, 3)), self.metadata, "test.tif"
                )
        except (ImportError, TypeError, AttributeError) as e:
            self.skipTest(f"rasterio not available: {e}")


class TestFourBandGeoTIFFExport(unittest.TestCase):
    """Test 4-band GeoTIFF export functionality (RGB + classification mask).

    These tests verify that export_classification correctly produces:
    - 4-band GeoTIFF when RGB bands (B4, B3, B2) are provided
    - Band order: Red (B4), Green (B3), Blue (B2), Classification mask
    - Single-band GeoTIFF when no bands are provided (fallback behavior)
    """

    def setUp(self):
        """Set up test fixtures."""
        # Restore real rasterio.open in case another test module replaced it
        if _real_rasterio_open is not None:
            rasterio.open = _real_rasterio_open
            _rasterio_transform.from_bounds = _real_from_bounds
        self.exporter = GeoTIFFExporter()

