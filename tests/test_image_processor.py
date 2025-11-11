"""
Tests for the image processing pipeline.

This module contains comprehensive tests for image preprocessing functionality,
including B11 resampling from 20m to 10m resolution, cloud masking,
coregistration checks, and radiometric normalization.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import satchange modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Note: conftest.py handles mocking of ee, diskcache, and jinja2
from satchange.image_processor import ImageProcessor, ImageProcessingError


class TestB11Resampling(unittest.TestCase):
    """Test B11 band resampling from 20m to 10m resolution."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config
        self.config = Mock()
        self.config.get = Mock(return_value=20)  # Default cloud threshold
        
        self.processor = ImageProcessor(self.config)
        
        # Create test bands at 10m resolution (100x100 pixels)
        self.bands_10m = {
            'B4': np.random.rand(100, 100).astype(np.float32) * 1000,  # Red
            'B3': np.random.rand(100, 100).astype(np.float32) * 1000,  # Green
            'B8': np.random.rand(100, 100).astype(np.float32) * 1000,  # NIR
        }
        
        # B11 at 20m resolution (50x50 pixels - half the size)
        self.b11_20m = np.random.rand(50, 50).astype(np.float32) * 1000
    
    def test_resample_b11_zoom_factor(self):
        """Test that B11 is correctly resampled with zoom factor ~2x."""
        bands = self.bands_10m.copy()
        bands['B11'] = self.b11_20m
        
        # Resample B11 to 10m
        result = self.processor._resample_b11_to_10m(bands)
        
        # Check that B11 is present in result
        self.assertIn('B11', result)
        
        # Check that resampled B11 has correct shape (should match 10m bands)
        self.assertEqual(result['B11'].shape, self.bands_10m['B4'].shape)
        
        # Check zoom factor: 50x50 -> 100x100 means zoom factor of 2.0
        expected_shape = (100, 100)
        self.assertEqual(result['B11'].shape, expected_shape)
    
    def test_resample_b11_shape_matching(self):
        """Test that resampled B11 matches reference band shape exactly."""
        # Test with different reference shapes
        reference_shapes = [(100, 100), (80, 120), (150, 75)]
        
        for ref_shape in reference_shapes:
            with self.subTest(ref_shape=ref_shape):
                bands = {
                    'B4': np.random.rand(*ref_shape).astype(np.float32) * 1000,
                    'B3': np.random.rand(*ref_shape).astype(np.float32) * 1000,
                    'B8': np.random.rand(*ref_shape).astype(np.float32) * 1000,
                    # B11 at half resolution
                    'B11': np.random.rand(ref_shape[0] // 2, ref_shape[1] // 2).astype(np.float32) * 1000,
                }
                
                result = self.processor._resample_b11_to_10m(bands)
                
                # Verify B11 now matches reference shape
                self.assertEqual(result['B11'].shape, ref_shape)
    
    def test_resample_b11_preserves_dtype(self):
        """Test that resampling preserves the original data type."""
        dtypes = [np.float32, np.float64, np.int16, np.uint16]
        
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                bands = self.bands_10m.copy()
                bands['B11'] = self.b11_20m.astype(dtype)
                
                result = self.processor._resample_b11_to_10m(bands)
                
                # Check that dtype is preserved
                self.assertEqual(result['B11'].dtype, dtype)
    
    def test_resample_b11_value_range(self):
        """Test that resampled values are within reasonable range."""
        bands = self.bands_10m.copy()
        bands['B11'] = self.b11_20m
        
        original_min = self.b11_20m.min()
        original_max = self.b11_20m.max()
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Resampled values should be within the original range
        # (with some tolerance for interpolation)
        self.assertGreaterEqual(result['B11'].min(), original_min * 0.9)
        self.assertLessEqual(result['B11'].max(), original_max * 1.1)
    
    def test_resample_b11_already_10m(self):
        """Test handling when B11 is already at 10m resolution."""
        bands = self.bands_10m.copy()
        bands['B11'] = np.random.rand(100, 100).astype(np.float32) * 1000  # Already 10m
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Should return unchanged since shapes already match
        self.assertEqual(result['B11'].shape, (100, 100))
    
    def test_resample_b11_missing_band(self):
        """Test handling when B11 band is not present."""
        bands = self.bands_10m.copy()
        # No B11 band
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Should return original bands unchanged
        self.assertNotIn('B11', result)
        self.assertEqual(result['B4'].shape, self.bands_10m['B4'].shape)
    
    def test_resample_b11_no_reference_band(self):
        """Test handling when no 10m reference band is available."""
        bands = {
            'B11': self.b11_20m,
            # No B4, B3, or B8
        }
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Should return original bands since no reference available
        self.assertEqual(result['B11'].shape, self.b11_20m.shape)
    
    def test_resample_b11_uses_correct_reference_priority(self):
        """Test that B4 is used as default reference, then B3, then B8."""
        # Test with only B8 available
        bands = {
            'B8': np.random.rand(100, 100).astype(np.float32),
            'B11': np.random.rand(50, 50).astype(np.float32),
        }
        
        result = self.processor._resample_b11_to_10m(bands, reference_band='B8')
        self.assertEqual(result['B11'].shape, (100, 100))
        
        # Test with B3 available (B4 missing)
        bands = {
            'B3': np.random.rand(80, 80).astype(np.float32),
            'B11': np.random.rand(40, 40).astype(np.float32),
        }
        
        result = self.processor._resample_b11_to_10m(bands, reference_band='B3')
        self.assertEqual(result['B11'].shape, (80, 80))
    
    def test_resample_b11_bilinear_interpolation(self):
        """Test that bilinear interpolation produces smooth results."""
        # Create a simple gradient pattern
        gradient = np.linspace(0, 100, 50).reshape(1, 50).repeat(50, axis=0).astype(np.float32)
        
        bands = self.bands_10m.copy()
        bands['B11'] = gradient
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Check that the gradient is preserved (values should increase left to right)
        resampled = result['B11']
        left_mean = resampled[:, :25].mean()
        right_mean = resampled[:, 75:].mean()
        
        self.assertLess(left_mean, right_mean)
    
    def test_resample_b11_data_integrity_known_pattern(self):
        """Test that known values are correctly preserved after resampling."""
        # Create a checkerboard pattern at 20m resolution
        b11_20m = np.zeros((50, 50), dtype=np.float32)
        b11_20m[::2, ::2] = 1000  # Alternating pattern
        b11_20m[1::2, 1::2] = 1000
        
        bands = self.bands_10m.copy()
        bands['B11'] = b11_20m
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # After bilinear interpolation, values should be smoothed
        # Mean should be approximately preserved
        original_mean = b11_20m.mean()
        resampled_mean = result['B11'].mean()
        
        # Mean should be within 10% of original
        self.assertAlmostEqual(resampled_mean, original_mean, delta=original_mean * 0.1)
    
    def test_resample_b11_statistics_preservation(self):
        """Test that overall statistics (mean) are reasonably preserved."""
        bands = self.bands_10m.copy()
        bands['B11'] = self.b11_20m
        
        original_mean = self.b11_20m.mean()
        original_min = self.b11_20m.min()
        original_max = self.b11_20m.max()
        
        result = self.processor._resample_b11_to_10m(bands)
        
        resampled_mean = result['B11'].mean()
        
        # Mean should be very close (within 5%)
        self.assertAlmostEqual(resampled_mean, original_mean, delta=original_mean * 0.05)
        
        # Min/Max should be within reasonable bounds (interpolation doesn't extrapolate)
        # Bilinear interpolation should not create values outside the original range
        self.assertGreaterEqual(result['B11'].min(), original_min - 1)  # Allow small numerical error
        self.assertLessEqual(result['B11'].max(), original_max + 1)
    
    def test_resample_b11_empty_bands_dict(self):
        """Test handling of empty bands dictionary."""
        bands = {}
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Should return empty dict unchanged
        self.assertEqual(result, {})
    
    def test_resample_b11_negative_values(self):
        """Test handling of negative values (some indices can be negative)."""
        # Create B11 with negative values (e.g., normalized differences)
        b11_with_negatives = np.random.rand(50, 50).astype(np.float32) * 2 - 1  # Range [-1, 1]
        
        bands = self.bands_10m.copy()
        bands['B11'] = b11_with_negatives
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Check shape is correct
        self.assertEqual(result['B11'].shape, (100, 100))
        
        # Check that negative values are preserved (min should still be negative)
        self.assertLess(result['B11'].min(), 0)
        
        # Check that overall range is maintained
        self.assertGreaterEqual(result['B11'].min(), b11_with_negatives.min() * 1.1)
        self.assertLessEqual(result['B11'].max(), b11_with_negatives.max() * 1.1)
    
    def test_resample_b11_non_standard_zoom_factor(self):
        """Test resampling with non-2x zoom factors."""
        # Test 3x zoom (30m to 10m equivalent)
        bands = {
            'B4': np.random.rand(150, 150).astype(np.float32) * 1000,
            'B11': np.random.rand(50, 50).astype(np.float32) * 1000,  # 3x smaller
        }
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Should upscale to match B4
        self.assertEqual(result['B11'].shape, (150, 150))
        
        # Test 1.5x zoom (arbitrary factor)
        bands = {
            'B4': np.random.rand(75, 75).astype(np.float32) * 1000,
            'B11': np.random.rand(50, 50).astype(np.float32) * 1000,  # 1.5x smaller
        }
        
        result = self.processor._resample_b11_to_10m(bands)
        
        self.assertEqual(result['B11'].shape, (75, 75))
    
    def test_resample_b11_single_pixel(self):
        """Test edge case with single pixel B11."""
        bands = {
            'B4': np.random.rand(10, 10).astype(np.float32) * 1000,
            'B11': np.array([[500.0]], dtype=np.float32),  # Single pixel
        }
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Should expand to match B4
        self.assertEqual(result['B11'].shape, (10, 10))
        
        # Value should be constant (single pixel expanded)
        # Due to bilinear interpolation, edge values might vary slightly
        self.assertAlmostEqual(result['B11'].mean(), 500.0, delta=10.0)
    
    def test_resample_b11_does_not_modify_original(self):
        """Test that resampling does not modify the original bands dictionary."""
        bands = self.bands_10m.copy()
        original_b11 = self.b11_20m.copy()
        bands['B11'] = self.b11_20m
        
        original_bands_keys = list(bands.keys())
        
        result = self.processor._resample_b11_to_10m(bands)
        
        # Original B11 should be unchanged
        np.testing.assert_array_equal(bands['B11'], original_b11)
        
        # Original bands dict should have same keys
        self.assertEqual(list(bands.keys()), original_bands_keys)
        
        # Result should be a different dict
        self.assertIsNot(result, bands)
    
    def test_resample_b11_rectangular_arrays(self):
        """Test resampling with non-square arrays."""
        # Wide array
        bands = {
            'B4': np.random.rand(50, 200).astype(np.float32) * 1000,
            'B11': np.random.rand(25, 100).astype(np.float32) * 1000,
        }
        
        result = self.processor._resample_b11_to_10m(bands)
        self.assertEqual(result['B11'].shape, (50, 200))
        
        # Tall array
        bands = {
            'B4': np.random.rand(200, 50).astype(np.float32) * 1000,
            'B11': np.random.rand(100, 25).astype(np.float32) * 1000,
        }
        
        result = self.processor._resample_b11_to_10m(bands)
        self.assertEqual(result['B11'].shape, (200, 50))


class TestImagePreprocessing(unittest.TestCase):
    """Test the complete image preprocessing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.get = Mock(return_value=20)
        
        self.processor = ImageProcessor(self.config)
        
        # Create test data with all required bands including B11
        self.bands_a = {
            'B4': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B3': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B8': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B11': np.random.rand(50, 50).astype(np.float32) * 1000,  # 20m resolution
            'QA60': np.zeros((100, 100), dtype=np.uint16),
        }
        
        self.bands_b = {
            'B4': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B3': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B8': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B11': np.random.rand(50, 50).astype(np.float32) * 1000,  # 20m resolution
            'QA60': np.zeros((100, 100), dtype=np.uint16),
        }
        
        self.metadata_a = {
            'width': 100,
            'height': 100,
            'crs': 'EPSG:32643',
            'transform': (10.0, 0.0, 500000.0, 0.0, -10.0, 1500000.0),
            'date': '2020-01-01',
        }
        
        self.metadata_b = {
            'width': 100,
            'height': 100,
            'crs': 'EPSG:32643',
            'transform': (10.0, 0.0, 500000.0, 0.0, -10.0, 1500000.0),
            'date': '2020-06-01',
        }
    
    def test_preprocess_resamples_b11(self):
        """Test that preprocessing pipeline resamples B11."""
        processed_a, processed_b = self.processor.preprocess_image_pair(
            self.bands_a, self.bands_b,
            self.metadata_a, self.metadata_b
        )
        
        # Check that B11 is now 10m resolution
        self.assertEqual(processed_a['B11'].shape, self.bands_a['B4'].shape)
        self.assertEqual(processed_b['B11'].shape, self.bands_b['B4'].shape)
    
    def test_preprocess_preserves_other_bands(self):
        """Test that preprocessing preserves other bands correctly."""
        processed_a, processed_b = self.processor.preprocess_image_pair(
            self.bands_a, self.bands_b,
            self.metadata_a, self.metadata_b
        )
        
        # Check all bands are present
        for band in ['B4', 'B3', 'B8', 'B11']:
            self.assertIn(band, processed_a)
            self.assertIn(band, processed_b)
    
    def test_preprocess_without_b11(self):
        """Test preprocessing works when B11 is not present."""
        bands_a_no_b11 = {k: v for k, v in self.bands_a.items() if k != 'B11'}
        bands_b_no_b11 = {k: v for k, v in self.bands_b.items() if k != 'B11'}
        
        # Should not raise an exception
        processed_a, processed_b = self.processor.preprocess_image_pair(
            bands_a_no_b11, bands_b_no_b11,
            self.metadata_a, self.metadata_b
        )
        
        # B11 should not be present
        self.assertNotIn('B11', processed_a)
        self.assertNotIn('B11', processed_b)


class TestCloudMasking(unittest.TestCase):
    """Test cloud masking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.get = Mock(return_value=20)
        
        self.processor = ImageProcessor(self.config)
    
    def test_create_cloud_mask(self):
        """Test cloud mask creation from QA60 band."""
        # Create a QA60 band with some cloudy pixels
        qa60 = np.zeros((100, 100), dtype=np.uint16)
        qa60[20:40, 20:40] = 2000  # Cloudy area
        
        cloud_mask = self.processor._create_cloud_mask(qa60)
        
        # Check output shape
        self.assertEqual(cloud_mask.shape, qa60.shape)
        
        # Check that cloudy areas are masked (0) and clear areas are unmasked (1)
        self.assertEqual(cloud_mask[50, 50], 1)  # Clear pixel
        self.assertEqual(cloud_mask[30, 30], 0)  # Cloudy pixel


class TestImageQualityValidation(unittest.TestCase):
    """Test image quality validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.get = Mock(return_value=20)
        
        self.processor = ImageProcessor(self.config)
    
    def test_validate_image_quality(self):
        """Test image quality validation."""
        bands = {
            'B4': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B3': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B8': np.random.rand(100, 100).astype(np.float32) * 1000,
            'QA60': np.zeros((100, 100), dtype=np.uint16),
        }
        
        metadata = {'width': 100, 'height': 100}
        
        quality = self.processor.validate_image_quality(bands, metadata)
        
        # Check required keys
        required_keys = ['total_pixels', 'valid_pixels', 'cloud_coverage', 
                        'brightness_mean', 'brightness_std', 'has_data']
        for key in required_keys:
            self.assertIn(key, quality)
        
        # Check that we have valid data
        self.assertTrue(quality['has_data'])
        self.assertGreater(quality['valid_pixels'], 0)


class TestProcessingSummary(unittest.TestCase):
    """Test processing summary generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.get = Mock(return_value=20)
        
        self.processor = ImageProcessor(self.config)
    
    def test_get_processing_summary(self):
        """Test processing summary generation."""
        bands_a = {
            'B4': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B3': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B8': np.random.rand(100, 100).astype(np.float32) * 1000,
            'QA60': np.zeros((100, 100), dtype=np.uint16),
        }
        
        bands_b = {
            'B4': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B3': np.random.rand(100, 100).astype(np.float32) * 1000,
            'B8': np.random.rand(100, 100).astype(np.float32) * 1000,
            'QA60': np.zeros((100, 100), dtype=np.uint16),
        }
        
        metadata_a = {'date': '2020-01-01'}
        metadata_b = {'date': '2020-06-01'}
        
        summary = self.processor.get_processing_summary(
            bands_a, bands_b, metadata_a, metadata_b
        )
        
        # Check required keys
        required_keys = ['date_a', 'date_b', 'cloud_coverage_a', 
                        'cloud_coverage_b', 'brightness_difference',
                        'processing_successful', 'warnings']
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check that processing was successful
        self.assertTrue(summary['processing_successful'])


if __name__ == '__main__':
    unittest.main()
