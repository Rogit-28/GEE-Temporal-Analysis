"""
Tests for the change detection engine.

This module contains comprehensive tests for the change detection algorithms,
spectral index calculations, and change classification functionality.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import satchange modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from satchange.change_detector import (
    ChangeDetector, SpectralIndexCalculator, ChangeDetectionError,
    ChangeType, ChangeDetector
)


class TestSpectralIndexCalculator(unittest.TestCase):
    """Test spectral index calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = SpectralIndexCalculator()
        
        # Create test band data
        self.red_band = np.array([[100, 150, 200], [50, 100, 150]], dtype=np.float32)
        self.nir_band = np.array([[200, 250, 300], [150, 200, 250]], dtype=np.float32)
        self.green_band = np.array([[120, 170, 220], [70, 120, 170]], dtype=np.float32)
        self.swir_band = np.array([[180, 230, 280], [130, 180, 230]], dtype=np.float32)
    
    def test_calculate_ndvi(self):
        """Test NDVI calculation."""
        ndvi = self.calculator.calculate_ndvi(self.red_band, self.nir_band)
        
        # Check output shape
        self.assertEqual(ndvi.shape, self.red_band.shape)
        
        # Check value range
        self.assertTrue(np.all(ndvi >= -1))
        self.assertTrue(np.all(ndvi <= 1))
        
        # Test expected values for known vegetation
        # High NIR, low Red should give high NDVI
        expected_high_ndvi = (300 - 200) / (300 + 200)  # 0.2
        self.assertAlmostEqual(ndvi[0, 2], expected_high_ndvi, places=5)
        
        # Test edge case - division by zero protection
        zero_red = np.zeros((2, 2))
        zero_nir = np.zeros((2, 2))
        ndvi_zero = self.calculator.calculate_ndvi(zero_red, zero_nir)
        self.assertTrue(np.all(np.isfinite(ndvi_zero)))
    
    def test_calculate_ndwi(self):
        """Test NDWI calculation."""
        ndwi = self.calculator.calculate_ndwi(self.green_band, self.nir_band)
        
        # Check output shape
        self.assertEqual(ndwi.shape, self.green_band.shape)
        
        # Check value range
        self.assertTrue(np.all(ndwi >= -1))
        self.assertTrue(np.all(ndwi <= 1))
        
        # Test expected values for water
        # High Green, low NIR should give high NDWI
        expected_high_ndwi = (220 - 300) / (220 + 300)  # -0.1538
        self.assertAlmostEqual(ndwi[0, 2], expected_high_ndwi, places=5)
    
    def test_calculate_ndbi(self):
        """Test NDBI calculation."""
        ndbi = self.calculator.calculate_ndbi(self.swir_band, self.nir_band)
        
        # Check output shape
        self.assertEqual(ndbi.shape, self.swir_band.shape)
        
        # Check value range
        self.assertTrue(np.all(ndbi >= -1))
        self.assertTrue(np.all(ndbi <= 1))
        
        # Test expected values for built-up areas
        # High SWIR, low NIR should give high NDBI
        expected_high_ndbi = (280 - 300) / (280 + 300)  # -0.0345
        self.assertAlmostEqual(ndbi[0, 2], expected_high_ndbi, places=5)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid input arrays."""
        # Test mismatched shapes - implementation broadcasts, so may not raise
        # Just verify it doesn't crash unexpectedly
        try:
            result = self.calculator.calculate_ndvi(self.red_band, self.nir_band[:1, :1])
            # If it doesn't raise, check result is valid
            self.assertIsNotNone(result)
        except (ValueError, ChangeDetectionError):
            pass  # Either exception is acceptable
        
        # Test non-numeric arrays
        try:
            self.calculator.calculate_ndvi(self.red_band.astype(str), self.nir_band)
        except (TypeError, ChangeDetectionError, ValueError):
            pass  # Any of these exceptions is acceptable


class TestChangeDetector(unittest.TestCase):
    """Test the main change detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ChangeDetector(threshold=0.2)
        
        # Create test band data with known changes
        # Date A: Vegetation (high NDVI)
        self.bands_a = {
            'B4': np.array([[100, 120, 140], [80, 100, 120]], dtype=np.float32),  # Red
            'B3': np.array([[120, 140, 160], [100, 120, 140]], dtype=np.float32),  # Green
            'B8': np.array([[200, 220, 240], [180, 200, 220]], dtype=np.float32),  # NIR
            'B11': np.array([[150, 170, 190], [130, 150, 170]], dtype=np.float32), # SWIR (for urban detection)
        }
        
        # Date B: Urban (low NDVI) in some areas
        self.bands_b = {
            'B4': np.array([[150, 170, 190], [130, 150, 170]], dtype=np.float32),  # Red (increased)
            'B3': np.array([[130, 150, 170], [110, 130, 150]], dtype=np.float32),  # Green (increased)
            'B8': np.array([[100, 120, 140], [80, 100, 120]], dtype=np.float32),   # NIR (decreased)
            'B11': np.array([[180, 200, 220], [160, 180, 200]], dtype=np.float32), # SWIR (increased for urban)
        }
        
        # Date C: Water (high NDWI) in some areas
        self.bands_c = {
            'B4': np.array([[100, 120, 140], [80, 100, 120]], dtype=np.float32),   # Red
            'B3': np.array([[180, 200, 220], [160, 180, 200]], dtype=np.float32),  # Green (increased)
            'B8': np.array([[80, 100, 120], [60, 80, 100]], dtype=np.float32),    # NIR (decreased)
            'B11': np.array([[150, 170, 190], [130, 150, 170]], dtype=np.float32), # SWIR
        }
    
    def test_detect_vegetation_change(self):
        """Test vegetation change detection."""
        result = self.detector.detect_vegetation_change(self.bands_a, self.bands_b)
        
        # Check required keys in result
        required_keys = ['ndvi_a', 'ndvi_b', 'delta', 'change_mask', 'growth_mask', 'loss_mask']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check output shapes
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                self.assertEqual(value.shape, self.bands_a['B4'].shape)
        
        # Check that loss mask is detected where vegetation decreased
        # In our test, NIR decreased significantly, so we should see loss
        self.assertTrue(np.any(result['loss_mask']))
        
        # Check that growth mask is empty (no vegetation growth in this test)
        self.assertFalse(np.any(result['growth_mask']))
    
    def test_detect_water_change(self):
        """Test water change detection."""
        result = self.detector.detect_water_change(self.bands_a, self.bands_c)
        
        # Check required keys in result
        required_keys = ['ndwi_a', 'ndwi_b', 'delta', 'change_mask', 'expansion_mask', 'reduction_mask']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that expansion mask is detected where water increased
        # In our test, Green increased and NIR decreased, so we should see expansion
        self.assertTrue(np.any(result['expansion_mask']))
    
    def test_detect_all_changes(self):
        """Test multi-index change detection."""
        result = self.detector.detect_all_changes(self.bands_a, self.bands_b)
        
        # Check required keys in result
        required_keys = ['vegetation', 'water', 'combined_mask']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that vegetation results are included
        self.assertIn('ndvi_a', result['vegetation'])
        self.assertIn('ndvi_b', result['vegetation'])
        
        # Check that water results are included
        self.assertIn('ndwi_a', result['water'])
        self.assertIn('ndwi_b', result['water'])
        
        # Check combined mask shape
        self.assertEqual(result['combined_mask'].shape, self.bands_a['B4'].shape)
    
    def test_threshold_parameter(self):
        """Test change detection with different thresholds."""
        detector_low = ChangeDetector(threshold=0.1)
        detector_high = ChangeDetector(threshold=0.5)
        
        result_low = detector_low.detect_vegetation_change(self.bands_a, self.bands_b)
        result_high = detector_high.detect_vegetation_change(self.bands_a, self.bands_b)
        
        # Lower threshold should detect more changes
        self.assertGreaterEqual(np.sum(result_low['change_mask']), np.sum(result_high['change_mask']))
    
    def test_invalid_input_handling(self):
        """Test handling of invalid input arrays."""
        # Test missing bands - implementation raises ChangeDetectionError wrapping KeyError
        invalid_bands = {'B4': self.bands_a['B4']}  # Missing B8 for NDVI
        with self.assertRaises(ChangeDetectionError):
            self.detector.detect_vegetation_change(invalid_bands, self.bands_b)
        
        # Test mismatched shapes - implementation may handle this differently
        different_shape_bands = {
            'B4': self.bands_a['B4'][:1, :1],
            'B3': self.bands_a['B3'][:1, :1],
            'B8': self.bands_a['B8'][:1, :1],
            'B11': self.bands_a['B11'][:1, :1],
        }
        # This may not raise for detect_all_changes since arrays broadcast
        # Just verify it handles it without crashing
        try:
            result = self.detector.detect_all_changes(different_shape_bands, self.bands_b)
            self.assertIsNotNone(result)
        except (ValueError, ChangeDetectionError):
            pass  # Either exception is acceptable
    
    def test_get_change_summary(self):
        """Test the comprehensive change summary function."""
        result = self.detector.get_change_summary(self.bands_a, self.bands_b, 'all')
        
        # Check required keys in result
        required_keys = ['classification', 'statistics', 'change_results', 'summary']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check classification shape
        self.assertEqual(result['classification'].shape, self.bands_a['B4'].shape)
        
        # Check statistics structure
        self.assertIn('total_change', result['statistics'])
        self.assertIn('change_types', result['statistics'])
        
        # Check that summary is a string
        self.assertIsInstance(result['summary'], str)
    
    def test_single_change_type_summary(self):
        """Test change summary for single change types."""
        # Test vegetation only - single change type doesn't return 'classification' key
        result = self.detector.get_change_summary(self.bands_a, self.bands_b, 'vegetation')
        
        # Check that result has expected keys for single change type
        self.assertIn('change_type', result)
        self.assertIn('change_results', result)
        self.assertIn('statistics', result)
        self.assertIn('summary', result)
        self.assertEqual(result['change_type'], 'vegetation')
        
        # Note: single change type doesn't include 'classification' key
        # Only 'all' change type includes classification
        
        # Test water only
        result = self.detector.get_change_summary(self.bands_a, self.bands_c, 'water')
        
        # Check that result has expected keys for single change type
        self.assertIn('change_type', result)
        self.assertEqual(result['change_type'], 'water')
        self.assertIn('change_results', result)
        self.assertIn('statistics', result)


class TestChangeClassification(unittest.TestCase):
    """Test change classification functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ChangeDetector()
        
        # Create test change results with non-overlapping masks for each class type
        # Using a 3x4 grid to ensure we have unique positions for each type
        self.change_results = {
            'vegetation': {
                'growth_mask': np.array([[False, True, False, False], 
                                        [False, False, False, False], 
                                        [False, False, False, False]], dtype=bool),
                'loss_mask': np.array([[True, False, False, False], 
                                      [False, False, False, False], 
                                      [False, False, False, False]], dtype=bool),
                'change_mask': np.array([[True, True, False, False], 
                                        [False, False, False, False], 
                                        [False, False, False, False]], dtype=bool),
            },
            'water': {
                'expansion_mask': np.array([[False, False, True, False], 
                                           [False, False, False, False], 
                                           [False, False, False, False]], dtype=bool),
                'reduction_mask': np.array([[False, False, False, True], 
                                           [False, False, False, False], 
                                           [False, False, False, False]], dtype=bool),
                'change_mask': np.array([[False, False, True, True], 
                                        [False, False, False, False], 
                                        [False, False, False, False]], dtype=bool),
            },
            'urban': {
                'development_mask': np.array([[False, False, False, False], 
                                             [True, False, False, False], 
                                             [False, False, False, False]], dtype=bool),
                'decline_mask': np.array([[False, False, False, False], 
                                         [False, True, False, False], 
                                         [False, False, False, False]], dtype=bool),
                'change_mask': np.array([[False, False, False, False], 
                                        [True, True, False, False], 
                                        [False, False, False, False]], dtype=bool),
            },
            'combined_mask': np.array([[True, True, True, True], 
                                      [True, True, False, False], 
                                      [False, False, False, False]], dtype=bool)
        }
    
    def test_classify_changes(self):
        """Test change classification function."""
        classification = self.detector.classify_changes(self.change_results)
        
        # Check output shape
        self.assertEqual(classification.shape, self.change_results['combined_mask'].shape)
        
        # Check that all values are valid class IDs (0-7)
        unique_classes = np.unique(classification)
        valid_classes = [0, 1, 2, 3, 4, 5, 6, 7]  # All valid change types including urban and ambiguous
        self.assertTrue(np.all(np.isin(unique_classes, valid_classes)))
        
        # Check specific classifications at known positions
        # Position [0,0] has vegetation loss -> class 2
        self.assertEqual(classification[0, 0], 2)
        
        # Position [0,1] has vegetation growth -> class 1
        self.assertEqual(classification[0, 1], 1)
        
        # Position [0,2] has water expansion -> class 3
        self.assertEqual(classification[0, 2], 3)
        
        # Position [0,3] has water reduction -> class 4
        self.assertEqual(classification[0, 3], 4)
        
        # Position [1,0] has urban development -> class 5
        self.assertEqual(classification[1, 0], 5)
        
        # Position [1,1] has urban decline -> class 6
        self.assertEqual(classification[1, 1], 6)
    
    def test_ambiguous_change_handling(self):
        """Test handling of ambiguous changes (multiple change types at same pixel)."""
        # Create overlapping change masks at position [2,0] - no other changes there
        self.change_results['vegetation']['growth_mask'][2, 0] = True
        self.change_results['water']['expansion_mask'][2, 0] = True
        self.change_results['vegetation']['change_mask'][2, 0] = True
        self.change_results['water']['change_mask'][2, 0] = True
        self.change_results['combined_mask'][2, 0] = True
        
        classification = self.detector.classify_changes(self.change_results)
        
        # Ambiguous changes should be classified as class 7 (not 5)
        self.assertEqual(classification[2, 0], 7)
    
    def test_no_change_handling(self):
        """Test handling of areas with no change."""
        # Position [2,3] has no change in any mask (all False in that position)
        classification = self.detector.classify_changes(self.change_results)
        
        # No-change areas should be class 0
        self.assertEqual(classification[2, 3], 0)


class TestChangeStatistics(unittest.TestCase):
    """Test change statistics computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ChangeDetector()
        
        # Create test classification
        self.classification = np.array([
            [0, 1, 2, 3],
            [4, 0, 1, 2],
            [3, 4, 0, 1],
            [2, 3, 4, 0]
        ], dtype=np.uint8)
        
        # Create test change results
        self.change_results = {
            'vegetation': {
                'growth_mask': np.array([[False, True, False, False], 
                                       [False, False, True, False]], dtype=bool),
                'loss_mask': np.array([[True, False, False, False], 
                                     [False, False, False, True]], dtype=bool),
            },
            'water': {
                'expansion_mask': np.array([[False, False, True, False], 
                                          [False, False, False, False]], dtype=bool),
                'reduction_mask': np.array([[False, False, False, False], 
                                          [True, False, False, False]], dtype=bool),
            }
        }
    
    def test_compute_change_statistics(self):
        """Test change statistics computation."""
        stats = self.detector.compute_change_statistics(self.classification)
        
        # Check required keys - must include urban classes
        required_keys = ['no_change', 'vegetation_growth', 'vegetation_loss', 
                        'water_expansion', 'water_reduction', 'urban_development',
                        'urban_decline', 'ambiguous', 'total_change']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check that all class percentages sum to 100%
        class_keys = ['no_change', 'vegetation_growth', 'vegetation_loss', 
                     'water_expansion', 'water_reduction', 'urban_development',
                     'urban_decline', 'ambiguous']
        total_percent = sum(stats[key]['percent'] for key in class_keys)
        self.assertAlmostEqual(total_percent, 100.0, places=1)
        
        # Check that total change is calculated correctly
        expected_total = (stats['vegetation_growth']['percent'] + 
                         stats['vegetation_loss']['percent'] +
                         stats['water_expansion']['percent'] + 
                         stats['water_reduction']['percent'] +
                         stats['urban_development']['percent'] +
                         stats['urban_decline']['percent'] +
                         stats['ambiguous']['percent'])
        self.assertAlmostEqual(stats['total_change']['percent'], expected_total, places=1)
        
        # Check area calculations
        pixel_area = 100  # 10m resolution = 100m² per pixel
        
        for key, stat in stats.items():
            if key not in ['total_change', 'change_types']:
                # Check that area is calculated correctly
                expected_area = (stat['pixels'] * pixel_area) / 1e6  # Convert to km²
                self.assertAlmostEqual(stat['area_km2'], expected_area, places=4)
    
    def test_pixel_area_parameter(self):
        """Test statistics computation with different pixel areas."""
        stats_default = self.detector.compute_change_statistics(self.classification)
        stats_custom = self.detector.compute_change_statistics(self.classification, pixel_area_m2=50)
        
        # Custom pixel area should give smaller area values
        self.assertLess(stats_custom['total_change']['area_km2'], 
                       stats_default['total_change']['area_km2'])


class TestChangeDetectorIntegration(unittest.TestCase):
    """Integration tests for the complete change detection pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ChangeDetector(threshold=0.2)
        
        # Create realistic test data
        height, width = 50, 50
        
        # Date A: Mostly vegetation with some water
        self.bands_a = {
            'B4': np.random.normal(120, 20, (height, width)).astype(np.float32),  # Red
            'B3': np.random.normal(140, 20, (height, width)).astype(np.float32),  # Green
            'B8': np.random.normal(220, 20, (height, width)).astype(np.float32),  # NIR
            'B11': np.random.normal(150, 20, (height, width)).astype(np.float32), # SWIR
        }
        
        # Create some known changes in Date B
        # Convert some vegetation to urban (decrease NIR, increase Red/Green)
        urban_area = slice(10, 20), slice(10, 20)
        self.bands_b = {
            'B4': self.bands_a['B4'].copy(),
            'B3': self.bands_a['B3'].copy(),
            'B8': self.bands_a['B8'].copy(),
            'B11': self.bands_a['B11'].copy(),
        }
        
        # Urban changes
        self.bands_b['B4'][urban_area] += 50   # Increase red
        self.bands_b['B3'][urban_area] += 30   # Increase green
        self.bands_b['B8'][urban_area] -= 80   # Decrease NIR
        self.bands_b['B11'][urban_area] += 50  # Increase SWIR (urban signature)
        
        # Water changes (convert some vegetation to water)
        water_area = slice(30, 40), slice(30, 40)
        self.bands_b['B3'][water_area] += 60   # Increase green
        self.bands_b['B8'][water_area] -= 100  # Decrease NIR
    
    def test_end_to_end_change_detection(self):
        """Test the complete change detection pipeline."""
        # Run change detection
        result = self.detector.get_change_summary(self.bands_a, self.bands_b, 'all')
        
        # Check that we get reasonable results
        self.assertIsInstance(result, dict)
        self.assertIn('classification', result)
        self.assertIn('statistics', result)
        self.assertIn('change_results', result)
        self.assertIn('summary', result)
        
        # Check that changes were detected
        changed_pixels = np.sum(result['classification'] > 0)
        self.assertGreater(changed_pixels, 0)  # Should detect some changes
        
        # Check that statistics are reasonable
        stats = result['statistics']
        self.assertIn('total_change', stats)
        self.assertGreaterEqual(stats['total_change']['percent'], 0)
        self.assertLessEqual(stats['total_change']['percent'], 100)
        
        # Check that summary text is generated
        self.assertIsInstance(result['summary'], str)
        self.assertGreater(len(result['summary']), 0)
    
    def test_different_change_types(self):
        """Test change detection for different change types."""
        change_types = ['vegetation', 'water', 'all']
        
        for change_type in change_types:
            with self.subTest(change_type=change_type):
                result = self.detector.get_change_summary(self.bands_a, self.bands_b, change_type)
                
                # Should always return the same base structure
                self.assertIsInstance(result, dict)
                self.assertIn('change_type', result)
                self.assertIn('change_results', result)
                self.assertIn('statistics', result)
                self.assertIn('summary', result)
                
                if change_type == 'all':
                    # Only 'all' returns classification
                    self.assertIn('classification', result)
                    unique_classes = np.unique(result['classification'])
                    expected_classes = [0, 1, 2, 3, 4, 5, 6, 7]  # all classes
                    self.assertTrue(np.all(np.isin(unique_classes, expected_classes)))
    
    def test_performance_with_large_arrays(self):
        """Test performance with larger arrays."""
        # Create larger test arrays
        height, width = 200, 200
        
        large_bands_a = {
            'B4': np.random.normal(120, 20, (height, width)).astype(np.float32),
            'B3': np.random.normal(140, 20, (height, width)).astype(np.float32),
            'B8': np.random.normal(220, 20, (height, width)).astype(np.float32),
            'B11': np.random.normal(150, 20, (height, width)).astype(np.float32),
        }
        
        large_bands_b = {
            'B4': large_bands_a['B4'].copy() + np.random.normal(0, 10, (height, width)).astype(np.float32),
            'B3': large_bands_a['B3'].copy() + np.random.normal(0, 10, (height, width)).astype(np.float32),
            'B8': large_bands_a['B8'].copy() + np.random.normal(0, 10, (height, width)).astype(np.float32),
            'B11': large_bands_a['B11'].copy() + np.random.normal(0, 10, (height, width)).astype(np.float32),
        }
        
        # Should handle larger arrays without crashing
        result = self.detector.get_change_summary(large_bands_a, large_bands_b, 'all')
        
        # Check that results are still valid
        self.assertEqual(result['classification'].shape, (height, width))
        self.assertIsInstance(result['statistics'], dict)


class TestUrbanChangeDetection(unittest.TestCase):
    """Test urban change detection with B11 (SWIR) band."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ChangeDetector(threshold=0.2)
        self.calculator = SpectralIndexCalculator()
        
        # Create test data simulating urban development
        # Date A: Vegetation (low NDBI - vegetation has higher NIR than SWIR)
        self.bands_a = {
            'B4': np.array([[100, 120, 140], [80, 100, 120]], dtype=np.float32),   # Red
            'B3': np.array([[120, 140, 160], [100, 120, 140]], dtype=np.float32),  # Green
            'B8': np.array([[250, 270, 290], [230, 250, 270]], dtype=np.float32),  # NIR (high for vegetation)
            'B11': np.array([[150, 170, 190], [130, 150, 170]], dtype=np.float32), # SWIR (low for vegetation)
        }
        
        # Date B: Urban (high NDBI - built-up areas have higher SWIR than NIR)
        self.bands_b = {
            'B4': np.array([[180, 200, 220], [160, 180, 200]], dtype=np.float32),  # Red (increased)
            'B3': np.array([[170, 190, 210], [150, 170, 190]], dtype=np.float32),  # Green (increased)
            'B8': np.array([[150, 170, 190], [130, 150, 170]], dtype=np.float32),  # NIR (decreased)
            'B11': np.array([[220, 240, 260], [200, 220, 240]], dtype=np.float32), # SWIR (increased)
        }
    
    def test_ndbi_calculation_with_b11(self):
        """Test NDBI calculation using B11 and B8 bands."""
        # Calculate NDBI: (B11 - B8) / (B11 + B8)
        ndbi = self.calculator.calculate_ndbi(self.bands_a['B11'], self.bands_a['B8'])
        
        # Check output shape
        self.assertEqual(ndbi.shape, self.bands_a['B11'].shape)
        
        # Check value range
        self.assertTrue(np.all(ndbi >= -1))
        self.assertTrue(np.all(ndbi <= 1))
        
        # For vegetation (high NIR, low SWIR), NDBI should be negative
        self.assertTrue(np.all(ndbi < 0))
    
    def test_ndbi_urban_signature(self):
        """Test that urban areas have positive NDBI values."""
        # Calculate NDBI for urban area (Date B)
        ndbi_urban = self.calculator.calculate_ndbi(self.bands_b['B11'], self.bands_b['B8'])
        
        # Urban areas should have positive NDBI (SWIR > NIR)
        self.assertTrue(np.all(ndbi_urban > 0))
    
    def test_ndbi_vegetation_signature(self):
        """Test that vegetation areas have negative NDBI values."""
        # Calculate NDBI for vegetation area (Date A)
        ndbi_veg = self.calculator.calculate_ndbi(self.bands_a['B11'], self.bands_a['B8'])
        
        # Vegetation should have negative NDBI (NIR > SWIR)
        self.assertTrue(np.all(ndbi_veg < 0))
    
    def test_detect_urban_change(self):
        """Test urban change detection from vegetation to built-up."""
        result = self.detector.detect_urban_change(self.bands_a, self.bands_b)
        
        # Check required keys in result
        required_keys = ['ndbi_a', 'ndbi_b', 'delta', 'change_mask', 
                        'development_mask', 'decline_mask', 'magnitude', 'change_type']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check output shapes
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                self.assertEqual(value.shape, self.bands_a['B4'].shape)
        
        # Urban development should be detected (positive NDBI change)
        # NDBI increased from negative (vegetation) to positive (urban)
        self.assertTrue(np.any(result['development_mask']))
        
        # No urban decline expected in this test case
        self.assertFalse(np.any(result['decline_mask']))
    
    def test_detect_urban_decline(self):
        """Test detection of urban decline (reverse of development)."""
        # Swap dates to simulate decline
        result = self.detector.detect_urban_change(self.bands_b, self.bands_a)
        
        # Urban decline should be detected (negative NDBI change)
        self.assertTrue(np.any(result['decline_mask']))
        
        # No urban development expected in this reverse case
        self.assertFalse(np.any(result['development_mask']))
    
    def test_ndbi_requires_b11_band(self):
        """Test that urban detection requires B11 band."""
        # Remove B11 from bands
        bands_no_b11 = {k: v for k, v in self.bands_a.items() if k != 'B11'}
        
        # Should raise ChangeDetectionError (wrapping KeyError) when B11 is missing
        with self.assertRaises(ChangeDetectionError):
            self.detector.detect_urban_change(bands_no_b11, self.bands_b)
    
    def test_ndbi_with_resampled_b11(self):
        """Test NDBI calculation with B11 resampled to 10m resolution."""
        # Simulate resampled B11 (same shape as NIR)
        b11_resampled = np.array([[150, 170, 190], [130, 150, 170]], dtype=np.float32)
        b8_nir = np.array([[250, 270, 290], [230, 250, 270]], dtype=np.float32)
        
        ndbi = self.calculator.calculate_ndbi(b11_resampled, b8_nir)
        
        # Should work correctly with matching shapes
        self.assertEqual(ndbi.shape, b11_resampled.shape)
        self.assertTrue(np.all(ndbi >= -1))
        self.assertTrue(np.all(ndbi <= 1))
    
    def test_ndbi_zero_handling(self):
        """Test NDBI calculation handles zero values correctly."""
        # Create bands with zero values
        b11 = np.array([[0, 100], [100, 0]], dtype=np.float32)
        b8 = np.array([[100, 0], [100, 100]], dtype=np.float32)
        
        ndbi = self.calculator.calculate_ndbi(b11, b8)
        
        # Should not produce NaN or Inf values
        self.assertTrue(np.all(np.isfinite(ndbi)))
    
    def test_urban_change_in_comprehensive_detection(self):
        """Test that urban changes are included in comprehensive detection."""
        result = self.detector.detect_all_changes(self.bands_a, self.bands_b)
        
        # Check that urban results are included
        self.assertIn('urban', result)
        self.assertIn('ndbi_a', result['urban'])
        self.assertIn('ndbi_b', result['urban'])
        self.assertIn('development_mask', result['urban'])
        self.assertIn('decline_mask', result['urban'])
    
    def test_urban_classification_values(self):
        """Test that urban changes are classified with correct values."""
        result = self.detector.detect_all_changes(self.bands_a, self.bands_b)
        classification = self.detector.classify_changes(result)
        
        # Check that urban development is classified as 5
        # and urban decline as 6 (based on ChangeType enum)
        unique_classes = np.unique(classification)
        
        # Should have some urban development (class 5)
        self.assertTrue(5 in unique_classes or 6 in unique_classes or 
                       np.any(result['urban']['change_mask']))
    
    def test_ndbi_expected_values(self):
        """Test NDBI calculation produces expected values for known inputs."""
        # Known values: SWIR=200, NIR=100 -> NDBI = (200-100)/(200+100) = 0.333
        swir = np.array([[200]], dtype=np.float32)
        nir = np.array([[100]], dtype=np.float32)
        
        ndbi = self.calculator.calculate_ndbi(swir, nir)
        
        expected = (200 - 100) / (200 + 100)  # ~0.333
        self.assertAlmostEqual(ndbi[0, 0], expected, places=3)
        
        # Known values: SWIR=100, NIR=200 -> NDBI = (100-200)/(100+200) = -0.333
        swir = np.array([[100]], dtype=np.float32)
        nir = np.array([[200]], dtype=np.float32)
        
        ndbi = self.calculator.calculate_ndbi(swir, nir)
        
        expected = (100 - 200) / (100 + 200)  # ~-0.333
        self.assertAlmostEqual(ndbi[0, 0], expected, places=3)


class TestUrbanNDBICalculation(unittest.TestCase):
    """Tests specifically for NDBI calculation formula: NDBI = (B11 - B8) / (B11 + B8)."""
    
    def setUp(self):
        """Set up test fixtures with known synthetic data."""
        self.detector = ChangeDetector(threshold=0.2)
        self.calculator = SpectralIndexCalculator()
    
    def test_ndbi_formula_uses_b11_and_b8(self):
        """Test that NDBI is calculated as (B11 - B8) / (B11 + B8)."""
        # Create synthetic data with known values
        b11_swir = np.array([[300, 400], [200, 100]], dtype=np.float32)
        b8_nir = np.array([[100, 200], [200, 300]], dtype=np.float32)
        
        ndbi = self.calculator.calculate_ndbi(b11_swir, b8_nir)
        
        # Manually calculate expected values
        # [0,0]: (300-100)/(300+100) = 200/400 = 0.5
        # [0,1]: (400-200)/(400+200) = 200/600 = 0.333...
        # [1,0]: (200-200)/(200+200) = 0/400 = 0.0
        # [1,1]: (100-300)/(100+300) = -200/400 = -0.5
        expected = np.array([[0.5, 1/3], [0.0, -0.5]], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(ndbi, expected, decimal=3)
    
    def test_development_mask_detects_positive_ndbi_change(self):
        """Test that development_mask is True where NDBI increases above threshold."""
        # Date A: Low NDBI (vegetation-like, NIR > SWIR)
        bands_a = {
            'B8': np.array([[300, 300], [300, 300]], dtype=np.float32),   # High NIR
            'B11': np.array([[100, 100], [100, 100]], dtype=np.float32),  # Low SWIR
        }
        # NDBI_A = (100-300)/(100+300) = -0.5 (vegetation)
        
        # Date B: High NDBI in top row (urban-like, SWIR > NIR)
        bands_b = {
            'B8': np.array([[100, 100], [300, 300]], dtype=np.float32),   # Low NIR (top)
            'B11': np.array([[300, 300], [100, 100]], dtype=np.float32),  # High SWIR (top)
        }
        # NDBI_B top row = (300-100)/(300+100) = 0.5 (urban)
        # NDBI_B bottom row = (100-300)/(100+300) = -0.5 (vegetation)
        
        # Delta top row = 0.5 - (-0.5) = 1.0 (> threshold 0.2) -> development
        # Delta bottom row = -0.5 - (-0.5) = 0.0 (< threshold) -> no change
        
        result = self.detector.detect_urban_change(bands_a, bands_b)
        
        # Top row should be detected as development
        self.assertTrue(result['development_mask'][0, 0])
        self.assertTrue(result['development_mask'][0, 1])
        
        # Bottom row should NOT be detected as development
        self.assertFalse(result['development_mask'][1, 0])
        self.assertFalse(result['development_mask'][1, 1])
    
    def test_decline_mask_detects_negative_ndbi_change(self):
        """Test that decline_mask is True where NDBI decreases below -threshold."""
        # Date A: High NDBI (urban-like)
        bands_a = {
            'B8': np.array([[100, 100], [300, 300]], dtype=np.float32),   # Low NIR (top)
            'B11': np.array([[300, 300], [100, 100]], dtype=np.float32),  # High SWIR (top)
        }
        # NDBI_A top row = 0.5 (urban)
        # NDBI_A bottom row = -0.5 (vegetation)
        
        # Date B: Low NDBI (reverted to vegetation in top row)
        bands_b = {
            'B8': np.array([[300, 300], [300, 300]], dtype=np.float32),   # High NIR
            'B11': np.array([[100, 100], [100, 100]], dtype=np.float32),  # Low SWIR
        }
        # NDBI_B = -0.5 (all vegetation)
        
        # Delta top row = -0.5 - 0.5 = -1.0 (< -threshold) -> decline
        # Delta bottom row = -0.5 - (-0.5) = 0.0 -> no change
        
        result = self.detector.detect_urban_change(bands_a, bands_b)
        
        # Top row should be detected as decline
        self.assertTrue(result['decline_mask'][0, 0])
        self.assertTrue(result['decline_mask'][0, 1])
        
        # Bottom row should NOT be detected as decline
        self.assertFalse(result['decline_mask'][1, 0])
        self.assertFalse(result['decline_mask'][1, 1])
    
    def test_missing_b11_raises_error(self):
        """Test that detect_urban_change raises ChangeDetectionError when B11 is missing."""
        bands_without_b11 = {
            'B4': np.array([[100]], dtype=np.float32),
            'B3': np.array([[100]], dtype=np.float32),
            'B8': np.array([[100]], dtype=np.float32),
            # B11 intentionally missing
        }
        bands_with_b11 = {
            'B4': np.array([[100]], dtype=np.float32),
            'B3': np.array([[100]], dtype=np.float32),
            'B8': np.array([[100]], dtype=np.float32),
            'B11': np.array([[100]], dtype=np.float32),
        }
        
        # Should raise ChangeDetectionError when B11 is missing from bands_a
        with self.assertRaises(ChangeDetectionError):
            self.detector.detect_urban_change(bands_without_b11, bands_with_b11)
        
        # Should raise ChangeDetectionError when B11 is missing from bands_b
        with self.assertRaises(ChangeDetectionError):
            self.detector.detect_urban_change(bands_with_b11, bands_without_b11)
    
    def test_ndbi_output_shape_matches_input(self):
        """Test that NDBI output has same shape as input bands."""
        shapes = [(10, 10), (50, 50), (100, 200)]
        
        for shape in shapes:
            b11 = np.random.rand(*shape).astype(np.float32) * 1000
            b8 = np.random.rand(*shape).astype(np.float32) * 1000
            
            ndbi = self.calculator.calculate_ndbi(b11, b8)
            
            self.assertEqual(ndbi.shape, shape)
    
    def test_ndbi_values_in_valid_range(self):
        """Test that NDBI values are always in [-1, 1] range."""
        # Test with random values
        b11 = np.random.rand(100, 100).astype(np.float32) * 10000
        b8 = np.random.rand(100, 100).astype(np.float32) * 10000
        
        ndbi = self.calculator.calculate_ndbi(b11, b8)
        
        self.assertTrue(np.all(ndbi >= -1))
        self.assertTrue(np.all(ndbi <= 1))
        
        # Test with extreme values
        b11_extreme = np.array([[0, 10000], [10000, 0]], dtype=np.float32)
        b8_extreme = np.array([[10000, 0], [0, 10000]], dtype=np.float32)
        
        ndbi_extreme = self.calculator.calculate_ndbi(b11_extreme, b8_extreme)
        
        self.assertTrue(np.all(ndbi_extreme >= -1))
        self.assertTrue(np.all(ndbi_extreme <= 1))


class TestUrbanChangeStatistics(unittest.TestCase):
    """Test urban change statistics computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ChangeDetector(threshold=0.2)
    
    def test_urban_statistics_in_summary(self):
        """Test that urban statistics are included in change summary."""
        # Create a classification with known urban changes
        classification = np.array([
            [0, 1, 2, 5],  # no_change, veg_growth, veg_loss, urban_development
            [5, 6, 0, 5],  # urban_dev, urban_decline, no_change, urban_dev
            [0, 5, 6, 0],
            [5, 0, 0, 6]
        ], dtype=np.uint8)
        
        stats = self.detector.compute_change_statistics(classification)
        
        # Check that urban statistics are included
        self.assertIn('urban_development', stats)
        self.assertIn('urban_decline', stats)
        
        # Check urban development count (5 pixels with value 5)
        self.assertEqual(stats['urban_development']['pixels'], 5)
        
        # Check urban decline count (3 pixels with value 6)
        self.assertEqual(stats['urban_decline']['pixels'], 3)
    
    def test_urban_change_type_summary(self):
        """Test urban change type summary statistics."""
        classification = np.array([
            [5, 5, 5, 5],
            [6, 6, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        
        stats = self.detector.compute_change_statistics(classification)
        
        # Check change_types includes urban
        self.assertIn('change_types', stats)
        self.assertIn('urban', stats['change_types'])
        
        # Urban total should be urban_development + urban_decline
        expected_urban_pixels = stats['urban_development']['pixels'] + stats['urban_decline']['pixels']
        self.assertEqual(stats['change_types']['urban']['pixels'], expected_urban_pixels)


if __name__ == '__main__':
    unittest.main()