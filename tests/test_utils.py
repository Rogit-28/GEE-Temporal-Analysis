"""
Tests for utility functions.
"""

import pytest
from datetime import datetime, timedelta

from satchange.utils import (
    parse_coordinates,
    parse_date,
    validate_pixel_size,
    validate_cloud_threshold,
    format_file_size,
    validate_threshold,
    setup_logging,
)


class TestCoordinateParsing:
    """Test coordinate parsing functions."""
    
    def test_parse_coordinates_valid(self):
        """Test parsing valid coordinate strings."""
        # Test comma-separated format
        lat, lon = parse_coordinates("13.0827,80.2707")
        assert lat == 13.0827
        assert lon == 80.2707
        
        # Test space-separated format
        lat, lon = parse_coordinates("13.0827 80.2707")
        assert lat == 13.0827
        assert lon == 80.2707
        
        # Test with extra whitespace
        lat, lon = parse_coordinates("  13.0827  ,  80.2707  ")
        assert lat == 13.0827
        assert lon == 80.2707
    
    def test_parse_coordinates_invalid_format(self):
        """Test parsing invalid coordinate formats."""
        with pytest.raises(ValueError, match="Coordinates must be in format"):
            parse_coordinates("13.0827")
        
        with pytest.raises(ValueError, match="Coordinates must be in format"):
            parse_coordinates("13.0827,80.2707,extra")
        
        with pytest.raises(ValueError, match="Coordinates must be in format"):
            parse_coordinates("invalid")
    
    def test_parse_coordinates_invalid_ranges(self):
        """Test parsing coordinates with invalid ranges."""
        # Invalid latitude
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            parse_coordinates("91.0,80.2707")
        
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            parse_coordinates("-91.0,80.2707")
        
        # Invalid longitude
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            parse_coordinates("13.0827,181.0")
        
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            parse_coordinates("13.0827,-181.0")


class TestDateParsing:
    """Test date parsing functions."""
    
    def test_parse_date_range(self):
        """Test parsing date range strings."""
        start_date, end_date = parse_date("2020-01-01:2020-12-31")
        assert start_date == datetime(2020, 1, 1)
        assert end_date == datetime(2020, 12, 31)
    
    def test_parse_single_date(self):
        """Test parsing single date strings."""
        start_date, end_date = parse_date("2020-01-01")
        assert start_date == datetime(2020, 1, 1)
        assert end_date == datetime(2020, 1, 1)
    
    def test_parse_date_with_whitespace(self):
        """Test parsing date strings with whitespace."""
        start_date, end_date = parse_date("  2020-01-01  :  2020-12-31  ")
        assert start_date == datetime(2020, 1, 1)
        assert end_date == datetime(2020, 12, 31)
    
    def test_parse_date_invalid_format(self):
        """Test parsing invalid date formats."""
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("2020/01/01:2020/12/31")
        
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("invalid-date")
    
    def test_parse_date_invalid_range(self):
        """Test parsing date with invalid range."""
        with pytest.raises(ValueError, match="Start date .* cannot be after end date"):
            parse_date("2020-12-31:2020-01-01")
    
    def test_parse_date_too_far_past(self):
        """Test parsing date that's too far in the past."""
        # Create a date more than 10 years ago
        past_date = datetime.now() - timedelta(days=365 * 11)
        past_date_str = past_date.strftime("%Y-%m-%d")
        
        with pytest.raises(ValueError, match="Start date too far in the past"):
            parse_date(f"{past_date_str}:2020-12-31")


class TestValidationFunctions:
    """Test validation functions."""
    
    def test_validate_pixel_size(self):
        """Test pixel size validation."""
        # Valid pixel sizes
        validate_pixel_size(100)
        validate_pixel_size(10)
        validate_pixel_size(1000)
        
        # Invalid pixel sizes
        with pytest.raises(ValueError, match="Pixel size must be a positive integer"):
            validate_pixel_size(0)
        
        with pytest.raises(ValueError, match="Pixel size must be a positive integer"):
            validate_pixel_size(-10)
        
        with pytest.raises(ValueError, match="Pixel size must be a positive integer"):
            validate_pixel_size("100")
        
        with pytest.raises(ValueError, match="Pixel size too large"):
            validate_pixel_size(1001)
    
    def test_validate_cloud_threshold(self):
        """Test cloud threshold validation."""
        # Valid thresholds
        validate_cloud_threshold(0)
        validate_cloud_threshold(20)
        validate_cloud_threshold(100)
        
        # Invalid thresholds
        with pytest.raises(ValueError, match="Cloud threshold must be between 0 and 100"):
            validate_cloud_threshold(-1)
        
        with pytest.raises(ValueError, match="Cloud threshold must be between 0 and 100"):
            validate_cloud_threshold(101)
        
        with pytest.raises(ValueError, match="Cloud threshold must be between 0 and 100"):
            validate_cloud_threshold("20")
    
    def test_validate_threshold(self):
        """Test general threshold validation."""
        # Valid thresholds
        validate_threshold(0.2, 0.0, 1.0)
        validate_threshold(0.0, 0.0, 1.0)
        validate_threshold(1.0, 0.0, 1.0)
        
        # Invalid thresholds
        with pytest.raises(ValueError, match="Threshold must be a number"):
            validate_threshold("0.2", 0.0, 1.0)
        
        with pytest.raises(ValueError, match="Threshold must be between"):
            validate_threshold(-0.1, 0.0, 1.0)
        
        with pytest.raises(ValueError, match="Threshold must be between"):
            validate_threshold(1.1, 0.0, 1.0)
    
class TestFormattingFunctions:
    """Test formatting functions."""
    
    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(0) == "0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"


class TestLogging:
    """Test logging setup."""
    
    def test_setup_logging(self):
        """Test logging configuration."""
        # This is mainly to ensure the function doesn't crash
        setup_logging(verbose=False)
        setup_logging(verbose=True)
        
        # Test that root logger is configured (not satchange logger)
        import logging
        root_logger = logging.getLogger()
        assert root_logger.level in (logging.INFO, logging.DEBUG)


if __name__ == '__main__':
    pytest.main([__file__])