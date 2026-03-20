#!/usr/bin/env python3
"""
Basic usage example for SatChange.

This example demonstrates how to use SatChange programmatically to:
1. Configure the system
2. Fetch satellite imagery
3. Run change detection analysis
4. Generate visualizations

Usage:
    python basic_usage.py
"""

import os
import sys
import tempfile
import numpy as np

# Add the parent directory to the path to import satchange modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from satchange.config import Config
from satchange.cache import CacheManager
from satchange.change_detector import ChangeDetector, ChangeDetectionError
from satchange.visualization import VisualizationManager, VisualizationError


def create_mock_data():
    """Create mock satellite data for demonstration purposes."""
    print("Creating mock satellite data...")

    # Create mock band arrays (simulating Sentinel-2 bands)
    bands_a = {
        "B4": np.random.rand(100, 100) * 0.8 + 0.1,  # Red band
        "B3": np.random.rand(100, 100) * 0.8 + 0.1,  # Green band
        "B8": np.random.rand(100, 100) * 0.8 + 0.1,  # NIR band
        "QA60": np.random.randint(0, 1000, (100, 100)),  # Quality band
    }

    bands_b = {
        "B4": np.random.rand(100, 100) * 0.8 + 0.1,
        "B3": np.random.rand(100, 100) * 0.8 + 0.1,
        "B8": np.random.rand(100, 100) * 0.8 + 0.1,
        "QA60": np.random.randint(0, 1000, (100, 100)),
    }

    # Create mock metadata
    metadata_a = {
        "date": "2020-01-15",
        "cloud_coverage": 15.0,
        "transform": (10.0, 0.0, 80.2707, 0.0, -10.0, 13.0827),
        "crs": "EPSG:4326",
    }

    metadata_b = {
        "date": "2024-01-15",
        "cloud_coverage": 12.0,
        "transform": (10.0, 0.0, 80.2707, 0.0, -10.0, 13.0827),
        "crs": "EPSG:4326",
    }

    return bands_a, bands_b, metadata_a, metadata_b


def setup_configuration():
    """Set up configuration for SatChange."""
    print("Setting up configuration...")

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
service_account_key: "/path/to/mock-service-account.json"
project_id: "mock-project-id"

cache_settings:
  max_size_gb: 1.0
  eviction_policy: "lru"

analysis_parameters:
  default_cloud_threshold: 20
  default_pixel_size: 100
  change_threshold: 0.2
  emboss_intensity: 1.0
  min_temporal_gap_days: 180
"""
        f.write(config_content)
        config_file = f.name

    try:
        # Initialize configuration
        config = Config(config_file)

        # Override service account key with mock data
        config.set("service_account_key", "/path/to/mock-service-account.json")
        config.set("project_id", "mock-project-id")

        return config, config_file

    except Exception as e:
        print(f"Configuration setup error: {e}")
        if os.path.exists(config_file):
            os.unlink(config_file)
        raise


def demonstrate_configuration():
    """Demonstrate configuration management."""
    print("\n" + "=" * 50)
    print("Configuration Management Demo")
    print("=" * 50)

    config, config_file = setup_configuration()

    try:
        # Get configuration values
        cloud_threshold = config.get("analysis_parameters.default_cloud_threshold")
        cache_size = config.get("cache_settings.max_size_gb")

        print(f"Cloud threshold: {cloud_threshold}")
        print(f"Cache size: {cache_size} GB")

        # Set new values
        config.set("analysis_parameters.change_threshold", 0.25)
        config.set("cache_settings.max_size_gb", 2.0)

        # Get updated values
        new_threshold = config.get("analysis_parameters.change_threshold")
        new_cache_size = config.get("cache_settings.max_size_gb")

        print(f"New change threshold: {new_threshold}")
        print(f"New cache size: {new_cache_size} GB")

        # Validate configuration
        config.validate()
        print("Configuration validation passed!")

    finally:
        # Clean up
        if os.path.exists(config_file):
            os.unlink(config_file)


def demonstrate_cache():
    """Demonstrate caching functionality."""
    print("\n" + "=" * 50)
    print("Caching System Demo")
    print("=" * 50)

    config, config_file = setup_configuration()

    try:
        # Initialize cache manager
        cache_manager = CacheManager(config)

        # Store mock data
        mock_data = {"test": "data", "numbers": [1, 2, 3]}
        cache_manager.set("test_key", mock_data)

        # Retrieve data
        retrieved_data = cache_manager.get("test_key")
        print(f"Retrieved data: {retrieved_data}")

        # Get cache statistics
        stats = cache_manager.get_cache_stats()
        print(f"Cache stats: {stats}")

        # Clear cache
        cache_manager.clear_cache()
        print("Cache cleared!")

    finally:
        # Clean up
        if os.path.exists(config_file):
            os.unlink(config_file)


def demonstrate_change_detection():
    """Demonstrate change detection functionality."""
    print("\n" + "=" * 50)
    print("Change Detection Demo")
    print("=" * 50)

    # Create mock data
    bands_a, bands_b, metadata_a, metadata_b = create_mock_data()

    try:
        # Initialize change detector
        detector = ChangeDetector(threshold=0.2)

        # Detect vegetation changes
        print("Detecting vegetation changes...")
        vegetation_changes = detector.detect_vegetation_change(bands_a, bands_b)

        print(
            f"Vegetation change mask shape: {vegetation_changes['change_mask'].shape}"
        )
        print(f"Vegetation change pixels: {np.sum(vegetation_changes['change_mask'])}")

        # Detect water changes
        print("Detecting water changes...")
        water_changes = detector.detect_water_change(bands_a, bands_b)

        print(f"Water change mask shape: {water_changes['change_mask'].shape}")
        print(f"Water change pixels: {np.sum(water_changes['change_mask'])}")

        # Detect all changes
        print("Detecting all changes...")
        all_changes = detector.detect_all_changes(bands_a, bands_b)

        print(f"Combined change mask shape: {all_changes['combined_mask'].shape}")
        print(f"Total change pixels: {np.sum(all_changes['combined_mask'])}")

        # Classify changes
        classification = detector.classify_changes(all_changes)
        print(f"Classification shape: {classification.shape}")
        print(f"Classification unique values: {np.unique(classification)}")

        # Get change statistics
        stats = detector.compute_change_statistics(classification)
        print(f"Change statistics: {stats}")

    except ChangeDetectionError as e:
        print(f"Change detection error: {e}")


def demonstrate_visualization():
    """Demonstrate visualization functionality."""
    print("\n" + "=" * 50)
    print("Visualization Demo")
    print("=" * 50)

    # Create mock data
    bands_a, bands_b, metadata_a, metadata_b = create_mock_data()

    try:
        # Initialize change detector and get results
        detector = ChangeDetector(threshold=0.2)
        all_changes = detector.detect_all_changes(bands_a, bands_b)
        classification = detector.classify_changes(all_changes)
        detector.compute_change_statistics(classification)

        # Initialize visualization manager
        viz_manager = VisualizationManager(emboss_intensity=1.0)

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")

            # Generate mock outputs (without actual image processing)
            print("Generating mock visualization outputs...")

            # Note: In a real scenario, these would generate actual files
            # For this demo, we'll just show what would be generated

            output_files = {
                "static_png": os.path.join(temp_dir, "visualization.png"),
                "interactive_html": os.path.join(temp_dir, "visualization.html"),
                "geotiff": os.path.join(temp_dir, "classification.tif"),
                "stats_json": os.path.join(temp_dir, "statistics.json"),
            }

            print("Would generate the following files:")
            for file_type, file_path in output_files.items():
                print(f"  {file_type}: {file_path}")

            # Apply emboss effect (mock)
            change_mask = classification > 0
            embossed = viz_manager.apply_emboss_effect(change_mask, intensity=1.0)
            print(f"Embossed mask shape: {embossed.shape}")
            print(f"Embossed mask range: {embossed.min()} - {embossed.max()}")

    except VisualizationError as e:
        print(f"Visualization error: {e}")


def demonstrate_error_handling():
    """Demonstrate error handling."""
    print("\n" + "=" * 50)
    print("Error Handling Demo")
    print("=" * 50)

    try:
        # Try to create a configuration with invalid values
        config, config_file = setup_configuration()

        # Set invalid cloud threshold
        config.set("analysis_parameters.default_cloud_threshold", 200)  # Too high

        # This should raise a ConfigError
        config.validate()

    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")

    try:
        # Try to create change detector with invalid threshold
        ChangeDetector(threshold=1.5)  # Invalid threshold

    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")


def main():
    """Main function to run all demonstrations."""
    print("SatChange Basic Usage Example")
    print("=" * 50)
    print("This example demonstrates the core functionality of SatChange.")
    print(
        "Note: This uses mock data since GEE authentication requires real credentials."
    )
    print()

    try:
        # Run demonstrations
        demonstrate_configuration()
        demonstrate_cache()
        demonstrate_change_detection()
        demonstrate_visualization()
        demonstrate_error_handling()

        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
