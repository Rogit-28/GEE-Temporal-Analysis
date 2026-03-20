#!/usr/bin/env python3
"""
Advanced analysis example for SatChange.

This example demonstrates advanced usage patterns including:
1. Batch processing multiple locations
2. Custom parameter tuning
3. Integration with external data sources
4. Performance optimization techniques

Usage:
    python advanced_analysis.py
"""

import os
import sys
import json
import time
import tempfile
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add the parent directory to the path to import satchange modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from satchange.config import Config
from satchange.gee_client import GEEClient
from satchange.cache import CacheManager
from satchange.image_processor import ImageProcessor
from satchange.change_detector import ChangeDetector
from satchange.visualization import VisualizationManager


class SatChangeAdvancedAnalyzer:
    """Advanced analyzer class for batch processing and custom analysis."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the advanced analyzer.

        Args:
            config_file: Path to configuration file
        """
        self.config = Config(config_file)
        self.gee_client = GEEClient(self.config)
        self.cache_manager = CacheManager(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.change_detector = ChangeDetector(
            threshold=self.config.get("analysis_parameters.change_threshold", 0.2)
        )
        self.visualization_manager = VisualizationManager(
            emboss_intensity=self.config.get(
                "analysis_parameters.emboss_intensity", 1.0
            )
        )

        # Performance tracking
        self.performance_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def analyze_location(self, location: Dict, analysis_params: Dict) -> Dict:
        """
        Analyze a single location with custom parameters.

        Args:
            location: Location dictionary with 'name', 'center', 'size'
            analysis_params: Analysis parameters

        Returns:
            Analysis results dictionary
        """
        start_time = time.time()
        self.performance_stats["total_operations"] += 1

        try:
            print(f"Analyzing {location['name']}...")

            # Extract parameters
            size = location.get("size", 100)
            start_date = analysis_params.get("start_date", "2020-01-01")
            end_date = analysis_params.get("end_date", "2024-12-31")
            change_type = analysis_params.get("change_type", "all")
            threshold = analysis_params.get("threshold", 0.2)

            # Update change detector threshold
            self.change_detector.threshold = threshold

            # Create cache key
            cache_key = f"{location['name']}_{start_date}_{end_date}_{threshold}"

            # Check cache first
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                print(f"Cache hit for {location['name']}")
                return cached_result

            self.performance_stats["cache_misses"] += 1

            # Create mock data for demonstration
            bands_a, bands_b, metadata_a, metadata_b = self._create_mock_data(size)

            # Preprocess images
            processed_a, processed_b = self.image_processor.preprocess_image_pair(
                bands_a, bands_b, metadata_a, metadata_b
            )

            # Detect changes
            change_summary = self.change_detector.detect_changes(
                processed_a, processed_b, change_type
            )

            # Generate statistics
            stats = self.change_detector.compute_change_statistics(
                change_summary["classification"]
            )

            # Create result
            result = {
                "location": location,
                "analysis_params": analysis_params,
                "change_summary": change_summary,
                "statistics": stats,
                "metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "processing_time": time.time() - start_time,
                    "cache_key": cache_key,
                },
            }

            # Cache result
            self.cache_manager.set(cache_key, result)

            self.performance_stats["successful_operations"] += 1
            print(
                f"Successfully analyzed {location['name']} in {result['metadata']['processing_time']:.2f}s"
            )

            return result

        except Exception as e:
            self.performance_stats["failed_operations"] += 1
            print(f"Failed to analyze {location['name']}: {e}")
            return {
                "location": location,
                "error": str(e),
                "metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "processing_time": time.time() - start_time,
                },
            }

    def batch_analyze(self, locations: List[Dict], analysis_params: Dict) -> List[Dict]:
        """
        Analyze multiple locations in batch.

        Args:
            locations: List of location dictionaries
            analysis_params: Analysis parameters

        Returns:
            List of analysis results
        """
        print(f"Starting batch analysis of {len(locations)} locations...")

        results = []
        for i, location in enumerate(locations):
            print(f"\nProcessing location {i + 1}/{len(locations)}: {location['name']}")

            result = self.analyze_location(location, analysis_params)
            results.append(result)

            # Progress update
            if (i + 1) % 5 == 0:
                progress = (i + 1) / len(locations) * 100
                print(f"Progress: {progress:.1f}%")

        return results

    def compare_analysis_parameters(
        self, location: Dict, param_sets: List[Dict]
    ) -> Dict:
        """
        Compare different analysis parameters for the same location.

        Args:
            location: Location dictionary
            param_sets: List of parameter dictionaries

        Returns:
            Comparison results
        """
        print(f"Comparing {len(param_sets)} parameter sets for {location['name']}...")

        comparison_results = {
            "location": location,
            "param_sets": param_sets,
            "results": [],
        }

        for i, params in enumerate(param_sets):
            print(f"Testing parameter set {i + 1}/{len(param_sets)}")

            result = self.analyze_location(location, params)
            comparison_results["results"].append(
                {"param_set": params, "result": result}
            )

        # Generate comparison summary
        comparison_results["summary"] = self._generate_comparison_summary(
            comparison_results["results"]
        )

        return comparison_results

    def _create_mock_data(self, size: int) -> Tuple[Dict, Dict, Dict, Dict]:
        """Create mock satellite data for demonstration."""
        # Create mock band arrays
        bands_a = {
            "B4": np.random.rand(size, size) * 0.8 + 0.1,
            "B3": np.random.rand(size, size) * 0.8 + 0.1,
            "B8": np.random.rand(size, size) * 0.8 + 0.1,
            "QA60": np.random.randint(0, 1000, (size, size)),
        }

        bands_b = {
            "B4": np.random.rand(size, size) * 0.8 + 0.1,
            "B3": np.random.rand(size, size) * 0.8 + 0.1,
            "B8": np.random.rand(size, size) * 0.8 + 0.1,
            "QA60": np.random.randint(0, 1000, (size, size)),
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

    def _generate_comparison_summary(self, results: List[Dict]) -> Dict:
        """Generate comparison summary from parameter test results."""
        summary = {
            "total_tests": len(results),
            "successful_tests": len([r for r in results if "error" not in r["result"]]),
            "failed_tests": len([r for r in results if "error" in r["result"]]),
            "best_performance": None,
            "most_sensitive": None,
            "most_conservative": None,
        }

        # Find best performing parameter set
        successful_results = [r for r in results if "error" not in r["result"]]
        if successful_results:
            processing_times = [
                r["result"]["metadata"]["processing_time"] for r in successful_results
            ]
            summary["best_performance"] = successful_results[
                np.argmin(processing_times)
            ]["param_set"]

            # Find most sensitive (highest change detection)
            total_changes = [
                r["result"]["statistics"]["total_change"]["percent"]
                for r in successful_results
            ]
            summary["most_sensitive"] = successful_results[np.argmax(total_changes)][
                "param_set"
            ]

            # Find most conservative (lowest change detection)
            summary["most_conservative"] = successful_results[np.argmin(total_changes)][
                "param_set"
            ]

        return summary

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self.performance_stats.copy()

    def clear_cache(self):
        """Clear the cache."""
        self.cache_manager.clear_cache()
        print("Cache cleared")


def create_sample_locations() -> List[Dict]:
    """Create sample locations for analysis."""
    return [
        {
            "name": "chennai",
            "center": "13.0827,80.2707",
            "description": "Chennai metropolitan area",
        },
        {
            "name": "mumbai",
            "center": "19.0760,72.8777",
            "description": "Mumbai metropolitan area",
        },
        {
            "name": "delhi",
            "center": "28.7041,77.1025",
            "description": "Delhi metropolitan area",
        },
        {
            "name": "bangalore",
            "center": "12.9716,77.5946",
            "description": "Bangalore metropolitan area",
        },
        {
            "name": "kolkata",
            "center": "22.5726,88.3639",
            "description": "Kolkata metropolitan area",
        },
    ]


def create_parameter_sets() -> List[Dict]:
    """Create different parameter sets for comparison."""
    return [
        {
            "name": "conservative",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "change_type": "all",
            "threshold": 0.3,  # Higher threshold = less sensitive
        },
        {
            "name": "moderate",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "change_type": "all",
            "threshold": 0.2,  # Moderate threshold
        },
        {
            "name": "sensitive",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "change_type": "all",
            "threshold": 0.1,  # Lower threshold = more sensitive
        },
        {
            "name": "vegetation_only",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "change_type": "vegetation",
            "threshold": 0.2,
        },
        {
            "name": "water_only",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "change_type": "water",
            "threshold": 0.2,
        },
    ]


def save_results(results: List[Dict], output_file: str):
    """Save analysis results to JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_file}")


def main():
    """Main function to run advanced analysis examples."""
    print("SatChange Advanced Analysis Example")
    print("=" * 50)
    print("This example demonstrates advanced usage patterns including:")
    print("- Batch processing multiple locations")
    print("- Parameter comparison")
    print("- Performance optimization")
    print("- Caching strategies")
    print()

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
service_account_key: "/path/to/mock-service-account.json"
project_id: "mock-project-id"

cache_settings:
  max_size_gb: 2.0
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
        # Initialize advanced analyzer
        analyzer = SatChangeAdvancedAnalyzer(config_file)

        # Create sample locations
        locations = create_sample_locations()

        # Create parameter sets
        param_sets = create_parameter_sets()

        # Example 1: Batch analysis
        print("\n" + "=" * 50)
        print("Example 1: Batch Analysis")
        print("=" * 50)

        batch_params = {
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "change_type": "all",
            "threshold": 0.2,
        }

        batch_results = analyzer.batch_analyze(locations[:3], batch_params)
        save_results(batch_results, "batch_analysis_results.json")

        # Example 2: Parameter comparison
        print("\n" + "=" * 50)
        print("Example 2: Parameter Comparison")
        print("=" * 50)

        comparison_location = locations[0]  # Use first location
        comparison_results = analyzer.compare_analysis_parameters(
            comparison_location, param_sets
        )
        save_results([comparison_results], "parameter_comparison_results.json")

        # Example 3: Performance optimization
        print("\n" + "=" * 50)
        print("Example 3: Performance Optimization")
        print("=" * 50)

        # Clear cache and run again to demonstrate caching
        print("Clearing cache and running analysis again...")
        analyzer.clear_cache()

        # Run the same analysis again to show caching benefits
        analyzer.batch_analyze(locations[:2], batch_params)

        # Get performance statistics
        stats = analyzer.get_performance_stats()
        print("\nPerformance Statistics:")
        print(f"  Total operations: {stats['total_operations']}")
        print(f"  Successful operations: {stats['successful_operations']}")
        print(f"  Failed operations: {stats['failed_operations']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(
            f"  Success rate: {stats['successful_operations'] / stats['total_operations'] * 100:.1f}%"
        )

        # Example 4: Custom analysis workflow
        print("\n" + "=" * 50)
        print("Example 4: Custom Analysis Workflow")
        print("=" * 50)

        # Create custom workflow for urban analysis
        urban_params = {
            "start_date": "2018-01-01",
            "end_date": "2023-12-31",
            "change_type": "all",
            "threshold": 0.25,
            "focus": "urban_development",
        }

        urban_results = analyzer.analyze_location(locations[1], urban_params)
        save_results([urban_results], "urban_analysis_results.json")

        print("\n" + "=" * 50)
        print("All advanced analysis examples completed!")
        print("=" * 50)

    except Exception as e:
        print(f"Error during advanced analysis: {e}")
        return 1

    finally:
        # Clean up
        if os.path.exists(config_file):
            os.unlink(config_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
