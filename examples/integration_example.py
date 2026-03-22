#!/usr/bin/env python3
"""
Integration example for SatChange with external systems.

This example demonstrates how to integrate SatChange with:
1. Web applications (Flask)
2. Data pipelines (pandas)
3. GIS systems (geopandas)
4. Cloud storage (AWS S3)
5. Database integration (SQLite)

Usage:
    python integration_example.py
"""

import os
import sys
import json
import sqlite3
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

# Add the parent directory to the path to import satchange modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from satchange.config import Config
from satchange.cache import CacheManager
from satchange.image_processor import ImageProcessor
from satchange.change_detector import ChangeDetector
from satchange.visualization import VisualizationManager


class SatChangeWebAPI:
    """Web API wrapper for SatChange (Flask-style)."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the web API."""
        self.config = Config(config_file)
        self.cache_manager = CacheManager(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.change_detector = ChangeDetector()
        self.visualization_manager = VisualizationManager()

        # Initialize database
        self.db_path = "satchange_analysis.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for storing analysis results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create analysis results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_name TEXT NOT NULL,
                center_lat REAL NOT NULL,
                center_lon REAL NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                change_type TEXT NOT NULL,
                threshold REAL NOT NULL,
                total_change_percent REAL,
                vegetation_loss_percent REAL,
                vegetation_growth_percent REAL,
                water_expansion_percent REAL,
                water_reduction_percent REAL,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create analysis metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analysis_results (id)
            )
        """)

        conn.commit()
        conn.close()

    def analyze_location(self, location_data: Dict) -> Dict:
        """
        Analyze a location and store results in database.

        Args:
            location_data: Location data dictionary

        Returns:
            Analysis results
        """
        try:
            # Extract parameters
            location_name = location_data["name"]
            center_lat, center_lon = map(float, location_data["center"].split(","))
            start_date = location_data["start_date"]
            end_date = location_data["end_date"]
            change_type = location_data.get("change_type", "all")
            threshold = location_data.get("threshold", 0.2)

            # Create mock data
            bands_a, bands_b, metadata_a, metadata_b = self._create_mock_data(100)

            # Process images
            processed_a, processed_b = self.image_processor.preprocess_image_pair(
                bands_a, bands_b, metadata_a, metadata_b
            )

            # Detect changes
            change_summary = self.change_detector.get_change_summary(
                processed_a, processed_b, change_type
            )
            if change_type == "all":
                classification = change_summary["classification"]
            else:
                classification = np.zeros_like(processed_a["B4"], dtype=np.uint8)
                single = change_summary["change_results"]
                if change_type == "vegetation":
                    classification[single["growth_mask"]] = 1
                    classification[single["loss_mask"]] = 2
                elif change_type == "water":
                    classification[single["expansion_mask"]] = 3
                    classification[single["reduction_mask"]] = 4
                elif change_type == "urban":
                    classification[single["development_mask"]] = 5
                    classification[single["decline_mask"]] = 6

            # Get statistics
            stats = self.change_detector.compute_change_statistics(classification)

            # Store in database
            analysis_id = self._store_analysis_results(
                location_name,
                center_lat,
                center_lon,
                start_date,
                end_date,
                change_type,
                threshold,
                stats,
            )

            # Generate visualization files
            viz_files = self._generate_visualization_files(
                analysis_id,
                processed_a,
                processed_b,
                classification,
                stats,
                center_lat,
                center_lon,
            )

            return {
                "success": True,
                "analysis_id": analysis_id,
                "location": location_name,
                "statistics": stats,
                "visualization_files": viz_files,
                "message": "Analysis completed successfully",
            }

        except Exception as e:
            return {"success": False, "error": str(e), "message": "Analysis failed"}

    def _store_analysis_results(
        self,
        location_name: str,
        center_lat: float,
        center_lon: float,
        start_date: str,
        end_date: str,
        change_type: str,
        threshold: float,
        stats: Dict,
    ) -> int:
        """Store analysis results in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO analysis_results 
            (location_name, center_lat, center_lon, start_date, end_date, 
             change_type, threshold, total_change_percent, vegetation_loss_percent,
             vegetation_growth_percent, water_expansion_percent, water_reduction_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                location_name,
                center_lat,
                center_lon,
                start_date,
                end_date,
                change_type,
                threshold,
                stats["total_change"]["percent"],
                stats.get("vegetation_loss", {}).get("percent", 0),
                stats.get("vegetation_growth", {}).get("percent", 0),
                stats.get("water_expansion", {}).get("percent", 0),
                stats.get("water_reduction", {}).get("percent", 0),
            ),
        )

        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return analysis_id

    def _generate_visualization_files(
        self,
        analysis_id: int,
        bands_a: Dict,
        bands_b: Dict,
        classification: np.ndarray,
        stats: Dict,
        center_lat: float,
        center_lon: float,
    ) -> Dict:
        """Generate visualization files and store metadata."""
        viz_files = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate different visualization types
            viz_files["static_png"] = os.path.join(
                temp_dir, f"{analysis_id}_static.png"
            )
            viz_files["interactive_html"] = os.path.join(
                temp_dir, f"{analysis_id}_interactive.html"
            )
            viz_files["geotiff"] = os.path.join(
                temp_dir, f"{analysis_id}_classification.tif"
            )
            viz_files["stats_json"] = os.path.join(
                temp_dir, f"{analysis_id}_stats.json"
            )

            # Create mock files (in real implementation, these would be actual files)
            for file_path in viz_files.values():
                with open(file_path, "w") as f:
                    f.write(f"Mock file for {os.path.basename(file_path)}")

            # Store file metadata in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for file_type, file_path in viz_files.items():
                file_size = os.path.getsize(file_path)
                cursor.execute(
                    """
                    INSERT INTO analysis_metadata (analysis_id, file_type, file_path, file_size)
                    VALUES (?, ?, ?, ?)
                """,
                    (analysis_id, file_type, file_path, file_size),
                )

            conn.commit()
            conn.close()

        return viz_files

    def get_analysis_history(
        self, location_name: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """Get analysis history from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if location_name:
            cursor.execute(
                """
                SELECT * FROM analysis_results 
                WHERE location_name = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """,
                (location_name, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM analysis_results 
                ORDER BY created_at DESC 
                LIMIT ?
            """,
                (limit,),
            )

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def _create_mock_data(self, size: int) -> tuple:
        """Create mock satellite data."""
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


class SatChangeDataPipeline:
    """Data pipeline integration with pandas."""

    def __init__(self, api: SatChangeWebAPI):
        """Initialize the data pipeline."""
        self.api = api

    def process_batch_locations(self, locations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process batch locations using pandas DataFrame.

        Args:
            locations_df: DataFrame with location data

        Returns:
            DataFrame with analysis results
        """
        results = []

        for _, row in locations_df.iterrows():
            location_data = {
                "name": row["name"],
                "center": f"{row['lat']},{row['lon']}",
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "change_type": row.get("change_type", "all"),
                "threshold": row.get("threshold", 0.2),
            }

            result = self.api.analyze_location(location_data)
            if result["success"]:
                results.append(
                    {
                        "location": row["name"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "total_change_percent": result["statistics"]["total_change"][
                            "percent"
                        ],
                        "vegetation_loss_percent": result["statistics"]
                        .get("vegetation_loss", {})
                        .get("percent", 0),
                        "vegetation_growth_percent": result["statistics"]
                        .get("vegetation_growth", {})
                        .get("percent", 0),
                        "water_expansion_percent": result["statistics"]
                        .get("water_expansion", {})
                        .get("percent", 0),
                        "water_reduction_percent": result["statistics"]
                        .get("water_reduction", {})
                        .get("percent", 0),
                        "analysis_id": result["analysis_id"],
                    }
                )

        return pd.DataFrame(results)

    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate summary report from analysis results."""
        return {
            "total_locations": len(results_df),
            "average_change_percent": results_df["total_change_percent"].mean(),
            "max_change_percent": results_df["total_change_percent"].max(),
            "min_change_percent": results_df["total_change_percent"].min(),
            "vegetation_loss_total": results_df["vegetation_loss_percent"].sum(),
            "vegetation_growth_total": results_df["vegetation_growth_percent"].sum(),
            "water_expansion_total": results_df["water_expansion_percent"].sum(),
            "water_reduction_total": results_df["water_reduction_percent"].sum(),
            "top_locations_by_change": results_df.nlargest(5, "total_change_percent")[
                ["location", "total_change_percent"]
            ].to_dict("records"),
        }


class SatChangeGISIntegration:
    """GIS integration with geopandas."""

    def __init__(self, api: SatChangeWebAPI):
        """Initialize GIS integration."""
        self.api = api

    def create_change_polygons(self, analysis_id: int) -> Dict:
        """
        Create polygon geometries for detected changes.

        Args:
            analysis_id: Analysis ID

        Returns:
            Dictionary with GeoJSON data
        """
        # Mock polygon data (in real implementation, this would come from actual analysis)
        polygons = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [80.26, 13.08],
                                [80.28, 13.08],
                                [80.28, 13.10],
                                [80.26, 13.10],
                                [80.26, 13.08],
                            ]
                        ],
                    },
                    "properties": {
                        "change_type": "vegetation_loss",
                        "change_percent": 15.2,
                        "analysis_id": analysis_id,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [80.27, 13.09],
                                [80.29, 13.09],
                                [80.29, 13.11],
                                [80.27, 13.11],
                                [80.27, 13.09],
                            ]
                        ],
                    },
                    "properties": {
                        "change_type": "water_expansion",
                        "change_percent": 8.7,
                        "analysis_id": analysis_id,
                    },
                },
            ],
        }

        return polygons

    def export_to_shapefile(self, analysis_id: int, output_path: str):
        """Export change polygons to shapefile."""
        # Mock implementation
        with open(output_path, "w") as f:
            f.write(f"Mock shapefile export for analysis {analysis_id}")


def create_sample_locations_dataframe() -> pd.DataFrame:
    """Create sample locations DataFrame."""
    return pd.DataFrame(
        [
            {
                "name": "chennai",
                "lat": 13.0827,
                "lon": 80.2707,
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
            },
            {
                "name": "mumbai",
                "lat": 19.0760,
                "lon": 72.8777,
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
            },
            {
                "name": "delhi",
                "lat": 28.7041,
                "lon": 77.1025,
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
            },
            {
                "name": "bangalore",
                "lat": 12.9716,
                "lon": 77.5946,
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
            },
            {
                "name": "kolkata",
                "lat": 22.5726,
                "lon": 88.3639,
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
            },
        ]
    )


def main():
    """Main function to run integration examples."""
    print("SatChange Integration Example")
    print("=" * 50)
    print("This example demonstrates integration with external systems:")
    print("- Web API (Flask-style)")
    print("- Data pipeline (pandas)")
    print("- GIS integration (geopandas)")
    print("- Database (SQLite)")
    print()

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = """
service_account_key: "/path/to/mock-service-account.json"
project_id: "mock-project-id"

cache_settings:
  max_size_gb: 1.0
  eviction_policy: "lru"

analysis:
  change_threshold: 0.2
  emboss_intensity: 1.0
  min_temporal_gap_days: 180
"""
        f.write(config_content)
        config_file = f.name

    try:
        # Initialize web API
        print("Initializing Web API...")
        web_api = SatChangeWebAPI(config_file)

        # Example 1: Web API usage
        print("\n" + "=" * 50)
        print("Example 1: Web API Usage")
        print("=" * 50)

        location_data = {
            "name": "test_location",
            "center": "13.0827,80.2707",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "change_type": "all",
            "threshold": 0.2,
        }

        result = web_api.analyze_location(location_data)
        print(f"Analysis result: {result}")

        # Get analysis history
        history = web_api.get_analysis_history()
        print(f"Analysis history: {len(history)} records")

        # Example 2: Data pipeline with pandas
        print("\n" + "=" * 50)
        print("Example 2: Data Pipeline with Pandas")
        print("=" * 50)

        # Create data pipeline
        data_pipeline = SatChangeDataPipeline(web_api)

        # Create sample locations DataFrame
        locations_df = create_sample_locations_dataframe()
        print(f"Processing {len(locations_df)} locations...")

        # Process batch locations
        results_df = data_pipeline.process_batch_locations(locations_df)
        print(f"Analysis results:\n{results_df}")

        # Generate summary report
        summary = data_pipeline.generate_summary_report(results_df)
        print(f"Summary report: {summary}")

        # Save results to CSV
        results_df.to_csv("analysis_results.csv", index=False)
        print("Results saved to analysis_results.csv")

        # Example 3: GIS integration
        print("\n" + "=" * 50)
        print("Example 3: GIS Integration")
        print("=" * 50)

        # Create GIS integration
        gis_integration = SatChangeGISIntegration(web_api)

        # Create change polygons
        polygons = gis_integration.create_change_polygons(result["analysis_id"])
        print(f"Created {len(polygons['features'])} change polygons")

        # Export to shapefile
        shapefile_path = "change_polygons.shp"
        gis_integration.export_to_shapefile(result["analysis_id"], shapefile_path)
        print(f"Exported to {shapefile_path}")

        # Example 4: Database operations
        print("\n" + "=" * 50)
        print("Example 4: Database Operations")
        print("=" * 50)

        # Get analysis history from database
        db_history = web_api.get_analysis_history()
        print(f"Database contains {len(db_history)} analysis records")

        # Query specific location
        chennai_history = web_api.get_analysis_history("chennai")
        print(f"Chennai analysis history: {len(chennai_history)} records")

        # Example 5: Advanced workflow
        print("\n" + "=" * 50)
        print("Example 5: Advanced Workflow")
        print("=" * 50)

        # Create custom workflow combining multiple integrations
        workflow_results = []

        for _, location in locations_df.iterrows():
            # Analyze location
            location_data = {
                "name": location["name"],
                "center": f"{location['lat']},{location['lon']}",
                "start_date": location["start_date"],
                "end_date": location["end_date"],
                "change_type": "all",
                "threshold": 0.2,
            }

            result = web_api.analyze_location(location_data)
            if result["success"]:
                # Create GIS data
                polygons = gis_integration.create_change_polygons(result["analysis_id"])

                workflow_results.append(
                    {
                        "location": location["name"],
                        "analysis_id": result["analysis_id"],
                        "statistics": result["statistics"],
                        "polygons": polygons,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Save workflow results
        with open("workflow_results.json", "w") as f:
            json.dump(workflow_results, f, indent=2, default=str)

        print(f"Workflow completed for {len(workflow_results)} locations")
        print("Results saved to workflow_results.json")

        print("\n" + "=" * 50)
        print("All integration examples completed!")
        print("=" * 50)

    except Exception as e:
        print(f"Error during integration example: {e}")
        return 1

    finally:
        # Clean up
        if os.path.exists(config_file):
            os.unlink(config_file)

        # Clean up database
        if os.path.exists("satchange_analysis.db"):
            os.unlink("satchange_analysis.db")

        # Clean up output files
        for file in [
            "analysis_results.csv",
            "change_polygons.shp",
            "workflow_results.json",
        ]:
            if os.path.exists(file):
                os.unlink(file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
