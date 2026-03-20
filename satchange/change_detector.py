"""
Change detection engine for SatChange.

This module implements the core change detection algorithms using spectral indices
to identify temporal changes in satellite imagery.
"""

import numpy as np
import logging
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Enumeration of change types."""

    NO_CHANGE = 0
    VEGETATION_GROWTH = 1
    VEGETATION_LOSS = 2
    WATER_EXPANSION = 3
    WATER_REDUCTION = 4
    URBAN_DEVELOPMENT = 5
    URBAN_DECLINE = 6
    AMBIGUOUS = 7


class ChangeDetectionError(Exception):
    """Exception raised for change detection errors."""

    pass


class SpectralIndexCalculator:
    """Calculate spectral indices for change detection."""

    @staticmethod
    def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            red_band: Red band array (B4 from Sentinel-2)
            nir_band: NIR band array (B8 from Sentinel-2)

        Returns:
            NDVI array in range [-1, 1]
        """
        try:
            # Add small epsilon to avoid division by zero
            numerator = nir_band.astype(float) - red_band.astype(float)
            denominator = nir_band.astype(float) + red_band.astype(float) + 1e-7

            ndvi = numerator / denominator
            # Replace NaN/Inf with 0
            ndvi = np.where(np.isfinite(ndvi), ndvi, 0.0)

            # Clip to valid range
            ndvi = np.clip(ndvi, -1, 1)

            return ndvi

        except Exception as e:
            logger.error(f"Failed to calculate NDVI: {e}")
            raise ChangeDetectionError(f"NDVI calculation failed: {e}")

    @staticmethod
    def calculate_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Water Index.

        NDWI = (Green - NIR) / (Green + NIR)

        Args:
            green_band: Green band array (B3 from Sentinel-2)
            nir_band: NIR band array (B8 from Sentinel-2)

        Returns:
            NDWI array in range [-1, 1]
        """
        try:
            numerator = green_band.astype(float) - nir_band.astype(float)
            denominator = green_band.astype(float) + nir_band.astype(float) + 1e-7

            ndwi = numerator / denominator
            # Replace NaN/Inf with 0
            ndwi = np.where(np.isfinite(ndwi), ndwi, 0.0)
            ndwi = np.clip(ndwi, -1, 1)

            return ndwi

        except Exception as e:
            logger.error(f"Failed to calculate NDWI: {e}")
            raise ChangeDetectionError(f"NDWI calculation failed: {e}")

    @staticmethod
    def calculate_ndbi(swir_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Built-up Index.

        NDBI = (SWIR - NIR) / (SWIR + NIR)

        Args:
            swir_band: SWIR band array (B11 from Sentinel-2, 20m resolution)
            nir_band: NIR band array (B8 from Sentinel-2)

        Returns:
            NDBI array in range [-1, 1]
        """
        try:
            # SWIR band needs to be resampled to 10m to match NIR resolution
            # For now, assume it's already been resampled
            numerator = swir_band.astype(float) - nir_band.astype(float)
            denominator = swir_band.astype(float) + nir_band.astype(float) + 1e-7

            ndbi = numerator / denominator
            # Replace NaN/Inf with 0
            ndbi = np.where(np.isfinite(ndbi), ndbi, 0.0)
            ndbi = np.clip(ndbi, -1, 1)

            return ndbi

        except Exception as e:
            logger.error(f"Failed to calculate NDBI: {e}")
            raise ChangeDetectionError(f"NDBI calculation failed: {e}")


class ChangeDetector:
    """Detect changes between two satellite images using spectral indices."""

    def __init__(self, threshold: float = 0.2):
        """Initialize change detector.

        Args:
            threshold: Minimum absolute difference to consider as significant change
        """
        self.threshold = threshold
        self.index_calculator = SpectralIndexCalculator()

    def detect_vegetation_change(
        self, bands_a: Dict[str, np.ndarray], bands_b: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Detect vegetation changes using NDVI differencing.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B

        Returns:
            Dictionary with change detection results
        """
        try:
            logger.info("Detecting vegetation changes using NDVI")

            # Validate band shapes match
            if bands_a["B4"].shape != bands_b["B4"].shape:
                raise ChangeDetectionError(
                    f"Band shape mismatch: Date A {bands_a['B4'].shape} vs Date B {bands_b['B4'].shape}"
                )

            # Calculate NDVI for both dates
            ndvi_a = self.index_calculator.calculate_ndvi(bands_a["B4"], bands_a["B8"])
            ndvi_b = self.index_calculator.calculate_ndvi(bands_b["B4"], bands_b["B8"])

            # Compute difference
            delta = ndvi_b - ndvi_a

            # Generate masks
            change_mask = np.abs(delta) > self.threshold
            growth_mask = delta > self.threshold  # Positive change = vegetation growth
            loss_mask = delta < -self.threshold  # Negative change = vegetation loss

            # Calculate change magnitude
            magnitude = np.abs(delta)

            return {
                "ndvi_a": ndvi_a,
                "ndvi_b": ndvi_b,
                "delta": delta,
                "change_mask": change_mask,
                "growth_mask": growth_mask,
                "loss_mask": loss_mask,
                "magnitude": magnitude,
                "change_type": "vegetation",
            }

        except Exception as e:
            logger.error(f"Vegetation change detection failed: {e}")
            raise ChangeDetectionError(f"Vegetation change detection failed: {e}")

    def detect_water_change(
        self, bands_a: Dict[str, np.ndarray], bands_b: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Detect water body changes using NDWI differencing.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B

        Returns:
            Dictionary with change detection results
        """
        try:
            logger.info("Detecting water changes using NDWI")

            # Validate band shapes match
            if bands_a["B3"].shape != bands_b["B3"].shape:
                raise ChangeDetectionError(
                    f"Band shape mismatch: Date A {bands_a['B3'].shape} vs Date B {bands_b['B3'].shape}"
                )

            # Calculate NDWI for both dates
            ndwi_a = self.index_calculator.calculate_ndwi(bands_a["B3"], bands_a["B8"])
            ndwi_b = self.index_calculator.calculate_ndwi(bands_b["B3"], bands_b["B8"])

            # Compute difference
            delta = ndwi_b - ndwi_a

            # Generate masks
            change_mask = np.abs(delta) > self.threshold
            expansion_mask = delta > self.threshold  # Water expansion/flooding
            reduction_mask = delta < -self.threshold  # Drought/water loss

            # Calculate change magnitude
            magnitude = np.abs(delta)

            return {
                "ndwi_a": ndwi_a,
                "ndwi_b": ndwi_b,
                "delta": delta,
                "change_mask": change_mask,
                "expansion_mask": expansion_mask,
                "reduction_mask": reduction_mask,
                "magnitude": magnitude,
                "change_type": "water",
            }

        except Exception as e:
            logger.error(f"Water change detection failed: {e}")
            raise ChangeDetectionError(f"Water change detection failed: {e}")

    def detect_urban_change(
        self, bands_a: Dict[str, np.ndarray], bands_b: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Detect urban changes using NDBI differencing.

        NDBI = (SWIR - NIR) / (SWIR + NIR)

        Uses B11 (SWIR) and B8 (NIR) bands from Sentinel-2.

        Args:
            bands_a: Band arrays for Date A (must include 'B11' and 'B8')
            bands_b: Band arrays for Date B (must include 'B11' and 'B8')

        Returns:
            Dictionary with change detection results
        """
        try:
            logger.info("Detecting urban changes using NDBI")

            # Validate band shapes match
            if bands_a["B11"].shape != bands_b["B11"].shape:
                raise ChangeDetectionError(
                    f"Band shape mismatch: Date A {bands_a['B11'].shape} vs Date B {bands_b['B11'].shape}"
                )

            # Calculate NDBI for both dates using B11 (SWIR) and B8 (NIR)
            ndbi_a = self.index_calculator.calculate_ndbi(bands_a["B11"], bands_a["B8"])
            ndbi_b = self.index_calculator.calculate_ndbi(bands_b["B11"], bands_b["B8"])

            # Compute difference
            delta = ndbi_b - ndbi_a

            # Generate masks
            change_mask = np.abs(delta) > self.threshold
            development_mask = delta > self.threshold  # Urban development
            decline_mask = delta < -self.threshold  # Urban decline

            # Calculate change magnitude
            magnitude = np.abs(delta)

            return {
                "ndbi_a": ndbi_a,
                "ndbi_b": ndbi_b,
                "delta": delta,
                "change_mask": change_mask,
                "development_mask": development_mask,
                "decline_mask": decline_mask,
                "magnitude": magnitude,
                "change_type": "urban",
            }

        except Exception as e:
            logger.error(f"Urban change detection failed: {e}")
            raise ChangeDetectionError(f"Urban change detection failed: {e}")

    def detect_all_changes(
        self, bands_a: Dict[str, np.ndarray], bands_b: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Run all change detection algorithms and combine results.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B

        Returns:
            Dictionary with combined change detection results
        """
        try:
            logger.info("Running comprehensive change detection")

            # Run individual detectors
            veg_changes = self.detect_vegetation_change(bands_a, bands_b)
            water_changes = self.detect_water_change(bands_a, bands_b)
            urban_changes = self.detect_urban_change(bands_a, bands_b)

            # Combine change masks
            combined_mask = (
                veg_changes["change_mask"]
                | water_changes["change_mask"]
                | urban_changes["change_mask"]
            )

            # Calculate overall statistics
            total_changed_pixels = np.sum(combined_mask)
            total_pixels = combined_mask.size

            return {
                "vegetation": veg_changes,
                "water": water_changes,
                "urban": urban_changes,
                "combined_mask": combined_mask,
                "total_changed_pixels": int(total_changed_pixels),
                "total_pixels": int(total_pixels),
                "change_percentage": (total_changed_pixels / total_pixels) * 100,
                "change_type": "all",
            }

        except Exception as e:
            logger.error(f"Comprehensive change detection failed: {e}")
            raise ChangeDetectionError(f"Comprehensive change detection failed: {e}")

    def classify_changes(self, change_results: Dict[str, Any]) -> np.ndarray:
        """Classify detected changes into categorical types.

        Args:
            change_results: Output from detect_all_changes()

        Returns:
            Integer classification array where:
                0 = No change
                1 = Vegetation growth
                2 = Vegetation loss (deforestation)
                3 = Water expansion (flooding)
                4 = Water reduction (drought)
                5 = Urban development
                6 = Urban decline
                7 = Ambiguous change (multiple indices triggered)
        """
        try:
            height, width = change_results["combined_mask"].shape
            classification = np.zeros((height, width), dtype=np.uint8)

            # Get individual masks
            veg_growth = change_results["vegetation"]["growth_mask"]
            veg_loss = change_results["vegetation"]["loss_mask"]
            water_expand = change_results["water"]["expansion_mask"]
            water_reduce = change_results["water"]["reduction_mask"]
            urban_dev = change_results["urban"].get(
                "development_mask", np.zeros((height, width), dtype=bool)
            )
            urban_dec = change_results["urban"].get(
                "decline_mask", np.zeros((height, width), dtype=bool)
            )

            # Compute ambiguous mask FIRST: pixels where 2+ change categories triggered
            veg_mask = change_results["vegetation"]["change_mask"]
            water_mask = change_results["water"]["change_mask"]
            urban_mask = change_results["urban"]["change_mask"]
            ambiguous = (
                veg_mask.astype(int) + water_mask.astype(int) + urban_mask.astype(int)
            ) >= 2

            # Assign individual classes ONLY where NOT ambiguous
            not_ambiguous = ~ambiguous
            classification[veg_growth & not_ambiguous] = 1
            classification[veg_loss & not_ambiguous] = 2
            classification[water_expand & not_ambiguous] = 3
            classification[water_reduce & not_ambiguous] = 4
            classification[urban_dev & not_ambiguous] = 5
            classification[urban_dec & not_ambiguous] = 6

            # Assign ambiguous last
            classification[ambiguous] = 7

            return classification

        except Exception as e:
            logger.error(f"Change classification failed: {e}")
            raise ChangeDetectionError(f"Change classification failed: {e}")

    def compute_change_statistics(
        self, classification: np.ndarray, pixel_area_m2: float = 100.0
    ) -> Dict[str, Any]:
        """Calculate summary statistics from classification map.

        Args:
            classification: Integer classification array from classify_changes()
            pixel_area_m2: Area per pixel in square meters (10m resolution = 100m²)

        Returns:
            Dictionary with statistics
        """
        try:
            total_pixels = classification.size

            # Define class names
            class_names = {
                0: "no_change",
                1: "vegetation_growth",
                2: "vegetation_loss",
                3: "water_expansion",
                4: "water_reduction",
                5: "urban_development",
                6: "urban_decline",
                7: "ambiguous",
            }

            stats = {}
            for class_id, class_name in class_names.items():
                count = np.sum(classification == class_id)
                percent = (count / total_pixels) * 100
                area_km2 = (count * pixel_area_m2) / 1e6

                stats[class_name] = {
                    "pixels": int(count),
                    "percent": round(percent, 2),
                    "area_km2": round(area_km2, 4),
                }

            # Total change percentage (all non-zero classes)
            changed_pixels = np.sum(classification > 0)
            stats["total_change"] = {
                "pixels": int(changed_pixels),
                "percent": round((changed_pixels / total_pixels) * 100, 2),
                "area_km2": round((changed_pixels * pixel_area_m2) / 1e6, 4),
            }

            # Add change type breakdown
            stats["change_types"] = {}
            for change_type in ["vegetation", "water", "urban"]:
                type_pixels = 0
                type_area = 0.0

                if change_type == "vegetation":
                    type_pixels = (
                        stats["vegetation_growth"]["pixels"]
                        + stats["vegetation_loss"]["pixels"]
                    )
                    type_area = (
                        stats["vegetation_growth"]["area_km2"]
                        + stats["vegetation_loss"]["area_km2"]
                    )
                elif change_type == "water":
                    type_pixels = (
                        stats["water_expansion"]["pixels"]
                        + stats["water_reduction"]["pixels"]
                    )
                    type_area = (
                        stats["water_expansion"]["area_km2"]
                        + stats["water_reduction"]["area_km2"]
                    )
                elif change_type == "urban":
                    type_pixels = (
                        stats["urban_development"]["pixels"]
                        + stats["urban_decline"]["pixels"]
                    )
                    type_area = (
                        stats["urban_development"]["area_km2"]
                        + stats["urban_decline"]["area_km2"]
                    )

                stats["change_types"][change_type] = {
                    "pixels": int(type_pixels),
                    "percent": round((type_pixels / total_pixels) * 100, 2),
                    "area_km2": round(type_area, 4),
                }

            return stats

        except Exception as e:
            logger.error(f"Statistics computation failed: {e}")
            raise ChangeDetectionError(f"Statistics computation failed: {e}")

    def detect_changes_by_type(
        self,
        bands_a: Dict[str, np.ndarray],
        bands_b: Dict[str, np.ndarray],
        change_type: str,
    ) -> Dict[str, Any]:
        """Detect changes for a specific change type.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B
            change_type: Type of change to detect ('vegetation', 'water', 'urban')

        Returns:
            Dictionary with change detection results
        """
        try:
            logger.info(f"Detecting {change_type} changes")

            if change_type == "vegetation":
                return self.detect_vegetation_change(bands_a, bands_b)
            elif change_type == "water":
                return self.detect_water_change(bands_a, bands_b)
            elif change_type == "urban":
                return self.detect_urban_change(bands_a, bands_b)
            else:
                raise ValueError(f"Unknown change type: {change_type}")

        except Exception as e:
            logger.error(f"Change detection for {change_type} failed: {e}")
            raise ChangeDetectionError(
                f"Change detection for {change_type} failed: {e}"
            )

    def get_change_summary(
        self,
        bands_a: Dict[str, np.ndarray],
        bands_b: Dict[str, np.ndarray],
        change_type: str = "all",
    ) -> Dict[str, Any]:
        """Get a summary of detected changes.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B
            change_type: Type of change to detect ('vegetation', 'water', 'urban', 'all')

        Returns:
            Dictionary with change summary
        """
        try:
            if change_type == "all":
                change_results = self.detect_all_changes(bands_a, bands_b)
                classification = self.classify_changes(change_results)
                stats = self.compute_change_statistics(classification)

                return {
                    "change_type": "all",
                    "change_results": change_results,
                    "classification": classification,
                    "statistics": stats,
                    "summary": self._generate_summary_text(stats),
                }
            else:
                change_results = self.detect_changes_by_type(
                    bands_a, bands_b, change_type
                )
                stats = self._compute_single_type_stats(change_results)

                return {
                    "change_type": change_type,
                    "change_results": change_results,
                    "statistics": stats,
                    "summary": self._generate_single_type_summary_text(
                        change_type, stats
                    ),
                }

        except Exception as e:
            logger.error(f"Change summary generation failed: {e}")
            raise ChangeDetectionError(f"Change summary generation failed: {e}")

    def _compute_single_type_stats(
        self, change_results: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compute statistics for a single change type.

        Args:
            change_results: Change detection results for a single type

        Returns:
            Dictionary with statistics
        """
        try:
            change_mask = change_results["change_mask"]
            total_pixels = change_mask.size
            changed_pixels = np.sum(change_mask)

            stats = {
                "total_pixels": int(total_pixels),
                "changed_pixels": int(changed_pixels),
                "change_percentage": round((changed_pixels / total_pixels) * 100, 2),
                "change_type": change_results["change_type"],
            }

            # Add specific statistics based on change type
            if change_results["change_type"] == "vegetation":
                stats["growth_pixels"] = int(np.sum(change_results["growth_mask"]))
                stats["loss_pixels"] = int(np.sum(change_results["loss_mask"]))
            elif change_results["change_type"] == "water":
                stats["expansion_pixels"] = int(
                    np.sum(change_results["expansion_mask"])
                )
                stats["reduction_pixels"] = int(
                    np.sum(change_results["reduction_mask"])
                )
            elif change_results["change_type"] == "urban":
                if "development_mask" in change_results:
                    stats["development_pixels"] = int(
                        np.sum(change_results["development_mask"])
                    )
                if "decline_mask" in change_results:
                    stats["decline_pixels"] = int(
                        np.sum(change_results["decline_mask"])
                    )

            return stats

        except Exception as e:
            logger.error(f"Single type statistics computation failed: {e}")
            raise ChangeDetectionError(
                f"Single type statistics computation failed: {e}"
            )

    def _generate_summary_text(self, stats: Dict[str, Any]) -> str:
        """Generate human-readable summary text.

        Args:
            stats: Statistics dictionary

        Returns:
            Summary text
        """
        try:
            summary_lines = [
                f"Total area changed: {stats['total_change']['percent']}%",
                f"Vegetation changes: {stats['change_types']['vegetation']['percent']}% (growth: {stats['vegetation_growth']['percent']}%, loss: {stats['vegetation_loss']['percent']}%)",
                f"Water changes: {stats['change_types']['water']['percent']}% (expansion: {stats['water_expansion']['percent']}%, reduction: {stats['water_reduction']['percent']}%)",
                f"Urban changes: {stats['change_types']['urban']['percent']}% (development: {stats['urban_development']['percent']}%, decline: {stats['urban_decline']['percent']}%)",
            ]

            return "\n".join(summary_lines)

        except Exception as e:
            logger.error(f"Summary text generation failed: {e}")
            return "Summary generation failed"

    def _generate_single_type_summary_text(
        self, change_type: str, stats: Dict[str, Any]
    ) -> str:
        """Generate summary text for single change type.

        Args:
            change_type: Type of change
            stats: Statistics dictionary

        Returns:
            Summary text
        """
        try:
            if change_type == "vegetation":
                summary_lines = [
                    f"Vegetation changes detected: {stats['change_percentage']}%",
                    f"Vegetation growth: {stats['growth_pixels']} pixels",
                    f"Vegetation loss: {stats['loss_pixels']} pixels",
                ]
            elif change_type == "water":
                summary_lines = [
                    f"Water changes detected: {stats['change_percentage']}%",
                    f"Water expansion: {stats['expansion_pixels']} pixels",
                    f"Water reduction: {stats['reduction_pixels']} pixels",
                ]
            elif change_type == "urban":
                summary_lines = [
                    f"Urban changes detected: {stats['change_percentage']}%",
                    f"Urban development: {stats.get('development_pixels', 0)} pixels",
                    f"Urban decline: {stats.get('decline_pixels', 0)} pixels",
                ]
            else:
                summary_lines = [f"Unknown change type: {change_type}"]

            return "\n".join(summary_lines)

        except Exception as e:
            logger.error(f"Single type summary text generation failed: {e}")
            return f"Summary generation failed for {change_type}"
