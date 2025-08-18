"""
Change detection engine for SatChange.

This module implements the core change detection algorithms using spectral indices
to identify temporal changes in satellite imagery.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
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

