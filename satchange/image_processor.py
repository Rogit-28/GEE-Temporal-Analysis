"""
Image processing pipeline for SatChange.

This module handles image preprocessing, including cloud masking,
coregistration, and radiometric normalization.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Exception raised for image processing errors."""

    pass


class ImageProcessor:
    """Image preprocessing pipeline for satellite imagery."""

    def __init__(self, config):
        """Initialize image processor.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.cloud_threshold = config.get("cloud_threshold", 20)

    def _resample_b11_to_10m(
        self, bands: Dict[str, np.ndarray], reference_band: str = "B4"
    ) -> Dict[str, np.ndarray]:
        """Resample B11 band from 20m to 10m resolution to match other bands.

        Sentinel-2 B11 (SWIR) is natively 20m resolution while B4/B3/B8 are 10m.
        This method upsamples B11 using bilinear interpolation to match the
        spatial resolution of the 10m bands.

        Args:
            bands: Dictionary of band arrays
            reference_band: Name of a 10m band to use as reference for target shape

        Returns:
            Dictionary of band arrays with B11 resampled to 10m
        """
        if "B11" not in bands:
            logger.debug("B11 band not present, skipping resampling")
            return bands

        # Get reference shape from a 10m band (B4, B3, or B8)
        reference_shape = None
        for ref_band in [reference_band, "B4", "B3", "B8"]:
            if ref_band in bands:
                reference_shape = bands[ref_band].shape
                logger.debug(
                    f"Using {ref_band} as reference for B11 resampling (shape: {reference_shape})"
                )
                break

        if reference_shape is None:
            logger.warning("No 10m reference band found for B11 resampling")
            return bands

        b11 = bands["B11"]

        # Check if resampling is needed (B11 at 20m should be ~half the size of 10m bands)
        if b11.shape == reference_shape:
            logger.debug("B11 already matches reference shape, skipping resampling")
            return bands

        try:
            # Calculate zoom factors for resampling
            zoom_factors = (
                reference_shape[0] / b11.shape[0],
                reference_shape[1] / b11.shape[1],
            )

            logger.info(
                f"Resampling B11 from {b11.shape} to {reference_shape} (zoom factors: {zoom_factors})"
            )

            # Use scipy.ndimage.zoom with bilinear interpolation (order=1)
            # order=1 is bilinear, order=3 is bicubic
            b11_resampled = zoom(b11, zoom_factors, order=3, mode="nearest")

            # Clamp to original value range to prevent bicubic overshoot/ringing
            b11_resampled = np.clip(b11_resampled, b11.min(), b11.max())

            # Ensure exact shape match (zoom may produce slightly different dimensions due to rounding)
            if b11_resampled.shape != reference_shape:
                logger.warning(
                    f"B11 resampled shape {b11_resampled.shape} differs from reference {reference_shape}, "
                    f"cropping/padding to match"
                )
                # Crop or pad to match exactly
                b11_final = np.zeros(reference_shape, dtype=b11.dtype)
                min_h = min(b11_resampled.shape[0], reference_shape[0])
                min_w = min(b11_resampled.shape[1], reference_shape[1])
                b11_final[:min_h, :min_w] = b11_resampled[:min_h, :min_w]
                b11_resampled = b11_final

            # Preserve original dtype
            b11_resampled = b11_resampled.astype(b11.dtype)

            # Update bands dictionary
            bands_resampled = bands.copy()
            bands_resampled["B11"] = b11_resampled

            logger.info(
                f"B11 resampling completed: {b11.shape} -> {b11_resampled.shape}"
            )
            return bands_resampled

        except Exception as e:
            logger.error(f"Failed to resample B11: {e}")
            return bands

    def preprocess_image_pair(
        self,
        bands_a: Dict[str, np.ndarray],
        bands_b: Dict[str, np.ndarray],
        metadata_a: Dict[str, Any],
        metadata_b: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Preprocess a pair of satellite images.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B
            metadata_a: Metadata for Date A
            metadata_b: Metadata for Date B

        Returns:
            Tuple of (processed_bands_a, processed_bands_b)

        Raises:
            ImageProcessingError: If preprocessing fails
        """
        try:
            logger.info("Starting image preprocessing pipeline")

            # Step 1: Resample B11 from 20m to 10m resolution
            logger.info("Resampling B11 to 10m resolution...")
            bands_a = self._resample_b11_to_10m(bands_a)
            bands_b = self._resample_b11_to_10m(bands_b)

            # Step 2: Cloud masking
            logger.info("Applying cloud masking...")
            bands_a, bands_b = self._apply_cloud_masking(bands_a, bands_b)

            # Step 3: Coregistration check
            logger.info("Checking coregistration...")
            self._check_coregistration(metadata_a, metadata_b)

            # Step 4: Radiometric normalization (if needed)
            logger.info("Applying radiometric normalization...")
            bands_a, bands_b = self._apply_radiometric_normalization(bands_a, bands_b)

            logger.info("Image preprocessing completed successfully")
            return bands_a, bands_b

        except Exception as e:
            error_msg = f"Image preprocessing failed: {e}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg)

    def _apply_cloud_masking(
        self, bands_a: Dict[str, np.ndarray], bands_b: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Apply cloud masking to image pairs.

        Args:
            bands_a: Band arrays for Date A
            bands_b: Band arrays for Date B

        Returns:
            Tuple of (cloud_masked_bands_a, cloud_masked_bands_b)
        """
        try:
            # Extract QA60 band for cloud detection
            qa60_a = bands_a.get("QA60")
            qa60_b = bands_b.get("QA60")

            if qa60_a is None or qa60_b is None:
                logger.warning("QA60 band not available, skipping cloud masking")
                return bands_a, bands_b

            # Create cloud masks
            cloud_mask_a = self._create_cloud_mask(qa60_a)
            cloud_mask_b = self._create_cloud_mask(qa60_b)

            # Calculate cloud coverage (cloud_mask: 1=clear, 0=cloudy)
            # So cloud coverage = percentage of cloudy pixels = sum(mask==0) / size
            cloud_coverage_a = (np.sum(cloud_mask_a == 0) / cloud_mask_a.size) * 100
            cloud_coverage_b = (np.sum(cloud_mask_b == 0) / cloud_mask_b.size) * 100

            logger.info(
                f"Cloud coverage - Date A: {cloud_coverage_a:.1f}%, Date B: {cloud_coverage_b:.1f}%"
            )

            # Apply cloud masks to all bands
            # Note: For change detection, we keep the original data and just track cloud coverage
            # The cloud mask can be used later to exclude cloudy pixels from analysis if needed
            bands_a_masked = {}
            bands_b_masked = {}

            for band_name, band_array in bands_a.items():
                if band_name != "QA60":  # Don't mask the QA band itself
                    # np.where(condition, value_if_true, value_if_false)
                    # cloud_mask: 1=clear, 0=cloudy
                    # Keep clear pixels, set cloudy pixels to 0
                    bands_a_masked[band_name] = np.where(
                        cloud_mask_a == 1, band_array, 0
                    )
                else:
                    bands_a_masked[band_name] = band_array

            for band_name, band_array in bands_b.items():
                if band_name != "QA60":
                    bands_b_masked[band_name] = np.where(
                        cloud_mask_b == 1, band_array, 0
                    )
                else:
                    bands_b_masked[band_name] = band_array

            # Check if cloud coverage exceeds threshold
            if cloud_coverage_a > self.cloud_threshold:
                logger.warning(
                    f"Date A cloud coverage ({cloud_coverage_a:.1f}%) exceeds threshold ({self.cloud_threshold}%)"
                )

            if cloud_coverage_b > self.cloud_threshold:
                logger.warning(
                    f"Date B cloud coverage ({cloud_coverage_b:.1f}%) exceeds threshold ({self.cloud_threshold}%)"
                )

            return bands_a_masked, bands_b_masked

        except Exception as e:
            logger.error(f"Cloud masking failed: {e}")
            # Return original bands if masking fails
            return bands_a, bands_b

    def _create_cloud_mask(self, qa60_band: np.ndarray) -> np.ndarray:
        """Create cloud mask from QA60 band.

        Args:
            qa60_band: QA60 band array

        Returns:
            Binary cloud mask (1 = clear, 0 = cloudy)
        """
        try:
            # QA60 band contains cloud and cirrus information in bit masks
            # Bit 10: Opaque clouds (1024)
            # Bit 11: Cirrus clouds (2048)

            # Create cloud mask using bitwise operations
            cloud_bit = 1 << 10  # 1024 - opaque clouds
            cirrus_bit = 1 << 11  # 2048 - cirrus clouds

            # Check if either cloud or cirrus bit is set
            qa60_int = qa60_band.astype(np.int32)
            cloud_present = ((qa60_int & cloud_bit) != 0) | (
                (qa60_int & cirrus_bit) != 0
            )

            # Clear pixels = 1, cloudy pixels = 0
            cloud_mask = (~cloud_present).astype(np.uint8)

            return cloud_mask

        except Exception as e:
            logger.error(f"Failed to create cloud mask: {e}")
            # Return all-clear mask if creation fails
            return np.ones_like(qa60_band, dtype=np.uint8)

    def _check_coregistration(
        self, metadata_a: Dict[str, Any], metadata_b: Dict[str, Any]
    ) -> None:
        """Check if images are properly coregistered.

        Args:
            metadata_a: Metadata for Date A
            metadata_b: Metadata for Date B

        Raises:
            ImageProcessingError: If images are not properly coregistered
        """
        try:
            # Check if images have the same dimensions
            width_a = metadata_a.get("width")
            height_a = metadata_a.get("height")
            width_b = metadata_b.get("width")
            height_b = metadata_b.get("height")

